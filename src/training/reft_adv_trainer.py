# src/training/reft_adv_trainer.py
from typing import Optional, Dict, Any
import torch
import copy, os, json
import pyvene as pv
import wandb
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from pyreft.reft_trainer import ReftTrainerForSequenceClassification, make_dataloader
from transformers.trainer_utils import EvalPrediction, has_length, denumpify_detensorize
from transformers.utils import logging
from src.attack.inner_attack import AttackConfig
from src.models.adv_intervention import AdversarialIntervention
from src.circuit.circuit_mask import load_circuit_mask
from src.circuit.surrogate_builder import build_surrogate_model

def get_inactive_heads_dict(circuit_mask_tensor):
    """把你的 0/1 Mask 转换成 Surrogate Builder 需要的格式：要删掉的 Heads"""
    inactive_dict = {}
    num_layers = circuit_mask_tensor.shape[0]
    for l in range(num_layers):
        # 找到值为 0 的索引（即不活跃的、需要被剪枝的 heads）
        inactive = torch.where(circuit_mask_tensor[l] == 0)[0].tolist()
        if inactive:
            inactive_dict[l] = inactive
    return inactive_dict

logger = logging.get_logger(__name__)

class ReftAdversarialTrainerForSequenceClassification(ReftTrainerForSequenceClassification):
    def __init__(self,
                 *args,
                 attack_config: Optional[AttackConfig] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config

        self._setup_adversarial()
        self._setup_circuit()
    
    def _setup_adversarial(self):
        """初始化对抗攻击模块"""
        self.adv_intervention = None
        for k, v in self.model.interventions.items():
            if isinstance(v, AdversarialIntervention):
                self.adv_intervention = v
                break
        
        if self.attack_config and self.attack_config.inner_attack != "none":
            if self.adv_intervention is None:
                raise RuntimeError("Attack enabled but no AdversarialIntervention found!")
            logger.info(f"[AdvTrainer] Attack Enabled: {self.attack_config.inner_attack}")
        else:
            logger.info("[AdvTrainer] Attack Disabled.")
    
    def _setup_circuit(self):
        """初始化 Surrogate 代理模型机制"""
        self.circuit_spec = None
        self.surrogate_intervenable = None
        self.surrogate_adv = None

        cfg = self.attack_config
        if cfg and getattr(cfg, "use_circuit_gate", False):
            base = self.model.model  # 这是 HF 的底层 GPT-2
            config = base.config
            
            # ==========================================================
            # 🚨 新增：Bypass 纯净沙箱模式拦截
            # 如果 yaml 里把 circuit_path 设为 "bypass" 或 "none"
            # ==========================================================
            if str(getattr(cfg, "circuit_path", "")).lower() in ["bypass", "none", "null"]:
                logger.info("[Surrogate] 🚨 触发纯净沙箱 (Bypass) 模式！不加载任何 JSON，不执行任何剪枝！")
                
                # 1. 纯粹的物理深拷贝
                surrogate_base = copy.deepcopy(base)
                
                # 2. 清理原模型的遗留 Hook (必须做，否则后续 pyvene 包装会出 bug)
                def clear_all_hooks(module):
                    module._forward_hooks.clear()
                    module._forward_pre_hooks.clear()
                    module._backward_hooks.clear()
                    for child in module.children():
                        clear_all_hooks(child)
                clear_all_hooks(surrogate_base)
                
                # 3. 精度对齐
                surrogate_base = surrogate_base.to(base.dtype)
                
            else:
                # ==========================================================
                # 原来的正常加载和剪枝逻辑 (保留不动)
                # ==========================================================
                spec = load_circuit_mask(
                    json_path=cfg.circuit_path,
                    num_layers=config.n_layer,
                    num_heads=config.n_head,
                    device=None
                )
                self.circuit_spec = spec
                inactive_heads = get_inactive_heads_dict(spec.mask)
                
                logger.info(f"[Surrogate] use_circuit_gate = {getattr(cfg, 'use_circuit_gate', False)}")
                
                mlp_mask_dict = {}
                if getattr(cfg, "mlp_mask_path", None) and str(cfg.mlp_mask_path).lower() not in ["none", "null"]:
                    if os.path.exists(cfg.mlp_mask_path):
                        with open(cfg.mlp_mask_path, "r") as f:
                            mlp_mask_dict = json.load(f)
                
                mlp_ratio = getattr(cfg, "mlp_keep_ratio", 1.0)
                
                from src.circuit.surrogate_builder import build_surrogate_model
                surrogate_base = build_surrogate_model(
                    base, 
                    inactive_heads, 
                    mlp_keep_ratio=mlp_ratio,
                    mlp_mask_dict=mlp_mask_dict
                )
            
            # ==========================================================
            # 后续挂载 Pyvene 的逻辑 (两边共用)
            # ==========================================================
            surrogate_config = copy.deepcopy(self.model.config)
            self.surrogate_intervenable = pv.IntervenableModel(surrogate_config, surrogate_base)

            for k, v in self.surrogate_intervenable.interventions.items():
                if isinstance(v, AdversarialIntervention):
                    self.surrogate_adv = v
                    break
                    
            if self.surrogate_adv is None:
                raise RuntimeError("Failed to find AdversarialIntervention on Surrogate Model!")
    
    def compute_loss(self, intervenable, inputs, return_outputs=False):
        if not getattr(self, "_wandb_watch_registered", False):
            if self.adv_intervention is not None:
                import wandb
                wandb.watch(
                    self.adv_intervention,
                    log="all",
                    log_freq=20,
                    log_graph=False
                )
            self._wandb_watch_registered = True
        
        # 1. 数据预处理
        if inputs["labels"].dim() == 2: inputs["labels"] = inputs["labels"].squeeze(1)
        
        unit_locations = None
        if "intervention_locations" in inputs:
            unit_locations = {"sources->base": (None, inputs["intervention_locations"].permute(1, 0, 2).tolist())}
                
        # 2. Clean Forward
        if self.adv_intervention: self.adv_intervention.reset_delta()
        
        _, cf_outputs = intervenable(
            {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )
        clean_loss = self._calculate_standard_loss(cf_outputs.logits, inputs["labels"])
        
        total_loss = clean_loss
        adv_loss = None

        # 3. Inner Attack Loop
        cfg = self.attack_config
        
        if cfg and cfg.inner_attack != "none":
            # 梯度优化 (PGD/GCG 等)
            if cfg.inner_attack in ("latent_pgd", "latent_gcg_coord", "random_noise"):
                
                if getattr(cfg, "use_circuit_gate", False) and self.surrogate_intervenable is not None:
                    # >>> Surrogate 模式 <<<
                    
                    # 1. 同步防御参数 (Shield Sync)
                    with torch.no_grad():
                        for (k_b, v_b), (k_s, v_s) in zip(self.model.interventions.items(), self.surrogate_intervenable.interventions.items()):
                            if not isinstance(v_b, AdversarialIntervention):
                                v_s.load_state_dict(v_b.state_dict())
                    
                    # 2. 内环攻击 (在 Surrogate 上寻找 delta)
                    self._run_inner_attack(
                        target_intervenable=self.surrogate_intervenable, 
                        target_adv=self.surrogate_adv,
                        inputs=inputs, 
                        unit_locations=unit_locations, 
                        labels=inputs["labels"]
                    )
                    
                    # 3. 交接武器 (Weapon Transfer)
                    self.adv_intervention.delta = self.surrogate_adv.delta.detach().clone()
                    
                else:
                    # >>> 全量 Baseline 模式 <<<
                    self._run_inner_attack(
                        target_intervenable=intervenable, 
                        target_adv=self.adv_intervention,
                        inputs=inputs, 
                        unit_locations=unit_locations, 
                        labels=inputs["labels"]
                    )
            else:
                raise ValueError(f"Unsupported inner_attack: {cfg.inner_attack}")
            
            # 4. Final Adversarial Forward (Update ReFT params)
            if self.adv_intervention.delta is not None:
                delta_tensor_for_log = self.adv_intervention.delta.detach()
                K = int(delta_tensor_for_log.shape[1])

                self.adv_intervention.delta = delta_tensor_for_log  # 固定 delta
                intervenable.zero_grad()
                
                _, cf_final = intervenable(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations=unit_locations,
                    labels=inputs["labels"],
                    subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
                )
                adv_loss = self._calculate_standard_loss(cf_final.logits, inputs["labels"])
                
                self.adv_intervention.reset_delta()
                total_loss = clean_loss + float(cfg.lambda_adv) * adv_loss
                        
        # 优化后的 Logging 逻辑
        if self.state.global_step % self.args.logging_steps == 0 and self.args.process_index == 0:

            c_loss = clean_loss.item()
            log_dict = {"train/loss_clean": c_loss}
            
            if adv_loss is not None:
                a_loss = adv_loss.item()
                c_loss = clean_loss.item()
                
                # 1. 计算 Loss 维度的指标
                loss_gap = a_loss - c_loss
                loss_ratio = a_loss / (c_loss + 1e-8)
                
                # 2. 计算 Delta 维度的标准化指标 (剥离 batch 和 seq_len 的影响)
                delta_l2_mean = 0.0
                delta_linf = 0.0
                
                if adv_loss is not None and "delta_tensor_for_log" in locals():
                    dt = delta_tensor_for_log.float()
                    token_l2_norms = torch.norm(dt, p=2, dim=-1)
                    delta_l2_mean = token_l2_norms.mean().item()
                    delta_linf = torch.norm(dt, p=float('inf')).item()
                    K_for_ratio = K
                
                # 3. 记录到 WandB
                log_dict.update({
                    "train/loss_adv_final": a_loss,
                    "train/loss_total": total_loss.item(),
                    "train/adv_clean_ratio": float(loss_ratio),
                    "train/loss_gap": float(loss_gap),
                    "train/adv_clean_ratio_per_tok": float(loss_ratio) / max(K_for_ratio, 1),
                    "train/attack_gain_per_tok": (loss_gap) / max(K_for_ratio, 1),
                    "metrics/delta_per_token_L2": delta_l2_mean,
                    "metrics/delta_Linf_max": delta_linf,
                })
                
                n_dirty = sum(
                    1 for k, v in self.model.interventions.items()
                    if not isinstance(v, AdversarialIntervention)
                    for p in v.parameters()
                    if p.grad is not None
                )
                log_dict["debug/n_dirty_intervention_params"] = n_dirty
            
            self.log(log_dict)

        return (total_loss, cf_outputs) if return_outputs else total_loss

    def _run_inner_attack(self, target_intervenable, target_adv, inputs, unit_locations, labels):
        """纯净版的 PGD 内环：不包含任何软切片/断梯度逻辑"""
        target_intervenable.model.eval()
        cfg = self.attack_config

        if cfg.inner_attack == "random_noise":
            current_delta = target_adv.delta
            if current_delta is not None:
                eps = float(cfg.eps)
                noise = torch.rand_like(current_delta) * 2 * eps - eps
                with torch.no_grad():
                    current_delta.add_(noise).clamp_(-eps, eps)
            return


        # 内环开始：冻结所有非 delta 参数
        frozen_params = []
        for k, v in target_intervenable.interventions.items():
            if not isinstance(v, AdversarialIntervention):
                for p in v.parameters():
                    if p.requires_grad:
                        p.requires_grad_(False)
                        frozen_params.append(p)
        try:
            steps = int(cfg.steps)
            eps = float(cfg.eps)
            step_size = 1 * (eps / max(1, steps))
            target_adv.reset_delta()

            for t in range(steps):
                target_intervenable.zero_grad()

                subspaces = inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None

                _, cf_adv = target_intervenable(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations=unit_locations,
                    labels=labels,
                    subspaces=subspaces,
                )
                
                loss = self._calculate_standard_loss(cf_adv.logits, labels)
                loss.backward()

                if t == 0:
                    initial_loss = loss.item()
                self._update_delta_grad(cfg, step_size, eps, target_adv=target_adv)

                with torch.no_grad():
                    d = target_adv.delta
                    K = d.shape[1]
                    H = d.shape[2]
                    l2 = d.view(d.size(0), -1).norm(p=2, dim=1).mean().item()
                    l2_per_tok = l2 / ((K * H) ** 0.5)
                    absmax = d.abs().max().item()

                if t == 0 or t == steps - 1 or t % 2 == 0:
                    self.log({
                        "inner/step": t,
                        "inner/loss": loss.item(),
                        "inner/delta_absmax": absmax,
                        "inner/delta_l2_mean": l2,
                        "inner/delta_l2_per_tok": l2_per_tok,
                        "inner/K": K,
                        "inner/eps": float(eps),
                        "inner/step_size": float(step_size),
                        "inner/global_step": self.state.global_step
                    })

                if t == steps - 1:
                    final_loss = loss.item()
                    self.log({
                        "train/attack_absolute_gain": final_loss - initial_loss,
                        "train/attack_relative_gain": final_loss / (initial_loss + 1e-8),
                        "train/attack_gain_per_tok": (final_loss - initial_loss) / max(K, 1),
                    })
        finally:
            # 内环结束：无论是否出错，一定恢复梯度
            for p in frozen_params:
                p.requires_grad_(True)
            target_intervenable.model.train()
            
       
    def _update_delta_grad(self, cfg, step_size, eps, target_adv=None):
        adv_module = target_adv if target_adv is not None else self.adv_intervention
        delta = adv_module.delta
        if delta is None or delta.grad is None: 
            print("[DELTA] grad is None")
            return

        with torch.no_grad():
            g = delta.grad
            
            if cfg.inner_attack == "latent_pgd":
                delta.add_(step_size * g.sign())
            elif cfg.inner_attack == "latent_gcg_coord":
                # GCG Coordinate Update
                g_flat = g.view(-1)
                k = min(int(getattr(cfg, "gcg_topk", 32)), g_flat.numel())
                topk_idx = g_flat.abs().topk(k=k).indices
                
                delta_flat = delta.view(-1)
                delta_flat[topk_idx] += step_size * g_flat[topk_idx].sign()
            
            delta.clamp_(-eps, eps)
            delta.grad.zero_()
            delta.requires_grad_(True)

    def _calculate_standard_loss(self, logits, labels):
        # 统一 Loss 计算
        if self.model.model.config.problem_type == "regression" or self.model.model.num_labels == 1:
            return MSELoss()(logits.squeeze(), labels.squeeze().to(torch.bfloat16))
        return CrossEntropyLoss()(logits.view(-1, self.model.model.num_labels), labels.view(-1))

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        self.model.model.eval()
        for k, v in self.model.interventions.items():
            if isinstance(v, (list, tuple)):
                for mod in v:
                    mod.eval()
            else:
                v.eval()

        if getattr(self, "adv_intervention", None) is not None:
            self.adv_intervention.reset_delta()

        batch_size = self.args.eval_batch_size
        dataloader = make_dataloader(eval_dataset, batch_size, self.data_collator, shuffle=False)

        logger.info("***** Running In-Training Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {batch_size}")

        from tqdm.auto import tqdm
        eval_iterator = tqdm(dataloader, position=0, leave=True)

        all_preds, all_labels = [], []
        device = self.model.get_device()

        with torch.no_grad():
            for step, inputs in enumerate(eval_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)

                unit_locations = None
                if "intervention_locations" in inputs:
                    intervention_locations = inputs["intervention_locations"].permute(1, 0, 2).tolist()
                    unit_locations = {"sources->base": (None, intervention_locations)}

                if getattr(self, "adv_intervention", None) is not None:
                    self.adv_intervention.reset_delta()

                _, cf_outputs = self.model(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations=unit_locations,
                    labels=inputs["labels"],
                )

                all_preds.append(cf_outputs.logits)
                all_labels.append(inputs["labels"])

        all_preds = torch.cat(all_preds, dim=0).cpu().to(torch.float32)
        all_labels = torch.cat(all_labels, dim=0).cpu().to(torch.float32)

        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        metrics = denumpify_detensorize(metrics)

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
