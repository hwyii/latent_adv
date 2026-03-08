# src/training/reft_adv_trainer.py
from typing import Optional, Dict, Any
from sympy import N, n_order
import torch
import pyvene as pv
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from pyreft.reft_trainer import ReftTrainerForSequenceClassification, make_dataloader
from transformers.trainer_utils import EvalPrediction, has_length, denumpify_detensorize
from transformers.utils import logging
from src.attack.inner_attack import AttackConfig
from src.models.adv_intervention import AdversarialIntervention
from src.circuit.circuit_mask import load_circuit_mask, mask_summary
from src.models.circuitable_attention import make_circuitable_neox, set_circuit_on_model

logger = logging.get_logger(__name__)

class ReftAdversarialTrainerForSequenceClassification(ReftTrainerForSequenceClassification):
    def __init__(self, *args, attack_config: Optional[AttackConfig] = None, **kwargs):
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
        """初始化 Circuit Gate"""
        self.circuit_spec = None
        self._base_model = None
        self._circuit_mask_cpu = None

        cfg = self.attack_config
        if cfg and getattr(cfg, "use_circuit_gate", False):
            if not getattr(cfg, "circuit_path", None):
                raise ValueError("use_circuit_gate=True but circuit_path is None")

            base = self.model.model
            self._base_model = base
            
            # 支持多种结构
            
            if not hasattr(base, "gpt_neox"):
                raise NotImplementedError("Currently only Pythia/GPT-NeoX is supported for circuit gate.")
            config = base.gpt_neox.config
            num_layers = len(base.gpt_neox.layers)
            num_heads = config.num_attention_heads
            
            # 加载 Mask (格式：configs/circuits/pythia-410m/Helpful_Patching_top20.json)
            spec = load_circuit_mask(
                json_path=cfg.circuit_path,
                num_layers=num_layers,
                num_heads=num_heads,
                device=None
            )
            total, per_layer = mask_summary(spec.mask)
            logger.info(f"[Circuit] Loaded {cfg.circuit_path} | Heads: {total}")

            self.circuit_spec = spec
            self._circuit_mask_cpu = spec.mask.detach().cpu()

            # 安装 Hook, 默认关闭状态
            make_circuitable_neox(base)

    def compute_loss(self, intervenable, inputs, return_outputs=False):
        
        # ###################### [debug]
        # lens = inputs["attention_mask"].sum(dim=1)  # [B]
        # loc = inputs["intervention_locations"]      # 形状一般 [B, n_rep, n_loc] 或 [B, 1, last_n]
        # B, S = inputs["input_ids"].shape

        # # 取第一组 rep 的位置（通常就是 0）
        # loc0 = loc[:, 0, :]  # [B, n_loc]

        # bad = (loc0 >= lens.unsqueeze(1)).any().item()
        # print(f"[LOC] B={B} S={S} lens(min/mean/max)={int(lens.min())}/{float(lens.float().mean()):.1f}/{int(lens.max())}")
        # print(f"[LOC] loc shape={tuple(loc.shape)} loc0 min/max={int(loc0.min())}/{int(loc0.max())}  any_loc_outside_len={bad}")

        # # 看前 3 条样本：最后一个有效 token index = len-1
        # for i in range(min(3, B)):
        #     last = int(lens[i].item()) - 1
        #     print(f"[LOC] i={i} len={int(lens[i])} last_idx={last} loc0_tail={loc0[i,-10:].tolist()}")
        # ########################
        
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
        
        #################### [debug]
        with torch.no_grad():
            cl = clean_loss.item()
            print(f"[LOSS] clean_loss={cl:.4f} logits_mean={cf_outputs.logits.float().mean().item():.4f} logits_std={cf_outputs.logits.float().std().item():.4f}")
        #################

        # 3. Inner Attack Loop
        cfg = self.attack_config
        if cfg and cfg.inner_attack != "none":
            self._run_inner_attack(intervenable, inputs, unit_locations, inputs["labels"])

            # 4. Final Adversarial Forward (Update ReFT params)
            if self.adv_intervention.delta is not None:
                delta_tensor_for_log = self.adv_intervention.delta.detach()
                K = int(delta_tensor_for_log.shape[1])

                self.adv_intervention.delta = delta_tensor_for_log  # 原来 detach 的逻辑
                intervenable.zero_grad()
            
                
                _, cf_final = intervenable(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations=unit_locations,
                    labels=inputs["labels"]
                )
                adv_loss = self._calculate_standard_loss(cf_final.logits, inputs["labels"])
                
                self.adv_intervention.reset_delta()
                total_loss = clean_loss + float(cfg.lambda_adv) * adv_loss
                
                #################### [debug]
                with torch.no_grad():
                    al = adv_loss.item()
                    print(f"[LOSS] adv_loss={al:.4f} (ratio={al/(cl+1e-8):.3f}) logits_mean={cf_final.logits.float().mean().item():.4f} logits_std={cf_final.logits.float().std().item():.4f}")
                ##################                
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
            
            self.log(log_dict)

        return (total_loss, cf_outputs) if return_outputs else total_loss

    def _run_inner_attack(self, intervenable, inputs, unit_locations, labels):
        """执行内部攻击循环 (GCG/PGD)"""
        cfg = self.attack_config
        
        # A. Random Noise (No Loop)
        if cfg.inner_attack == "random_noise":
            current_delta = self.adv_intervention.delta
            if current_delta is not None:
                eps = float(cfg.eps)
                noise = torch.rand_like(current_delta) * 2 * eps - eps
                with torch.no_grad():
                    current_delta.add_(noise).clamp_(-eps, eps)
            return

        # B. Gradient-based Attack (Loop)
        steps = int(cfg.steps)
        eps = float(cfg.eps)
        step_size = 1 * (eps / max(1, steps)) 
        self.adv_intervention.reset_delta()

        # Gate Control
        gate_mode = getattr(cfg, "gate_mode", "inner_only")
        do_stopgrad = (self.circuit_spec is not None and gate_mode in ("inner_only", "inner+final"))
        
        if do_stopgrad:
            set_circuit_on_model(self._base_model, self._circuit_mask_cpu, enabled=True)

        try:
            for t in range(steps):
                intervenable.zero_grad()
                
                # Forward
                _, cf_adv = intervenable(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations=unit_locations,
                    labels=labels,
                )
                loss = self._calculate_standard_loss(cf_adv.logits, labels)

                # Backward
                loss.backward()
                    
                if t == 0: 
                    initial_loss = loss.item()
                self._update_delta_grad(cfg, step_size, eps)
                # === 记录 delta 尺度：建议每步都记录轻量版 ===
                with torch.no_grad():
                    d = self.adv_intervention.delta
                    K = d.shape[1]; H = d.shape[2]
                    l2 = d.view(d.size(0), -1).norm(p=2, dim=1).mean().item()
                    l2_per_tok = l2 / ((K * H) ** 0.5)
                    absmax = d.abs().max().item()

                # 每步都 log（如果担心 wandb 卡，就每隔几步 log）
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

                # 最后一轮：记录 gain（用 initial_loss）
                if t == steps - 1:
                    final_loss = loss.item()
                    self.log({
                        "train/attack_absolute_gain": final_loss - initial_loss,
                        "train/attack_relative_gain": final_loss / (initial_loss + 1e-8),
                        "train/attack_gain_per_tok": (final_loss - initial_loss) / max(K, 1),
                    })
                
        finally:
            # 无论攻击过程中发生什么，都确保最后关闭电路门，恢复正常梯度流
            if do_stopgrad:
                set_circuit_on_model(self._base_model, None, enabled=False)

    def _update_delta_grad(self, cfg, step_size, eps):
        delta = self.adv_intervention.delta
        if delta is None or delta.grad is None: 
            print("[DELTA] grad is None")
            return

        with torch.no_grad():
            g = delta.grad
            
            ################## [debug]
            g_abs = g.abs()
            print(f"[DELTA] shape={tuple(delta.shape)} "
                f"g_abs_mean={g_abs.mean().item():.3e} g_abs_max={g_abs.max().item():.3e} "
                f"g_nonzero={(g_abs>0).float().mean().item():.4f} "
                f"delta_absmax_before={delta.abs().max().item():.4f}")
            ###################
            
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
            #################### [debug]
            print(f"[DELTA] delta_absmax_after={delta.abs().max().item():.4f} eps={eps:.4f} step={step_size:.4f}")
            ###################
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
