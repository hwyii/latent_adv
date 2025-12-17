from typing import Optional, Dict, Any
import torch, pyreft, os, json
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import TrainingArguments
from pyreft.reft_trainer import ReftTrainerForSequenceClassification, make_dataloader
import pyvene as pv
from transformers.trainer_utils import EvalPrediction, has_length, denumpify_detensorize
from transformers.utils import logging
import pdb
logger = logging.get_logger(__name__)

from src.attack.inner_attack import AttackConfig
from src.models.adv_intervention import AdversarialIntervention


    
class ReftAdversarialTrainerForSequenceClassification(ReftTrainerForSequenceClassification):
    # 继承父类并重写compute_loss
    def __init__(self, *args, attack_config: Optional[AttackConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config
        # --- 找出我们刚才挂上的 AdversarialIntervention ---
        self.adv_intervention = None
        for k, v in self.model.interventions.items():
            if isinstance(v, AdversarialIntervention):
                self.adv_intervention = v
                print(f"[AdvTrainer] found AdversarialIntervention at key: {k}")
                break

        if self.adv_intervention is None and self.attack_config and self.attack_config.inner_attack != "none":
            raise RuntimeError("Attack enabled but no AdversarialIntervention found in reft_model.interventions")

        if self.attack_config and self.attack_config.inner_attack != "none":
            logger.info(f"[AdvTrainer] adversarial training enabled: {self.attack_config.inner_attack}")
        else:
            logger.info("[AdvTrainer] adversarial training disabled.")
        
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ):
        #print("[Debug] dim labels:", inputs["labels"].dim())
        if inputs["labels"].dim() == 2:  
            inputs["labels"] = inputs["labels"].squeeze(1) 
        #print("[Debug] labels after squeeze:", inputs["labels"]) 
        """
        重写 compute_loss：
        1) 复制 ReftTrainer 的 clean loss 逻辑
        2) 添加 latent adv loss
        """
        # clean forward (原始pyreft逻辑)
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
        # --------- 1. clean loss：确保 adv_intervention 不加扰动 ---------
        if self.adv_intervention is not None:
            self.adv_intervention.reset_delta()
        _, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )
        # classification loss on counterfactual labels
        logits = cf_outputs.logits
        labels = inputs["labels"]

        if self.model.model.config.problem_type is None:
            if self.model.model.num_labels == 1:
                problem_type = "regression"
            elif self.model.model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
        else:
            problem_type = self.model.model.config.problem_type
            
        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.model.model.num_labels == 1:
                clean_loss = loss_fct(logits.squeeze(), labels.squeeze().to(torch.bfloat16))
            else:
                clean_loss = loss_fct(logits, labels.to(torch.bfloat16))
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            clean_loss = loss_fct(logits.view(-1, self.model.model.num_labels), labels.view(-1))
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            clean_loss = loss_fct(logits, labels)
        total_loss = clean_loss
        adv_loss = None
        #pdb.set_trace()
        # --------- 2. latent adv loss：在 adv_intervention 上加扰动 ---------
        cfg = self.attack_config
        if cfg is not None and cfg.inner_attack != "none":
            steps = cfg.steps
            eps = cfg.eps
            step_size = 2.5 * (eps / max(1, steps)) # step size 设置为 eps 和 steps 比值 的 2.5 倍
        
            self.adv_intervention.reset_delta()

            # [新增] 准备一个列表来存当前 Batch 的攻击轨迹
            # 仅在需要记录的 step 开启，避免 I/O 瓶颈
            should_log_indices = (self.state.global_step % 10 == 0) # 每10个batch记录一次
            batch_attack_trace = []
            
            for t in range(steps):
                intervenable.zero_grad()

                # forward：此时 AdversarialIntervention.forward 会：
                #   - 发现 self.delta is None -> 初始化为 zeros_like(base) 可导
                #   - 下一步再 forward 时，沿用同一个 delta
                _, cf_adv = intervenable(
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    },
                    unit_locations=unit_locations,
                    labels=labels,
                )
                logits_adv = cf_adv.logits
                loss_step = CrossEntropyLoss()(
                logits_adv.view(-1, self.model.model.num_labels),
                    labels.view(-1)
                )

                loss_step.backward()

                # 关键：现在 graph 里的是 self.adv_intervention.delta，而不是外面一个局部 delta
                delta_tensor = self.adv_intervention.delta
                if delta_tensor is None or delta_tensor.grad is None:
                    print("[PGD][ERROR] delta.grad is None, skip adversarial step")
                    break
                
                #print(f"delta norm = {delta_tensor.norm().item():.4f}, step loss = {loss_step.item():.4f}")

                with torch.no_grad():
                    g = delta_tensor.grad
                    g_abs = g.abs().view(-1)
                    
                    top_vals, top_ids = g_abs.topk(50)
                    mean_val = g_abs.mean()
                    
                    if t == 0: # 只看第一步
                        print(f"Top-1 Gradient: {top_vals[0].item():.4f}")
                        print(f"Top-32 Gradient: {top_vals[31].item():.4f}")
                        print(f"Top-33 Gradient: {top_vals[32].item():.4f}") # 落选的第一名
                        print(f"Average Gradient: {mean_val.item():.4f}")
                        
                        # 简单的比率检查
                        ratio = top_vals[31] / (top_vals[32] + 1e-9)
                        print(f"Dominance Ratio (Top32 / Rest): {ratio.item():.2f}")
                    
                    if cfg.inner_attack == "latent_pgd":
                        print("[inner attack] using latent_pgd")
                        # PGD update, 所有坐标全维 sign update
                        delta_tensor.add_(step_size * g.sign())
                    elif cfg.inner_attack == "latent_gcg_coord":
                        # GCG coordinate update, 每次只更新|grad|最大的top-k维
                        print("[inner attack] using latent_gcg_coord")
                        g_flat = g.view(-1)
                        k = min(cfg.gcg_topk, g_flat.numel())
                        
                        topk_obj = g_flat.abs().topk(k=k)
                        topk_idx = topk_obj.indices
                        
                        # === [新增] 记录 indices ===
                        if should_log_indices:
                            # 必须转成 list 才能存 json
                            # 记录由: (inner_step, indices_list) 组成
                            ids_list = topk_idx.cpu().tolist()
                            batch_attack_trace.append({
                                "t": t,
                                "ids": ids_list
                            })
                        
                        delta_flat = delta_tensor.view(-1)
                        delta_flat[topk_idx] += step_size * g_flat[topk_idx].sign()
                        
                    else:
                        raise ValueError(f"Unknown inner_attack type: {cfg.inner_attack}")
                        
                    
                    delta_tensor.clamp_(-eps, eps)
                    delta_tensor.grad.zero_()
                    # 下一步继续对这个 delta 求梯度
                    delta_tensor.requires_grad_(True)
            
            if should_log_indices and len(batch_attack_trace) > 0:
                log_path = os.path.join(self.args.output_dir, "attack_neurons_log.jsonl")
                # 使用 'a' (append) 模式
                with open(log_path, "a") as f:
                    record = {
                        "global_step": self.state.global_step,
                        "trace": batch_attack_trace
                    }
                    f.write(json.dumps(record) + "\n")
            
            
            # 用最终的 delta 再 forward 一次，算真正的 adv_loss（对模型参数求梯度）
            if self.adv_intervention.delta is not None:
            # 防止再对 delta 求梯度：我们只想对参数求梯度
                self.adv_intervention.delta = self.adv_intervention.delta.detach()
                intervenable.zero_grad()

                _, cf_final = intervenable(
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    },
                    unit_locations=unit_locations,
                    labels=labels,
                )
                logits_final = cf_final.logits
                adv_loss = CrossEntropyLoss()(
                    logits_final.view(-1, self.model.model.num_labels),
                    labels.view(-1)
                )

                # 用完之后可以重置，避免下一 batch 残留
                self.adv_intervention.reset_delta()

                total_loss = clean_loss + cfg.lambda_adv * adv_loss

        if (self.state.is_world_process_zero
                and self.state.global_step > 0
                and self.state.global_step % self.args.logging_steps == 0):
                
                log_dict = {"loss_clean": clean_loss.item()}
                if adv_loss is not None:
                    log_dict["loss_adv"] = adv_loss.item()
                    log_dict["loss_total"] = total_loss.item()
                self.log(log_dict)

        if return_outputs:
            cf_outputs.loss = total_loss
            return (total_loss, cf_outputs)
        return total_loss
    
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix: str = "eval"):
        """
        重写 evaluate：评估前重置 adv_intervention 的 delta
        兼容 ReftConfig 里的 adversarial intervention
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        # base classifier eval 模式
        self.model.model.eval()
        for k, v in self.model.interventions.items():
            # 兼容两种结构：v 是单个 intervention 或 list[intervention]
            if isinstance(v, (list, tuple)):
                for mod in v:
                    mod.eval()
            else:
                v.eval()
        # 如果有 adversarial intervention，把 delta 清掉，防止 eval 时加扰动
        if getattr(self, "adv_intervention", None) is not None:
            self.adv_intervention.reset_delta()
            
        # -------- 1. 构建 dataloader --------
        batch_size = self.args.eval_batch_size
        data_collator = self.data_collator
        intervenable = self.model

        dataloader = make_dataloader(
            eval_dataset,
            batch_size,
            data_collator,
            shuffle=False
        )
        
        logger.info("***** Running In-Training Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")
        from tqdm.auto import tqdm
        eval_iterator = tqdm(dataloader, position=0, leave=True)

        all_preds = []
        all_labels = []

        device = self.model.get_device()

        with torch.no_grad():
            for step, inputs in enumerate(eval_iterator):
                # 把 batch 搬到正确 device
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)

                # 构造 unit_locations（和 compute_loss 里保持一致）
                unit_locations = None
                if "intervention_locations" in inputs:
                    # inputs["intervention_locations"]: [B, L, P]
                    # 需要变成 [L, B, P]
                    intervention_locations = (
                        inputs["intervention_locations"].permute(1, 0, 2).tolist()
                    )
                    unit_locations = {
                        "sources->base": (None, intervention_locations)
                    }

                # forward：这里只做“正常 ReFT”，不加 adversarial delta
                if getattr(self, "adv_intervention", None) is not None:
                    self.adv_intervention.reset_delta()

                _, cf_outputs = intervenable(
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    },
                    unit_locations=unit_locations,
                    labels=inputs["labels"],
                )

                all_preds.append(cf_outputs.logits)
                all_labels.append(inputs["labels"])
        #pdb.set_trace()
        # -------- 3. 汇总 + 调用 compute_metrics --------
        all_preds = torch.cat(all_preds, dim=0).cpu().to(torch.float32)
        all_labels = torch.cat(all_labels, dim=0).cpu().to(torch.float32)

        metrics = self.compute_metrics(
            EvalPrediction(predictions=all_preds, label_ids=all_labels)
        )
        metrics = denumpify_detensorize(metrics)

        # 对 key 加上 "eval_" 前缀（和原版一样）
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # 日志 + callbacks
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
        
        
        