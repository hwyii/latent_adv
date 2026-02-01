# src/training/reft_adv_trainer.py
from typing import Optional, Dict, Any
import os, json
import torch
import pyreft
import pyvene as pv

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from pyreft.reft_trainer import ReftTrainerForSequenceClassification, make_dataloader
from transformers.trainer_utils import EvalPrediction, has_length, denumpify_detensorize
from transformers.utils import logging

from src.attack.inner_attack import AttackConfig
from src.models.adv_intervention import AdversarialIntervention

from src.circuit.circuit_mask import load_circuit_mask, mask_summary
from src.circuit.grad_gate import CircuitGradGate  # optional (focus grad slices)
from src.models.circuitable_attention import make_circuitable_neox, set_circuit_on_model

logger = logging.get_logger(__name__)


class ReftAdversarialTrainerForSequenceClassification(ReftTrainerForSequenceClassification):
    def __init__(self, *args, attack_config: Optional[AttackConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config

        # --- find AdversarialIntervention ---
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

        # ===== circuit setup =====
        self.circuit_spec = None
        self.circuit_gate = None   # optional grad slice gate (not for FLOPs)
        self._base_model = None
        self._circuit_mask_cpu = None

        cfg = self.attack_config
        if cfg is not None and getattr(cfg, "use_circuit_gate", False):
            if not getattr(cfg, "circuit_path", None):
                raise ValueError("use_circuit_gate=True but circuit_path is None")

            base = self.model.model  # HF base model inside IntervenableModel
            self._base_model = base

            if not hasattr(base, "gpt_neox"):
                raise ValueError("Circuit gate currently supports GPT-NeoX/Pythia (model.gpt_neox.layers).")

            num_layers = len(base.gpt_neox.layers)
            num_heads = base.gpt_neox.config.num_attention_heads
            head_dim = base.gpt_neox.config.hidden_size // num_heads

            # IMPORTANT: your load_circuit_mask() does NOT accept top_k kwarg in your pasted version.
            # Use JSON's own "top_k" or pre-trim the list in JSON.
            spec = load_circuit_mask(
                json_path=cfg.circuit_path,
                num_layers=num_layers,
                num_heads=num_heads,
                device=None,
            )
            total, per_layer = mask_summary(spec.mask)
            logger.info(f"[Circuit] loaded {cfg.circuit_path} | total_heads={total} | layers={len(per_layer)}")

            self.circuit_spec = spec
            self._circuit_mask_cpu = spec.mask.detach().cpu()  # bool [L,H]

            # 1) install forward-time stopgrad hooks once
            make_circuitable_neox(base)
            # default OFF
            set_circuit_on_model(base, None, enabled=False)

            # 2) optional: gradient slice gate (does not reduce FLOPs, just focus grads)
            #    If you want pure FLOPs saving only, you can comment this out.
            self.circuit_gate = CircuitGradGate(
                model=base,
                head_mask=spec.mask,
                head_dim=head_dim,
                enabled=True,
                verbose=False,
            )

    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ):
        # labels shape fix
        if inputs["labels"].dim() == 2:
            inputs["labels"] = inputs["labels"].squeeze(1)

        # unit_locations
        unit_locations = None
        if "intervention_locations" in inputs:
            unit_locations = {
                "sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )
            }

        # -------- 1) clean forward (no delta) --------
        if self.adv_intervention is not None:
            self.adv_intervention.reset_delta()

        _, cf_outputs = intervenable(
            {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )

        logits = cf_outputs.logits
        labels = inputs["labels"]

        # problem type
        if self.model.model.config.problem_type is None:
            if self.model.model.num_labels == 1:
                problem_type = "regression"
            elif self.model.model.num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
        else:
            problem_type = self.model.model.config.problem_type

        # clean loss
        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.model.model.num_labels == 1:
                clean_loss = loss_fct(logits.squeeze(), labels.squeeze().to(torch.bfloat16))
            else:
                clean_loss = loss_fct(logits, labels.to(torch.bfloat16))
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            clean_loss = loss_fct(logits.view(-1, self.model.model.num_labels), labels.view(-1))
        else:
            loss_fct = BCEWithLogitsLoss()
            clean_loss = loss_fct(logits, labels)

        total_loss = clean_loss
        adv_loss = None

        # -------- 2) inner adversarial loop (build delta) --------
        cfg = self.attack_config
        if cfg is not None and cfg.inner_attack != "none":
            # random noise delta
            if cfg.inner_attack == "random_noise":
            
                current_delta = self.adv_intervention.delta
                if current_delta is None:
                    # 如果 clean forward 没生成 delta (某些特定的 reft 配置)，需要特殊处理
                    # 这里假设它不为 None
                    pass
                else:
                    eps = float(cfg.eps)
                    
                    # 生成随机噪声
                    # 方式 A: 均匀分布 U[-eps, eps]
                    noise = torch.rand_like(current_delta) * 2 * eps - eps
                    
                    # 方式 B: 高斯噪声并截断 (更猛一点)
                    # noise = torch.randn_like(current_delta)
                    # noise = noise / (noise.norm(p=2, dim=-1, keepdim=True) + 1e-10) * eps
                    
                    # 应用噪声
                    # 注意：PyReFT 的 delta 通常是 Parameter 或 Tensor，若是 Parameter 需用 data
                    with torch.no_grad():
                        current_delta.add_(noise)
                        # 再次做 clamp 确保不越界 (如果是针对 norm 的限制)
                        current_delta.clamp_(-eps, eps)
                
                # Random 模式不需要 steps 循环，一次即可
                # 但为了代码兼容性，我们直接跳过下面的 for t in range(steps) 
                
            else:
            
                steps = int(cfg.steps)
                eps = float(cfg.eps)
                step_size = 2.5 * (eps / max(1, steps))

                self.adv_intervention.reset_delta()

                should_log_indices = (self.state.global_step % 10 == 0)
                batch_attack_trace = []

                # enable stopgrad gate only for inner loop
                gate_mode = getattr(cfg, "gate_mode", "inner_only")
                do_stopgrad_inner = (
                    self.circuit_spec is not None and gate_mode in ("inner_only", "inner+final")
                )
                if do_stopgrad_inner:
                    set_circuit_on_model(self._base_model, self._circuit_mask_cpu, enabled=True)

                try:
                    

                    for t in range(steps):
                        if do_stopgrad_inner and t == 0:
                            print(f"[DEBUG] stopgrad gate ENABLED at global_step={self.state.global_step}")
                        intervenable.zero_grad()

                        _, cf_adv = intervenable(
                            {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                            unit_locations=unit_locations,
                            labels=labels,
                        )
                        logits_adv = cf_adv.logits
                        loss_step = CrossEntropyLoss()(
                            logits_adv.view(-1, self.model.model.num_labels),
                            labels.view(-1)
                        )

                        # optional grad slice gate (focus only; not FLOPs saving)
                        do_grad_mask_inner = (
                            self.circuit_gate is not None and gate_mode in ("inner_only", "inner+final")
                        )
                        if do_grad_mask_inner:
                            with self.circuit_gate:
                                loss_step.backward()
                        else:
                            loss_step.backward()

                        # Debug point 3: inspect dense input-gradient sparsity after backward
                        if t == 0 and (self.state.global_step % 50 == 0):
                            try:
                                base = self._base_model
                                layer = 13
                                dense = base.gpt_neox.layers[layer].attention.dense

                                gW = dense.weight.grad
                                if gW is None:
                                    print("[DEBUG] dense.weight.grad is None")
                                else:
                                    head_mask = self._circuit_mask_cpu[layer].to(gW.device)  # [H]
                                    num_heads = base.gpt_neox.config.num_attention_heads
                                    head_dim = base.gpt_neox.config.hidden_size // num_heads

                                    keep_cols = torch.zeros(num_heads * head_dim, dtype=torch.bool, device=gW.device)
                                    for h in range(num_heads):
                                        if head_mask[h]:
                                            keep_cols[h*head_dim:(h+1)*head_dim] = True

                                    cols_keep = gW[:, keep_cols].abs().mean().item()
                                    cols_drop = gW[:, ~keep_cols].abs().mean().item()
                                    print(f"[DEBUG] layer{layer} dense.grad mean keep={cols_keep:.4e} drop={cols_drop:.4e}")
                            except Exception as e:
                                print("[DEBUG] grad-inspect error:", e)

                        delta_tensor = self.adv_intervention.delta
                        if delta_tensor is None or delta_tensor.grad is None:
                            print("[InnerAttack][ERROR] delta.grad is None, abort inner loop")
                            break

                        with torch.no_grad():
                            g = delta_tensor.grad

                            if cfg.inner_attack == "latent_pgd":
                                delta_tensor.add_(step_size * g.sign())

                            elif cfg.inner_attack == "latent_gcg_coord":
                                g_flat = g.view(-1)
                                k = min(int(getattr(cfg, "gcg_topk", 2)), g_flat.numel())

                                topk_idx = g_flat.abs().topk(k=k).indices

                                if should_log_indices:
                                    batch_attack_trace.append({"t": t, "ids": topk_idx.detach().cpu().tolist()})

                                delta_flat = delta_tensor.view(-1)
                                delta_flat[topk_idx] += step_size * g_flat[topk_idx].sign()

                            else:
                                raise ValueError(f"Unknown inner_attack type: {cfg.inner_attack}")

                            delta_tensor.clamp_(-eps, eps)
                            delta_tensor.grad.zero_()
                            delta_tensor.requires_grad_(True)

                finally:
                    if do_stopgrad_inner:
                        set_circuit_on_model(self._base_model, None, enabled=False)
                        print(f"[DEBUG] stopgrad gate DISABLED at global_step={self.state.global_step}")


                if should_log_indices and len(batch_attack_trace) > 0:
                    log_path = os.path.join(self.args.output_dir, "attack_neurons_log.jsonl")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"global_step": self.state.global_step, "trace": batch_attack_trace}) + "\n")

            # -------- 3) final adversarial loss (update ReFT params) --------
            if self.adv_intervention.delta is not None:
                self.adv_intervention.delta = self.adv_intervention.delta.detach()
                intervenable.zero_grad()

                # Option A (recommended): final backward NOT gated (base frozen anyway, only ReFT grads)
                _, cf_final = intervenable(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations=unit_locations,
                    labels=labels,
                )
                logits_final = cf_final.logits
                adv_loss = CrossEntropyLoss()(
                    logits_final.view(-1, self.model.model.num_labels),
                    labels.view(-1)
                )

                self.adv_intervention.reset_delta()
                total_loss = clean_loss + float(cfg.lambda_adv) * adv_loss

        # logging
        if (
            self.state.is_world_process_zero
            and self.state.global_step > 0
            and self.state.global_step % self.args.logging_steps == 0
        ):
            log_dict = {"loss_clean": clean_loss.item()}
            if adv_loss is not None:
                log_dict["loss_adv"] = adv_loss.item()
                log_dict["loss_total"] = total_loss.item()
            self.log(log_dict)

        if return_outputs:
            cf_outputs.loss = total_loss
            return (total_loss, cf_outputs)
        return total_loss

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
