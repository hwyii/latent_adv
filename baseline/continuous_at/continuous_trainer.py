# baseline/continuous_at/continuous_trainer.py
import os
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from baseline.continuous_at.embedding_attack import AttackConfig, pgd_attack_embeddings

# deepspeed FLOPs profiler
from deepspeed.profiling.flops_profiler import FlopsProfiler


class HFDatasetFromItems(Dataset):
    def __init__(self, items, tokenizer, max_length=512):
        self.items = items
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]["content"]
        y = int(self.items[i]["clf_label"])
        enc = self.tok(
            x,
            truncation=True,
            max_length=min(self.max_length, getattr(self.tok, "model_max_length", self.max_length)),
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(y, dtype=torch.long),
        }


def make_collate_fn(pad_id: int):
    def collate(batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch], dim=0)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate


def _maybe_profile_flops_full_step(
    model,
    do_profile: bool,
    step: int,
    input_ids,
    attention_mask,
    labels,
    attack_cfg: AttackConfig,
    mix_adv_frac: float,
    opt: torch.optim.Optimizer,
):
    """
    Profile FULL training step FLOPs:
      - inner PGD (k steps) with autograd.grad
      - clean forward
      - adv forward
      - outer backward + opt.step

    Returns:
      (loss, loss_clean, loss_adv, flops_step, macs_step, params, t_batch)
    """
    profiler = None
    flops_step = None
    macs_step = None
    params = None

    t_batch = int(input_ids.shape[1])  # padded length

    if do_profile:
        profiler = FlopsProfiler(model)
        profiler.start_profile()

    # -------- FULL STEP COMPUTE --------
    # 1) inner max: build adversarial embeddings
    adv_embeds = pgd_attack_embeddings(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        attack=attack_cfg,
    )

    # 2) forward clean + adv
    model.train()
    out_clean = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss_clean = out_clean.loss if out_clean.loss is not None else F.cross_entropy(out_clean.logits, labels)

    out_adv = model(inputs_embeds=adv_embeds, attention_mask=attention_mask, labels=labels)
    loss_adv = out_adv.loss if out_adv.loss is not None else F.cross_entropy(out_adv.logits, labels)

    loss = (1.0 - mix_adv_frac) * loss_clean + mix_adv_frac * loss_adv

    # 3) outer min: update params
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if do_profile and profiler is not None:
        profiler.stop_profile()
        # total flops/macs for everything between start/stop, including backward
        flops_step = profiler.get_total_flops(as_string=False)
        macs_step = profiler.get_total_macs(as_string=False)
        params = profiler.get_total_params(as_string=False)
        profiler.end_profile()

    return loss, loss_clean, loss_adv, flops_step, macs_step, params, t_batch


def train_continuous_at(
    model,
    tokenizer,
    train_items,
    device,
    out_dir,
    attack_cfg: AttackConfig,
    mix_adv_frac: float = 0.5,
    lr: float = 2e-5,
    batch_size: int = 8,
    max_steps: int = 1000,
    weight_decay: float = 0.0,
    log_every: int = 50,
    # FLOPs profiling
    profile_flops_steps: int = 0,
    profile_flops_every: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)

    # pad token fix (pythia/gpt-neox needs it for batch>1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    ds = HFDatasetFromItems(train_items, tokenizer)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    model.train()

    step = 0
    log_path = os.path.join(out_dir, "train_log.jsonl")
    with open(log_path, "w") as f:
        f.write("")

    flops_records = []  # store per-profile step flops to summarize

    it = iter(dl)
    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        avg_seq_len = attention_mask.sum(dim=1).float().mean().item()

        labels = batch["labels"].to(device)

        do_profile = (
            profile_flops_steps > 0
            and step < profile_flops_steps
            and (profile_flops_every > 0 and (step % profile_flops_every) == 0)
        )

        t0 = time.time()
        loss, loss_clean, loss_adv, flops_step, macs_step, params, t_batch = _maybe_profile_flops_full_step(
            model=model,
            do_profile=do_profile,
            step=step,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            attack_cfg=attack_cfg,
            mix_adv_frac=mix_adv_frac,
            opt=opt,
        )
        dt = time.time() - t0

        if (step % log_every) == 0:
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "loss_clean": float(loss_clean.item()),
                "loss_adv": float(loss_adv.item()),
                "eps": attack_cfg.eps,
                "alpha": attack_cfg.alpha,
                "k": attack_cfg.steps,
                "norm": attack_cfg.norm,
                "avg_seq_len": avg_seq_len,
                "mix_adv_frac": mix_adv_frac,
                "batch_size": int(input_ids.shape[0]),
                "t_batch": int(t_batch),
                "sec_step": float(dt),
            }

            if flops_step is not None:
                rec["flops_step"] = float(flops_step)
                rec["macs_step"] = float(macs_step) if macs_step is not None else None
                rec["params"] = float(params) if params is not None else None
                flops_records.append(float(flops_step))

            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

            msg = f"[step {step}] loss={rec['loss']:.4f} clean={rec['loss_clean']:.4f} adv={rec['loss_adv']:.4f}"
            if flops_step is not None:
                msg += f" | FLOPs/step={flops_step:.3e} T={t_batch}"
            print(msg)

        step += 1

    # Summarize FLOPs
    if len(flops_records) > 0:
        flops_mean = sum(flops_records) / len(flops_records)
        flops_total_est = flops_mean * max_steps
        summ = {
            "profile_flops_steps": int(profile_flops_steps),
            "profile_flops_every": int(profile_flops_every),
            "n_profiled": int(len(flops_records)),
            "flops_step_mean": float(flops_mean),
            "flops_total_est_max_steps": float(flops_total_est),
        }
        with open(os.path.join(out_dir, "flops_summary.json"), "w") as f:
            json.dump(summ, f, indent=2)
        print("[FLOPs] wrote:", os.path.join(out_dir, "flops_summary.json"))

    # save HF checkpoint
    save_dir = os.path.join(out_dir, "final_ckpt")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[saved] {save_dir}")
    print(f"[log] {log_path}")
    return save_dir
