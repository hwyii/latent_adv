# baseline/scaling/adv_pool_trainer.py
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


@dataclass
class AdvExample:
    item: Dict[str, Any]   # {"content": ..., "clf_label": ...}
    loss: float
    time_id: int
    round_id: int
    success: bool


class AdvPool:
    def __init__(self):
        self.pool: List[AdvExample] = []
        self.time_counter = 0

    def add_many(self, items: List[Tuple[Dict[str, Any], float, bool]], round_id: int):
        for it, loss, success in items:
            self.time_counter += 1
            self.pool.append(AdvExample(item=it, loss=float(loss), time_id=self.time_counter,
                                        round_id=round_id, success=bool(success)))

    def __len__(self):
        return len(self.pool)

    def sample_indices(self, n: int, lam: float = 0.005) -> List[int]:
        nadv = len(self.pool)
        if n >= nadv:
            return list(range(nadv))

        loss_vals = np.array([ex.loss for ex in self.pool], dtype=np.float64)
        order_loss = np.argsort(loss_vals)  # asc
        rloss = np.empty(nadv, dtype=np.float64)
        rloss[order_loss] = np.arange(nadv)  # high loss -> high rank

        time_vals = np.array([ex.time_id for ex in self.pool], dtype=np.int64)
        order_time = np.argsort(time_vals)  # old->new
        rtime = np.empty(nadv, dtype=np.float64)
        rtime[order_time] = np.arange(nadv)  # recent -> high rank

        ri = 0.5 * rloss + 0.5 * rtime
        w = np.exp(lam * ri)
        w = w / w.sum()

        chosen = np.random.choice(np.arange(nadv), size=n, replace=False, p=w)
        return chosen.tolist()


class SimpleTextClsDataset(Dataset):
    """
    Minimal dataset: tokenizes on the fly.
    Expects items as {"content": str, "clf_label": int}
    """
    def __init__(self, items: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        enc = self.tokenizer(
            it["content"],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(int(it["clf_label"]), dtype=torch.long),
        }


def make_collate_fn(pad_id: int):
    def collate_fn(batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch], dim=0)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate_fn


def linear_schedule_k(round_id: int, R: int, k_start: int, k_end: int) -> int:
    frac = float(round_id) / float(R)
    return int(round(k_start + frac * (k_end - k_start)))


def build_round_items(
    clean_items: List[Dict[str, Any]],
    adv_pool: AdvPool,
    naug: int = 1000,
    adv_frac: float = 0.8,
    lam: float = 0.005,
) -> List[Dict[str, Any]]:
    sadv = min(int(round(adv_frac * naug)), len(adv_pool))
    sclean = naug - sadv

    clean_batch = random.sample(clean_items, k=sclean) if sclean > 0 else []
    if sadv > 0:
        adv_idx = adv_pool.sample_indices(sadv, lam=lam)
        adv_batch = [adv_pool.pool[i].item for i in adv_idx]
    else:
        adv_batch = []

    return clean_batch + adv_batch


def train_one_round(
    model,
    tokenizer,
    round_items: List[Dict[str, Any]],
    device: torch.device,
    lr: float,
    batch_size: int,
    max_steps: int,
    weight_decay: float,
    save_dir: str,
    round_id: int,
):
    model.train()
    ds = SimpleTextClsDataset(round_items, tokenizer=tokenizer, max_length=512)
    collate = make_collate_fn(tokenizer.pad_token_id)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max_steps
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)

    step = 0
    while step < max_steps:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

            step += 1
            if step >= max_steps:
                break

    ckpt_dir = os.path.join(save_dir, f"round_{round_id:03d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    return ckpt_dir
