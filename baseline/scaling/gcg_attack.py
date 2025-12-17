# baseline/scaling/gcg_attack.py
import random
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F

from src.attack.eval_reft_adv import token_level_gcg_single_baseline

def gcg_attack_text(
    model,
    tokenizer,
    text: str,
    label: int,
    device: torch.device,
    rounds: int,
    n_attack_tokens: int = 10,
    attack_mode: str = "suffix",
    beam_k: int = 256,
    n_candidates_per_it: int = 128,
    max_length: int = 512,
) -> Tuple[str, float, bool]:
    """
    Convenience wrapper: raw text -> attacked text
    Returns: (attacked_text, adv_loss, success)
    """
    enc = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]

    _, _, _, adv_loss, success, adv_ids = token_level_gcg_single_baseline(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        true_label=int(label),
        device=device,
        n_attack_tokens=n_attack_tokens,
        beam_k=beam_k,
        rounds=rounds,
        forbid_special=True,
        attack_mode=attack_mode,
        n_candidates_per_it=n_candidates_per_it,
    )
    adv_text = tokenizer.decode(adv_ids, skip_special_tokens=True)
    return adv_text, float(adv_loss), bool(success)