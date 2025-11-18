"""
评估由train_reft_adv.py保存的final_intervention目录
"""
import argparse, json, math, os, random, numpy as np
from typing import Tuple, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer  
from tqdm import tqdm
import pdb
from transformers.utils import logging

logger = logging.get_logger(__name__)

from src.utils.tools import set_seed

def token_level_gcg_single_baseline(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    true_label: int,
    device: torch.device,
    n_attack_tokens: int = 5, # 一次攻击的token数
    attack_start: int = 5,
    beam_k: int = 20, # 每个位置按梯度筛出的候选数
    rounds: int = 20,
    forbid_special: bool = True,
):
    """
    Perform token-level GCG attack for baseline model (HF classifier, no ReFT).
    """
    model.eval()
    ids = input_ids.unsqueeze(0).to(device)  # (1, T)
    mask = attention_mask.unsqueeze(0).to(device)  # (1, T)
    label_t = torch.tensor([true_label], device=device)  # (1,)

    # clean forward
    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=mask, labels=label_t)
        orig_logits = outputs.logits
        orig_pred = int(orig_logits.argmax(-1).item())
        orig_loss = float(F.cross_entropy(orig_logits, label_t).item())

    T = ids.size(1)
    attack_len = min(n_attack_tokens, max(0, T - attack_start))
    if attack_len == 0:
        return orig_pred, orig_loss, orig_pred, orig_loss, False, ids.squeeze(0)

    cur_ids = ids.clone()

    # vocab embeddings
    vocab_emb = model.get_input_embeddings().weight.detach()  # [V,H]
    V, H = vocab_emb.shape

    # mask special tokens if desired
    if forbid_special:
        try:
            specials = set(tokenizer.all_special_ids)
        except Exception:
            specials = set()
        if tokenizer.pad_token_id is not None:
            specials.add(tokenizer.pad_token_id)
        vocab_mask = torch.ones(V, dtype=torch.bool, device=device)
        for s in specials:
            if 0 <= s < V:
                vocab_mask[s] = False
    else:
        vocab_mask = torch.ones(V, dtype=torch.bool, device=device)

    # main GCG loop
    for rnd in range(rounds):
        # compute gradient wrt embeddings
        embeds = model.get_input_embeddings()(cur_ids)  # [1,T,H]
        embeds = embeds.clone().detach().requires_grad_(True)

        outputs = model(inputs_embeds=embeds, attention_mask=mask, labels=label_t)
        loss = F.cross_entropy(outputs.logits, label_t)
        grad = torch.autograd.grad(loss, embeds)[0].squeeze(0)  # [T,H]

        changed_any = False
        # for each position in attack slice
        for pos in range(attack_start, attack_start + attack_len):
            g_pos = grad[pos]  # [H]
            # compute scores over vocab: v @ g_pos
            scores = torch.mv(vocab_emb, g_pos.to(vocab_emb.dtype))  # [V]
            scores_masked = scores.clone()
            scores_masked[~vocab_mask] = -1e30
            
            k = min(beam_k, V)
            topk = torch.topk(scores_masked, k=k, largest=True).indices.tolist()
            
            best_loss = None
            best_tok = None
            
            for cand_tok in topk:
                cand_ids = cur_ids.clone()
                cand_ids[0, pos] = cand_tok
                with torch.no_grad():
                    outputs_c = model(input_ids=cand_ids, attention_mask=mask, labels=label_t)
                    loss_c = float(F.cross_entropy(outputs_c.logits, label_t).item())
                # untargeted: pick candidate that maximizes loss
                if (best_loss is None) or (loss_c > best_loss):
                    best_loss = loss_c
                    best_tok = cand_tok 
            # check current loss (for cur_ids)
            with torch.no_grad():
                outputs_cur = model(input_ids=cur_ids, attention_mask=mask, labels=label_t)
                cur_loss_val = float(F.cross_entropy(outputs_cur.logits, label_t).item())
            
            if best_loss is not None and best_loss > cur_loss_val + 1e-9:
                cur_ids[0, pos] = best_tok
                changed_any = True
        
        if not changed_any:
            break   
    # final eval
    with torch.no_grad():
        outputs_fin = model(input_ids=cur_ids, attention_mask=mask, labels=label_t)
        adv_logits = outputs_fin.logits
        adv_pred = int(adv_logits.argmax(-1).item())
        adv_loss = float(F.cross_entropy(adv_logits, label_t).item())
    success = (adv_pred != orig_pred)
    return orig_pred, orig_loss, adv_pred, adv_loss, success, cur_ids.squeeze(0)

# token-level GCG attack for reft model
def token_level_gcg_single_reft(
    reft_model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    true_label: int,
    intervention_locations: torch.Tensor,
    device: torch.device,
    n_attack_tokens: int = 5, # 一次攻击的token数
    attack_start: int = 5,
    beam_k: int = 20, # 每个位置按梯度筛出的候选数
    rounds: int = 20, 
    forbid_special: bool = True,
):
    """
    Perform token-level GCG attack as per the original paper.
    """
    reft_model.eval()
    ids = input_ids.unsqueeze(0).to(device)  # (1, L)
    mask = attention_mask.unsqueeze(0).to(device)  # (1, L)
    label_t = torch.tensor([true_label], device=device)  # (1,)

    unit_locations = {
        "sources->base": (
            None,
            intervention_locations.unsqueeze(1).permute(0 ,1, 2).tolist(),
        )
    }

    with torch.no_grad():
        _, cf_outputs = reft_model(
            {"input_ids": ids, "attention_mask": mask},
            unit_locations=unit_locations,
            labels=label_t,
        )
        
        orig_logits = cf_outputs.logits
        orig_pred = int(orig_logits.argmax(-1).item())
        orig_loss = float(F.cross_entropy(orig_logits, label_t).item())
        

    T = ids.size(1)
    attack_len = min(n_attack_tokens, max(0, T - attack_start))
    if attack_len == 0:
        return orig_pred, orig_loss, orig_pred, orig_loss, False, ids.squeeze(0)

    attack_slice = slice(attack_start, attack_start + attack_len)
    cur_ids = ids.clone()
    
    # vocab embeddings
    vocab_emb = reft_model.model.get_input_embeddings().weight.detach()  # [V,H]
    V, H = vocab_emb.shape

    # mask special tokens if desired
    if forbid_special:
        try:
            specials = set(tokenizer.all_special_ids)
        except Exception:
            specials = set()
        if tokenizer.pad_token_id is not None:
            specials.add(tokenizer.pad_token_id)
        vocab_mask = torch.ones(V, dtype=torch.bool, device=device)
        for s in specials:
            if 0 <= s < V:
                vocab_mask[s] = False
    else:
        vocab_mask = torch.ones(V, dtype=torch.bool, device=device)

    for rnd in range(rounds):
        # compute gradient wrt embeddings
        embeds = reft_model.model.get_input_embeddings()(cur_ids)  # [1,T,H]
        embeds = embeds.clone().detach().requires_grad_(True)
        
        _, cf_outputs = reft_model(
            {"inputs_embeds": embeds, "attention_mask": mask},
            unit_locations=unit_locations,
            labels=label_t,
        )
        loss = F.cross_entropy(cf_outputs.logits, label_t)
        grad = torch.autograd.grad(loss, embeds)[0].squeeze(0)  # [T,H]

        changed_any = False
        # for each position in attack slice
        for pos in range(attack_start, attack_start + attack_len):
            g_pos = grad[pos]  # [H]
            # compute scores over vocab: v @ g_pos
            scores = torch.mv(vocab_emb, g_pos.to(vocab_emb.dtype))  # [V]
            scores_masked = scores.clone()
            scores_masked[~vocab_mask] = -1e30

            k = min(beam_k, V)
            topk = torch.topk(scores_masked, k=k, largest=True).indices.tolist()

            best_loss = None
            best_tok = None
            for cand_tok in topk:
                cand_ids = cur_ids.clone()
                cand_ids[0, pos] = cand_tok
                with torch.no_grad():
                    _, cf_cand = reft_model(
                        {"input_ids": cand_ids, "attention_mask": mask},
                        unit_locations=unit_locations,
                        labels=label_t,
                    )
                    loss_c = float(F.cross_entropy(cf_cand.logits, label_t).item())
                # untargeted: pick candidate that maximizes loss
                if (best_loss is None) or (loss_c > best_loss):
                    best_loss = loss_c
                    best_tok = cand_tok

            # check current loss (for cur_ids)
            with torch.no_grad():
                _, cf_cur = reft_model(
                    {"input_ids": cur_ids, "attention_mask": mask},
                    unit_locations=unit_locations,
                    labels=label_t,
                )
                cur_loss_val = float(F.cross_entropy(cf_cur.logits, label_t).item())

            if best_loss is not None and best_loss > cur_loss_val + 1e-9:
                cur_ids[0, pos] = best_tok
                changed_any = True

        if not changed_any:
            break

    # final eval
    with torch.no_grad():
        _, cf_fin = reft_model(
            {"input_ids": cur_ids, "attention_mask": mask},
            unit_locations=unit_locations,
            labels=label_t,
        )
        adv_logits = cf_fin.logits
        adv_pred = int(adv_logits.argmax(-1).item())
        adv_loss = float(F.cross_entropy(adv_logits, label_t).item())

    success = (adv_pred != orig_pred)
    return orig_pred, orig_loss, adv_pred, adv_loss, success, cur_ids.squeeze(0)
