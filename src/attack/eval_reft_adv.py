"""
评估由train_reft_adv.py保存的final_intervention目录
"""
from multiprocessing import pool
import argparse, json, math, os, random, numpy as np
from typing import Tuple, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer  
from tqdm import tqdm
import pdb
from transformers.utils import logging

logger = logging.get_logger(__name__)

from src import attack
from src.utils.tools import set_seed

def token_level_gcg_single_baseline(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    true_label: int,
    device: torch.device,
    n_attack_tokens: int = 10, # 一次攻击的token数
    attack_start: int = 5, # 仅在replace mode下有效
    beam_k: int = 256, # 每个位置按梯度筛出的候选数
    rounds: int = 20,
    forbid_special: bool = True,
    attack_mode: str = "suffix", # "suffix" or "replace" of "infix"
    n_candidates_per_it: int = 128, # 每轮全局随机评估的候选数
):
    """
    Perform token-level GCG attack for baseline model (HF classifier, no ReFT).
    attack_mode = "suffix":  在原 prompt 末尾追加 N 个 token，并只修改这些后缀。
    attack_mode = "replace": 保持原来的“从 attack_start 开始替换”的行为。
    attack_mode = "infix":  在原 prompt 90% 位置插入 N 个 token。
    
    候选策略（与 scaling 论文一致的两步法）：
      1. 对每个攻击位置，取 top-k (beam_k) token 作为候选；
      2. 把所有 (pos, token) 丢进一个全局 pool；
      3. 从 pool 中随机采样 n_candidates_per_it 个进行前向评估，
         只应用全局最优的那一个修改。
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

    T_orig = ids.size(1)
    
    # 构造攻击用的输入
    if attack_mode == "suffix":
        suffix_len = n_attack_tokens
        if suffix_len <= 0:
            return orig_pred, orig_loss, orig_pred, orig_loss, False, ids.squeeze(0)
        
        # 初始化suffix为"!"
        init_tok = tokenizer.encode("!")[0]

        suffix_ids = torch.full(
            (1, suffix_len),
            init_tok,
            dtype=ids.dtype,
            device=device,
        )
        suffix_mask = torch.ones_like(suffix_ids, device=device)

        cur_ids = torch.cat([ids, suffix_ids], dim=1)   # (1, T_orig + N)
        cur_mask = torch.cat([mask, suffix_mask], dim=1)
        attack_positions = range(T_orig, T_orig + suffix_len)
    elif attack_mode == "infix":
        infix_len = n_attack_tokens
        if infix_len <= 0:
            return orig_pred, orig_loss, orig_pred, orig_loss, False, ids.squeeze(0)

        # 插入位置：句子 90% 处（token 序列层面）
        insert_pos = int(0.9 * T_orig)

        # 可选：避免插得太极端（比如 BOS/开头、EOS/末尾）
        insert_pos = max(1, min(insert_pos, T_orig - 1))

        init_tok = tokenizer.encode("!")[0]
        infix_ids = torch.full((1, infix_len), init_tok, dtype=ids.dtype, device=device)
        infix_mask = torch.ones_like(infix_ids, device=device)

        left_ids = ids[:, :insert_pos]
        right_ids = ids[:, insert_pos:]
        left_mask = mask[:, :insert_pos]
        right_mask = mask[:, insert_pos:]

        cur_ids = torch.cat([left_ids, infix_ids, right_ids], dim=1)   # (1, T_orig + N)
        cur_mask = torch.cat([left_mask, infix_mask, right_mask], dim=1)

        attack_positions = range(insert_pos, insert_pos + infix_len)
    elif attack_mode == "replace":
        # 保持原有逻辑
        cur_ids = ids.clone()
        cur_mask = mask.clone()
        attack_len = min(n_attack_tokens, max(0, T_orig - attack_start))
        if attack_len == 0:
            return orig_pred, orig_loss, orig_pred, orig_loss, False, ids.squeeze(0)
        attack_positions = range(attack_start, attack_start + attack_len)
    else:
        raise ValueError(f"Unknown attack_mode: {attack_mode}")
    
    # 预计算 vocab embedding 和mask

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

        outputs = model(inputs_embeds=embeds, attention_mask=cur_mask, labels=label_t)
        loss = F.cross_entropy(outputs.logits, label_t)
        grad = torch.autograd.grad(loss, embeds)[0].squeeze(0)  # [T,H]

        cur_loss_val = float(loss.item())
        
        # generate candidate pool
        pool = []
        top_k = min(beam_k, V)
        for pos in attack_positions:
            g_pos = grad[pos]  # [H]
            # compute scores over vocab: v @ g_pos
            scores = torch.mv(vocab_emb, g_pos.to(vocab_emb.dtype))  # [V]
            scores_masked = scores.clone()
            scores_masked[~vocab_mask] = -1e30
            
            topk_indices = torch.topk(scores_masked, k=top_k, largest=True).indices.tolist()
            for tok_id in topk_indices:
                pool.append( (pos, tok_id) )
                
        if len(pool) == 0:
            break
        
        # 4.3 从候选池随机抽 n_candidates_per_it 个进行评估
        if len(pool) <= n_candidates_per_it:
            sampled_candidates = pool
        else:
            sampled_candidates = random.sample(pool, n_candidates_per_it)

        
        best_loss = None
        best_pos = None
        best_tok = None

        for pos, cand_tok in sampled_candidates:
            cand_ids = cur_ids.clone()
            cand_ids[0, pos] = cand_tok
            with torch.no_grad():
                outputs_c = model(
                    input_ids=cand_ids,
                    attention_mask=cur_mask,
                    labels=label_t,
                )
                loss_c = float(F.cross_entropy(outputs_c.logits, label_t).item())
            # 无目标攻击：选择让 loss 最大的那个候选
            if (best_loss is None) or (loss_c > best_loss):
                best_loss = loss_c
                best_pos = pos
                best_tok = cand_tok

        # 4.4 如果最优候选比当前 loss 大，就接受这一步；否则提前停止
        changed_any = False
        if best_loss is not None and best_loss > cur_loss_val + 1e-9:
            cur_ids[0, best_pos] = best_tok
            changed_any = True

        if not changed_any:
            break

    # ===== 5) 最终对抗预测 =====
    with torch.no_grad():
        outputs_fin = model(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            labels=label_t,
        )
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
    n_attack_tokens: int = 10, # 一次攻击的token数
    attack_start: int = 5,
    beam_k: int = 256, # 每个位置按梯度筛出的候选数
    rounds: int = 20, 
    forbid_special: bool = True,
    attack_mode: str = "suffix",
    n_candidates_per_it: int = 128,
    reft_loc_mode: str = "train" # "train": 固定用训练时的位置；"last_token": 总是指向最后一个token
):
    """  
    Perform token-level GCG attack for ReFT model.  
    支持两种模式：  
    - suffix: 在原始输入后追加 N 个 token（与 baseline 一致）  
    - replace: 替换指定位置的 token（原有逻辑）  
    
    reft_loc_mode:  
    - "train": 固定使用训练时的 intervention_locations  
    - "last_token": 动态更新 intervention_locations，总是指向当前输入的最后一个 token
    """  
    reft_model.eval()
    ids = input_ids.unsqueeze(0).to(device)  # (1, L)
    mask = attention_mask.unsqueeze(0).to(device)  # (1, L)
    label_t = torch.tensor([true_label], device=device)  # (1,)
    
    T_orig = ids.size(1)

    # ===== 1) 构造“基准”训练位置（只有 reft_loc_mode="train" 时会用到） =====
    base_intervention = intervention_locations.to(torch.long)  # [1, I] 或 [1,1]

    def make_unit_locations_train():
        # 使用训练时的位置（对齐原有代码的维度变形）
        return {
            "sources->base": (
                None,
                base_intervention.unsqueeze(1).permute(0, 1, 2).tolist(),
            )
        }

    def make_unit_locations_last_token(seq_len: int):
        # 忽略 intervention_locations，永远挂在当前序列最后一个 token 上
        last_pos = seq_len - 1
        loc = torch.tensor([[last_pos]], dtype=torch.long)
        return {
            "sources->base": (
                None,
                loc.unsqueeze(1).permute(0, 1, 2).tolist(),  # [1,1,1] -> list
            )
        }

    # 一个小 helper，根据模式和当前长度生成 unit_locations
    def get_unit_locations(seq_len: int):
        if reft_loc_mode == "train":
            return make_unit_locations_train()
        elif reft_loc_mode == "last_token":
            return make_unit_locations_last_token(seq_len)
        else:
            raise ValueError(f"Unknown reft_loc_mode: {reft_loc_mode}")

    # ===== 2) 先在干净输入上跑一次，得到 orig_pred / orig_loss =====
    unit_locations_clean = get_unit_locations(T_orig)
    with torch.no_grad():
        _, cf_outputs = reft_model(
            {"input_ids": ids, "attention_mask": mask},
            unit_locations=unit_locations_clean,
            labels=label_t,
        )
        orig_logits = cf_outputs.logits
        orig_pred = int(orig_logits.argmax(-1).item())
        orig_loss = float(F.cross_entropy(orig_logits, label_t).item())

    # ===== 3) 根据 attack_mode 构造攻击输入（注意：不在这里改变 ReFT 位置） =====
    if attack_mode == "suffix":
        suffix_len = n_attack_tokens
        if suffix_len <= 0:
            return orig_pred, orig_loss, orig_pred, orig_loss, False, ids.squeeze(0)

        # 初始化 suffix，用一个正常 token 即可
        if tokenizer.eos_token_id is not None:
            init_tok = tokenizer.eos_token_id
        elif tokenizer.pad_token_id is not None:
            init_tok = tokenizer.pad_token_id
        else:
            init_tok = tokenizer.encode("!")[0]

        suffix_ids = torch.full(
            (1, suffix_len),
            init_tok,
            dtype=ids.dtype,
            device=device,
        )
        suffix_mask = torch.ones_like(suffix_ids, device=device)

        cur_ids = torch.cat([ids, suffix_ids], dim=1)   # (1, T_orig + N)
        cur_mask = torch.cat([mask, suffix_mask], dim=1)

        attack_positions = range(T_orig, T_orig + suffix_len)

    elif attack_mode == "replace":
        cur_ids = ids.clone()
        cur_mask = mask.clone()

        attack_len = min(n_attack_tokens, max(0, T_orig - attack_start))
        if attack_len == 0:
            return orig_pred, orig_loss, orig_pred, orig_loss, False, ids.squeeze(0)

        attack_positions = range(attack_start, attack_start + attack_len)

    else:
        raise ValueError(f"Unknown attack_mode: {attack_mode}")

    # ===== 4) vocab embeddings & special token mask =====
    vocab_emb = reft_model.model.get_input_embeddings().weight.detach()  # [V, H]
    V, H = vocab_emb.shape

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

    # ===== 5) 主 GCG 循环（两步法：top-k → pool → 随机采样） =====
    for rnd in range(rounds):
        T_cur = cur_ids.size(1)
        unit_locations = get_unit_locations(T_cur)

        # 5.1 当前 loss & grad
        embeds = reft_model.model.get_input_embeddings()(cur_ids)  # [1, T_cur, H]
        embeds = embeds.clone().detach().requires_grad_(True)

        _, cf_outputs = reft_model(
            {"inputs_embeds": embeds, "attention_mask": cur_mask},
            unit_locations=unit_locations,
            labels=label_t,
        )
        loss = F.cross_entropy(cf_outputs.logits, label_t)
        grad = torch.autograd.grad(loss, embeds)[0].squeeze(0)  # [T_cur, H]

        cur_loss_val = float(loss.item())

        # 5.2 构造候选池：所有 attack_positions × top-k token
        pool = []
        top_k = min(beam_k, V)
        for pos in attack_positions:
            g_pos = grad[pos]  # [H]
            scores = torch.mv(vocab_emb, g_pos.to(vocab_emb.dtype))  # [V]
            scores_masked = scores.clone()
            scores_masked[~vocab_mask] = -1e30

            topk_indices = torch.topk(scores_masked, k=top_k, largest=True).indices
            for tok_id in topk_indices.tolist():
                pool.append((pos, tok_id))

        if len(pool) == 0:
            break

        # 5.3 随机采样 n_candidates_per_it 个候选进行评估
        if len(pool) <= n_candidates_per_it:
            sampled_candidates = pool
        else:
            sampled_candidates = random.sample(pool, n_candidates_per_it)

        best_loss = None
        best_pos = None
        best_tok = None

        for pos, cand_tok in sampled_candidates:
            cand_ids = cur_ids.clone()
            cand_ids[0, pos] = cand_tok

            T_cand = cand_ids.size(1)
            unit_locations_cand = get_unit_locations(T_cand)

            with torch.no_grad():
                _, cf_cand = reft_model(
                    {"input_ids": cand_ids, "attention_mask": cur_mask},
                    unit_locations=unit_locations_cand,
                    labels=label_t,
                )
                loss_c = float(F.cross_entropy(cf_cand.logits, label_t).item())

            if (best_loss is None) or (loss_c > best_loss):
                best_loss = loss_c
                best_pos = pos
                best_tok = cand_tok

        # 5.4 接受最优候选（如果能增大 loss）
        changed_any = False
        if best_loss is not None and best_loss > cur_loss_val + 1e-9:
            cur_ids[0, best_pos] = best_tok
            changed_any = True

        if not changed_any:
            break

    # ===== 6) 最终对抗评估 =====
    T_final = cur_ids.size(1)
    unit_locations_final = get_unit_locations(T_final)

    with torch.no_grad():
        _, cf_fin = reft_model(
            {"input_ids": cur_ids, "attention_mask": cur_mask},
            unit_locations=unit_locations_final,
            labels=label_t,
        )
        adv_logits = cf_fin.logits
        adv_pred = int(adv_logits.argmax(-1).item())
        adv_loss = float(F.cross_entropy(adv_logits, label_t).item())

    success = (adv_pred != orig_pred)
    return orig_pred, orig_loss, adv_pred, adv_loss, success, cur_ids.squeeze(0)