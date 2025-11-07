#!/usr/bin/env python3
"""
eval_spam_gcg.py

Evaluate two checkpoints (baseline and adv-trained) on EnronSpam attack split, using:
 - token-level GCG (per-sample paper-style, untargeted)
"""
import argparse, json, math, os, random, numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer  
from typing import Tuple, Dict, Any
from tqdm import tqdm
import pdb

from src import data

from ..data.adv_dataset import AdvDataset

# load model and tokenizer

def set_global_seed(seed: int):
    """Make runs reproducible as far as possible."""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN deterministic (may slow down). Keep benchmark=False.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def load_model_from_state(model_name, ckpt_path, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    state = torch.load(ckpt_path, map_location=device)
    
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    
    model.to(device)
    model.eval()
    return model

def calculate_and_print_stats(model_key: str, s: Dict[str, int], n_samples: int) -> Dict[str, Any]:
    """
    计算并打印所有指标, 并返回一个用于 JSON 存储的字典。
    """
    n_total = n_samples
    
    # --- 派生 4 种情况 ---
    # (来自你的请求：展示所有 4 种情况)
    # 1. 对 -> 错 (被攻破)
    n_C_to_W = s["n_flipped_to_wrong_C_to_W"]
    # 2. 错 -> 对 (被修正)
    n_W_to_C = s["n_flipped_to_correct_W_to_C"]
    # 3. 对 -> 对 (未攻破)
    n_C_to_C = s["n_initially_correct"] - n_C_to_W
    # 4. 错 -> 错 (保持错)
    n_W_to_W = (n_total - s["n_initially_correct"]) - n_W_to_C
    
    # 交叉验证
    assert n_C_to_C + n_C_to_W == s["n_initially_correct"]
    assert n_W_to_W + n_W_to_C == n_total - s["n_initially_correct"]
    assert n_C_to_C + n_W_to_C == s["n_correct_after_attack"]

    # --- 计算 3 个黄金指标 ---
    # 指标 A: 干净准确率 (ACC)
    clean_acc = s["n_initially_correct"] / (n_total + 1e-9)
    # 指标 B: 鲁棒准确率 (Rob-Acc)
    robust_acc = s["n_correct_after_attack"] / (n_total + 1e-9)
    # 指标 C: 攻击成功率 (ASR)
    asr = s["n_flipped_to_wrong_C_to_W"] / (s["n_initially_correct"] + 1e-9)

    # --- 打印到 stdout.log (用于调试) ---
    print(f"\n--- {model_key.upper()} Model ---")
    print("  --- 性能指标 ---")
    print(f"  Clean Accuracy (ACC)      = {clean_acc:.4f}  ({s['n_initially_correct']}/{n_total})")
    print(f"  Robust Accuracy (Rob-Acc) = {robust_acc:.4f}  ({s['n_correct_after_attack']}/{n_total})")
    print(f"  Attack Success Rate (ASR) = {asr:.4f}  ({s['n_flipped_to_wrong_C_to_W']}/{s['n_initially_correct']})")
    print("  --- 详细原始计数 (分母=40) ---")
    print(f"  Correct -> Correct (未攻破): {n_C_to_C}")
    print(f"  Correct -> Wrong (被攻破):   {n_C_to_W}")
    print(f"  Wrong   -> Wrong (保持错):   {n_W_to_W}")
    print(f"  Wrong   -> Correct (被修正): {n_W_to_C}")
    print(f"  ---------------------------------")
    print(f"  总计 (Check): {n_C_to_C + n_C_to_W + n_W_to_W + n_W_to_C}")

    # --- 返回用于 JSON 的数据 ---
    summary_metrics = {
        "Clean Accuracy (ACC)": clean_acc,
        "Robust Accuracy (Rob-Acc)": robust_acc,
        "Attack Success Rate (ASR)": asr,
    }
    # (我们返回最原始的计数, 派生的可以在 summarize.py 中计算)
    raw_counts = s.copy()
    raw_counts.update({
        "n_correct_after_attack_C_to_C": n_C_to_C,
        "n_stayed_wrong_W_to_W": n_W_to_W,
    })
    
    return {"summary_metrics": summary_metrics, "raw_counts": raw_counts}


# token-level GCG attack (per-sample paper-style, untargeted)
def token_level_gcg_single(
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
    Perform token-level GCG attack as per the original paper.
    """
    model.eval()
    #pdb.set_trace()
    ids = input_ids.unsqueeze(0).to(device)  # (1, L)
    mask = attention_mask.unsqueeze(0).to(device)  # (1, L)
    label_t = torch.tensor([true_label], device=device)  # (1,)

    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
        orig_logits = out.logits  # (1, num_labels)
        orig_pred = int(orig_logits.argmax(-1).item())
        orig_loss = float(F.cross_entropy(orig_logits, label_t).item())

    T = ids.size(1)
    attack_len = min(n_attack_tokens, max(0, T - attack_start))
    if attack_len == 0:
        return orig_pred, orig_loss, orig_pred, orig_loss, False, ids.squeeze(0)

    attack_slice = slice(attack_start, attack_start + attack_len)
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

    for rnd in range(rounds):
        # compute gradient wrt embeddings
        embeds = model.get_input_embeddings()(cur_ids)  # [1,T,H]
        embeds = embeds.clone().detach().requires_grad_(True)
        out = model(inputs_embeds=embeds, attention_mask=mask)
        loss = F.cross_entropy(out.logits, label_t)
        grad = torch.autograd.grad(loss, embeds)[0].detach()[0]  # [T,H]

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
                    out_cand = model(input_ids=cand_ids, attention_mask=mask)
                    loss_c = float(F.cross_entropy(out_cand.logits, label_t).item())
                # untargeted: pick candidate that maximizes loss
                if (best_loss is None) or (loss_c > best_loss):
                    best_loss = loss_c
                    best_tok = cand_tok

            # check current loss (for cur_ids)
            with torch.no_grad():
                out_cur = model(input_ids=cur_ids, attention_mask=mask)
                cur_loss_val = float(F.cross_entropy(out_cur.logits, label_t).item())

            if best_loss is not None and best_loss > cur_loss_val + 1e-9:
                cur_ids[0, pos] = best_tok
                changed_any = True

        if not changed_any:
            break

    # final eval
    with torch.no_grad():
        out_fin = model(input_ids=cur_ids, attention_mask=mask)
        adv_logits = out_fin.logits
        adv_pred = int(adv_logits.argmax(-1).item())
        adv_loss = float(F.cross_entropy(adv_logits, label_t).item())

    success = (adv_pred != orig_pred)
    return orig_pred, orig_loss, adv_pred, adv_loss, success, cur_ids.squeeze(0)


# ------------------ high-level evaluation harness ------------------
def build_dataset_from_splits(splits_json: str, split_name: str, dataset_name: str):
    with open(splits_json, "r") as f:
        splits = json.load(f)
    if split_name not in splits:
        raise ValueError(f"split {split_name} not found in {splits_json}")
    idxs = splits[split_name]
    ds = AdvDataset(split_indices=idxs, dataset_name=dataset_name)
    return ds


def run_evaluations(
    model_name: str,
    baseline_ckpt: str,
    adv_ckpt: str,
    splits_json: str,
    dataset_name: str,
    split_name: str,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_samples: int = 200,
    n_attack_tokens: int = 5,
    attack_start: int = 0,
    beam_k: int = 20,
    rounds: int = 20,
    max_length: int = 256,
    # (移除了未使用的 latent_... 参数)
) -> Dict[str, Any]:
    
    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading models...")
    # baseline = load_model_from_state(model_name, baseline_ckpt, device)
    adv = load_model_from_state(model_name, adv_ckpt, device)
    # models = {"baseline": baseline, "adv": adv}
    models = {"adv": adv}

    ds = build_dataset_from_splits(splits_json, split_name, dataset_name)
    print(f"Dataset split '{split_name}' size = {len(ds)}. Using first {n_samples} samples for evaluation.")
    n_samples = min(n_samples, len(ds))

    # === [计时修改 1]：GPU 预热 ===
    # (非常重要，否则第一次运行会包含 CUDA 内核编译时间)
    print(">>> [Timer] 开始 GPU 预热 (运行 2 个样本)...")
    try:
        for i in range(min(2, n_samples)): # 预热 2 个样本
            item = ds[i]
            label = int(item["label"])
            enc = tokenizer(item["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            # Squeeze(0) 以匹配 gcg 函数的期望
            input_ids_s = enc["input_ids"].squeeze(0)
            attention_mask_s = enc["attention_mask"].squeeze(0)
            
            # 假设 'adv' 模型总是存在
            _ = token_level_gcg_single(
                models['adv'], tokenizer, input_ids_s, attention_mask_s, label, device,
                n_attack_tokens=n_attack_tokens, attack_start=attack_start, beam_k=beam_k, rounds=rounds
            )
        torch.cuda.synchronize(device=device) # 等待预热完成
        print(">>> [Timer] 预热完成。")
    except Exception as e:
        print(f"WARN: 预热失败, {e}. 计时可能不准。")
    # === 预热结束 ===    
        
    # (新) 详细的统计字典
    stats = {
        # "baseline": {
        #     "n_total": n_samples,
        #     "n_initially_correct": 0,
        #     "n_flipped_to_wrong_C_to_W": 0,
        #     "n_flipped_to_correct_W_to_C": 0,
        #     "n_correct_after_attack": 0,
        #     "n_flipped_total": 0, # GCG 导致了多少次翻转
        # },
        "adv": {
            "n_total": n_samples,
            "n_initially_correct": 0,
            "n_flipped_to_wrong_C_to_W": 0,
            "n_flipped_to_correct_W_to_C": 0,
            "n_correct_after_attack": 0,
            "n_flipped_total": 0,
        },
    }
    
    # (新) 存储定性样本
    # qualitative_samples = {"baseline": [], "adv": []}
    qualitative_samples = {"adv": []}
    MAX_QUAL_SAMPLES = 5 # 最多存 5 个
    
    # === [计时修改 2]：创建事件 ===
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"--- 开始对 {n_samples} 个样本进行计时评估 ---")
    
    # === [计时修改 3]：在循环前打点 ===
    start_event.record()

    for i in tqdm(range(n_samples), desc="samples"):
        item = ds[i]
        text = item["text"]
        label = int(item["label"])

        enc = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        
        if input_ids.dim() == 1:
            input_ids_b = input_ids.unsqueeze(0).to(device)
            attention_mask_b = attention_mask.unsqueeze(0).to(device)
        else:
            input_ids_b = input_ids.to(device)
            attention_mask_b = attention_mask.to(device)

        # 针对两个模型进行评估
        for model_key, model in models.items():
            
            # 1. 检查干净样本的准确率
            with torch.no_grad():
                out_clean = model(input_ids=input_ids_b, attention_mask=attention_mask_b)
                pred_clean = int(out_clean.logits.argmax(-1).item())
            
            is_initially_correct = (pred_clean == label)

            if is_initially_correct:
                stats[model_key]["n_initially_correct"] += 1
                
            # 2. 发起 GCG 攻击
            orig_pred_gcg, _, adv_pred_gcg, _, gcg_success_flag, adv_ids = token_level_gcg_single(
                model, tokenizer, input_ids, attention_mask, label, device,
                n_attack_tokens=n_attack_tokens, attack_start=attack_start, beam_k=beam_k, rounds=rounds
            )
            
            # (健全性检查: GCG 的 orig_pred 应该等于我们的 pred_clean)
            if orig_pred_gcg != pred_clean:
                print(f"警告: Sample {i}, Model {model_key}: GCG 内部的 pred ({orig_pred_gcg}) "
                      f"与外部的 pred ({pred_clean}) 不匹配。这不应该发生。")

            if gcg_success_flag:
                stats[model_key]["n_flipped_total"] += 1

            is_correct_after_attack = (adv_pred_gcg == label)

            # 3. 统计所有 4 种情况
            if is_correct_after_attack:
                stats[model_key]["n_correct_after_attack"] += 1

            if is_initially_correct and not is_correct_after_attack:
                stats[model_key]["n_flipped_to_wrong_C_to_W"] += 1
            elif not is_initially_correct and is_correct_after_attack:
                stats[model_key]["n_flipped_to_correct_W_to_C"] += 1
            
            # 4. (新) 保存定性样本
            is_attack_successful = (is_initially_correct and not is_correct_after_attack)
            if is_attack_successful and len(qualitative_samples[model_key]) < MAX_QUAL_SAMPLES:
                orig_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                adv_text = tokenizer.decode(adv_ids, skip_special_tokens=True)
                qualitative_samples[model_key].append({
                    "sample_index": i,
                    "label": label,
                    "pred_clean": pred_clean,
                    "pred_adv": adv_pred_gcg,
                    "original_text": orig_text,
                    "attacked_text": adv_text
                })

    # === [计时修改 4]：在循环后打点并同步 ===
    end_event.record()
    torch.cuda.synchronize(device=device) # 关键：等待 GPU 完成所有工作

    # === [计时修改 5]：计算时间 ===
    total_gpu_time_ms = start_event.elapsed_time(end_event)
    total_gpu_time_sec = total_gpu_time_ms / 1000.0
    avg_gpu_time_per_sample_ms = total_gpu_time_ms / max(1, n_samples)
    
    # 打印到 stdout.log 以便调试
    print(f"\n[计时统计 - GPU Wall-Clock Time]")
    print(f"  总共用时 (Total GPU Time): {total_gpu_time_sec:.4f} 秒 (for {n_samples} samples)")
    print(f"  平均每样本 (Avg. GPU Time/Sample): {avg_gpu_time_per_sample_ms:.4f} 毫秒")

    # --- 4. 计算并打印最终总结 ---
    print("\n=== 总结 (已修复 ASR 逻辑, 增加 Rob-Acc) ===")
    print(f"总样本数: {n_samples}")
    
    #bsum = calculate_and_print_stats("BASELINE", stats["baseline"], n_samples)
    asum = calculate_and_print_stats("ADV", stats["adv"], n_samples)

    # --- 5. 返回所有数据用于 JSON 保存 ---
    return {
        #"baseline": bsum,
        "adv": asum,
        "qualitative_samples": qualitative_samples,
        "config": {
            "n_samples": n_samples,
            "n_attack_tokens": n_attack_tokens,
            "attack_start": attack_start,
            "beam_k": beam_k,
            "rounds": rounds
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--adv_ckpt", required=True)
    parser.add_argument("--splits_json", required=True, help="enron_splits.json")
    parser.add_argument("--split", default="attack", help="which split array name in enron_splits.json")
    parser.add_argument("--dataset_name", default="AlignmentResearch/EnronSpam")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_attack_tokens", type=int, default=5)
    parser.add_argument("--attack_start", type=int, default=0)
    parser.add_argument("--beam_k", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json_path", type=str, default=None,
                        help="如果提供, 将把详细结果保存到此 JSON 文件")
    args = parser.parse_args()
    
    set_global_seed(args.seed)

    final_results = run_evaluations(
        model_name=args.model_name,
        baseline_ckpt=args.baseline_ckpt,
        adv_ckpt=args.adv_ckpt,
        splits_json=args.splits_json,
        dataset_name=args.dataset_name,
        split_name=args.split,
        device_str=args.device,
        n_samples=args.n_samples,
        n_attack_tokens=args.n_attack_tokens,
        attack_start=args.attack_start,
        beam_k=args.beam_k,
        rounds=args.rounds,
        max_length=args.max_length,
    )
    if args.out_json_path:
        try:
            os.makedirs(os.path.dirname(args.out_json_path) or ".", exist_ok=True)
            with open(args.out_json_path, "w", encoding="utf-8") as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            print(f"\n[eval_spam_gcg] 详细评估结果已保存到: {args.out_json_path}")
        except Exception as e:
            print(f"\n[eval_spam_gcg] 错误: 无法保存 JSON 结果到 {args.out_json_path}: {e}")

if __name__ == "__main__":
    main()
    
    
# python -m src.attack.eval_spam_gcg   --model_name EleutherAI/pythia-410m   --baseline_ckpt out/pythia410m/harmless/best_Harmless.pt   --adv_ckpt out/spam_pythia410m/reft_lat_gcg_rank4_layer14_debug.pt   --splits_json src/data/harmless_splits.json   --split attack  --dataset_name AlignmentResearch/Harmless  --n_samples 40   --n_attack_tokens 10   --attack_start 0   --beam_k 20   --rounds 20   --latent_eps 0.05   --latent_steps 20
# python -m src.attack.eval_spam_gcg   --model_name EleutherAI/pythia-410m   --baseline_ckpt out/spam_pythia410m/best_new.pt   --adv_ckpt out/spam_pythia410m/reft_lat_gcg_rank4_layer14_randomR.pt   --splits_json src/data/enron_splits.json   --split attack   --n_samples 200   --n_attack_tokens 10   --attack_start 0   --beam_k 20   --rounds 20   --latent_eps 0.05   --latent_steps 20