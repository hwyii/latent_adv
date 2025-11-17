import os
import json
import argparse
import pdb
import token
from typing import Dict, Any

import torch
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from src.models.reft_latent import load_reft_model_for_eval
from src.data.datasets_reft import (
    build_reft_classification_datasets,
)
from src.utils.tools import set_seed

from src.attack.eval_reft_adv import token_level_gcg_single_reft 

logger = hf_logging.get_logger(__name__)

def calculate_and_print_stats(model_key: str, s: Dict[str, int], n_samples: int) -> Dict[str, Any]:
    n_total = n_samples

    n_C_to_W = s["n_flipped_to_wrong_C_to_W"]
    n_W_to_C = s["n_flipped_to_correct_W_to_C"]

    n_C_to_C = s["n_initially_correct"] - n_C_to_W
    n_W_to_W = (n_total - s["n_initially_correct"]) - n_W_to_C

    assert n_C_to_C + n_C_to_W == s["n_initially_correct"]
    assert n_W_to_W + n_W_to_C == n_total - s["n_initially_correct"]
    assert n_C_to_C + n_W_to_C == s["n_correct_after_attack"]

    clean_acc = s["n_initially_correct"] / (n_total + 1e-9) # 在这里evaluate的样本上(e.g. 40)
    robust_acc = s["n_correct_after_attack"] / (n_total + 1e-9) # 攻击后正确率
    # 条件ASR：在原本分类正确的样本中，被攻击成功的比例
    asr_cond = s["n_flipped_to_wrong_C_to_W"] / (s["n_initially_correct"] + 1e-9)
    # 总体ASR: 分母是所有样本
    asr_overall = s["n_flipped_to_wrong_C_to_W"] / (n_total + 1e-9)

    print(f"\n--- {model_key.upper()} Model ---")
    print("  --- 性能指标 ---")
    print(f"  Clean Accuracy (ACC)              = {clean_acc:.4f}  ({s['n_initially_correct']}/{n_total})")
    print(f"  Robust Accuracy (Rob-Acc)         = {robust_acc:.4f}  ({s['n_correct_after_attack']}/{n_total})")
    print(f"  Attack Success Rate (ASR_cond)    = {asr_cond:.4f}  ({s['n_flipped_to_wrong_C_to_W']}/{s['n_initially_correct']})")
    print(f"  Attack Success Rate (ASR_overall) = {asr_overall:.4f}  ({s['n_flipped_to_wrong_C_to_W']}/{n_total})")
    print("  --- 详细计数 ---")
    print(f"  Correct -> Correct (未攻破): {n_C_to_C}")
    print(f"  Correct -> Wrong (被攻破):   {n_C_to_W}")
    print(f"  Wrong   -> Wrong (保持错):   {n_W_to_W}")
    print(f"  Wrong   -> Correct (被修正): {n_W_to_C}")
    print(f"  ---------------------------------")
    print(f"  总计 (Check): {n_C_to_C + n_C_to_W + n_W_to_W + n_W_to_C}")


    summary_metrics = {
        "Clean Accuracy (ACC)": clean_acc,
        "Robust Accuracy (Rob-Acc)": robust_acc,
        "ASR_cond_on_correct": asr_cond,
        "ASR_overall": asr_overall,
    }
    raw_counts = s.copy()
    raw_counts.update({
        "n_correct_after_attack_C_to_C": n_C_to_C,
        "n_stayed_wrong_W_to_W": n_W_to_W,
    })

    return {"summary_metrics": summary_metrics, "raw_counts": raw_counts}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--baseline_ckpt", type=str, required=True,
                        help="如 out/pythia410m/harmless/best_Harmless.pt")
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--rank_r", type=int, default=4)
    parser.add_argument("--run_dir", type=str, required=True,
                        help="训练 run 的 out_dir，里面有 final_intervention/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="AlignmentResearch/Harmless")
    parser.add_argument("--input_field", type=str, default="content")
    parser.add_argument("--label_field", type=str, default="clf_label")
    parser.add_argument("--position", type=str, default="l1")
    parser.add_argument("--max_eval_samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default=None)
    
    # GCG 超参
    parser.add_argument("--attack_start", type=int, default=5)
    parser.add_argument("--n_attack_tokens", type=int, default=10)
    parser.add_argument("--beam_k", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)

    args = parser.parse_args()
    set_seed(args.seed)

    # === Logging：类似 train，把 log 写到 run_dir 里 ===
    log_file = os.path.join(args.run_dir, "eval_token_gcg.log")
    os.makedirs(args.run_dir, exist_ok=True)

    hf_logging.set_verbosity_info()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    # 简单起见，用 transformers 的 logger + 手动写一行说明
    logger.info(f"[Eval-GCG] Logging to {log_file}")

    with open("src/data/harmless_splits.json", "r") as f:
        splits = json.load(f)

    attack_indices = splits["attack"] 
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    final_intervention_dir = os.path.join(args.run_dir, "final_intervention")
    if not os.path.isdir(final_intervention_dir):
        raise FileNotFoundError(f"{final_intervention_dir} 不存在（确认 run_dir 填的是训练 run 的 out_dir）")

    # 1. 加载 ReFT 模型
    reft_model, tokenizer = load_reft_model_for_eval(
        model_name=args.model_name,
        baseline_ckpt=args.baseline_ckpt,
        layer_idx=args.layer_idx,
        rank_r=args.rank_r,
        adv_intervention_dir=final_intervention_dir,
        device=str(device),
        attack_layer=None,
    )
    reft_model.to(device)

    

    # 2. 构建 attack dataset
    attack_dataset, _ = build_reft_classification_datasets(
        tokenizer=tokenizer,
        data_path=args.dataset,
        train_split="train",        
        eval_split="validation",
        train_indices=attack_indices,
        input_field=args.input_field,
        label_field=args.label_field,
        position=args.position,
        num_interventions=1,
    )

    n_total = len(attack_dataset)
    n_eval = min(args.max_eval_samples, n_total) if args.max_eval_samples > 0 else n_total
    print(f"[Token GCG Attack] 总样本={n_total}, 本次评估={n_eval}")

    stats = {
        "n_initially_correct": 0,
        "n_correct_after_attack": 0,
        "n_flipped_to_wrong_C_to_W": 0,
        "n_flipped_to_correct_W_to_C": 0,
    }

    all_results = []

    for idx in tqdm(range(n_eval), desc="Token-level GCG eval"):
        #pdb.set_trace()
        sample = attack_dataset[idx]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        label = int(sample["labels"].item())
        inter_locs = sample["intervention_locations"]

        orig_pred, orig_loss, adv_pred, adv_loss, success, adv_ids = token_level_gcg_single_reft(
            reft_model=reft_model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            true_label=label,
            intervention_locations=inter_locs,
            device=device,
            n_attack_tokens=args.n_attack_tokens,
            attack_start=args.attack_start,
            beam_k=args.beam_k,
            rounds=args.rounds,
        )

        # 更新统计
        if orig_pred == label:
            stats["n_initially_correct"] += 1
        if adv_pred == label:
            stats["n_correct_after_attack"] += 1
        if (orig_pred == label) and (adv_pred != label):
            stats["n_flipped_to_wrong_C_to_W"] += 1
        if (orig_pred != label) and (adv_pred == label):
            stats["n_flipped_to_correct_W_to_C"] += 1

        orig_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        adv_text = tokenizer.decode(adv_ids, skip_special_tokens=True)

        MAX_PRINT = 10

        all_results.append({
            "idx": idx,
            "label": label,
            "orig_pred": orig_pred,
            "adv_pred": adv_pred,
            "orig_loss": orig_loss,
            "adv_loss": adv_loss,
            "success": bool(success),
            "orig_text": orig_text,
            "adv_text": adv_text,
        })
        
        if success and stats["n_flipped_to_wrong_C_to_W"] <= MAX_PRINT:
            print("\n====== [Attack Success Example] ======")
            print(f"Sample idx = {idx}, label = {label}, orig_pred = {orig_pred}, adv_pred = {adv_pred}")
            print("[ORIG]")
            print(orig_text)
            print("[ADV]")
            print(adv_text)
            print("======================================")

    # 3. 计算并打印指标
    summary = calculate_and_print_stats("reft_latent_adv", stats, n_samples=n_eval)

    # 4. 可选：存 JSON
    if args.output_json is not None:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        out_obj = {
            "summary": summary,
            "per_example": all_results,
            "config": vars(args),
        }
        with open(args.output_json, "w") as f:
            json.dump(out_obj, f, indent=2, ensure_ascii=False)
        print(f"[Eval] 结果已保存到 {args.output_json}")


if __name__ == "__main__":
    print("=== Token-level GCG Attack on ReFT Model ===")
    main()
# CUDA_VISIBLE_DEVICES=0 python -m src.attack.eval_token_gcg   --baseline_ckpt out/pythia410m/harmless/best_Harmless.pt   --layer_idx 16   --rank_r 4   --run_dir out/pythia410m/harmless/reft_latent_adv/seed42_L16_A11/seed42_Rrandom_L16_A11_r4_atklatent_pgd_task9   --max_eval_samples 40   --output_json results/gcg_token_L16_A11.json
