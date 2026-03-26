from html import parser
import os
import json
import argparse
from re import split
from typing import Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.data.datasets_reft import build_reft_classification_datasets
from src.utils.tools import set_seed
from src.attack.eval_reft_adv import token_level_gcg_single_baseline
from src.attack.eval_token_gcg import calculate_and_print_stats

logger = hf_logging.get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="AlignmentResearch/Harmless")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--input_field", type=str, default="content")
    parser.add_argument("--label_field", type=str, default="clf_label")
    parser.add_argument("--position", type=str, default="l1")
    parser.add_argument("--max_eval_samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default=None)

    # GCG 超参
    parser.add_argument("--attack_start", type=int, default=0)
    parser.add_argument("--n_attack_tokens", type=int, default=5)
    parser.add_argument("--beam_k", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--attack_mode", type=str, default="suffix")
    parser.add_argument("--n_candidates_per_it", type=int, default=128)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 加载模型与 Tokenizer
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    state = torch.load(args.baseline_ckpt, map_location="cpu")
    base_model.load_state_dict(state, strict=False)
    base_model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # GPT-2 特殊处理
    if "gpt2" in args.model_name:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512

    # 2) 加载攻击数据集
    with open(args.split, "r") as f:
        splits = json.load(f)
    attack_indices = splits["attack"]

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

    n_eval = min(args.max_eval_samples, len(attack_dataset)) if args.max_eval_samples > 0 else len(attack_dataset)
    print(f"[Token GCG Attack] 评估样本数: {n_eval}")

    # 统计初始化
    stats = {
        "n_initially_correct": 0,
        "n_correct_after_attack": 0,
        "n_flipped_to_wrong_C_to_W": 0,
        "n_flipped_to_correct_W_to_C": 0,
    }
    performance_metrics = {"total_fwd": 0, "total_bwd": 0, "total_toks": 0}
    all_results = []

    for idx in tqdm(range(n_eval), desc="GCG eval"):
        sample = attack_dataset[idx]
        input_ids = sample["input_ids"]
        attention_mask = sample.get("attention_mask", torch.ones_like(input_ids))
        label = int(sample["labels"].item())

        # 调用函数并解包 9 个值
        (orig_pred, orig_loss, adv_pred, adv_loss, success, adv_ids, 
         n_fwd, n_bwd, n_toks) = token_level_gcg_single_baseline(
            model=base_model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            true_label=label,
            device=device,
            n_attack_tokens=args.n_attack_tokens,
            attack_start=args.attack_start,
            beam_k=args.beam_k,
            rounds=args.rounds,
            attack_mode=args.attack_mode,
            n_candidates_per_it=args.n_candidates_per_it,
        )

        # 更新统计
        performance_metrics["total_fwd"] += n_fwd
        performance_metrics["total_bwd"] += n_bwd
        performance_metrics["total_toks"] += n_toks

        if orig_pred == label: stats["n_initially_correct"] += 1
        if adv_pred == label: stats["n_correct_after_attack"] += 1
        if (orig_pred == label) and (adv_pred != label): stats["n_flipped_to_wrong_C_to_W"] += 1
        if (orig_pred != label) and (adv_pred == label): stats["n_flipped_to_correct_W_to_C"] += 1

        all_results.append({
            "idx": idx, "label": label, "orig_pred": orig_pred, "adv_pred": adv_pred,
            "orig_loss": orig_loss, "adv_loss": adv_loss, "success": bool(success),
            "n_fwd": n_fwd, "n_bwd": n_bwd,
            "orig_text": tokenizer.decode(input_ids, skip_special_tokens=True),
            "adv_text": tokenizer.decode(adv_ids, skip_special_tokens=True),
        })

    # 计算 Summary
    summary = calculate_and_print_stats("baseline", stats, n_samples=n_eval)
    summary.update({
        "avg_fwd": performance_metrics["total_fwd"] / n_eval,
        "avg_bwd": performance_metrics["total_bwd"] / n_eval,
        "total_fwd": performance_metrics["total_fwd"],
        "total_bwd": performance_metrics["total_bwd"]
    })

    # 保存
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"summary": summary, "per_example": all_results, "config": vars(args)}, f, indent=2)


if __name__ == "__main__":
    print("=== Token-level GCG Attack on BASELINE Harmless Model ===")
    main()

'''
CUDA_VISIBLE_DEVICES=2 python -m src.attack.eval_token_gcg_baseline \
  --model_name gpt2 \
  --baseline_ckpt outputs/imdb/best_IMDB.pt \
  --dataset AlignmentResearch/IMDB \
  --split src/data/imdb_splits.json \
  --max_eval_samples 100 \
  --n_attack_tokens 10 \
  --beam_k 256 \
  --rounds 20 \
  --attack_mode suffix \
  --n_candidates_per_it 128 \
  --output_json outputs/imdb/suffix_baseline_eval_10tokens_20rounds.json
'''

