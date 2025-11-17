from html import parser
import os
import json
import argparse
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
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--baseline_ckpt", type=str, required=True,
                        help="如 out/pythia410m/harmless/best_Harmless.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="AlignmentResearch/Harmless")
    parser.add_argument("--input_field", type=str, default="content")
    parser.add_argument("--label_field", type=str, default="clf_label")
    parser.add_argument("--position", type=str, default="l1")
    parser.add_argument("--max_eval_samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default=None)

    # GCG 超参（保持和 ReFT eval 一致）
    parser.add_argument("--attack_start", type=int, default=0)
    parser.add_argument("--n_attack_tokens", type=int, default=5)
    parser.add_argument("--beam_k", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=20)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 加载 baseline 模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    state = torch.load(args.baseline_ckpt, map_location="cpu")
    base_model.load_state_dict(state, strict=False)
    base_model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512  # 和训练一致

    # 2) attack split
    with open("src/data/harmless_splits.json", "r") as f:
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

    n_total = len(attack_dataset)
    n_eval = min(args.max_eval_samples, n_total) if args.max_eval_samples > 0 else n_total
    print(f"[Token GCG Attack BASELINE] 总样本={n_total}, 本次评估={n_eval}")

    stats = {
        "n_initially_correct": 0,
        "n_correct_after_attack": 0,
        "n_flipped_to_wrong_C_to_W": 0,
        "n_flipped_to_correct_W_to_C": 0,
    }
    all_results = []

    for idx in tqdm(range(n_eval), desc="Token-level GCG eval (baseline)"):
        sample = attack_dataset[idx]
        input_ids = sample["input_ids"]
        attention_mask = sample.get("attention_mask", torch.ones_like(input_ids))
        label = int(sample["labels"].item())

        orig_pred, orig_loss, adv_pred, adv_loss, success, adv_ids = token_level_gcg_single_baseline(
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
        )

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

    summary = calculate_and_print_stats("baseline", stats, n_samples=n_eval)

    if args.output_json is None:
        args.output_json = "out/pythia410m/harmless/baseline_eval_token_gcg.json"

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"summary": summary, "per_example": all_results, "config": vars(args)},
                  f, indent=2, ensure_ascii=False)
    print(f"[Eval-Baseline] 结果已保存到 {args.output_json}")


if __name__ == "__main__":
    print("=== Token-level GCG Attack on BASELINE Harmless Model ===")
    main()

'''
CUDA_VISIBLE_DEVICES=0 python -m src.attack.eval_token_gcg_baseline \
  --baseline_ckpt out/pythia410m/harmless/best_Harmless.pt \
  --max_eval_samples 40 \
  --attack_start 0 \
  --n_attack_tokens 5 \
  --beam_k 512 \
  --rounds 20
'''

