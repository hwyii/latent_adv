import os
import json
import random
from tqdm import tqdm
import argparse
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.attack.eval_reft_adv import token_level_gcg_single_baseline


def load_model_and_tokenizer(ckpt_dir: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)

    # Pythia/GPT-NeoX: pad token 必须定义，否则 batch>1/某些 forward 会报错
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


def eval_one_checkpoint(
    ckpt_dir: str,
    dataset_name: str,
    splits_json: str,
    split_name: str,
    device: torch.device,
    max_eval: int,
    seed: int,
    gcg_kwargs: Dict[str, Any],
):
    model, tok = load_model_and_tokenizer(ckpt_dir, device)

    ds = load_dataset(dataset_name)["train"]
    with open(splits_json, "r") as f:
        splits = json.load(f)

    idxs: List[int] = splits[split_name][:100]
    rng = random.Random(seed)
    if max_eval > 0:
        idxs = rng.sample(idxs, k=min(max_eval, len(idxs)))

    n = 0
    n_flip = 0
    n_correct_clean = 0
    clean_losses = []
    adv_losses = []

    for idx in tqdm(idxs, desc=f"GCG eval on {os.path.basename(ckpt_dir)}", ncols=100):
        item = ds[idx]
        text = item["content"]
        y = int(item["clf_label"])

        enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        # clean forward (可选：算 clean acc/loss)
        with torch.no_grad():
            out = model(
                input_ids=input_ids.unsqueeze(0).to(device),
                attention_mask=attention_mask.unsqueeze(0).to(device),
                labels=torch.tensor([y], device=device),
            )
            clean_logits = out.logits
            clean_pred = int(clean_logits.argmax(-1).item())
            clean_loss = float(F.cross_entropy(clean_logits, torch.tensor([y], device=device)).item())

        # GCG attack
        orig_pred, orig_loss, adv_pred, adv_loss, success, adv_ids = token_level_gcg_single_baseline(
            model=model,
            tokenizer=tok,
            input_ids=input_ids,
            attention_mask=attention_mask,
            true_label=y,
            device=device,
            **gcg_kwargs,
        )

        # 统计：你现在的 success 定义是 adv_pred != orig_pred（预测翻转）
        n += 1
        n_flip += int(success)
        n_correct_clean += int(clean_pred == y)
        clean_losses.append(clean_loss)
        adv_losses.append(float(adv_loss))

        if (n % 50) == 0:
            print(f"[{os.path.basename(ckpt_dir)}] {n}/{len(idxs)} ASR={n_flip/n:.3f}")

    result = {
        "ckpt_dir": ckpt_dir,
        "split": split_name,
        "n_eval": n,
        "asr_flip_pred": n_flip / max(1, n),          # adv_pred != orig_pred
        "clean_acc": n_correct_clean / max(1, n),
        "clean_loss_mean": sum(clean_losses) / max(1, n),
        "adv_loss_mean": sum(adv_losses) / max(1, n),
        "gcg": gcg_kwargs,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, nargs="+", required=True,
                    help="run dirs, e.g. baseline/continuous_at/runs/eps0.05_k10_l2_mix0.5_seed27")
    ap.add_argument("--ckpt_subdir", type=str, default="final_ckpt",
                    help="subdir inside each run dir that stores HF checkpoint")
    ap.add_argument("--dataset", type=str, default="AlignmentResearch/Harmless")
    ap.add_argument("--splits_json", type=str, default="src/data/harmless_splits.json")
    ap.add_argument("--split", type=str, default="attack", choices=["attack", "probe", "ft_train"])
    ap.add_argument("--max_eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="baseline/continuous_at/runs_harmless/gcg_eval_results.jsonl")

    # GCG params
    ap.add_argument("--attack_mode", type=str, default="suffix", choices=["suffix", "replace", "infix"])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--beam_k", type=int, default=256)
    ap.add_argument("--n_candidates_per_it", type=int, default=128)
    ap.add_argument("--n_attack_tokens", type=int, default=10)
    ap.add_argument("--attack_start", type=int, default=5)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gcg_kwargs = dict(
        n_attack_tokens=args.n_attack_tokens,
        attack_start=args.attack_start,
        beam_k=args.beam_k,
        rounds=args.rounds,
        forbid_special=True,
        attack_mode=args.attack_mode,
        n_candidates_per_it=args.n_candidates_per_it,
    )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    all_results = []
    for run_dir in args.runs:
        ckpt_dir = os.path.join(run_dir, args.ckpt_subdir)
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

        res = eval_one_checkpoint(
            ckpt_dir=ckpt_dir,
            dataset_name=args.dataset,
            splits_json=args.splits_json,
            split_name=args.split,
            device=device,
            max_eval=args.max_eval,
            seed=args.seed,
            gcg_kwargs=gcg_kwargs,
        )
        all_results.append(res)
        print(json.dumps(res, indent=2))

        with open(args.out_json, "a") as f:
            f.write(json.dumps(res) + "\n")

    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
