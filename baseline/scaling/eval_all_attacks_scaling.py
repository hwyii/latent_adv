# baseline/scaling/eval_all_attacks_scaling.py
"""
Unified eval script for advpool checkpoints.
Supports --attack_type {gcg, beast, random}.
Output: {out_root}/{round_name}/{attack_type}_eval_per_example.jsonl
        {out_root}/{round_name}/{attack_type}_eval_summary.json
"""
import os
import json
import random
import argparse
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from src.attack.eval_reft_adv import (
    token_level_gcg_single_baseline,
    token_level_random_single_baseline,
    beast_attack_single_baseline,
)


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_model_and_tokenizer(ckpt_dir: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


@torch.no_grad()
def clean_forward_one(model, input_ids, attention_mask, y, device):
    ids = input_ids.unsqueeze(0).to(device)
    mask = attention_mask.unsqueeze(0).to(device)
    labels = torch.tensor([y], device=device)
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    pred = int(out.logits.argmax(-1).item())
    loss = float(F.cross_entropy(out.logits, labels).item())
    return pred, loss


def pick_eval_indices(splits: dict, split_name: str, max_eval: int, seed: int) -> List[int]:
    idxs = splits[split_name][:100]
    if max_eval > 0 and max_eval < len(idxs):
        rng = random.Random(seed)
        idxs = rng.sample(idxs, k=max_eval)
    return idxs


_DEFAULT_BEAST_BASE = {
    "pythia": "EleutherAI/pythia-14m",
    "qwen":   "Qwen/Qwen2.5-0.5B",
    "llama":  "meta-llama/Llama-3.2-1B",
}


def infer_beast_base_model(ckpt_dir: str) -> str:
    """Infer the small base-LM name from the HF config in a round checkpoint."""
    try:
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg = json.load(f)
        name = (cfg.get("_name_or_path", "") + cfg.get("model_type", "")).lower()
    except Exception:
        name = ""
    if "pythia" in name:
        return _DEFAULT_BEAST_BASE["pythia"]
    if "qwen" in name:
        return _DEFAULT_BEAST_BASE["qwen"]
    if "llama" in name:
        return _DEFAULT_BEAST_BASE["llama"]
    return _DEFAULT_BEAST_BASE["pythia"]  # fallback


# -----------------------------------------------------------------------
# core eval loop (one round directory)
# -----------------------------------------------------------------------

def eval_one_round(
    round_dir: str,
    dataset_name: str,
    splits_json: str,
    split_name: str,
    device: torch.device,
    max_eval: int,
    seed: int,
    attack_type: str,
    attack_kwargs: Dict[str, Any],
    out_root: str,
    save_text: bool,
    base_model=None,
    base_device: Optional[torch.device] = None,
):
    model, tok = load_model_and_tokenizer(round_dir, device)

    ds = load_dataset(dataset_name)["train"]
    with open(splits_json) as f:
        splits = json.load(f)

    idxs = pick_eval_indices(splits, split_name, max_eval, seed)
    round_name = os.path.basename(round_dir.rstrip("/"))
    out_dir = os.path.join(out_root, round_name)
    ensure_dir(out_dir)

    per_ex_path = os.path.join(out_dir, f"{attack_type}_eval_per_example.jsonl")
    summary_path = os.path.join(out_dir, f"{attack_type}_eval_summary.json")

    n = n_clean_correct = n_asr = n_flip = 0
    clean_losses: List[float] = []
    adv_losses: List[float] = []

    random.seed(seed)
    torch.manual_seed(seed)

    with open(per_ex_path, "w") as fout:
        for idx in tqdm(idxs, desc=f"{attack_type} eval {round_name}", ncols=100):
            item = ds[idx]
            text = item["content"]
            y = int(item["clf_label"])

            enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]

            clean_pred, clean_loss = clean_forward_one(model, input_ids, attention_mask, y, device)

            if attack_type == "gcg":
                orig_pred, orig_loss, adv_pred, adv_loss, success, adv_ids = (
                    token_level_gcg_single_baseline(
                        model=model, tokenizer=tok,
                        input_ids=input_ids, attention_mask=attention_mask,
                        true_label=y, device=device,
                        **attack_kwargs,
                    )
                )
            elif attack_type == "random":
                orig_pred, orig_loss, adv_pred, adv_loss, success, adv_ids, _ = (
                    token_level_random_single_baseline(
                        model=model, tokenizer=tok,
                        input_ids=input_ids, attention_mask=attention_mask,
                        true_label=y, device=device,
                        **attack_kwargs,
                    )
                )
            elif attack_type == "beast":
                orig_pred, orig_loss, adv_pred, adv_loss, success, adv_ids, _ = (
                    beast_attack_single_baseline(
                        model=model, tokenizer=tok,
                        input_ids=input_ids, attention_mask=attention_mask,
                        true_label=y, device=device,
                        base_model=base_model, base_device=base_device,
                        **attack_kwargs,
                    )
                )

            clean_correct = (clean_pred == y)
            adv_wrong = (adv_pred != y)
            asr_hit = clean_correct and adv_wrong
            flip = (adv_pred != clean_pred)

            n += 1
            n_clean_correct += int(clean_correct)
            n_asr += int(asr_hit)
            n_flip += int(flip)
            clean_losses.append(clean_loss)
            adv_losses.append(adv_loss)

            rec = {
                "ds_idx": int(idx),
                "label": int(y),
                "clean_pred": int(clean_pred),
                "clean_correct": bool(clean_correct),
                "clean_loss": float(clean_loss),
                "adv_pred": int(adv_pred),
                "adv_wrong": bool(adv_wrong),
                "adv_loss": float(adv_loss),
                "asr_overall_hit": bool(asr_hit),
                "flip_pred": bool(flip),
                "attack_type": attack_type,
                "attack_kwargs": {k: v for k, v in attack_kwargs.items()},
                "eval_seed": int(seed),
            }
            if save_text:
                rec["text"] = text
                try:
                    rec["adv_text"] = tok.decode(adv_ids.tolist(), skip_special_tokens=True)
                except Exception:
                    rec["adv_text"] = ""
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if n % 50 == 0:
                print(
                    f"[{round_name}] {n}/{len(idxs)} "
                    f"ASR={n_asr/n:.3f} clean_acc={n_clean_correct/n:.3f} flip={n_flip/n:.3f}"
                )

    summary = {
        "round_dir": round_dir,
        "dataset": dataset_name,
        "split": split_name,
        "attack_type": attack_type,
        "n_eval": n,
        "clean_acc": n_clean_correct / max(1, n),
        "asr_overall": n_asr / max(1, n),
        "flip_rate": n_flip / max(1, n),
        "clean_loss_mean": sum(clean_losses) / max(1, n),
        "adv_loss_mean": sum(adv_losses) / max(1, n),
        "attack_kwargs": {k: v for k, v in attack_kwargs.items()},
        "eval_seed": int(seed),
        "per_example_jsonl": per_ex_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {per_ex_path}")
    print(f"[saved] {summary_path}")
    return summary


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack_type", type=str, required=True, choices=["gcg", "beast", "random"])
    ap.add_argument("--round_dirs", type=str, nargs="+", required=True,
                    help="HF checkpoint dirs saved by train_one_round, e.g. .../round_001")
    ap.add_argument("--dataset", type=str, default="AlignmentResearch/Harmless")
    ap.add_argument("--splits_json", type=str, default="src/data/harmless_splits.json")
    ap.add_argument("--split", type=str, default="attack", choices=["attack", "probe", "ft_train"])
    ap.add_argument("--max_eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--save_text", action="store_true")

    # GCG params
    ap.add_argument("--attack_mode", type=str, default="suffix", choices=["suffix", "replace", "infix"])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--beam_k", type=int, default=256)
    ap.add_argument("--n_candidates_per_it", type=int, default=128)
    ap.add_argument("--n_attack_tokens", type=int, default=10)
    ap.add_argument("--attack_start", type=int, default=5)

    # BEAST params
    ap.add_argument("--beast_base_model", type=str, default=None,
                    help="Small causal LM for BEAST token sampling. Auto-inferred if omitted.")
    ap.add_argument("--beast_n_tokens", type=int, default=25)
    ap.add_argument("--beast_beam_size", type=int, default=7)

    # RandomToken params
    ap.add_argument("--random_rounds", type=int, default=200)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_root)

    # Validate round dirs
    for rd in args.round_dirs:
        if not os.path.isdir(rd):
            raise FileNotFoundError(f"Round dir not found: {rd}")

    # Build attack kwargs and (optionally) load base model for BEAST
    base_model = None
    base_device = None

    if args.attack_type == "gcg":
        attack_kwargs = dict(
            n_attack_tokens=args.n_attack_tokens,
            attack_start=args.attack_start,
            beam_k=args.beam_k,
            rounds=args.rounds,
            forbid_special=True,
            attack_mode=args.attack_mode,
            n_candidates_per_it=args.n_candidates_per_it,
        )

    elif args.attack_type == "random":
        attack_kwargs = dict(
            n_attack_tokens=args.n_attack_tokens,
            rounds=args.random_rounds,
            attack_mode=args.attack_mode,
        )

    elif args.attack_type == "beast":
        attack_kwargs = dict(
            n_tokens=args.beast_n_tokens,
            beam_size=args.beast_beam_size,
        )
        beast_base = args.beast_base_model or infer_beast_base_model(args.round_dirs[0])
        print(f"[BEAST] loading base model: {beast_base}")
        base_model = AutoModelForCausalLM.from_pretrained(beast_base).to(device).eval()
        base_device = device

    for rd in args.round_dirs:
        eval_one_round(
            round_dir=rd,
            dataset_name=args.dataset,
            splits_json=args.splits_json,
            split_name=args.split,
            device=device,
            max_eval=args.max_eval,
            seed=args.seed,
            attack_type=args.attack_type,
            attack_kwargs=attack_kwargs,
            out_root=args.out_root,
            save_text=args.save_text,
            base_model=base_model,
            base_device=base_device,
        )


if __name__ == "__main__":
    main()
