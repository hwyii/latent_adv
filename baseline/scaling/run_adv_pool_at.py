# baseline/scaling/run_adv_pool_at.py
import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

from baseline.scaling.gcg_attack import gcg_attack_text
from baseline.scaling.adv_pool_trainer import (
    AdvPool,
    build_round_items,
    train_one_round,
    linear_schedule_k,
)


def load_state_dict_into_model(model: torch.nn.Module, ckpt_path: str) -> Tuple[int, int]:
    """
    Load a .pt checkpoint into a HF model.
    Supports:
      - state dict directly
      - {'state_dict': ...}
      - {'model_state_dict': ...}
    Also strips common prefixes: 'module.' and 'model.'.

    Returns: (num_missing_keys, num_unexpected_keys)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in {ckpt_path}: {type(state)}")

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return len(missing), len(unexpected)


def get_item(ds_train, idx: int) -> Dict[str, Any]:
    return {
        "content": ds_train[idx]["content"],
        "clf_label": int(ds_train[idx]["clf_label"]),
    }


def collect_clean_items(ds_train, indices: List[int], n: int) -> List[Dict[str, Any]]:
    n = min(n, len(indices))
    chosen = indices[:n]  # deterministic slice; randomness happens per-round
    return [get_item(ds_train, i) for i in chosen]


def main():
    ap = argparse.ArgumentParser()

    # model loading
    ap.add_argument("--base_model_path", type=str, required=True,
                    help="Base HF model path or hub id, e.g. EleutherAI/pythia-410m")
    ap.add_argument("--model_pt", type=str, required=True,
                    help="Path to finetuned .pt checkpoint, e.g. out/.../best_Harmless.pt")

    # data
    ap.add_argument("--dataset", type=str, default="AlignmentResearch/Harmless")
    ap.add_argument("--splits_json", type=str, required=True,
                    help="e.g. src/data/harmless_splits.json")
    ap.add_argument("--clean_pool_size", type=int, default=20000)

    # output
    ap.add_argument("--output_dir", type=str, default="baseline/scaling/outputs_advpool")

    # adversarial training rounds / pool
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--n_new_adv", type=int, default=200)
    ap.add_argument("--naug", type=int, default=1000)
    ap.add_argument("--adv_frac", type=float, default=0.8)
    ap.add_argument("--lam", type=float, default=0.005)

    # attack schedule (GCG iterations per round)
    ap.add_argument("--kstart", type=int, default=8)
    ap.add_argument("--kend", type=int, default=64)

    # GCG params
    ap.add_argument("--attack_mode", type=str, default="suffix", choices=["suffix", "infix", "replace"])
    ap.add_argument("--n_attack_tokens", type=int, default=10)
    ap.add_argument("--beam_k", type=int, default=256)
    ap.add_argument("--n_candidates_per_it", type=int, default=128)

    # training params per round (fixed compute)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    # misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_fp16", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load base tokenizer + config + model skeleton
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    
    config = AutoConfig.from_pretrained(args.base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model_path, config=config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # common practice for causal LMs
        # Ensure model knows pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    # 2) load finetuned .pt weights
    n_missing, n_unexpected = load_state_dict_into_model(model, args.model_pt)
    print(f"[load] loaded {args.model_pt} into base model {args.base_model_path}")
    print(f"[load_state_dict] missing={n_missing}, unexpected={n_unexpected}")

    model.to(device)
    if args.use_fp16 and device.type == "cuda":
        model = model.half()
    model.eval()

    # 3) load dataset + splits json
    ds_train = load_dataset(args.dataset)["train"]
    with open(args.splits_json, "r") as f:
        splits = json.load(f)

    ft_train_idx: List[int] = splits["ft_train"]
    attack_idx: List[int] = splits["attack"]

    if len(ft_train_idx) == 0 or len(attack_idx) == 0:
        raise ValueError("Split indices are empty. Check harmless_splits.json")

    # clean pool (nclean)
    nclean = min(args.clean_pool_size, len(ft_train_idx))
    clean_pool_items = collect_clean_items(ds_train, ft_train_idx, nclean)

    # adv pool
    adv_pool = AdvPool()

    # logs
    meta_path = os.path.join(args.output_dir, "adv_pool_log.jsonl")
    with open(meta_path, "w") as f:
        f.write("")

    # 4) adversarial training rounds
    for r in range(1, args.R + 1):
        # attack iterations schedule
        k = linear_schedule_k(r, args.R, args.kstart, args.kend)

        # (a) attack n_new_adv examples from attack split against CURRENT model
        n_new = min(args.n_new_adv, len(attack_idx))
        sampled_attack_ids = random.sample(attack_idx, k=n_new)

        new_adv = []
        for idx in sampled_attack_ids:
            raw = get_item(ds_train, idx)

            adv_text, adv_loss, success = gcg_attack_text(
                model=model,
                tokenizer=tokenizer,
                text=raw["content"],
                label=raw["clf_label"],
                device=device,
                rounds=k,
                n_attack_tokens=args.n_attack_tokens,
                attack_mode=args.attack_mode,
                beam_k=args.beam_k,
                n_candidates_per_it=args.n_candidates_per_it,
            )

            attacked_item = {"content": adv_text, "clf_label": raw["clf_label"]}
            new_adv.append((attacked_item, float(adv_loss), bool(success)))

            with open(meta_path, "a") as f:
                f.write(json.dumps({
                    "round": r,
                    "k": k,
                    "attack_seed_ds_idx": int(idx),

                    # before / after text
                    "orig_text": raw["content"],
                    "adv_text": adv_text,

                    # labels & outcome
                    "label": int(raw["clf_label"]),
                    "success": bool(success),
                    "adv_loss": float(adv_loss),

                    # attack config
                    "attack_mode": args.attack_mode,
                }, ensure_ascii=False) + "\n")

        # (b) add to pool
        adv_pool.add_many(new_adv, round_id=r)

        # (c) sample training set for this round (size fixed naug)
        round_items = build_round_items(
            clean_items=clean_pool_items,
            adv_pool=adv_pool,
            naug=args.naug,
            adv_frac=args.adv_frac,
            lam=args.lam,
        )

        # (d) train fixed compute & save checkpoint
        ckpt_dir = train_one_round(
            model=model,
            tokenizer=tokenizer,
            round_items=round_items,
            device=device,
            lr=args.lr,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            weight_decay=args.weight_decay,
            save_dir=args.output_dir,
            round_id=r,
        )

        print(f"[round {r}/{args.R}] k={k} pool={len(adv_pool)} ckpt={ckpt_dir}")

    print(f"Done.\nLogs: {meta_path}\nCheckpoints: {args.output_dir}/round_*/")


if __name__ == "__main__":
    main()


"""
CUDA_VISIABLE_DEVICES=2 python -m baseline.scaling.run_adv_pool_at \
  --base_model_path EleutherAI/pythia-410m \
  --model_pt out/pythia410m/harmless/best_Harmless.pt \
  --splits_json src/data/harmless_splits.json \
  --output_dir baseline/scaling/advpool_runs/harmless_gcg_suffix \
  --attack_mode suffix \
  --R 8 \
  --seed 42


"""