# baseline/continuous_at/run_continuous_at.py
import os
import json
import random
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from baseline.continuous_at.embedding_attack import AttackConfig
from baseline.continuous_at.continuous_trainer import train_continuous_at


def load_state_dict_into_model(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_path", type=str, required=True)
    ap.add_argument("--model_pt", type=str, required=True)

    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)

    ap.add_argument("--output_dir", type=str, default="baseline/continuous_at/outputs")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train_examples", type=int, default=20000)

    # continuous attack hyperparams
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--norm", type=str, default="l2", choices=["l2", "linf"])
    ap.add_argument("--random_init", action="store_true")

    # outer training
    ap.add_argument("--mix_adv_frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    config = AutoConfig.from_pretrained(args.base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model_path, config=config)

    load_state_dict_into_model(model, args.model_pt)

    model.to(device)

    # load data + splits
    ds = load_dataset(args.dataset)["train"]
    with open(args.splits_json, "r") as f:
        splits = json.load(f)

    ft_train_idx = splits["ft_train"]
    if args.max_train_examples > 0:
        ft_train_idx = ft_train_idx[: min(len(ft_train_idx), args.max_train_examples)]

    train_items = [{"content": ds[i]["content"], "clf_label": int(ds[i]["clf_label"])} for i in ft_train_idx]

    out_dir = os.path.join(args.output_dir, f"eps{args.eps}_k{args.k}_{args.norm}_mix{args.mix_adv_frac}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    attack_cfg = AttackConfig(
        eps=args.eps,
        alpha=args.alpha,
        steps=args.k,
        norm=args.norm,
        random_init=args.random_init,
    )

    train_continuous_at(
        model=model,
        tokenizer=tokenizer,
        train_items=train_items,
        device=device,
        out_dir=out_dir,
        attack_cfg=attack_cfg,
        mix_adv_frac=args.mix_adv_frac,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()

"""CUDA_VISIBLE_DEVICES=3 python -m baseline.continuous_at.run_continuous_at \
  --base_model_path EleutherAI/pythia-410m \
  --model_pt out/pythia410m/helpful/best_Helpful.pt \
  --dataset AlignmentResearch/Helpful \
  --splits_json src/data/helpful_splits.json \
  --output_dir baseline/continuous_at/runs_helpful \
  --seed 42 \
  --mix_adv_frac 0.5 \
  --batch_size 8 --max_steps 1500
"""