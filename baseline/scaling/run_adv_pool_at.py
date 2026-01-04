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
    ap.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Base HF model path or hub id, e.g. EleutherAI/pythia-410m",
    )
    ap.add_argument(
        "--model_pt",
        type=str,
        required=True,
        help="Path to finetuned .pt checkpoint, e.g. out/.../best_Harmless.pt",
    )

    # data
    ap.add_argument("--dataset", type=str, default="AlignmentResearch/Harmless")
    ap.add_argument("--splits_json", type=str, required=True, help="e.g. src/data/harmless_splits.json")
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

    # FLOPs profiling (DeepSpeed) for TRAINING ONLY (optional)
    ap.add_argument(
        "--profile_flops_steps",
        type=int,
        default=0,
        help="DeepSpeed profile first N training steps per round (0 disables)",
    )
    ap.add_argument(
        "--profile_flops_every",
        type=int,
        default=1,
        help="DeepSpeed profile every k steps within the profiled window",
    )

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

    # pad token fix
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # 2) load finetuned .pt weights
    n_missing, n_unexpected = load_state_dict_into_model(model, args.model_pt)
    print(f"[load] loaded {args.model_pt} into base model {args.base_model_path}")
    print(f"[load_state_dict] missing={n_missing}, unexpected={n_unexpected}")

    model.to(device)
    if args.use_fp16 and device.type == "cuda":
        model = model.half()
    model.eval()

    # Kaplan: param count (N)
    N_params = sum(p.numel() for p in model.parameters())
    print(f"[kaplan] N_params={N_params}")

    # 3) load dataset + splits json
    ds_train = load_dataset(args.dataset)["train"]
    with open(args.splits_json, "r") as f:
        splits = json.load(f)

    ft_train_idx: List[int] = splits["ft_train"]
    attack_idx: List[int] = splits["attack"]

    if len(ft_train_idx) == 0 or len(attack_idx) == 0:
        raise ValueError("Split indices are empty. Check splits_json")

    # clean pool (nclean)
    nclean = min(args.clean_pool_size, len(ft_train_idx))
    clean_pool_items = collect_clean_items(ds_train, ft_train_idx, nclean)

    # adv pool
    adv_pool = AdvPool()

    # logs: per-attacked-example record
    meta_path = os.path.join(args.output_dir, "adv_pool_log.jsonl")
    with open(meta_path, "w") as f:
        f.write("")

    # logs: per-round summary (appended)
    round_log_path = os.path.join(args.output_dir, "round_kaplan_log.jsonl")
    with open(round_log_path, "w") as f:
        f.write("")

    # global totals across rounds (Kaplan)
    kaplan_Ctrain_total = 0.0
    kaplan_Csearch_total = 0.0
    total_attacked_examples = 0

    # 4) adversarial training rounds
    for r in range(1, args.R + 1):
        # attack iterations schedule
        k = linear_schedule_k(r, args.R, args.kstart, args.kend)

        # round accumulators (Kaplan)
        kaplan_Csearch_round = 0.0
        kaplan_attack_n = 0

        # (a) attack n_new_adv examples from attack split against CURRENT model
        n_new = min(args.n_new_adv, len(attack_idx))
        sampled_attack_ids = random.sample(attack_idx, k=n_new)

        new_adv = []
        for idx in sampled_attack_ids:
            raw = get_item(ds_train, idx)

            # NOTE: requires gcg_attack_text to return:
            #   (adv_text, adv_loss, success, n_forward, n_backward, D_tokens)
            adv_text, adv_loss, success, nF, nB, D_tokens = gcg_attack_text(
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

            # Kaplan Csearch for this attacked example:
            # C = (2*nF + 4*nB) * N * D
            kaplan_Csearch_ex = (2.0 * float(nF) + 4.0 * float(nB)) * float(N_params) * float(D_tokens)
            kaplan_Csearch_round += kaplan_Csearch_ex
            kaplan_attack_n += 1

            with open(meta_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "round": int(r),
                            "k": int(k),
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
                            "n_attack_tokens": int(args.n_attack_tokens),
                            "beam_k": int(args.beam_k),
                            "n_candidates_per_it": int(args.n_candidates_per_it),

                            # pass counting (Kaplan)
                            "gcg_n_forward": int(nF),
                            "gcg_n_backward": int(nB),
                            "gcg_D_tokens": int(D_tokens),
                            "kaplan_Csearch_ex": float(kaplan_Csearch_ex),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

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
        # NOTE: requires train_one_round to return: (ckpt_dir, round_train_compute_dict)
        ckpt_dir, train_compute = train_one_round(
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
            kaplan_N_params=N_params,
            profile_flops_steps=args.profile_flops_steps,
            profile_flops_every=args.profile_flops_every,
        )

        # round totals
        kaplan_Ctrain_round = float(train_compute.get("kaplan_Ctrain_round", 0.0))
        kaplan_Cadv_round = kaplan_Ctrain_round + kaplan_Csearch_round

        kaplan_Ctrain_total += kaplan_Ctrain_round
        kaplan_Csearch_total += kaplan_Csearch_round
        total_attacked_examples += kaplan_attack_n

        round_summary = {
            "round": int(r),
            "R": int(args.R),
            "k": int(k),
            "pool_size": int(len(adv_pool)),
            "n_new_adv": int(kaplan_attack_n),

            # kaplan totals
            "kaplan_N_params": int(N_params),
            "kaplan_Ctrain_round": float(kaplan_Ctrain_round),
            "kaplan_Csearch_round": float(kaplan_Csearch_round),
            "kaplan_Cadv_round": float(kaplan_Cadv_round),
            "kaplan_Csearch_per_attacked_example": float(kaplan_Csearch_round) / max(int(kaplan_attack_n), 1),

            # training token totals (optional but useful)
            "kaplan_Dtrain_round_tokens": int(train_compute.get("kaplan_Dtrain_round_tokens", 0)),

            # deepspeed profiler summary (training only)
            "ds_prof_n": int(train_compute.get("ds_prof_n", 0)),
            "ds_prof_flops_step_mean": train_compute.get("ds_prof_flops_step_mean", None),

            # checkpoint
            "ckpt_dir": ckpt_dir,
        }

        # write per-round json (one file) + append to round log jsonl
        with open(os.path.join(args.output_dir, f"round_{r:03d}_kaplan_summary.json"), "w") as f:
            json.dump(round_summary, f, indent=2)
        with open(round_log_path, "a") as f:
            f.write(json.dumps(round_summary) + "\n")

        print(
            f"[round {r}/{args.R}] k={k} pool={len(adv_pool)} ckpt={ckpt_dir} | "
            f"Ctrain={kaplan_Ctrain_round:.3e} Csearch={kaplan_Csearch_round:.3e} Cadv={kaplan_Cadv_round:.3e} | "
            f"ds_mean={round_summary['ds_prof_flops_step_mean']}"
        )

    # global summary
    final_summary = {
        "dataset": args.dataset,
        "base_model_path": args.base_model_path,
        "model_pt": args.model_pt,
        "seed": int(args.seed),

        "R": int(args.R),
        "naug": int(args.naug),
        "adv_frac": float(args.adv_frac),
        "n_new_adv": int(args.n_new_adv),
        "kstart": int(args.kstart),
        "kend": int(args.kend),

        "kaplan_N_params": int(N_params),
        "kaplan_Ctrain_total": float(kaplan_Ctrain_total),
        "kaplan_Csearch_total": float(kaplan_Csearch_total),
        "kaplan_Cadv_total": float(kaplan_Ctrain_total + kaplan_Csearch_total),
        "kaplan_Ctrain_over_Csearch": float(kaplan_Ctrain_total / max(kaplan_Csearch_total, 1e-12)),

        "total_attacked_examples": int(total_attacked_examples),
        "kaplan_Csearch_per_attacked_example_overall": (
            float(kaplan_Csearch_total) / max(int(total_attacked_examples), 1)
        ),

        "logs": {
            "per_example_log": meta_path,
            "per_round_log": round_log_path,
        },
    }

    with open(os.path.join(args.output_dir, "kaplan_compute_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    print(f"Done.\nLogs: {meta_path}\nRound logs: {round_log_path}\nSummary: {os.path.join(args.output_dir, 'kaplan_compute_summary.json')}")
    print(f"Checkpoints: {args.output_dir}/round_*/")


if __name__ == "__main__":
    main()


"""
Example:

CUDA_VISIBLE_DEVICES=0 python -m baseline.scaling.run_adv_pool_at \
  --base_model_path EleutherAI/pythia-410m \
  --model_pt out/pythia410m/imdb/best_IMDB.pt \
  --dataset AlignmentResearch/IMDB \
  --splits_json src/data/imdb_splits.json \
  --output_dir baseline/scaling/advpool_runs/imdb_gcg_suffix_flops \
  --attack_mode suffix \
  --R 3 \
  --seed 42 \
  --profile_flops_steps 20 \
  --profile_flops_every 5
"""