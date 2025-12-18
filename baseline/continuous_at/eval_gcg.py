import os
import json
import random
import argparse
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.attack.eval_reft_adv import token_level_gcg_single_baseline


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_model_and_tokenizer(ckpt_dir: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    # Pythia/GPT-NeoX: 必须有 pad token，否则 batch>1/部分 forward 会报错
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


@torch.no_grad()
def clean_forward_one(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    y: int,
    device: torch.device,
):
    ids = input_ids.unsqueeze(0).to(device)
    mask = attention_mask.unsqueeze(0).to(device)
    labels = torch.tensor([y], device=device)

    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    logits = out.logits
    pred = int(logits.argmax(-1).item())
    loss = float(F.cross_entropy(logits, labels).item())
    return pred, loss


def pick_eval_indices(
    splits: Dict[str, Any],
    split_name: str,
    max_eval: int,
    seed: int,
    force_first_n: int = 100,
):
    """
    你之前说想确保是 attack split 前100个，这里默认取 splits[split][:100]。
    之后如果 max_eval < 100，则在这100里随机 sample。
    """
    idxs = splits[split_name][:force_first_n]
    if max_eval > 0 and max_eval < len(idxs):
        rng = random.Random(seed)
        idxs = rng.sample(idxs, k=max_eval)
    return idxs


def eval_one_checkpoint(
    run_dir: str,
    ckpt_dir: str,
    dataset_name: str,
    splits_json: str,
    split_name: str,
    device: torch.device,
    max_eval: int,
    seed: int,
    gcg_kwargs: Dict[str, Any],
    out_root: str,
    save_text: bool,
):
    model, tok = load_model_and_tokenizer(ckpt_dir, device)

    ds = load_dataset(dataset_name)["train"]
    with open(splits_json, "r") as f:
        splits = json.load(f)

    idxs = pick_eval_indices(splits, split_name, max_eval=max_eval, seed=seed, force_first_n=100)

    # 输出路径：每个 run 一个子目录
    run_name = os.path.basename(run_dir.rstrip("/"))
    out_dir = os.path.join(out_root, run_name)
    ensure_dir(out_dir)

    per_ex_path = os.path.join(out_dir, "gcg_eval_per_example.jsonl")
    summary_path = os.path.join(out_dir, "gcg_eval_summary.json")

    # 统计
    n = 0
    n_clean_correct = 0
    n_adv_wrong_given_clean_correct = 0  # ASR overall 分子
    n_flip = 0

    clean_losses = []
    adv_losses = []
    orig_losses = []

    # 为了复现攻击随机性（你的 GCG 里用 random.sample）
    random.seed(seed)
    torch.manual_seed(seed)

    with open(per_ex_path, "w") as f_out:
        for idx in tqdm(idxs, desc=f"GCG eval: {run_name}", ncols=100):
            item = ds[idx]
            text = item["content"]
            y = int(item["clf_label"])

            enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]

            # clean
            clean_pred, clean_loss = clean_forward_one(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                y=y,
                device=device,
            )

            # attack
            orig_pred, orig_loss, adv_pred, adv_loss, success_flip_attackfn, adv_ids = token_level_gcg_single_baseline(
                model=model,
                tokenizer=tok,
                input_ids=input_ids,
                attention_mask=attention_mask,
                true_label=y,
                device=device,
                **gcg_kwargs,
            )

            # 定义：ASR overall = clean正确 且 adv错误
            clean_correct = (clean_pred == y)
            adv_wrong = (adv_pred != y)
            asr_overall_hit = (clean_correct and adv_wrong)

            # flip-rate（用 clean_pred 更一致）
            flip = (adv_pred != clean_pred)

            # 统计
            n += 1
            n_clean_correct += int(clean_correct)
            n_adv_wrong_given_clean_correct += int(asr_overall_hit)
            n_flip += int(flip)

            clean_losses.append(float(clean_loss))
            adv_losses.append(float(adv_loss))
            orig_losses.append(float(orig_loss))

            # 尽可能多存内容
            rec: Dict[str, Any] = {
                "dataset": dataset_name,
                "split": split_name,
                "ds_idx": int(idx),

                "label": int(y),

                # clean
                "clean_pred": int(clean_pred),
                "clean_correct": bool(clean_correct),
                "clean_loss": float(clean_loss),

                # attack fn 输出（方便排查不一致）
                "orig_pred_attackfn": int(orig_pred),
                "orig_loss_attackfn": float(orig_loss),

                # adv
                "adv_pred": int(adv_pred),
                "adv_wrong": bool(adv_wrong),
                "adv_loss": float(adv_loss),

                # metrics per example
                "flip_pred": bool(flip),
                "asr_overall_hit": bool(asr_overall_hit),

                # GCG config + randomness control
                "gcg": dict(gcg_kwargs),
                "eval_seed": int(seed),

                # token info
                "orig_len": int(input_ids.numel()),
                "adv_len": int(adv_ids.numel()) if isinstance(adv_ids, torch.Tensor) else None,
                "adv_ids": adv_ids.tolist() if isinstance(adv_ids, torch.Tensor) else None,
            }

            if save_text:
                rec["text"] = text
                if isinstance(adv_ids, torch.Tensor):
                    try:
                        rec["adv_text"] = tok.decode(
                            adv_ids.tolist(),
                            skip_special_tokens=True
                        )
                    except Exception:
                        rec["adv_text"] = ""


            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (n % 50) == 0:
                asr_overall = n_adv_wrong_given_clean_correct / max(1, n)
                clean_acc = n_clean_correct / max(1, n)
                flip_rate = n_flip / max(1, n)
                print(f"[{run_name}] {n}/{len(idxs)} ASR_overall={asr_overall:.3f} clean_acc={clean_acc:.3f} flip={flip_rate:.3f}")

    # summary
    summary = {
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "dataset": dataset_name,
        "split": split_name,
        "n_eval": n,
        "clean_acc": n_clean_correct / max(1, n),

        # 你要的指标
        "asr_overall": n_adv_wrong_given_clean_correct / max(1, n),

        # 额外指标
        "flip_rate": n_flip / max(1, n),

        "clean_loss_mean": sum(clean_losses) / max(1, n),
        "adv_loss_mean": sum(adv_losses) / max(1, n),
        "orig_loss_attackfn_mean": sum(orig_losses) / max(1, n),

        "gcg": dict(gcg_kwargs),
        "eval_seed": int(seed),

        "per_example_jsonl": per_ex_path,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {per_ex_path}")
    print(f"[saved] {summary_path}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, nargs="+", required=True,
                    help="run dirs, e.g. baseline/continuous_at/runs_harmless/eps0.05_k10_l2_mix0.5_seed27")
    ap.add_argument("--ckpt_subdir", type=str, default="final_ckpt",
                    help="subdir inside each run dir that stores HF checkpoint")
    ap.add_argument("--dataset", type=str, default="AlignmentResearch/Harmless")
    ap.add_argument("--splits_json", type=str, default="src/data/harmless_splits.json")
    ap.add_argument("--split", type=str, default="attack", choices=["attack", "probe", "ft_train"])

    # eval sampling
    ap.add_argument("--max_eval", type=int, default=100,
                    help="evaluate at most max_eval examples from the first 100 indices of the split")
    ap.add_argument("--seed", type=int, default=42)

    # output
    ap.add_argument("--out_root", type=str, default="baseline/continuous_at/runs_harmless",
                    help="root dir to store per-run eval outputs")
    ap.add_argument("--save_text", action="store_true",
                    help="store original/adv text in jsonl (can be large)")

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

    ensure_dir(args.out_root)

    for run_dir in args.runs:
        ckpt_dir = os.path.join(run_dir, args.ckpt_subdir)
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

        eval_one_checkpoint(
            run_dir=run_dir,
            ckpt_dir=ckpt_dir,
            dataset_name=args.dataset,
            splits_json=args.splits_json,
            split_name=args.split,
            device=device,
            max_eval=args.max_eval,
            seed=args.seed,
            gcg_kwargs=gcg_kwargs,
            out_root=args.out_root,
            save_text=args.save_text,
        )


if __name__ == "__main__":
    main()

    """
    CUDA_VISIBLE_DEVICES=1 python -m baseline.continuous_at.eval_gcg \
  --runs baseline/continuous_at/runs_imdb/eps0.05_k10_l2_mix0.5_seed27 \
        baseline/continuous_at/runs_imdb/eps0.05_k10_l2_mix0.5_seed42 \
        baseline/continuous_at/runs_imdb/eps0.05_k10_l2_mix0.5_seed56 \
  --dataset AlignmentResearch/IMDB \
  --splits_json src/data/imdb_splits.json \
  --split attack \
  --out_root baseline/continuous_at/runs_imdb \
  --max_eval 100 \
  --seed 42 \
  --attack_mode suffix \
  --save_text

    """