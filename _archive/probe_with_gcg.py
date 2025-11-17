#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.adv_dataset import AdvDataset
from src.data.Collator import Collator
from src.attack.eval_spam_gcg import token_level_gcg_single

# -------------------------------
# 通用工具
# -------------------------------
def build_model_and_tokenizer(model_name: str, ckpt: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.config.pad_token_id = tok.pad_token_id
    if model.config.eos_token_id is None and tok.eos_token_id is not None:
        model.config.eos_token_id = tok.eos_token_id
    model.to(device).eval()
    return model, tok

def last_nonpad_pos(attn_mask: torch.Tensor) -> torch.Tensor:
    return attn_mask.sum(dim=1) - 1  # (B,)

class AdvTensorDataset(Dataset):
    def __init__(self, ids, attn, labels):
        self.ids = ids; self.attn = attn; self.labels = labels
    def __len__(self): return self.ids.size(0)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.attn[i], "labels": self.labels[i]}

# -------------------------------
# 阶段 1：生成 x_adv
# -------------------------------
def stage_make_adv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_model_and_tokenizer(args.model_name, args.ckpt, device)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.data_splits, "r") as f:
        splits = json.load(f)
    ds = AdvDataset(split_indices=splits['probe'], dataset_name=args.dataset_name)
    coll = Collator(model_name=args.model_name, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=coll, num_workers=0)
    used_indices = []  # 新增：记录实际攻击到的 probe 原始索引
    adv_ids_list, adv_attn_list, adv_labels, meta = [], [], [], []
    for i, batch in enumerate(tqdm(loader, desc="GCG attacking")):
        # 新增：若设置了 limit，则只处理前 N 个
        if args.limit and i >= args.limit:
            break
        ids = batch["input_ids"].squeeze(0)
        attn = batch["attention_mask"].squeeze(0)
        lab = int(batch["labels"].item())

        orig_pred, orig_loss, adv_pred, adv_loss, success, adv_ids = token_level_gcg_single(
            model, tokenizer, ids, attn, lab, device,
            n_attack_tokens=args.n_attack_tokens,
            attack_start=args.attack_start,
            beam_k=args.beam_k,
            rounds=args.rounds,
            forbid_special=True
        )
        adv_ids_list.append(adv_ids.cpu())
        adv_attn_list.append(attn.cpu())
        adv_labels.append(lab)
        meta.append({
            "idx": i, "orig_pred": orig_pred, "adv_pred": adv_pred, "success": bool(success),
            "orig_loss": orig_loss, "adv_loss": adv_loss
        })
        used_indices.append(splits["probe"][i])

        if (i+1) % args.save_every == 0:
            torch.save({
                "input_ids": torch.stack(adv_ids_list),
                "attention_mask": torch.stack(adv_attn_list),
                "labels": torch.tensor(adv_labels, dtype=torch.long),
                "meta": meta,
                # 新增：中途也存 indices，防止中途断掉
                "indices": torch.tensor(used_indices, dtype=torch.long),
            }, os.path.join(args.out_dir, f"adv_partial_{i+1}.pt"))

    adv_ids_t = torch.stack(adv_ids_list, dim=0)
    adv_attn_t = torch.stack(adv_attn_list, dim=0)
    adv_labels_t = torch.tensor(adv_labels, dtype=torch.long)
    save_path = os.path.join(args.out_dir, "adv_all.pt")
    torch.save({"input_ids": adv_ids_t, "attention_mask": adv_attn_t, "labels": adv_labels_t, "meta": meta, "indices": torch.tensor(used_indices, dtype=torch.long),}, save_path)
    print(f"[Stage1] Saved ADV dataset to {save_path}")


# -------------------------------
# 阶段 2：提取各层表示
# -------------------------------
def collect_hidden(model, loader, device, fname):
    all_hiddens: Dict[int, list] = {}
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"collect {fname}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states  # tuple len = num_layers+1
            last_pos = last_nonpad_pos(batch["attention_mask"])
            idx = torch.arange(hs[0].size(0), device=device)
            for l, h in enumerate(hs):
                h_last = h[idx, last_pos, :].detach().cpu()
                all_hiddens.setdefault(l, []).append(h_last)
            all_labels.append(batch["labels"].detach().cpu())

    for l in list(all_hiddens.keys()):
        all_hiddens[l] = torch.cat(all_hiddens[l], dim=0).numpy().astype("float32")
    labels = torch.cat(all_labels, dim=0).numpy().astype("int64")
    torch.save({"hiddens": all_hiddens, "labels": labels}, fname)
    return fname

def stage_extract(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_model_and_tokenizer(args.model_name, args.ckpt, device)

    os.makedirs(args.out_dir, exist_ok=True)

    # CLEAN
    # 先读取 adv_all.pt，拿到实际用过的 indices
    adv_data = torch.load(args.adv_pt)
    used_idx = adv_data.get("indices", None)

    with open(args.data_splits, "r") as f:
        splits = json.load(f)
    if used_idx is not None:
        used_idx = used_idx.cpu().tolist()
        clean_ds = AdvDataset(split_indices=used_idx, dataset_name=args.dataset_name)
    else:
        # 回退（不推荐，但兼容旧文件）
        clean_ds = AdvDataset(split_indices=splits["probe"], dataset_name=args.dataset_name)
    #clean_ds = AdvDataset(split_indices=splits["probe"], dataset_name=args.dataset_name)
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=Collator(model_name=args.model_name, max_length=args.max_length))
    clean_path = os.path.join(args.out_dir, "clean_hiddens.pt")
    collect_hidden(model, clean_loader, device, clean_path)
    print(f"[Stage2] Saved CLEAN hiddens to {clean_path}")

    # ADV
    adv_data = torch.load(args.adv_pt)
    adv_loader = DataLoader(
        AdvTensorDataset(adv_data["input_ids"], adv_data["attention_mask"], adv_data["labels"]),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda x: {
            "input_ids": torch.stack([i["input_ids"] for i in x]),
            "attention_mask": torch.stack([i["attention_mask"] for i in x]),
            "labels": torch.tensor([i["labels"] for i in x], dtype=torch.long)
        }
    )
    adv_path = os.path.join(args.out_dir, "adv_hiddens.pt")
    collect_hidden(model, adv_loader, device, adv_path)
    print(f"[Stage2] Saved ADV hiddens to {adv_path}")


# -------------------------------
# 阶段 3：逐层线性探针 + 选层
# -------------------------------
def stage_probe(args):
    clean = torch.load(os.path.join(args.hid_dir, "clean_hiddens.pt"))
    adv   = torch.load(os.path.join(args.hid_dir, "adv_hiddens.pt"))

    Hc, yc = clean["hiddens"], clean["labels"]
    Ha, ya = adv["hiddens"], adv["labels"]

    assert yc.shape == ya.shape and (yc == ya).all(), "clean/adv labels mismatch"

    layers = sorted(Hc.keys())
    results = {}

    for l in layers:
        Xc = Hc[l]  # (N, D)
        Xa = Ha[l]
        y  = yc

        # split (同样的 random_state 但独立 split，近似)
        Xtr_c, Xte_c, ytr, yte = train_test_split(Xc, y, test_size=0.3, random_state=42, stratify=y)
        Xtr_a, Xte_a, _,  _    = train_test_split(Xa, y, test_size=0.3, random_state=42, stratify=y)

        scaler = StandardScaler().fit(Xtr_c)
        Xtr_s = scaler.transform(Xtr_c); Xte_s = scaler.transform(Xte_c)
        Xte_a_s = scaler.transform(Xte_a)

        clf = LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1).fit(Xtr_s, ytr)
        yhat_c = clf.predict(Xte_s)
        acc_clean = accuracy_score(yte, yhat_c)
        try:
            prob_c = clf.predict_proba(Xte_s)[:,1]; auc_clean = roc_auc_score(yte, prob_c)
        except Exception:
            auc_clean = float("nan")

        yhat_a = clf.predict(Xte_a_s)
        acc_adv = accuracy_score(yte, yhat_a)
        try:
            prob_a = clf.predict_proba(Xte_a_s)[:,1]; auc_adv = roc_auc_score(yte, prob_a)
        except Exception:
            auc_adv = float("nan")

        results[l] = {
            "acc_clean": float(acc_clean), "auc_clean": float(auc_clean),
            "acc_adv": float(acc_adv),     "auc_adv":  float(auc_adv),
            "drop_acc": float(acc_clean - acc_adv),
            "drop_auc": float(auc_clean - auc_adv),
            "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()
        }

        print(f"Layer {l:02d}: acc_clean={acc_clean:.3f} acc_adv={acc_adv:.3f} "
              f"drop={acc_clean-acc_adv:.3f} auc_clean={auc_clean:.3f} auc_adv={auc_adv:.3f}")

    # 选层：先按 drop_acc 升序，再按 acc_adv 降序
    K = args.topk
    candidates = sorted(results.items(), key=lambda x: (x[1]["drop_acc"], -x[1]["acc_adv"]))
    selected = [int(l) for l, _ in candidates[:K]]
    print("Selected layers:", selected)

    os.makedirs(args.hid_dir, exist_ok=True)
    with open(os.path.join(args.hid_dir, "probe_layer_results.json"), "w") as f:
        json.dump({"selected": selected, "results": {str(k): v for k, v in results.items()}}, f, indent=2)

    # 同时把 u,v 存成 npz（便于后面写入 R）
    uvs = {}
    for l, v in results.items():
        u = np.asarray(v["coef"], dtype=np.float32).reshape(-1)   # (1,D) -> (D,)
        v0 = float(np.asarray(v["intercept"], dtype=np.float32).reshape(()))
        uvs[f"L{l}"] = np.concatenate([u, np.array([v0], dtype=np.float32)], axis=0)
    np.savez(os.path.join(args.hid_dir, "probe_uv_clean_adv.npz"), **uvs)
    print(f"[Stage3] Saved probe results to {args.hid_dir}")


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GCG -> extract hiddens -> probe & select layers")
    p.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    p.add_argument("--ckpt", type=str, required=True, help="ft checkpoint, e.g., out/spam_pythia410m/best_new.pt")
    p.add_argument("--dataset_name", type=str, default="AlignmentResearch/EnronSpam")
    p.add_argument("--data_splits", type=str, default="src/data/enron_splits.json")

    sub = p.add_subparsers(dest="stage", required=True)

    # stage 1
    p1 = sub.add_parser("gen-adv", help="generate x_adv with token-level GCG")
    p1.add_argument("--out_dir", type=str, default="out/spam_pythia410m/adv_data")
    p1.add_argument("--max_length", type=int, default=512)
    p1.add_argument("--n_attack_tokens", type=int, default=5)
    p1.add_argument("--attack_start", type=int, default=0)
    p1.add_argument("--beam_k", type=int, default=20)
    p1.add_argument("--rounds", type=int, default=20)
    p1.add_argument("--save_every", type=int, default=200)
    p1.add_argument("--limit", type=int, default=0,
                help="attack only the first N samples from the probe split (0 means all)")

    # stage 2
    p2 = sub.add_parser("extract", help="extract per-layer hiddens for clean & adv")
    p2.add_argument("--adv_pt", type=str, default="out/spam_pythia410m/adv_data/adv_all.pt")
    p2.add_argument("--out_dir", type=str, default="out/spam_pythia410m/probe_hiddens")
    p2.add_argument("--max_length", type=int, default=512)
    p2.add_argument("--batch_size", type=int, default=8)

    # stage 3
    p3 = sub.add_parser("probe", help="logreg probes on clean vs adv & select layers")
    p3.add_argument("--hid_dir", type=str, default="out/spam_pythia410m/probe_hiddens")
    p3.add_argument("--topk", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "gen-adv":
        stage_make_adv(args)
    elif args.stage == "extract":
        stage_extract(args)
    elif args.stage == "probe":
        stage_probe(args)
    else:
        raise ValueError("Unknown stage:", args.stage)


if __name__ == "__main__":
    main()
