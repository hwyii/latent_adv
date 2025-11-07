# layer_selection.py
# 目标：对每一层同时计算三种指标：Logistic、核SVM、Sliced-Wasserstein，用于“选层”

import os, json, yaml, random
import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data.adv_dataset import AdvDataset
from src.data.Collator import Collator

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

from scipy.stats import wasserstein_distance

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============== 工具函数 ===============

def balanced_subsample(X, y, max_per_class=5000, seed=42):
    """
    从两类中各取相同数量，避免不平衡；用于SW和SVM训练更稳定
    """
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n = min(len(idx_pos), len(idx_neg), max_per_class)
    idx_pos = rng.choice(idx_pos, size=n, replace=False)
    idx_neg = rng.choice(idx_neg, size=n, replace=False)
    idx = np.concatenate([idx_pos, idx_neg])
    rng.shuffle(idx)
    return X[idx], y[idx]

def compute_sliced_wasserstein(X, y, n_proj=128, seed=42, max_per_class=5000):
    """
    计算二类在特征空间的 Sliced-Wasserstein-2（使用1D W1作为近似统计，再平方求均值）
    - 流程：随机投影 -> 1D Wasserstein 距离 -> 平均（可视为SW(W1)的经验估计）
    - 返回：平均值 mean_w1, std_w1, mean_w1_sq（可当作“W2^2”的proxy）
    """
    Xb, yb = balanced_subsample(X, y, max_per_class=max_per_class, seed=seed)
    XA = Xb[yb == 1]
    XB = Xb[yb == 0]
    d = Xb.shape[1]
    rng = np.random.RandomState(seed)
    ws = []
    for _ in range(n_proj):
        v = rng.normal(size=(d,))
        v = v / (np.linalg.norm(v) + 1e-12)
        a = XA @ v
        b = XB @ v
        # 1D Wasserstein 距离（W1）：越大表示两类沿该方向分离越大
        w = wasserstein_distance(a, b)
        ws.append(w)
    ws = np.asarray(ws, dtype=np.float64)
    return float(ws.mean()), float(ws.std(ddof=1)), float((ws**2).mean())

def fit_logistic_probe(X, y, seed=42):
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(solver="saga", penalty="l2", C=1.0, max_iter=5000, n_jobs=-1)
    )
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_va)
    y_proba = pipe.predict_proba(X_va)[:, 1]
    acc = accuracy_score(y_va, y_pred)
    try:
        auc = roc_auc_score(y_va, y_proba)
    except ValueError:
        auc = float("nan")
    # 取最后一层 Logistic 的权重（可作为线性方向 u, v）
    clf = pipe.named_steps["logisticregression"]
    u = clf.coef_.ravel().copy()
    v = float(clf.intercept_[0])
    return acc, auc, u, v
# === ADD: MLP probe (2-layer FFN) ===
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler

class _MLP(nn.Module):
    def __init__(self, d_in, hidden=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def fit_mlp_probe(X, y, seed=42, max_epochs=30, batch_size=256, lr=3e-3, hidden=128, dropout=0.1):
    """
    轻量 MLP-probe：标准化 -> MLP -> 早停（基于 val AUC）
    返回: acc, auc
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    y_tr = y_tr.astype(np.float32); y_va = y_va.astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _MLP(X_tr.shape[1], hidden=hidden, p=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    def _batcher(X, y):
        n = len(X)
        idx = np.arange(n); rng.shuffle(idx)
        for i in range(0, n, batch_size):
            j = idx[i:i+batch_size]
            yield torch.from_numpy(X[j]).to(device), torch.from_numpy(y[j]).to(device)

    best_auc, patience, best_state = 0.0, 5, None
    import math
    for ep in range(max_epochs):
        model.train()
        for xb, yb in _batcher(X_tr, y_tr):
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X_va).to(device)
            logits = model(xb).cpu().numpy()
            proba = 1/(1+np.exp(-logits))
        y_pred = (proba >= 0.5).astype(np.int64)
        acc = accuracy_score(y_va, y_pred)
        try:
            auc = roc_auc_score(y_va, proba)
        except ValueError:
            auc = float("nan")
        # early stop on AUC
        if auc >= best_auc - 1e-4:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 5
        else:
            patience -= 1
            if patience == 0:
                break
    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return float(acc), float(best_auc)

def fit_kernel_svm(X, y, seed=42):
    """
    RBF 核 SVM：给 AUC + 支持向量占比 + 决策值绝对值的中位数（作为 margin 代理）
    """
    # 训练前做一个 class-balanced subsample，防止n太大核O(n^2)炸内存
    Xb, yb = balanced_subsample(X, y, max_per_class=8000, seed=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        Xb, yb, test_size=0.2, random_state=seed, stratify=yb
    )
    svc = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=seed,
    )
    svc.fit(X_tr, y_tr)
    y_pred = svc.predict(X_va)
    y_proba = svc.predict_proba(X_va)[:, 1]
    acc = accuracy_score(y_va, y_pred)
    try:
        auc = roc_auc_score(y_va, y_proba)
    except ValueError:
        auc = float("nan")
    sv_ratio = len(svc.support_) / len(X_tr)  # 支持向量越少，说明间隔通常越“干净”
    # 决策函数绝对值的中位数可近似反映“离超曲面有多远”，越大越好
    margins = np.abs(svc.decision_function(X_va))
    med_margin = float(np.median(margins))
    return acc, auc, sv_ratio, med_margin, svc

def minmax_scale(x_list):
    x = np.asarray(x_list, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

# =============== 主流程 ===============

with open("configs/baseline.yaml") as f:
    cfg = yaml.safe_load(f)

dataset = cfg["data"]["dataset"].replace("AlignmentResearch/","")
device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
model_name = cfg["model"]["name"]
ckpt_path = f"{cfg['out']['dir']}/best_{dataset}.pt"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
state = torch.load(ckpt_path, map_location=device)
model.config.pad_token_id = tokenizer.pad_token_id
model.load_state_dict(state, strict=True)
if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
    model.config.eos_token_id = tokenizer.eos_token_id
model.to(device).eval()

# 验证集（或 probe split）
with open(cfg["data"]["split_file"], "r") as f:
    splits = json.load(f)
ds = AdvDataset(splits["probe"], cfg["data"]["dataset"])
collate = Collator(model_name=model_name, max_length=512)
val_loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)

# 收集每层句向量（取最后非pad token）与标签
all_hiddens = {}
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
        hiddens = outputs.hidden_states  # list of tensors, len = num_layers+1
        attn = batch['attention_mask']
        last_idx = attn.sum(dim=1) - 1
        bsz = last_idx.size(0)
        idx = torch.arange(bsz, device=device)
        for l, h in enumerate(hiddens):
            sent = h[idx, last_idx, :]  # (bsz, hidden_size)
            all_hiddens.setdefault(l, []).append(sent.detach().cpu())
        all_labels.append(batch['labels'].detach().cpu())

y = torch.cat(all_labels, dim=0).numpy().astype(np.int64).ravel()
for l in all_hiddens:
    all_hiddens[l] = torch.cat(all_hiddens[l], dim=0).numpy().astype(np.float32)

latent_out_dir = os.path.join(cfg['out']['dir'], "latents")
os.makedirs(latent_out_dir, exist_ok=True)
    
# 四路评价
results = {}  # layer -> dict
print(f"\n=== Dataset: {dataset} | Layers: {len(all_hiddens)} ===")
for l, X in all_hiddens.items():
    print(f"\n[Layer {l:02d}] dim={X.shape[1]} n={X.shape[0]}")
    # 保存每层的 latent 和 label，供后续分析或可视化
    np.savez_compressed(os.path.join(latent_out_dir, f"{dataset}_L{l}.npz"), X=X, y=y)
    # 1) Logistic probe
    log_acc, log_auc, u, v = fit_logistic_probe(X, y, seed=42)
    # 2) 核 SVM
    svm_acc, svm_auc, sv_ratio, med_margin, svc = fit_kernel_svm(X, y, seed=42)
    np.savez_compressed(os.path.join(latent_out_dir, f"{dataset}_L{l}_svm_margin.npz"), margin=svc.decision_function(X), y=y)
    
    
    # 3) Sliced-Wasserstein
    sw_mean, sw_std, sw_mean_sq = compute_sliced_wasserstein(X, y, n_proj=128, seed=42, max_per_class=5000)

    # 4) === ADD: MLP probe ===
    mlp_acc, mlp_auc = fit_mlp_probe(X, y, seed=42, max_epochs=30, batch_size=256, lr=3e-3, hidden=128, dropout=0.1)
    mlp_save = os.path.join(latent_out_dir, f"{dataset}_L{l}_mlp.pt")
    torch.save(model.state_dict(), mlp_save)
    results[l] = dict(
        log_acc=log_acc, log_auc=log_auc, u_norm=float(np.linalg.norm(u)), v=v,
        svm_acc=svm_acc, svm_auc=svm_auc, sv_ratio=sv_ratio, svm_med_margin=med_margin,
        sw_mean=sw_mean, sw_std=sw_std, sw_mean_sq=sw_mean_sq,
        mlp_acc=mlp_acc, mlp_auc=mlp_auc,   # <<< 新增
    )

    print(f"  Logistic  : acc={log_acc:.3f} | auc={log_auc:.3f} | ||u||={np.linalg.norm(u):.3f}")
    print(f"  Kernel SVM: acc={svm_acc:.3f} | auc={svm_auc:.3f} | sv_ratio={sv_ratio:.3f} | med|margin|={med_margin:.3f}")
    print(f"  Sliced-W  : mean(W1)={sw_mean:.4f} ± {sw_std:.4f} | mean(W1^2)={sw_mean_sq:.5f}")
    print(f"  MLP-probe : acc={mlp_acc:.3f} | auc={mlp_auc:.3f}")  # <<< 新增

# 排行（单指标 & 综合）
layers = sorted(results.keys())
log_auc_rank  = sorted(layers, key=lambda l: results[l]["log_auc"], reverse=True)
svm_auc_rank  = sorted(layers, key=lambda l: results[l]["svm_auc"], reverse=True)
# SVM 支持向量越少越好 -> 反向排序（少排前）
sv_ratio_rank = sorted(layers, key=lambda l: results[l]["sv_ratio"])
# margin 越大越好
margin_rank   = sorted(layers, key=lambda l: results[l]["svm_med_margin"], reverse=True)
# 几何分离度：SW 的 mean(W1) 或 mean(W1^2) 越大越好
sw_rank       = sorted(layers, key=lambda l: results[l]["sw_mean_sq"], reverse=True)
mlp_auc_rank = sorted(layers, key=lambda l: results[l]["mlp_auc"], reverse=True)


print("\n=== Ranking by metrics ===")
print("Logistic AUC  (desc):", log_auc_rank)
print("Kernel SVM AUC(desc):", svm_auc_rank)
print("SVM SV ratio  (asc):", sv_ratio_rank)
print("SVM median|margin|(desc):", margin_rank)
print("Sliced-W mean(W1^2)(desc):", sw_rank)
print("MLP  AUC     (desc):", mlp_auc_rank)

# 综合打分（可调权重）：侧重几何 + 判别 + 线性基线
w_mlp = 0.10
w_sw, w_svm_auc, w_svratio, w_margin, w_logauc = 0.32, 0.23, 0.15, 0.15, 0.05  # 让位给 MLP

sw_scores   = minmax_scale([results[l]["sw_mean_sq"] for l in layers])             # 大好
svm_auc_s   = minmax_scale([results[l]["svm_auc"] for l in layers])                # 大好
svratio_s   = 1.0 - minmax_scale([results[l]["sv_ratio"] for l in layers])         # 小好 -> 取 1 - scaled
margin_s    = minmax_scale([results[l]["svm_med_margin"] for l in layers])         # 大好
logauc_s    = minmax_scale([results[l]["log_auc"] for l in layers])                # 大好
mlpauc_s  = minmax_scale([results[l]["mlp_auc"] for l in layers])

composite = {}
for i, l in enumerate(layers):
    score = (w_sw * sw_scores[i] + w_svm_auc * svm_auc_s[i]
             + w_svratio * svratio_s[i] + w_margin * margin_s[i]
             + w_logauc * logauc_s[i] + w_mlp * mlpauc_s[i])   # <<< 新增
    composite[l] = float(score)

best_by_composite = max(composite, key=composite.get)
print("\n=== Composite score (higher is better) ===")
for l in sorted(layers):
    print(f"Layer {l:02d}: score={composite[l]:.3f} | SW={results[l]['sw_mean_sq']:.5f} "
          f"| SVM AUC={results[l]['svm_auc']:.3f} | SV%={results[l]['sv_ratio']:.3f} "
          f"| med|margin|={results[l]['svm_med_margin']:.3f} | LogAUC={results[l]['log_auc']:.3f}| MLP AUC={results[l]['mlp_auc']:.3f}")

print(f"\n>>> Recommended layer for boundary & LAT: L{best_by_composite} "
      f"(composite={composite[best_by_composite]:.3f}) on dataset {dataset}")

# 可选：保存逐层结果到 JSON/CSV，供可视化或后续脚本读取
import json as _json
out_dir = cfg['out']['dir']
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, f"layer_tri_metrics_{dataset}.json"), "w") as f:
    _json.dump({"results": results, "composite": composite}, f, indent=2)
