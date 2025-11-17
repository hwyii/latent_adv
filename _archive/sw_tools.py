# --- sw_tools.py ---
# Utilities for sliced-Wasserstein layer analysis and visualization
# Author: Zilong + GPT-5 Thinking
# Usage:
#   from sw_tools import (
#       balanced_subsample, compute_sliced_wasserstein,
#       plot_sw_over_layers, plot_layer_topk_1d, plot_layer_scatter_by_top2,
#       save_sw_details, load_sw_details,
#   )
#
# This module is self-contained (numpy, scipy, matplotlib only).

from __future__ import annotations
import os
import numpy as np
from typing import Dict, Any, Tuple, List
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# ------------------------------
# Data helpers
# ------------------------------

def balanced_subsample(X: np.ndarray, y: np.ndarray, max_per_class: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a class-balanced subsample up to max_per_class per class.
    Assumes binary labels {0,1}. Returns (X_sub, y_sub).
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    Xa, Xb = X[y == 1], X[y == 0]
    na = min(len(Xa), max_per_class)
    nb = min(len(Xb), max_per_class)
    ia = rng.choice(len(Xa), size=na, replace=False)
    ib = rng.choice(len(Xb), size=nb, replace=False)
    X_sub = np.vstack([Xa[ia], Xb[ib]])
    y_sub = np.concatenate([np.ones(na, dtype=int), np.zeros(nb, dtype=int)])
    return X_sub, y_sub

# ------------------------------
# Core SW computation
# ------------------------------

def compute_sliced_wasserstein(
    X: np.ndarray,
    y: np.ndarray,
    n_proj: int = 256,
    seed: int = 42,
    max_per_class: int = 5000,
    topk: int = 8,
) -> Dict[str, Any]:
    """
    Compute sliced-Wasserstein by random projections.
    - Projects class 1 and class 0 onto n_proj random unit vectors v_k.
    - For each projection, compute 1D W1 distance (scipy.stats.wasserstein_distance).
    - Returns summary stats and Top-K directions with largest W1.

    Returns dict with keys:
      mean, std, mean_sq, ws, vecs, topk_idx, topk_ws, topk_vecs,
      topk_proj_A, topk_proj_B, class_sizes
    """
    Xb, yb = balanced_subsample(X, y, max_per_class=max_per_class, seed=seed)
    XA = Xb[yb == 1]
    XB = Xb[yb == 0]
    d = Xb.shape[1]
    rng = np.random.RandomState(seed)

    ws = np.empty(n_proj, dtype=np.float64)
    vecs = np.empty((n_proj, d), dtype=np.float32)
    projA_cache: List[np.ndarray] = [None] * n_proj  # type: ignore
    projB_cache: List[np.ndarray] = [None] * n_proj  # type: ignore

    for k in range(n_proj):
        v = rng.normal(size=(d,))
        v /= (np.linalg.norm(v) + 1e-12)
        a = XA @ v
        b = XB @ v
        w = wasserstein_distance(a, b)
        ws[k] = w
        vecs[k] = v.astype(np.float32)
        projA_cache[k] = a.astype(np.float32)
        projB_cache[k] = b.astype(np.float32)

    order = np.argsort(ws)[::-1]
    topk = min(topk, len(order))
    topk_idx = order[:topk]
    topk_ws = ws[topk_idx].astype(np.float32)
    topk_vecs = vecs[topk_idx]
    topk_proj_A = [projA_cache[i] for i in topk_idx]
    topk_proj_B = [projB_cache[i] for i in topk_idx]

    return {
        "mean": float(ws.mean()),
        "std": float(ws.std(ddof=1)),
        "mean_sq": float((ws ** 2).mean()),
        "ws": ws,
        "vecs": vecs,
        "topk_idx": topk_idx.astype(np.int32),
        "topk_ws": topk_ws,
        "topk_vecs": topk_vecs,
        "topk_proj_A": topk_proj_A,
        "topk_proj_B": topk_proj_B,
        "class_sizes": (int((yb == 1).sum()), int((yb == 0).sum())),
    }

# ------------------------------
# I/O helpers for SW artifacts
# ------------------------------

def save_sw_details(path: str, sw: Dict[str, Any]) -> None:
    """Save SW results (including Top-K projections) to a .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        ws=sw["ws"], vecs=sw["vecs"],
        topk_idx=sw["topk_idx"], topk_ws=sw["topk_ws"], topk_vecs=sw["topk_vecs"],
        topk_proj_A=np.array(sw["topk_proj_A"], dtype=object),
        topk_proj_B=np.array(sw["topk_proj_B"], dtype=object),
        class_sizes=np.array(sw["class_sizes"], dtype=np.int32),
        mean=np.float64(sw["mean"]), std=np.float64(sw["std"]), mean_sq=np.float64(sw["mean_sq"]),
    )

def load_sw_details(path: str) -> Dict[str, Any]:
    dat = np.load(path, allow_pickle=True)
    out = {k: dat[k] for k in dat.files}
    # cast back some fields
    out["ws"] = out["ws"].astype(float)
    out["vecs"] = out["vecs"].astype(np.float32)
    out["topk_idx"] = out["topk_idx"].astype(int)
    out["topk_ws"] = out["topk_ws"].astype(float)
    return out

# ------------------------------
# Visualization
# ------------------------------

def plot_sw_over_layers(results: Dict[int, Dict[str, float]], save_path: str, ylabel: str = "SW mean of W1^2") -> None:
    layers = sorted(results.keys())
    vals = [results[l]["mean_sq"] for l in layers]
    plt.figure(figsize=(8, 3.2))
    plt.plot(layers, vals, marker='o')
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title("Sliced-Wasserstein summary across layers")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_layer_topk_1d(layer: int, latent_dir: str, dataset: str, bins: int = 60, save_path: str | None = None) -> None:
    dat = load_sw_details(os.path.join(latent_dir, f"{dataset}_L{layer}_sw.npz"))
    topk_ws = dat["topk_ws"]
    topk_proj_A = dat["topk_proj_A"]
    topk_proj_B = dat["topk_proj_B"]

    K = len(topk_ws)
    ncols = min(4, K)
    nrows = int(np.ceil(K / ncols))
    plt.figure(figsize=(4 * ncols, 2.8 * nrows))
    for i in range(K):
        plt.subplot(nrows, ncols, i + 1)
        a = np.asarray(topk_proj_A[i])
        b = np.asarray(topk_proj_B[i])
        plt.hist(a, bins=bins, alpha=0.5, density=True, label="class=1")
        plt.hist(b, bins=bins, alpha=0.5, density=True, label="class=0")
        plt.title(f"Top-{i + 1}: W1={topk_ws[i]:.3f}")
        if i % ncols == 0:
            plt.ylabel("Density")
        if i // ncols == nrows - 1:
            plt.xlabel("Projection value")
        if i == 0:
            plt.legend(loc="upper right")
    plt.suptitle(f"Layer {layer}: Top-K projections with largest 1D W1")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=220)
        plt.close()
    else:
        plt.show()

def plot_layer_scatter_by_top2(layer: int, latent_dir: str, dataset: str, save_path: str | None = None, max_points: int = 8000, seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    data_lat = np.load(os.path.join(latent_dir, f"{dataset}_L{layer}.npz"))
    X = data_lat["X"]
    y = data_lat["y"].astype(int)

    dat = load_sw_details(os.path.join(latent_dir, f"{dataset}_L{layer}_sw.npz"))
    topk_idx = dat["topk_idx"]
    vecs = dat["vecs"]
    if len(topk_idx) < 2:
        raise ValueError("Need at least Top-2 directions to make a 2D scatter.")
    v1 = vecs[topk_idx[0]]
    v2 = vecs[topk_idx[1]]
    z1 = X @ v1
    z2 = X @ v2

    n = len(y)
    idx = np.arange(n)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
    z1 = z1[idx]
    z2 = z2[idx]
    yy = y[idx]

    plt.figure(figsize=(5, 4.2))
    plt.scatter(z1[yy == 0], z2[yy == 0], s=6, alpha=0.5, label="class=0")
    plt.scatter(z1[yy == 1], z2[yy == 1], s=6, alpha=0.5, label="class=1")
    plt.xlabel("Proj on Top-1 SW direction")
    plt.ylabel("Proj on Top-2 SW direction")
    plt.title(f"Layer {layer}: Top-2 SW directions (2D view)")
    plt.legend(markerscale=2)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=220)
        plt.close()
    else:
        plt.show()

# --- visualize_sw.py ---
# CLI script to summarize SW across layers and visualize Top-K 1D histograms & Top-2 scatter for a chosen layer.
# Example:
#   python visualize_sw.py \
#     --latent_dir /mnt/scratch/zhan2210/output/<exp>/latents \
#     --dataset SLN111 \
#     --summary_out /mnt/scratch/zhan2210/output/<exp>/latents/SLN111_SW_layers.png \
#     --layer 21 \
#     --topk1d_out /mnt/scratch/zhan2210/output/<exp>/latents/SLN111_L21_topk_1D.png \
#     --scatter2d_out /mnt/scratch/zhan2210/output/<exp>/latents/SLN111_L21_top2_scatter.png

if __name__ == "__main__":
    import argparse, json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Sliced-Wasserstein layer analysis viewer")
    parser.add_argument("--latent_dir", type=str, required=True, help="Directory of saved layer latents and *_sw.npz")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset key used in file names (e.g., SLN111)")
    parser.add_argument("--summary_out", type=str, required=True, help="Path to save layer-wise SW curve image")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to visualize Top-K projections")
    parser.add_argument("--topk1d_out", type=str, required=True, help="Path to save Top-K 1D histograms figure")
    parser.add_argument("--scatter2d_out", type=str, required=False, default="", help="Path to save Top-2 2D scatter figure (optional)")

    args = parser.parse_args()
    latent_dir = args.latent_dir
    dataset = args.dataset

    # Reconstruct results dict from per-layer *_sw.npz (mean_sq saved inside)
    results: Dict[int, Dict[str, float]] = {}
    for p in sorted(Path(latent_dir).glob(f"{dataset}_L*_sw.npz")):
        name = p.stem  # e.g., SLN111_L21_sw
        # extract layer index
        try:
            layer = int(name.split("_L")[1].split("_")[0])
        except Exception:
            continue
        dat = load_sw_details(str(p))
        # mean_sq may not be in older files; if absent, recompute from ws
        mean_sq = float((np.asarray(dat["ws"]) ** 2).mean()) if "mean_sq" not in dat else float(dat["mean_sq"]) 
        results[layer] = {"mean_sq": mean_sq, "mean": float(np.asarray(dat["ws"]).mean()), "std": float(np.asarray(dat["ws"]).std(ddof=1))}

    # 1) layer-wise SW curve
    plot_sw_over_layers(results, save_path=args.summary_out)

    # 2) chosen layer Top-K 1D histograms
    plot_layer_topk_1d(layer=args.layer, latent_dir=latent_dir, dataset=dataset, save_path=args.topk1d_out)

    # 3) optional Top-2 2D scatter
    if args.scatter2d_out:
        plot_layer_scatter_by_top2(layer=args.layer, latent_dir=latent_dir, dataset=dataset, save_path=args.scatter2d_out)

    print("Saved:")
    print(" -", args.summary_out)
    print(" -", args.topk1d_out)
    if args.scatter2d_out:
        print(" -", args.scatter2d_out)


# =============================
# Add-ons: batch export + auto-compute + boundary lines
# =============================
# Paste below anywhere in sw_tools.py if you want a single file; otherwise keep as an add-on.

from typing import Iterable

def _discover_layers(latent_dir: str, dataset: str):
    from pathlib import Path
    layers = []
    for p in Path(latent_dir).glob(f"{dataset}_L*.npz"):
        name = p.stem
        try:
            L = int(name.split("_L")[1])
            layers.append(L)
        except Exception:
            pass
    return sorted(set(layers))


def export_sw_for_all_layers(latent_dir: str, dataset: str, layers: Iterable[int] | None = None,
                             n_proj: int = 256, seed: int = 42, max_per_class: int = 5000, topk: int = 8):
    """Compute SW for each layer that has `{dataset}_L{L}.npz` and save to `{dataset}_L{L}_sw.npz`."""
    results = {}
    if layers is None:
        layers = _discover_layers(latent_dir, dataset)
    for L in layers:
        lat_path = os.path.join(latent_dir, f"{dataset}_L{L}.npz")
        if not os.path.exists(lat_path):
            print(f"[skip] latent not found: {lat_path}")
            continue
        dat = np.load(lat_path)
        X, y = dat["X"], dat["y"].astype(int)
        sw = compute_sliced_wasserstein(X, y, n_proj=n_proj, seed=seed, max_per_class=max_per_class, topk=topk)
        save_sw_details(os.path.join(latent_dir, f"{dataset}_L{L}_sw.npz"), sw)
        results[L] = {"mean": sw["mean"], "std": sw["std"], "mean_sq": sw["mean_sq"]}
        print(f"[ok] L{L:02d} -> SW saved")
    return results


def recompute_missing_sw(latent_dir: str, dataset: str, n_proj: int = 256, seed: int = 42,
                          max_per_class: int = 5000, topk: int = 8):
    from pathlib import Path
    layers = _discover_layers(latent_dir, dataset)
    done = set()
    for p in Path(latent_dir).glob(f"{dataset}_L*_sw.npz"):
        try:
            L = int(p.stem.split("_L")[1].split("_")[0])
            done.add(L)
        except Exception:
            pass
    todo = [L for L in layers if L not in done]
    print("Missing SW files for layers:", todo)
    return export_sw_for_all_layers(latent_dir, dataset, layers=todo, n_proj=n_proj, seed=seed, max_per_class=max_per_class, topk=topk)

# Add decision boundary + SW axis on scatter (requires scikit-learn for boundary)
try:
    from sklearn.linear_model import LogisticRegression
    _HAS_SK = True
except Exception:
    _HAS_SK = False

def plot_layer_scatter_by_top2_with_boundary(layer: int, latent_dir: str, dataset: str, save_path: str | None = None,
                                             max_points: int = 8000, seed: int = 0, draw_boundary: bool = True,
                                             draw_sw_axis: bool = True):
    rng = np.random.RandomState(seed)
    data_lat = np.load(os.path.join(latent_dir, f"{dataset}_L{layer}.npz"))
    X = data_lat["X"]; y = data_lat["y"].astype(int)
    dat = load_sw_details(os.path.join(latent_dir, f"{dataset}_L{layer}_sw.npz"))
    topk_idx = dat["topk_idx"]; vecs = dat["vecs"]
    assert len(topk_idx) >= 2, "Need at least Top-2 directions"
    v1 = vecs[topk_idx[0]]; v2 = vecs[topk_idx[1]]
    z1 = X @ v1; z2 = X @ v2

    n = len(y)
    idx = np.arange(n)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
    z1 = z1[idx]; z2 = z2[idx]; yy = y[idx]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.6, 4.4))
    plt.scatter(z1[yy == 0], z2[yy == 0], s=6, alpha=0.5, label="class=0")
    plt.scatter(z1[yy == 1], z2[yy == 1], s=6, alpha=0.5, label="class=1")
    plt.xlabel("Proj on Top-1 SW direction"); plt.ylabel("Proj on Top-2 SW direction")
    plt.title(f"Layer {layer}: Top-2 SW dirs + boundary")

    if draw_boundary and _HAS_SK:
        clf = LogisticRegression(max_iter=1000).fit(np.c_[z1, z2], yy)
        gx, gy = np.meshgrid(np.linspace(z1.min(), z1.max(), 200), np.linspace(z2.min(), z2.max(), 200))
        grid = np.c_[gx.ravel(), gy.ravel()]
        prob = clf.predict_proba(grid)[:, 1].reshape(gx.shape)
        plt.contour(gx, gy, prob, levels=[0.5], colors='k', linewidths=1.2)

    if draw_sw_axis:
        m = X.mean(axis=0)
        x0, y0 = float(m @ v1), float(m @ v2)
        dx = float(v1 @ v1); dy = float(v2 @ v1)
        scale = 5.0
        plt.plot([x0 - scale * dx, x0 + scale * dx], [y0 - scale * dy, y0 + scale * dy], 'r--', lw=1.0, label='Top-1 SW axis')

    plt.legend(markerscale=2); plt.tight_layout()
    if save_path: os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=220); plt.close()
    else: plt.show()

# CLI helper: compute missing SW before plotting
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--compute_only", action="store_true")
    parser.add_argument("--n_proj", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_class", type=int, default=5000)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--scatter_out", type=str, default="")
    args, _ = parser.parse_known_args()
    if args.latent_dir and args.dataset:
        recompute_missing_sw(args.latent_dir, args.dataset, n_proj=args.n_proj, seed=args.seed, max_per_class=args.max_per_class, topk=args.topk)
        if not args.compute_only and args.layer >= 0 and args.scatter_out:
            plot_layer_scatter_by_top2_with_boundary(layer=args.layer, latent_dir=args.latent_dir, dataset=args.dataset, save_path=args.scatter_out)
