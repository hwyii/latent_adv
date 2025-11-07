#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot per-layer metrics (AUC / ACC / SW / Margin etc.)
Usage:
    python plot_layer_metrics.py --json /path/to/layer_tri_metrics_PBMC-Li.json
"""
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12
})

def plot_line(metrics_dict, dataset, outdir):
    """折线图（每层一个点）"""
    layers = sorted(metrics_dict.keys(), key=int)
    L = np.array([int(l) for l in layers])

    def _plot_metric(metric, label, color):
        y = np.array([metrics_dict[str(l)][metric] for l in L])
        plt.plot(L, y, marker="o", color=color, label=label)

    plt.figure(figsize=(8, 4))
    if "log_auc" in next(iter(metrics_dict.values())):
        _plot_metric("log_auc", "Logistic AUC", "tab:blue")
    if "svm_auc" in next(iter(metrics_dict.values())):
        _plot_metric("svm_auc", "SVM AUC", "tab:orange")
    if "mlp_auc" in next(iter(metrics_dict.values())):
        _plot_metric("mlp_auc", "MLP AUC", "tab:green")
    if "sw_mean_sq" in next(iter(metrics_dict.values())):
        _plot_metric("sw_mean_sq", "SW (W1²)", "tab:red")

    plt.xlabel("Layer index")
    plt.ylabel("Score")
    plt.title(f"Layer-wise separability ({dataset})")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(outdir, f"{dataset}_layer_auc_plot.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")

def plot_heatmap(metrics_dict, dataset, outdir): #to modify
    """热力图：每行是层，每列是指标"""
    layers = sorted(metrics_dict.keys(), key=int)
    metrics = list(next(iter(metrics_dict.values())).keys())
    data = np.array([[metrics_dict[str(l)].get(m, np.nan) for m in metrics] for l in layers])

    plt.figure(figsize=(10, len(layers)*0.35 + 2))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=metrics, yticklabels=layers)
    plt.xlabel("Metric")
    plt.ylabel("Layer")
    plt.title(f"Per-layer metric heatmap ({dataset})")
    plt.tight_layout()
    out_path = os.path.join(outdir, f"{dataset}_layer_heatmap.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True,
                        help="Path to JSON file (e.g., layer_tri_metrics_xx.json or *_gnet_results.json)")
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dataset = os.path.basename(args.json).replace(".json", "").replace("layer_tri_metrics_", "")

    with open(args.json, "r") as f:
        data = json.load(f)

    # 自动适配格式
    if "results" in data:
        metrics_dict = data["results"]
    else:
        metrics_dict = data  # gnet_results.json 情况

    plot_line(metrics_dict, dataset, args.outdir)
    plot_heatmap(metrics_dict, dataset, args.outdir)

# python plot_layer_metrics.py 
