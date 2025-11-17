#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sw_analysis.py
一键执行：计算每层 SW + 输出曲线 + 绘制指定层的图
示例：
  python -m src.utils.run_sw_analysis --latent_dir out/pythia410m/harmless/latents --dataset harmless --layer 21
"""
import os, argparse, numpy as np
from src.utils.sw_tools import (
    recompute_missing_sw, load_sw_details,
    plot_sw_over_layers, plot_layer_topk_1d,
    plot_layer_scatter_by_top2_with_boundary,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--n_proj", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_class", type=int, default=5000)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.latent_dir
    ds = args.dataset
    print(f"=== Running SW analysis for {ds} ===")

    # 1️⃣ 计算所有缺失的 *_sw.npz
    results = recompute_missing_sw(
        latent_dir=out_dir, dataset=ds,
        n_proj=args.n_proj, seed=args.seed,
        max_per_class=args.max_per_class, topk=args.topk,
    )

    # 2️⃣ 汇总层间 SW 曲线
    if not results:
        # 如果都存在，就从文件重载
        from pathlib import Path
        results = {}
        for p in Path(out_dir).glob(f"{ds}_L*_sw.npz"):
            try:
                L = int(p.stem.split("_L")[1].split("_")[0])
            except Exception:
                continue
            dat = load_sw_details(str(p))
            results[L] = {"mean_sq": float(dat.get("mean_sq", (dat["ws"]**2).mean()))}
    plot_sw_over_layers(results, save_path=os.path.join(out_dir, f"{ds}_SW_layers.png"))

    # 3️⃣ 绘制指定层的 Top-K 方向与 2D 散点（带边界）
    plot_layer_topk_1d(
        layer=args.layer, latent_dir=out_dir,
        dataset=ds, save_path=os.path.join(out_dir, f"{ds}_L{args.layer}_topk_1D.png"),
    )
    plot_layer_scatter_by_top2_with_boundary(
        layer=args.layer, latent_dir=out_dir, dataset=ds,
        save_path=os.path.join(out_dir, f"{ds}_L{args.layer}_top2_scatter.png"),
    )
    print(f"✅ 完成：输出保存在 {out_dir}")

if __name__ == "__main__":
    main()
# python -m src.utils.run_sw_analysis   --latent_dir out/pythia410m/harmless/latents   --dataset Harmless   --layer 23  --n_proj 256   --topk 8
