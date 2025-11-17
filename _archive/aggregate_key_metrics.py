# src/utils/aggregate_key_metrics.py
import pandas as pd
import numpy as np

def summarize_key_metrics(csv_path: str,
                          out_csv: str = None,
                          group_by_cols=None) -> pd.DataFrame:
    """
    读取 summarize_runs 产出的CSV，按配置（不含seed）聚合，对seed做平均，输出关键指标表。
    默认分组键：
      - model_name, layer_idx, rank_r, R_init_mode,
      - train_inner_attack（优先使用 *_from_dir 列）
    关键输出列：
      - seeds（该配置下seed数）
      - attack_asr_mean/std
      - eval_last_mean/std
      - prop_pretraining_mean (= Adversarial Training Compute (Proportion of Pretraining))
      - Cadv_total_mean, Ctrain_mean, Csearch_mean, search_over_train_mean
      - flops_per_sample_mean
    """
    df = pd.read_csv(csv_path)

    # 兼容不同列名：从目录名解析得到的列优先
    atk_col = "train_inner_attack_from_dir" if "train_inner_attack_from_dir" in df.columns else "train_inner_attack"
    rinit_col = "R_init_mode" if "R_init_mode" in df.columns else None
    layer_col = "layer_idx"
    rank_col = "rank_r"
    model_col = "model_name"

    # 默认分组键
    if group_by_cols is None:
        group_by_cols = [c for c in [model_col, layer_col, rank_col, rinit_col, atk_col] if c is not None and c in df.columns]

    # 需要数值化的列（有时会以字符串写入CSV）
    num_cols = [
        "attack_asr", "eval_last",
        "prop_of_pretraining",   # proportion_of_pretraining
        "Cadv_total", "Ctrain", "Csearch", "Caux",
        "search_over_train", "flops_per_sample",
        "elapsed_seconds", "steps_recorded", "steps_per_sec",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 选择我们关心的最小列集，避免别的杂项干扰
    keep_cols = list(set(group_by_cols + [
        "seed", "seed_from_dir",
        "attack_asr", "eval_last",
        "prop_of_pretraining",
        "Cadv_total", "Ctrain", "Csearch",
        "search_over_train", "flops_per_sample",
    ]) & set(df.columns))
    d = df[keep_cols].copy()

    # 选择 seed 列（优先 seed_from_dir）
    seed_col = "seed_from_dir" if "seed_from_dir" in d.columns else ("seed" if "seed" in d.columns else None)

    # 分组聚合：对 seed 求均值；同时统计 seeds 数
    agg_map = {
        "attack_asr": ["mean", "std"],
        "eval_last": ["mean", "std"],
        "prop_of_pretraining": ["mean"],          # Adversarial Training Compute (Proportion of Pretraining)
        "Cadv_total": ["mean"],
        "Ctrain": ["mean"],
        "Csearch": ["mean"],
        "search_over_train": ["mean"],
        "flops_per_sample": ["mean"],
    }
    # 丢掉全 NaN 的列
    agg_map = {k: v for k, v in agg_map.items() if k in d.columns}

    g = d.groupby(group_by_cols, dropna=False)
    out = g.agg(agg_map)

    # 展平多级列
    out.columns = ["{}_{}".format(k, s) for (k, s) in out.columns]
    out = out.reset_index()

    # 统计 seeds 数
    if seed_col:
        seeds_count = g[seed_col].nunique().reset_index(name="seeds")
        out = out.merge(seeds_count, on=group_by_cols, how="left")
    else:
        out["seeds"] = np.nan

    # 友好列名
    rename_map = {
        "attack_asr_mean": "attack_asr_mean",
        "attack_asr_std": "attack_asr_std",
        "eval_last_mean": "eval_last_mean",
        "eval_last_std": "eval_last_std",
        "prop_of_pretraining_mean": "prop_pretraining_mean",
        "Cadv_total_mean": "Cadv_total_mean",
        "Ctrain_mean": "Ctrain_mean",
        "Csearch_mean": "Csearch_mean",
        "search_over_train_mean": "search_over_train_mean",
        "flops_per_sample_mean": "flops_per_sample_mean",
    }
    out = out.rename(columns=rename_map)

    # 最终列顺序（尽量简洁）
    ordered_cols = [c for c in group_by_cols if c in out.columns] + [
        "seeds",
        "attack_asr_mean", "attack_asr_std",
        "eval_last_mean", "eval_last_std",
        "prop_pretraining_mean",       # <= 你要的 Adversarial Training Compute (Proportion of Pretraining)
        "Cadv_total_mean", "Ctrain_mean", "Csearch_mean", "search_over_train_mean",
        "flops_per_sample_mean",
    ]
    # 补上存在但不在有序列表的列（防止遗漏）
    ordered_cols += [c for c in out.columns if c not in ordered_cols]
    out = out[ordered_cols]

    if out_csv:
        out.to_csv(out_csv, index=False)
        print(f"[aggregate] wrote {out_csv} with {len(out)} rows")

    return out

summarize_key_metrics('out_spam_summary.csv', 'out_spam_key_summary.csv')
