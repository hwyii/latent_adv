#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Gemini v3 - 最终版]
此脚本现在*只*读取 JSON 文件，不再解析 stdout.log。
它会:
 1. 查找 训练 JSON (Rinit...json)
 2. 查找 *所有* 评估 JSON (attacks/attack_results_seed*.json)
 3. 计算评估指标的 均值(mean) 和 标准差(std)
 4. 将所有结果汇总到 CSV
"""
import os, re, json, argparse, glob, csv
import sys
import numpy as np
from statistics import mean

def parse_attack_results(run_dir: str):
    """
    解析 attacks/ 目录下的 *所有* attack_results_seed*.json 文件,
    并计算 均值(mean) 和 标准差(stddev)。
    """
    search_pattern = os.path.join(run_dir, "attacks", "attack_results_seed*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        return {} # 没有找到评估 JSON

    # 存储每个 seed 跑出来的指标
    metrics = {
        'baseline_acc': [],
        'baseline_robust_acc': [],
        'baseline_asr': [],
        'adv_acc': [],
        'adv_robust_acc': [],
        'adv_asr': [],
        'adv_C_to_C': [],
        'adv_C_to_W': [],
        'adv_W_to_W': [],
        'adv_W_to_C': [],
    }

    try:
        for f_path in json_files:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 提取 baseline 和 adv 的统计数据
            bsum = data.get("baseline", {}).get("summary_metrics", {})
            asum = data.get("adv", {}).get("summary_metrics", {})
            acounts = data.get("adv", {}).get("raw_counts", {})

            metrics['baseline_acc'].append(bsum.get("Clean Accuracy (ACC)"))
            metrics['baseline_robust_acc'].append(bsum.get("Robust Accuracy (Rob-Acc)"))
            metrics['baseline_asr'].append(bsum.get("Attack Success Rate (ASR)"))
            
            metrics['adv_acc'].append(asum.get("Clean Accuracy (ACC)"))
            metrics['adv_robust_acc'].append(asum.get("Robust Accuracy (Rob-Acc)"))
            metrics['adv_asr'].append(asum.get("Attack Success Rate (ASR)"))

            # 从 raw_counts 中提取 adv 模型的 4 种情况
            metrics['adv_C_to_C'].append(acounts.get("n_correct_after_attack_C_to_C"))
            metrics['adv_C_to_W'].append(acounts.get("n_flipped_to_wrong_C_to_W"))
            metrics['adv_W_to_W'].append(acounts.get("n_stayed_wrong_W_to_W"))
            metrics['adv_W_to_C'].append(acounts.get("n_flipped_to_correct_W_to_C"))
            
        # --- 计算均值和标准差 ---
        results = {}
        for key, values in metrics.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                results[f"{key}_mean"] = np.mean(valid_values)
                results[f"{key}_std"] = np.std(valid_values)
                results[f"{key}_n_runs"] = len(valid_values)
        
        return results
        
    except Exception as e:
        print(f"解析 {run_dir} 的 JSON 时出错: {e}", file=sys.stderr)
        return {}


def try_load_report_json(run_dir: str):
    """
    在 run_dir 中查找 Rinit...json
    """
    candidates = glob.glob(os.path.join(run_dir, "Rinit*.json"))
    if not candidates:
        return None, None
    # 修复：确保我们选择最新的或唯一的, 而不是列表中的第一个
    candidates.sort(key=os.path.getmtime, reverse=True)
    rep_path = candidates[0] 
    try:
        with open(rep_path, "r") as f:
            data = json.load(f)
        if ("Ctrain" in data or "Cadv_total" in data) and ("cfg_summary" in data or "meta" in data):
            return rep_path, data
    except Exception:
        return None, None
    return None, None


def summarize_root(root: str, out_csv: str, out_json: str = None):
    rows = []
    # 路径匹配 '.../param_grid_task_*/seed*.../'
    search_pattern = os.path.join(root, "param_grid_task_*", "seed*")
    run_dirs = sorted(glob.glob(search_pattern))
    
    print(f"在 {root} 下找到 {len(run_dirs)} 个潜在的 run 目录...")
    
    found_runs_count = 0

    for cur_dir in run_dirs:
        if not os.path.isdir(cur_dir) or os.path.basename(cur_dir) == "attacks":
            continue

        # 1. 解析训练 JSON
        rep_path, data = try_load_report_json(cur_dir)
        if rep_path is None or data is None:
            print(f"  - 在 {cur_dir} 中未找到训练 JSON, 跳过。")
            continue

        # 2. 解析 *所有* ASR 评估 JSON 并计算均值/标准差
        attack_results = parse_attack_results(cur_dir)
        if not attack_results:
             print(f"  - 在 {cur_dir} 中未找到 ASR JSON 结果, 跳过。")
             continue

        # --- 成功：我们拿到了所有数据 ---
        found_runs_count += 1
        
        meta = data.get("meta", {})
        cfgs = data.get("cfg_summary", {})
        eval_accs = data.get("eval_accs", [])
        eval_last = eval_accs[-1] if eval_accs else None
        
        row = {
            "run_dir": cur_dir,
            "reft_layer": cfgs.get("layer_idx"),
            "attack_layer": cfgs.get("attack_layer"),
            "rank_r": cfgs.get("rank_r"),
            "train_inner_attack": cfgs.get("inner_attack"),
            "clean_acc_validation": eval_last, 
            "elapsed_seconds": meta.get("elapsed_seconds"),
        }

        # 添加所有均值/标准差指标
        row.update(attack_results)
        rows.append(row)

    # 写入 CSV
    if not rows:
        print(f"[总结] 在 {root} 下没有找到任何 *完整* 的运行 (同时包含 训练JSON 和 ASR JSON)。")
        return
        
    fieldnames = sorted({k for r in rows for k in r.keys()})
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    
    print(f"\n[总结] {found_runs_count}/{len(run_dirs)} 次运行已成功汇总 -> {out_csv}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"               & {out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, 
                    help="root dir: out/pythia410m/harmless/reft_att_heatmap")
    ap.add_argument("--out_csv", type=str, default="summary_all_metrics_v3.csv")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()
    summarize_root(args.root, args.out_csv, args.out_json)