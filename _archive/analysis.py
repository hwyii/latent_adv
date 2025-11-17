import os
import glob
import json
import csv
import re
import argparse
from typing import Dict, Any, List

def parse_experiment_path(json_path: str) -> Dict[str, Any]:
    """
    从 JSON 文件的父目录路径中智能提取实验参数。
    
    预期路径结构:
    .../param_grid_task_XX/seed<TRAIN_SEED>_R..._L<REFT_L>_A<ATTACK_L>.../attacks_high/attack_results_seed<EVAL_SEED>.json
    """
    try:
        parts = json_path.split(os.sep)
        
        # 1. 从文件名中提取 Eval Seed
        filename = os.path.basename(json_path)
        eval_seed_match = re.search(r"seed(\d+)\.json", filename)
        eval_seed = eval_seed_match.group(1) if eval_seed_match else None
        
        # 2. 从倒数第三个目录名中提取训练参数
        # (e.g., "seed42_Rrandom_L21_A6_r4_atklatent_pgd_task22")
        dir_name = parts[-3]
        
        train_seed_match = re.search(r"seed(\d+)", dir_name)
        reft_layer_match = re.search(r"_L(\d+)", dir_name)
        attack_layer_match = re.search(r"_A(\d+)", dir_name)
        rank_match = re.search(r"_r(\d+)", dir_name)
        
        return {
            "eval_seed": int(eval_seed) if eval_seed else None,
            "train_seed": int(train_seed_match.group(1)) if train_seed_match else None,
            "reft_layer": int(reft_layer_match.group(1)) if reft_layer_match else None,
            "attack_layer": int(attack_layer_match.group(1)) if attack_layer_match else None,
            "rank": int(rank_match.group(1)) if rank_match else None,
            "full_path": json_path
        }
    except Exception as e:
        print(f"警告: 解析路径 {json_path} 失败: {e}")
        return {"full_path": json_path}

def flatten_json_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    展平 JSON 内容, 只提取 adv 模型的结果。
    """
    flat_data = {}
    
    # 提取 config
    config = data.get("config", {})
    for key, value in config.items():
        flat_data[f"config_{key}"] = value
        
    # 提取 adv.summary_metrics
    summary = data.get("adv", {}).get("summary_metrics", {})
    for key, value in summary.items():
        # 清理键名, e.g., "Clean Accuracy (ACC)" -> "adv_clean_acc"
        clean_key = f"adv_{key.split(' (')[0].lower().replace(' ', '_')}"
        flat_data[clean_key] = value

    # 提取 adv.raw_counts
    counts = data.get("adv", {}).get("raw_counts", {})
    for key, value in counts.items():
        flat_data[f"adv_count_{key}"] = value
        
    return flat_data

def main():
    parser = argparse.ArgumentParser(
        description="聚合所有 'attack_results_seed*.json' 文件到一个 CSV 中。"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="out/pythia410m/harmless/reft_att_heatmap",
        help="包含所有 'param_grid_task_*' 文件夹的根目录。"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="aggregated_attack_results.csv",
        help="输出 CSV 文件的名称。"
    )
    args = parser.parse_args()

    # --- 1. 查找所有结果文件 ---
    # (我们使用 'attacks_high' 来匹配你的新路径)
    glob_pattern = os.path.join(
        args.root_dir, 
        "param_grid_task_*", 
        "seed*", 
        "attacks_high", 
        "attack_results_seed*.json"
    )
    
    print(f"正在搜索: {glob_pattern}")
    json_files = glob.glob(glob_pattern, recursive=True)
    
    if not json_files:
        print(f"错误: 在 {args.root_dir} 下未找到任何匹配的结果文件。")
        print("请检查你的 --root_dir 路径。")
        return

    print(f"找到了 {len(json_files)} 个结果文件。正在处理...")

    all_results: List[Dict[str, Any]] = []
    
    for json_path in json_files:
        # --- 2. 解析路径和 JSON ---
        try:
            path_data = parse_experiment_path(json_path)
            
            with open(json_path, 'r') as f:
                json_content = json.load(f)
            
            json_data = flatten_json_data(json_content)
            
            # --- 3. 合并数据 ---
            combined_data = {**path_data, **json_data}
            all_results.append(combined_data)
            
        except Exception as e:
            print(f"处理文件 {json_path} 时失败: {e}")
            continue

    # --- 4. 写入 CSV ---
    if not all_results:
        print("没有成功解析任何数据。")
        return

    # 动态获取所有可能的列名
    fieldnames = sorted(list(set(key for row in all_results for key in row.keys())))

    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n成功！所有 {len(all_results)} 条结果已聚合到: {args.output_csv}")
    
    except Exception as e:
        print(f"\n错误: 无法写入 CSV 文件: {e}")

if __name__ == "__main__":
    main()