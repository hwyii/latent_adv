# import json
# import glob
# import os
# import sys
# import csv

# # --- 关键常数 ---
# # "EleutherAI/pythia-410m" 的总层数是 24
# N_TOTAL_LAYERS = 24 

# def recalculate_flops_for_run(json_path):
#     """
#     加载单个 JSON 文件, 并使用“可变成本”公式重新计算 FLOPs,
#     同时提取最终的 eval_acc。
#     """
#     try:
#         with open(json_path, 'r') as f:
#             data = json.load(f)

#         # --- 1. 提取我们需要的“原材料” ---
#         cfg = data.get("cfg_summary")
#         counts = data.get("counts")
        
#         if not cfg or not counts:
#             print(f"警告: {json_path} 中缺少 'cfg_summary' 或 'counts' 键。跳过。", file=sys.stderr)
#             return None
            
#         attack_layer = int(cfg["attack_layer"])
#         reft_layer = int(cfg["layer_idx"])
        
#         # 确保 attack_layer 在有效范围内
#         if not (1 <= attack_layer <= N_TOTAL_LAYERS):
#              print(f"警告: 发现 attack_layer={attack_layer}, "
#                    f"超出了模型层数 {N_TOTAL_LAYERS}。跳过 {json_path}", file=sys.stderr)
#              return None

#         # FLOPs 常数 (从 JSON 中读取)
#         F_fwd_full = float(data["F_fwd_full"])
#         F_bwd_full = F_fwd_full * 2.0 # 标准假设: BWD ≈ 2 * FWD

#         # 计数值 (从 JSON 中读取)
#         n_train_fwd = counts.get("n_train_full_fwd_samples", 0)
#         n_train_bwd = counts.get("n_train_full_bwd_samples", 0)
#         n_search_fwd = counts.get("n_search_full_fwd_samples", 0)
#         n_search_bwd = counts.get("n_search_full_bwd_samples", 0)

#         # --- 2. 重新计算 FLOPs ---
        
#         # Ctrain (这部分成本是固定的)
#         Ctrain = (n_train_fwd * F_fwd_full) + (n_train_bwd * F_bwd_full)
        
#         # Csearch (这部分是可变的)
#         Csearch_fwd = n_search_fwd * F_fwd_full
        
#         # --- 这是我们的“魔法公式” ---
#         bwd_cost_fraction = (N_TOTAL_LAYERS - attack_layer + 1) / float(N_TOTAL_LAYERS)
#         F_bwd_partial = F_bwd_full * bwd_cost_fraction
        
#         Csearch_bwd_corrected = n_search_bwd * F_bwd_partial
#         Csearch_corrected = Csearch_fwd + Csearch_bwd_corrected

#         # Cadv_total (修正后的总和)
#         Cadv_total_corrected = Ctrain + Csearch_corrected

#         # --- 3. 提取 Eval Acc ---
#         eval_accs_list = data.get("eval_accs", [])
#         final_eval_acc = None
#         if eval_accs_list:
#             final_eval_acc = eval_accs_list[-1] # 获取列表中的最后一个准确率

#         return {
#             "reft_layer": reft_layer,
#             "attack_layer": attack_layer,
#             "final_eval_acc": final_eval_acc,
#             "corrected_adv_total_flops": Cadv_total_corrected,
#             "original_adv_total_flops": data.get("Cadv_total", 0),
#             "elapsed_seconds": data.get("meta", {}).get("elapsed_seconds", 0)
#         }

#     except Exception as e:
#         print(f"处理 {json_path} 时出错: {e}", file=sys.stderr)
#         return None

# def main():
#     # 1. 设置你的实验根目录
#     base_dir = "out/pythia410m/harmless/reft_att_heatmap"
    
#     # 2. 构建新的 glob 模式以匹配你的嵌套目录结构
#     #    它会查找 "base_dir/param_grid_task_*/seed*/Rinit*.json"
#     search_pattern = os.path.join(base_dir, "param_grid_task_*", "seed*", "Rinit*.json")
    
#     print(f"正在搜索: {search_pattern}\n")
#     json_files = glob.glob(search_pattern)
    
#     if not json_files:
#         print(f"错误: 在 {base_dir} 中没有找到匹配的 '.json' 文件。")
#         print("请检查你的 `base_dir` 路径和目录结构。")
#         return

#     results = []
#     for f_path in json_files:
#         result = recalculate_flops_for_run(f_path)
#         if result:
#             results.append(result)

#     if not results:
#         print("没有找到可以成功处理的 JSON 文件。")
#         return

#     # 3. 按 ReFT 层和 Attack 层排序，方便查看
#     results.sort(key=lambda x: (x["reft_layer"], x["attack_layer"]))
    
#     # 4. 将结果保存到 CSV 文件 (最好的存储方式)
#     output_csv_file = "flops_and_acc_report.csv"
#     csv_header = [
#         "ReFT_Layer", 
#         "Attack_Layer", 
#         "Final_Eval_Acc", 
#         "Corrected_Total_FLOPs", 
#         "Elapsed_Seconds", 
#         "Original_Total_FLOPs"
#     ]
    
#     try:
#         with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow(csv_header)
#             for res in results:
#                 writer.writerow([
#                     res["reft_layer"],
#                     res["attack_layer"],
#                     f"{res['final_eval_acc']:.6f}" if res['final_eval_acc'] is not None else "N/A",
#                     f"{res['corrected_adv_total_flops']:.4e}",
#                     f"{res['elapsed_seconds']:.1f}",
#                     f"{res['original_adv_total_flops']:.4e}"
#                 ])
#         print(f"\n--- 报告已成功保存到 {output_csv_file} ---")

#     except Exception as e:
#         print(f"\n--- 写入 CSV 文件时出错: {e} ---")


#     # 5. 在终端打印漂亮的表格 (方便快速查看)
#     print(f"\n成功处理了 {len(results)}/{len(json_files)} 个实验的 FLOPs。")
#     print("\n--- 修正后的 FLOPs 与准确率报告 ---")
#     print("ReFT_Layer\tAttack_Layer\tFinal_Eval_Acc\tCorrected_FLOPs\t(Original_FLOPs)\tElapsed_Secs")
#     print("-" * 110)
#     for res in results:
#         acc_str = f"{res['final_eval_acc']:.6f}" if res['final_eval_acc'] is not None else "N/A"
#         print(f"{res['reft_layer']}\t\t{res['attack_layer']}\t\t{acc_str}\t\t"
#               f"{res['corrected_adv_total_flops']:.4e}\t\t"
#               f"({res['original_adv_total_flops']:.4e})\t\t"
#               f"{res['elapsed_seconds']:.1f}")

# if __name__ == "__main__":
#     main()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_heatmaps(csv_file="flops_and_acc_report.csv"):
    """
    加载 CSV 报告, 将数据转换为 5x5 网格, 
    并绘制三个并排的热力图：
    1. 最终评估准确率 (Final Eval Acc)
    2. 修正后的总 FLOPs (Corrected Total FLOPs)
    3. 运行时间 (Elapsed Seconds)
    """
    
    # --- 1. 加载和准备数据 ---
    if not os.path.exists(csv_file):
        print(f"错误: 未找到文件 '{csv_file}'。")
        print("请确保此脚本与你的 CSV 报告在同一目录中。")
        return

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"读取 CSV 时出错: {e}")
        return

    # 将数据转换为数值类型，以防万一
    df['ReFT_Layer'] = pd.to_numeric(df['ReFT_Layer'])
    df['Attack_Layer'] = pd.to_numeric(df['Attack_Layer'])
    df['Final_Eval_Acc'] = pd.to_numeric(df['Final_Eval_Acc'])
    df['Corrected_Total_FLOPs'] = pd.to_numeric(df['Corrected_Total_FLOPs'])
    df['Elapsed_Seconds'] = pd.to_numeric(df['Elapsed_Seconds'])

    # --- 2. 将数据“透视”为 2D 网格 (Heatmap 的核心) ---
    try:
        acc_pivot = df.pivot(
            index="ReFT_Layer", 
            columns="Attack_Layer", 
            values="Final_Eval_Acc"
        )
        flops_pivot = df.pivot(
            index="ReFT_Layer", 
            columns="Attack_Layer", 
            values="Corrected_Total_FLOPs"
        )
        time_pivot = df.pivot(
            index="ReFT_Layer", 
            columns="Attack_Layer", 
            values="Elapsed_Seconds"
        )
    except Exception as e:
        print(f"Pivoting data failed. Do you have duplicate (ReFT_Layer, Attack_Layer) pairs? Error: {e}")
        return

    # --- 3. 绘制热力图 ---
    
    # 设置一个 1x3 的子图布局，(宽, 高)
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    
    # ----- 图 1: 准确率 (Acc) -----
    sns.heatmap(
        acc_pivot, 
        ax=axes[0], 
        annot=True,     # 在格子上显示数字
        fmt=".4f",      # 数字格式化 (4位小数)
        cmap="YlGnBu",  # 颜色：黄-绿-蓝 (越高越好)
        linewidths=.5   # 格子间的线条
    )
    axes[0].set_title("Final Eval Accuracy", fontsize=16)
    axes[0].set_xlabel("Attack_Layer", fontsize=12)
    axes[0].set_ylabel("ReFT_Layer", fontsize=12)

    # ----- 图 2: FLOPs 成本 -----
    sns.heatmap(
        flops_pivot, 
        ax=axes[1], 
        annot=True,     # 在格子上显示数字
        fmt=".2e",      # 数字格式化 (科学计数法)
        cmap="Reds",    # 颜色：红色 (越低越好)
        linewidths=.5
    )
    axes[1].set_title("Corrected Total FLOPs", fontsize=16)
    axes[1].set_xlabel("Attack_Layer", fontsize=12)
    axes[1].set_ylabel("ReFT_Layer", fontsize=12)

    # ----- 图 3: 运行时间 -----
    sns.heatmap(
        time_pivot, 
        ax=axes[2], 
        annot=True,     # 在格子上显示数字
        fmt=".0f",      # 数字格式化 (整数)
        cmap="Oranges", # 颜色：橘色 (越低越好)
        linewidths=.5
    )
    axes[2].set_title("Elapsed Seconds", fontsize=16)
    axes[2].set_xlabel("Attack_Layer", fontsize=12)
    axes[2].set_ylabel("ReFT_Layer", fontsize=12)

    # --- 4. 保存和显示 ---
    plt.suptitle("Latent Adversarial ReFT", fontsize=20, y=1.03)
    plt.tight_layout()
    
    output_filename = "experiment_heatmaps.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    
    print(f"\n--- 热力图已成功保存为 {output_filename} ---")
    
    # 自动打开图像 (可选)
    try:
        if os.name == 'nt': # Windows
            os.startfile(output_filename)
        elif os.uname().sysname == 'Darwin': # MacOS
            os.system(f'open {output_filename}')
        else: # Linux
            os.system(f'xdg-open {output_filename}')
    except Exception:
        print(f"无法自动打开图像。请手动打开 {output_filename}")

if __name__ == "__main__":
    create_heatmaps()