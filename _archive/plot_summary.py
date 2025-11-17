import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def create_summary_heatmaps(csv_file="summary.csv"):
    """
    加载 summarize.py 生成的 summary.csv, 
    并绘制 3 个关键的热力图:
    1. 干净验证集准确率 (Clean Acc - Validation)
    2. 对抗训练模型的 ASR (Adv-Trained ASR)
    3. ASR 降低率 (ASR Reduction %)
    """
    
    # --- 1. 加载和准备数据 ---
    if not os.path.exists(csv_file):
        print(f"错误: 未找到文件 '{csv_file}'。")
        print("请先运行 summarize.py 来生成它。")
        return

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"读取 CSV 时出错: {e}")
        return

    # --- 2. 计算我们的“钱”图指标 ---
    # (我们使用 .mean() 来处理 baseline_asr, 因为25个实验的 baseline 值可能略有不同)
    avg_baseline_asr = df['baseline_asr'].mean()
    print(f"已加载数据。检测到平均 Baseline ASR 为: {avg_baseline_asr:.4f}")
    
    # ASR 降低率 = (Baseline - Adv) / Baseline
    df['asr_reduction_pct'] = (avg_baseline_asr - df['adv_asr']) / avg_baseline_asr

    # --- 3. 将数据“透视”为 2D 网格 ---
    try:
        # 图 A: 干净性能 (来自训练 JSON)
        clean_acc_pivot = df.pivot(
            index="reft_layer", 
            columns="attack_layer", 
            values="clean_acc_validation"
        )
        
        # 图 B: 鲁棒性 (来自 ASR 日志)
        adv_asr_pivot = df.pivot(
            index="reft_layer", 
            columns="attack_layer", 
            values="adv_asr"
        )
        
    except Exception as e:
        print(f"Pivoting data failed. Do you have duplicate (reft_layer, attack_layer) pairs? Error: {e}")
        return

    # --- 4. 绘制热力图 ---
    
    # 设置一个 1x3 的子图布局
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # ----- 图 1: 干净准确率 (来自 验证集) -----
    sns.heatmap(
        clean_acc_pivot, 
        ax=axes[0], 
        annot=True,     # 显示数字
        fmt=".4f",      # 4位小数
        cmap="YlGnBu",  # 颜色: 越高越好
        linewidths=.5,
        vmin=0.50,      # 锁定最小颜色范围 (突出 0.501 的灾难)
        vmax=0.70       # 锁定最大颜色范围
    )
    axes[0].set_title("Figure 1: Clean accuracy", fontsize=16)
    axes[0].set_xlabel("Attack_Layer", fontsize=15)
    axes[0].set_ylabel("ReFT_Layer", fontsize=15)

    # ----- 图 2: 对抗成功率 (Adv ASR) -----
    sns.heatmap(
        adv_asr_pivot, 
        ax=axes[1], 
        annot=True,
        fmt=".4f",
        cmap="Reds",    # 颜色: 越低越好 (ASR 越低 = 防御越好)
        linewidths=.5
    )
    axes[1].set_title("Figure 2: Attack Sucess Rate", fontsize=16)
    axes[1].set_xlabel("Attack_Layer", fontsize=15)
    axes[1].set_ylabel("ReFT_Layer", fontsize=15)

    # --- 5. 保存和显示 ---
    plt.suptitle("Latent ReFT", fontsize=20, y=1.03)
    plt.tight_layout()
    
    output_filename = "summary_heatmaps.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    
    print(f"\n--- 总结热力图已成功保存为 {output_filename} ---")
    

if __name__ == "__main__":
    create_summary_heatmaps()
