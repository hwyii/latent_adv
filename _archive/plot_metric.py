import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt # 导入 matplotlib
import os # 导入 os 用于检查文件路径

# ---------- 1. 加载和准备数据 ----------
# (假设您的JSON数据保存在名为 'layer_metrics.json' 的文件中)
json_file_path = 'out/pythia410m/harmless/layer_tri_metrics_Harmless.json'

# 确保目录存在（如果需要）
# os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

try:
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    # 提取 'results' 部分的数据
    results = data.get('results', {})
    if not results:
        print("错误：JSON数据中未找到 'results' 键或其内容为空。")
        # 如果在本地运行，您可能希望在这里退出
        # exit()
        
except FileNotFoundError:
    print(f"错误：未找到文件 '{json_file_path}'。")
    print("警告：将使用随机生成的示例数据继续执行以演示代码。")
    # 创建假的 data 字典以便代码的其余部分可以运行
    results = {str(i): {'log_auc': np.random.rand(), 
                       'svm_auc': np.random.rand(), 
                       'mlp_auc': np.random.rand(), 
                       'sw_mean_sq': np.random.rand()} 
               for i in range(24)} # 假设有24层

except json.JSONDecodeError:
    print(f"错误：'{json_file_path}' 文件中的JSON格式无效，请检查。")
    # exit()
    print("警告：JSON 解码错误。将使用示例数据继续执行以演示代码。")
    results = {str(i): {'log_auc': np.random.rand(), 
                       'svm_auc': np.random.rand(), 
                       'mlp_auc': np.random.rand(), 
                       'sw_mean_sq': np.random.rand()} 
               for i in range(24)}

except Exception as e:
    # 捕获其他潜在错误
    print(f"发生意外错误: {e}")
    print("警告：发生错误。将使用示例数据继续执行以演示代码。")
    results = {str(i): {'log_auc': np.random.rand(), 
                       'svm_auc': np.random.rand(), 
                       'mlp_auc': np.random.rand(), 
                       'sw_mean_sq': np.random.rand()} 
               for i in range(24)}


# 将数据转换为 Pandas DataFrame，方便处理
# 确保 'results' 不是空的
if not results:
    print("错误：'results' 数据为空，无法继续。请检查JSON文件。")
    # exit()
else:
    layer_indices = sorted([int(k) for k in results.keys()])
    metrics_data = []
    for layer in layer_indices:
        layer_str = str(layer)
        metrics = results.get(layer_str, {})
        metrics_data.append({
            'Layer': layer,
            'LogReg AUC': metrics.get('log_auc'),
            'SVM AUC': metrics.get('svm_auc'),
            'MLP AUC': metrics.get('mlp_auc'),
            'SWD²': metrics.get('sw_mean_sq') # 使用 sw_mean_sq
        })

    df = pd.DataFrame(metrics_data)

    # 检查是否有缺失值
    if df.isnull().values.any():
        print("警告：数据中存在缺失值，可能影响归一化和绘图。")
        print("缺失值所在的行:")
        print(df[df.isnull().any(axis=1)])
        # 简单处理：用0填充缺失值，您可能需要更复杂的处理方式
        print("用 0 填充缺失值...")
        df = df.fillna(0)

    # ---------- 2. 归一化处理 ----------
    metrics_to_normalize = ['LogReg AUC', 'SVM AUC', 'MLP AUC', 'SWD²']
    
    # 确保要归一化的列都存在
    valid_metrics = [col for col in metrics_to_normalize if col in df.columns]
    
    if len(valid_metrics) < len(metrics_to_normalize):
        print(f"警告：并非所有指定的指标都在DataFrame中。将只归一化：{valid_metrics}")

    if not valid_metrics:
        print("错误：没有可用于归一化的有效指标列。")
        # exit()
    else:
        scaler = MinMaxScaler()
        
        # 对每个指标进行 Min-Max 归一化
        df_normalized = df.copy()
        df_normalized[valid_metrics] = scaler.fit_transform(df[valid_metrics])

        # ---------- 3. 可视化对比 (使用 Matplotlib) ----------

        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(12, 7)) # 设置图像大小

        # 循环绘制每一条线
        for metric in valid_metrics:
            ax.plot(
                df_normalized['Layer'],     # X 轴
                df_normalized[metric],      # Y 轴
                label=metric,               # 图例标签
                marker='o',                 # 标记点
                markersize=5,               # 标记大小
                linestyle='-'               # 线条样式
            )

        # 设置图表标题和坐标轴标签
        ax.set_title('Normalized Performance Metrics vs. Layer Index', fontsize=16)
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Normalized Metric Value (0 to 1)', fontsize=12)

        # 添加图例
        ax.legend(title='Metric Type')

        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.6)

        # 设置 X 轴刻度为整数（如果层数不多的话）
        if len(layer_indices) <= 30: # 仅当层数较少时
             ax.set_xticks(layer_indices[::2]) # 每隔一层显示一个刻度
        
        # 自动调整布局，防止标签重叠
        plt.tight_layout()

        # 保存图表为PNG文件
        chart_path_png = 'normalized_metrics_comparison.png'
        try:
            plt.savefig(chart_path_png)
            print(f"\n对比图表已保存为 '{chart_path_png}'")
            
            # (可选) 显示图表
            # plt.show() 
            
        except Exception as e:
            print(f"保存图表时出错: {e}")

        print("\n数据归一化后前5行:")
        print(df_normalized.head())
        print("\n数据归一化后后5行:")
        print(df_normalized.tail())