#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_pca.py

(步骤 A: Offline)
运行一次此脚本，用于:
 1. 加载干净模型和数据
 2. 在指定的 `attack_layer` 提取 N 个样本的隐状态 `h`
 3. 运行 PCA 找到方差最大的 `k` 个方向
 4. 将这 `k` 个方向 (R_pca 矩阵) 保存到 .pt 文件
 5. (回答 Q3) 可视化 "Scree Plot"，展示 k 维捕获了多少信息
"""
import argparse, os, torch, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader

# 导入你现有的辅助函数
# (假设此脚本与你的 src/ 目录在同一级别，或者 src/ 在 PYTHONPATH 中)
try:
    from src.attack.inner_attack import get_module_by_name, find_target_component_name
    from src.data.adv_dataset import AdvDataset
    from src.data.Collator import Collator
except ImportError:
    print("错误: 无法导入 src.utils.tools 或 src.data。")
    print("请确保此脚本的运行路径正确，或者 src/ 位于 PYTHONPATH。")
    exit(1)

def get_activations(model, loader, attack_layer, n_samples, device):
    """ 在 attack_layer 抓取 n_samples 个激活 """
    model.eval()
    
    # 找到 hook 模块
    attack_module_name = find_target_component_name(model, attack_layer)
    attack_module = get_module_by_name(model.model, attack_module_name)
    
    activations_list = []
    h_storage = {}
    def get_h_hook(module, input, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        h_storage['h'] = out.detach().cpu() # 移动到 CPU 节省 VRAM
        
    hook_handle = attack_module.register_forward_hook(get_h_hook)
    
    samples_collected = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="收集激活"):
            if samples_collected >= n_samples:
                break
                
            batch_size = batch['input_ids'].size(0)
            batch_for_model = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            _ = model.model(**batch_for_model)
            
            if 'h' in h_storage:
                # h 形状: [B, S, D]
                # 我们只关心非 [PAD] token 的激活
                mask = batch['attention_mask'].bool() # [B, S]
                h_unpadded = h_storage['h'][mask] # [N_tokens, D]
                activations_list.append(h_unpadded)
                samples_collected += batch_size
            
            h_storage.clear()
            
    hook_handle.remove()
    
    print(f"收集了 {samples_collected} 个样本的激活。")
    # 将所有激活拼接为一个大张量
    all_h = torch.cat(activations_list, dim=0) # [TotalTokens, D]
    print(f"总 token 数: {all_h.shape[0]}, 维度: {all_h.shape[1]}")
    return all_h

def run_pca_and_visualize(H, k, attack_layer):
    """ 运行 PCA 并可视化 (回答 Q3) """
    print(f"在 [N, D] = {H.shape} 上运行 PCA (q=k={k})...")
    
    # D 维度太高 (e.g., 1024), N 可能更大 (e.g., 512*200)
    # 我们使用 pca_lowrank 来高效计算
    # H 必须是 float
    H = H.float()
    
    # U, S, V = torch.pca_lowrank(H, q=k)
    # U: [N, k] (scores)
    # S: [k] (singular values)
    # V: [D, k] (components / a.k.a. R_pca)
    U, S, V = torch.pca_lowrank(H, q=k)
    
    print(f"PCA 完成。R_pca 矩阵 (V) 形状: {V.shape}")

    # --- (回答 Q3) 可视化 Scree Plot ---
    explained_variance = S.pow(2)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance
    cumulative_explained_variance = explained_variance_ratio.cumsum(0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, k + 1), cumulative_explained_variance.numpy(), marker='o')
    plt.title(f'PCA 可解释方差 (Attack Layer {attack_layer})')
    plt.xlabel('主成分数量 (k)')
    plt.ylabel('累积可解释方差比例')
    plt.grid(True)
    
    # 打印一些关键点
    k_90_pct_index = (cumulative_explained_variance > 0.90).nonzero()
    if k_90_pct_index.numel() > 0:
        k_90_pct = k_90_pct_index.min().item() + 1
        print(f"需要 {k_90_pct} 维来捕获 90% 的方差。")
    else:
        print("在前 k 维中未达到 90% 的方差。")
        
    print(f"前 {k} 维捕获了 {cumulative_explained_variance[-1]*100:.2f}% 的方差。")
    
    plot_filename = f'pca_L{attack_layer}_k{k}_scree_plot.png'
    plt.savefig(plot_filename)
    print(f"Scree plot 已保存到 {plot_filename}")
    
    return V # [D, k]

def main():
    parser = argparse.ArgumentParser(description="步骤 A: 离线计算 PCA 方向")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--baseline_ckpt", type=str, required=True, help="用于提取激活的干净模型 .pt 路径")
    parser.add_argument("--splits_json", type=str, default="src/data/enron_splits.json")
    parser.add_argument("--split", default="ft_train", help="用于计算 PCA 的数据分割 (e.g., 'train')")
    parser.add_argument("--dataset_name", default="AlignmentResearch/EnronSpam")
    
    parser.add_argument("--attack_layer", type=int, required=True, help="要提取激活的攻击层 (e.g., 21)")
    parser.add_argument("--k", type=int, default=50, help="要保留的 PCA 主成分数量")
    parser.add_argument("--n_samples", type=int, default=200, help="用于计算 PCA 的样本数")
    
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. 加载模型
    print(f"加载模型 {args.model_name} (来自 {args.baseline_ckpt})...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    state = torch.load(args.baseline_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载数据
    print(f"加载数据 {args.dataset_name} (split: {args.split})...")
    with open(args.splits_json, "r") as f:
        splits = json.load(f)
    dataset = AdvDataset(splits[args.split], dataset_name=args.dataset_name)
    collator = Collator(args.model_name, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # 3. 提取激活
    H = get_activations(model, loader, args.attack_layer, args.n_samples, device)
    
    # 4. 运行 PCA 并可视化
    R_pca = run_pca_and_visualize(H, args.k, args.attack_layer)
    
    # 5. 保存 R_pca 矩阵
    output_filename = f'R_pca_L{args.attack_layer}_k{args.k}.pt'
    torch.save(R_pca, output_filename)
    print(f"PCA 矩阵 (R_pca) [D, k] 已保存到 {output_filename}")

if __name__ == "__main__":
    import json # 确保 json 已导入
    main()

