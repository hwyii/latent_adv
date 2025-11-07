"""
In latent space, do GCG attack or PGD attack
GCG may need less steps to achieve similary performance with PGD
"""

from regex import B
import torch
import pdb
import torch.nn.functional as F
from typing import Dict
from src.utils.tools import find_loreft_for_layer, get_outputs_from_reft, inject_hidden_at_layer
from src.utils.compute import incr_count
import functools

def get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    for name, module in model.named_modules():
        if name == module_name:
            return module
    raise ValueError(f"Module {module_name} not found in model.")

def find_target_component_name(reft_model: torch.nn.Module, layer_idx: int) -> str:
    if "gpt_neox" in reft_model.model.config.model_type:
        comp_idx = layer_idx - 1
        return f"gpt_neox.layers.{comp_idx}"
    else:
        raise ValueError(f"Model type {reft_model.model.config.model_type} not supported for finding target component.")


def manual_reft_defense_hook(module, input, output, loreft_module):
    """
    这个钩子在 reft_layer (例如 L6) 触发。
    它接收原始输出 'output'，将其通过 'loreft_module'，
    并返回干预后的结果。
    """
    # 1. 获取原始激活
    original_h = output[0] if isinstance(output, (tuple, list)) else output
    
    # 2. 手动应用 LoReFT 干预 (保持计算图连接)
    intervened_h = loreft_module(original_h)
    
    # 3. 将干预后的激活放回原处
    if torch.is_tensor(output):
        return intervened_h
    if isinstance(output,(tuple,list)):
        if isinstance(output, tuple):
            return (intervened_h,) + output[1:]
        else:
            return [intervened_h] + list(output[1:])
    return intervened_h

# hook1: fullspace attack
def dynamic_adv_hook(module, input, output, delta_high_dim):
    #pdb.set_trace()
    if torch.is_tensor(output):
        return output + delta_high_dim
    if isinstance(output,(tuple,list)):
        first = output[0]
        if torch.is_tensor(first):
            adv_first = first + delta_high_dim
            if isinstance(output, tuple):
                return (adv_first,) + output[1:]
            else:
                return [adv_first] + list(output[1:])  
# Hook 2: PCA 空间攻击
def dynamic_adv_hook_pca(module, input, output, R_pca, alpha):
    """
    在 L_atk 触发。
    它使用 [D, k] 的 R_pca 和 [B,S,k] 的 alpha 来重建 [B,S,D] 的 delta。
    """
    delta_high_dim = alpha @ R_pca.T # 重建攻击 [B, S, k] @ [k, D] = [B, S, D]
    
    if torch.is_tensor(output):
        return output + delta_high_dim
    if isinstance(output,(tuple,list)):
        first = output[0]
        if torch.is_tensor(first):
            adv_first = first + delta_high_dim
            if isinstance(output, tuple):
                return (adv_first,) + output[1:]
            else:
                return [adv_first] + list(output[1:])
    return output  

# *** 修改后的 compute_adv_loss_via_hooks ***
def compute_adv_loss_via_hooks(
    reft_model: torch.nn.Module,
    batch,
    attack_layer: int,
    reft_layer: int,
    attack_mode: str,
    eps: float = 0.05,
    steps: int = 3,
    lr: float = None,
    gcg_topk: int = 2,
    gcg_alpha: float = 0.01,
    pca_matrix_path: str = None # from attack layer
):
    #pdb.set_trace()  
    if attack_mode == "none":
        return torch.tensor(0.0, device=reft_model.device)

    # --- 1. 找到所有需要的模块 ---
    
    # (A) 攻击模块 (例如 L1)
    attack_module_name = find_target_component_name(reft_model, attack_layer)
    attack_module = get_module_by_name(reft_model.model, attack_module_name)
    
    # (B) 防御模块 (LoReFT 模块本身，例如 L6)
    target_loreft_module = find_loreft_for_layer(reft_model, reft_layer)
    
    # (C) 防御模块在 HF 模型中的 "挂钩点" (例如 L6 的 'gpt_neox.layers[5].output')
    # 我们假设 reft_config 中只有一个干预
    #pdb.set_trace()
    reft_rep_key = list(reft_model.representations.keys())[0]
    defense_component_name = reft_model.representations[reft_rep_key].layer
    defense_component_name_correct = find_target_component_name(reft_model, defense_component_name)
    # --- 修复结束 ---
    defense_module_hook_point = get_module_by_name(reft_model.model, defense_component_name_correct)

    B = batch["input_ids"].size(0)
    S = batch["input_ids"].size(1)

    # --- 2. 获取 h_clean (在 attack_layer) ---
    # 你的修复是正确的：我们必须调用 .model
    if attack_mode == "pgd_fullspace" or attack_mode == "gcg":
        batch_for_clean = batch.copy()
        h_clean_storage = {}
        def get_h_clean_hook(module, input, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            h_clean_storage['h_clean'] = out.detach() 
            
        h_hook_handle = attack_module.register_forward_hook(get_h_clean_hook)
        with torch.no_grad():
            _ = reft_model.model(
                input_ids=batch_for_clean['input_ids'],
                attention_mask=batch_for_clean['attention_mask']
            )
        h_hook_handle.remove()
        
        if 'h_clean' not in h_clean_storage:
            raise RuntimeError(f"get_h_clean_hook failed at {attack_module_name}. 'h_clean' not found.")
            
        h_clean = h_clean_storage['h_clean']
        B, S, D = h_clean.shape
        incr_count("n_search_full_fwd_samples", B)
        
        # --- 3. 初始化 Delta ---
        delta = torch.zeros_like(h_clean, device=h_clean.device, dtype=h_clean.dtype).uniform_(-eps, eps)
        delta.requires_grad_(True)
    elif attack_mode == "pca_pgd":
        # (A) 加载 R_pca
        if pca_matrix_path is None:
            raise ValueError("`pca_pgd` 攻击需要 `pca_matrix_path`。")
        try:
            R_pca = torch.load(pca_matrix_path, map_location=reft_model.device)
            D, k = R_pca.shape
        except Exception as e:
            raise FileNotFoundError(f"无法加载 PCA 矩阵: {pca_matrix_path}. Error: {e}")
        
        # (B) 初始化 alpha [B, S, k]
        alpha = torch.zeros((B, S, k), device=reft_model.device, dtype=R_pca.dtype)
        alpha.uniform_(-eps, eps) # eps 现在约束 alpha
        alpha.requires_grad_(True)
    else:
        raise ValueError(f"未知的 attack_mode: {attack_mode}")
        
    # 准备 batch 用于 PGD 循环
    pgd_batch = batch.copy()
    if "labels" in pgd_batch:
        del pgd_batch["labels"] # loss 在循环外计算

    # --- 4. 攻击循环 (使用双钩子 + reft_model.model) ---
    for i in range(steps):
        
        # (A) 准备攻击钩子 (L1)
        if attack_mode == "pgd_fullspace" or attack_mode == "gcg":
            hook_fn_attack = functools.partial(dynamic_adv_hook, delta_high_dim=delta)
        else:
            hook_fn_attack = functools.partial(dynamic_adv_hook_pca, R_pca=R_pca.detach(), alpha=alpha)
        attack_handle = attack_module.register_forward_hook(hook_fn_attack)
        
        # (B) 准备防御钩子 (L6)
        hook_fn_defense = functools.partial(manual_reft_defense_hook, loreft_module=target_loreft_module)
        defense_handle = defense_module_hook_point.register_forward_hook(hook_fn_defense)
        
        # (C) *** 关键改动：调用 .model, 而不是 .forward() ***
        # 这将强制执行一个单一的、统一的、端到端的前向传播
        # 它会先触发 L1 钩子，然后触发 L6 钩子，保持计算图完整
        out = reft_model.model(**pgd_batch, return_dict=True)
        incr_count("n_search_full_fwd_samples", B)
        
        # (D) 立即移除钩子
        attack_handle.remove()
        defense_handle.remove()
        
        logits = out.logits
        loss = F.cross_entropy(logits, batch["labels"]) 
        
        if attack_mode == "pgd_fullspace" or attack_mode == "gcg":
            grad_variable = delta
        else:
            grad_variable = alpha
        
        grad = torch.autograd.grad(loss, grad_variable, only_inputs=True, retain_graph=True, allow_unused=False)[0]
        #pdb.set_trace()
        if grad is None:
            print("⚠️ grad is None (PGD 循环). 检查钩子逻辑。")
        # # --- 在这里加入你的“冒烟测试” ---
        # if grad is None:
        #     print(f"[VERIFY ERROR] PGD 步骤 {i}: grad 依然是 None! 停止。")
        #     # 提前退出，没必要继续了
        #     break 
        # else:
        #     # 这是一个好迹象，打印它
        #     print(f"[VERIFY OK] PGD 步骤 {i}: grad norm = {grad.norm().item():.4f}")
        # # --- 测试结束 ---
                 
        incr_count("n_search_full_bwd_samples", B)
        
        # (E) 更新 delta (逻辑不变)
        with torch.no_grad():
            if attack_mode == "pgd_fullspace":
                delta = (delta + lr * torch.sign(grad)).detach()
                delta = torch.clamp(delta, -eps, eps)
                delta.requires_grad_(True)
            elif attack_mode == "gcg":
                grad_norms = torch.norm(grad, p=2, dim=-1) # [B, S]
                grad_norms_flat = grad_norms.view(B, -1)
                topk_indices = torch.topk(grad_norms_flat, k=gcg_topk, dim=-1).indices # [B, k]
                
                update_sparse = torch.zeros_like(delta)
                batch_indices = torch.arange(B, device=delta.device)[:, None]
                token_indices = topk_indices
                update_sparse[batch_indices, token_indices, :] = grad[batch_indices, token_indices, :]
                
                delta = (delta + gcg_alpha * update_sparse).detach()
                delta = torch.clamp(delta, -eps, eps)
                delta.requires_grad_(True)
            elif attack_mode == "pca_pgd":
                alpha = (alpha + lr * torch.sign(grad)).detach()
                alpha = torch.clamp(alpha, -eps, eps)
                alpha.requires_grad_(True)
           
    # --- 5. 计算最终的 adv_loss ---
    if attack_mode == "pgd_fullspace" or attack_mode == "gcg":
        delta_final = delta.detach()
        hook_fn_attack = functools.partial(dynamic_adv_hook, delta_high_dim=delta_final)
    else: # pca_pgd
        alpha_final = alpha.detach()
        hook_fn_attack = functools.partial(dynamic_adv_hook_pca, R_pca=R_pca, alpha=alpha_final)
    
    attack_handle = attack_module.register_forward_hook(hook_fn_attack)
    
    hook_fn_defense = functools.partial(manual_reft_defense_hook, loreft_module=target_loreft_module)
    defense_handle = defense_module_hook_point.register_forward_hook(hook_fn_defense)
    
    # 再次调用 reft_model.model
    out = reft_model.model(**pgd_batch, return_dict=True)
    incr_count("n_search_full_fwd_samples", B)
    
    attack_handle.remove()
    defense_handle.remove()
    
    logits = out.logits
    adv_loss = F.cross_entropy(logits, batch["labels"])
    
    return adv_loss