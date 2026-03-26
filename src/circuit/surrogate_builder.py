# src/circuit/surrogate_builder.py
import torch
import torch.nn as nn
import copy
import random
from transformers.pytorch_utils import prune_conv1d_layer
from transformers.utils import logging
logger = logging.get_logger(__name__)

class DummyAttention(nn.Module):
    """当一层的所有 Head 都被砍掉时，用来占位并完美伪装接口的空壳层"""
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        
    def forward(
        self, 
        hidden_states, 
        layer_past=None, 
        attention_mask=None, 
        head_mask=None, 
        encoder_hidden_states=None, 
        encoder_attention_mask=None, 
        use_cache=False, 
        output_attentions=False,
        **kwargs
    ):
        bz, seq_len, _ = hidden_states.shape
        # 1. 伪造计算输出 (全 0)
        zero_output = torch.zeros(
            bz, seq_len, self.embed_dim, 
            device=hidden_states.device, 
            dtype=hidden_states.dtype
        )
        
        # 2. 伪造 KV Cache (如果不给，HF 外层大循环就会崩溃)
        present = None
        if use_cache:
            # 伪造一个 num_heads = 0 的合法 Cache Tensor
            empty_kv = torch.zeros(
                bz, 0, seq_len, self.head_dim, 
                device=hidden_states.device, 
                dtype=hidden_states.dtype
            )
            present = (empty_kv, empty_kv)
            
        outputs = (zero_output, present)
        
        # 3. 伪造 Attention Weights (应对 output_attentions=True)
        if output_attentions:
            outputs += (None,)
            
        return outputs

# def prune_gpt2_mlp(mlp_module, keep_ratio: float):
#     """
#     按比例随机保留 GPT-2 MLP 的神经元 (中间维度)。
#     GPT-2 的 MLP 包含两个 Conv1D 层：c_fc 和 c_proj。
#     默认全量维度 d_ff = 3072。
#     """
#     if keep_ratio >= 1.0:
#         return mlp_module
    
#     # 1. 获取当前 MLP 的中间维度大小 (即 c_fc 的输出维度 / nf)
#     current_ff_dim = mlp_module.c_fc.nf 
#     keep_dim = max(1, int(current_ff_dim * keep_ratio))
    
#     # 2. 随机生成要保留的神经元索引
#     all_indices = list(range(current_ff_dim))
#     keep_indices = random.sample(all_indices, keep_dim)
#     keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.long)
    
#     # 3. 物理切片：c_fc 削减输出维度 (dim=1)
#     # c_fc.weight 形状是 [d_model, d_ff]
#     mlp_module.c_fc = prune_conv1d_layer(mlp_module.c_fc, keep_indices_tensor, dim=1)
#     mlp_module.c_fc.nf = keep_dim # 更新内部超参数
    
#     # 4. 物理切片：c_proj 削减输入维度 (dim=0)
#     # c_proj.weight 形状是 [d_ff, d_model]
#     mlp_module.c_proj = prune_conv1d_layer(mlp_module.c_proj, keep_indices_tensor, dim=0)
    
#     return mlp_module



# def build_surrogate_model(base_model, inactive_heads_dict, mlp_keep_ratio: float = 1.0):
#     # 1. 复制一个完全独立的模型
#     surrogate = copy.deepcopy(base_model)
#     orig_embed_dim = surrogate.config.n_embd
    
#     for layer_idx, layer in enumerate(surrogate.transformer.h):
#         # 2. 剪枝 Attention Heads
#         heads_to_prune = inactive_heads_dict.get(layer_idx, [])
        
#         if heads_to_prune:
#             # 判断是不是这层的 Head 绝户了？
#             if len(heads_to_prune) == layer.attn.num_heads:
#                 # 换上配置了全套“假证件”的空壳层
#                 layer.attn = DummyAttention(
#                     embed_dim=orig_embed_dim, 
#                     head_dim=layer.attn.head_dim
#                 ).to(base_model.device)
#             else:
#                 layer.attn.prune_heads(heads_to_prune)
#                 layer.attn.embed_dim = layer.attn.num_heads * layer.attn.head_dim
            
#         # 3. 剪枝 MLP
#         if mlp_keep_ratio < 1.0:
#             # 假设你之前已经有了 prune_gpt2_mlp 函数
#             layer.mlp = prune_gpt2_mlp(layer.mlp, mlp_keep_ratio)
            
#     # 强制把整个替身刷回和基座一样的精度
#     surrogate = surrogate.to(base_model.dtype)
    
#     return surrogate

def prune_gpt2_mlp_with_indices(mlp_module, prune_indices):
    """根据 JSON 里提供的索引，对 MLP 进行精确的物理切除 (使用 HF 原生接口)"""
    if not prune_indices:  # 如果列表为空，说明这层 100% 保留
        return mlp_module
        
    # 1. 获取当前 MLP 的中间维度大小
    inner_dim = mlp_module.c_fc.weight.shape[1]
    
    # 2. 计算需要保留的索引
    all_indices = set(range(inner_dim))
    keep_indices = sorted(list(all_indices - set(prune_indices)))
    keep_dim = len(keep_indices)
    
    if keep_dim == inner_dim:
        return mlp_module
        
    # 3. 转换为 PyTorch Tensor (注意要放到对应的 device 上)
    device = mlp_module.c_fc.weight.device
    keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.long, device=device)
    
    # 4. 物理切片：c_fc 削减输出维度 (dim=1)
    mlp_module.c_fc = prune_conv1d_layer(mlp_module.c_fc, keep_indices_tensor, dim=1)
    mlp_module.c_fc.nf = keep_dim  # 必须更新 GPT-2 特有的内部超参数
    
    # 5. 物理切片：c_proj 削减输入维度 (dim=0)
    mlp_module.c_proj = prune_conv1d_layer(mlp_module.c_proj, keep_indices_tensor, dim=0)
    
    return mlp_module


def build_surrogate_model(base_model, inactive_heads_dict, mlp_keep_ratio: float = 1.0, mlp_mask_dict: dict = None):
    surrogate = copy.deepcopy(base_model)

    def clear_all_hooks(module):
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
        for child in module.children():
            clear_all_hooks(child)

    clear_all_hooks(surrogate)

    orig_embed_dim = surrogate.config.n_embd
    total_orig_mlp = 0
    total_kept_mlp = 0

    print("\n" + "=" * 60)
    print(f"🛠️  Surrogate Builder Runtime Profiling (MLP Keep Ratio Target: {mlp_keep_ratio})")
    print("=" * 60)

    for layer_idx, layer in enumerate(surrogate.transformer.h):
        orig_heads = layer.attn.num_heads
        orig_mlp_dim = layer.mlp.c_fc.weight.shape[1]

        total_orig_mlp += orig_mlp_dim

        # 1. prune heads
        heads_to_prune = inactive_heads_dict.get(layer_idx, [])
        if heads_to_prune:
            if len(heads_to_prune) == layer.attn.num_heads:
                layer.attn = DummyAttention(orig_embed_dim, layer.attn.head_dim).to(base_model.device)
            else:
                layer.attn.prune_heads(heads_to_prune)
                layer.attn.embed_dim = layer.attn.num_heads * layer.attn.head_dim

        # 2. prune mlp
        prune_indices = []
        if mlp_keep_ratio < 1.0 and mlp_mask_dict is not None:
            prune_indices = mlp_mask_dict.get(str(layer_idx), [])
            prune_indices = sorted(set(int(i) for i in prune_indices))

            bad = [i for i in prune_indices if i < 0 or i >= orig_mlp_dim]
            if bad:
                raise ValueError(
                    f"[MLP MASK] layer={layer_idx} has invalid indices, "
                    f"first_bad={bad[:10]}, orig_mlp_dim={orig_mlp_dim}"
                )

            expected_keep_dim = orig_mlp_dim - len(prune_indices)
            if expected_keep_dim <= 0:
                raise ValueError(
                    f"[MLP MASK] layer={layer_idx} would prune all MLP neurons "
                    f"(orig={orig_mlp_dim}, prune={len(prune_indices)})."
                )

            logger.info(
                f"[MLP MASK] layer={layer_idx} | prune={len(prune_indices)} "
                f"| keep={expected_keep_dim}/{orig_mlp_dim} ({expected_keep_dim / orig_mlp_dim:.4f}) "
                f"| first10={prune_indices[:10]}"
            )

            layer.mlp = prune_gpt2_mlp_with_indices(layer.mlp, prune_indices)

        # runtime checks
        if isinstance(layer.attn, DummyAttention):
            current_heads = 0
        else:
            current_heads = layer.attn.num_heads

        current_mlp_dim = layer.mlp.c_fc.weight.shape[1]
        expected_mlp_dim = orig_mlp_dim - len(prune_indices)
        assert current_mlp_dim == expected_mlp_dim, (
            f"Layer {layer_idx} MLP cut failed! Expected {expected_mlp_dim}, got {current_mlp_dim}"
        )

        total_kept_mlp += current_mlp_dim

        print(
            f"Layer {layer_idx:02d} | Heads: {orig_heads} -> {current_heads} | "
            f"MLP Dim: {orig_mlp_dim} -> {current_mlp_dim} | "
            f"Pruned: {orig_mlp_dim - current_mlp_dim} | "
            f"Keep Ratio: {current_mlp_dim / orig_mlp_dim:.4f}"
        )

    surrogate = surrogate.to(base_model.dtype)

    global_keep_ratio = total_kept_mlp / total_orig_mlp
    print("=" * 60)
    print(
        f"[GLOBAL MLP] kept {total_kept_mlp}/{total_orig_mlp} "
        f"({global_keep_ratio:.4f}), target={mlp_keep_ratio:.4f}"
    )
    print("=" * 60 + "\n")

    logger.info(
        f"[MLP MASK][GLOBAL] kept {total_kept_mlp}/{total_orig_mlp} "
        f"({global_keep_ratio:.4f}), target={mlp_keep_ratio:.4f}"
    )

    return surrogate