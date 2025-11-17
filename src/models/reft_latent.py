# 统一的模型构建和加载工具

import torch
from torch import dropout
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pyreft
from pyreft import ReftConfig, LoreftIntervention
from pyreft.reft_model import ReftModel
from typing import Tuple
from src.models.adv_intervention import AdversarialIntervention

def build_base_classifier(model_name: str, num_labels: int = 2) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """(A) 加载 HF 基础模型和 Tokenizer"""
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def build_reft_model(
    model_name: str,
    baseline_ckpt: str | None,
    reft_layer: int,
    rank_r: int,
    device: str,
    attack_layer: int | None = None,
    disable_model_grads: bool = True,
) -> Tuple[ReftModel, AutoTokenizer]:
    # HF 分类模型
    base_model, tokenizer = build_base_classifier(model_name, num_labels=2)
    
    # 载入 full FT baseline
    if baseline_ckpt is not None:
        print(f"[build_reft_model] 正在从 {baseline_ckpt} 加载 (B) 基线权重...")
        state = torch.load(baseline_ckpt, map_location="cpu")
        base_model.load_state_dict(state, strict=False)
    else:
        print("[build_reft_model] 未提供 baseline_ckpt，使用 (A) 原始权重。")

    base_model.to(device)
    model_dtype = next(base_model.parameters()).dtype
    hidden_size = base_model.config.hidden_size
    reft_comp_idx = reft_layer - 1
    attack_comp_idx = attack_layer - 1
    
    # ReFT config: 指定插入层和 LoReFT 
    reft_config = ReftConfig(
        representations=[
            # 1) LoReFT 本身
            {
                "layer": reft_layer,
                "component": f"gpt_neox.layers[{reft_comp_idx}].output",
                "low_rank_dimension": rank_r,
                "intervention": LoreftIntervention(
                    embed_dim=hidden_size,
                    low_rank_dimension=rank_r,
                    dtype=model_dtype,
                    dropout=0.0,
                    act_fn=None,
                ),
            },
            # 2) 新加的 adversarial 插件
            {
                "layer": attack_layer,
                "component": f"gpt_neox.layers[{attack_comp_idx}].output",
                # 这个不用 low_rank_dimension，只要能拿到 embed_dim 就行
                "intervention": AdversarialIntervention(
                    embed_dim=hidden_size,
                ),
            },
        ]
    )
    
    # 包成 ReftModel
    reft_model = pyreft.get_reft_model(
        base_model,
        reft_config,
        set_device=True,
        disable_model_grads=disable_model_grads,
    )
    reft_model.set_device(device)
    # 调试一下，看看两个 intervention 都挂上了
    print("[build_reft_model] interventions keys & types:")
    for k, v in reft_model.interventions.items():
        print("  ", k, "->", type(v))
    
    return reft_model, tokenizer

def load_reft_model_for_eval(
    model_name: str,
    baseline_ckpt: str,
    layer_idx: int,
    rank_r: int,
    adv_intervention_dir: str,
    device: str,
) -> Tuple[ReftModel, AutoTokenizer]:
    # evaluate的时候要
    print(f"[load_reft_model] 1. 正在构建框架")
    reft_model, tokenizer = build_reft_model(
        model_name=model_name,
        baseline_ckpt=baseline_ckpt,
        layer_idx=layer_idx,
        rank_r=rank_r,
        device=device,
        disable_model_grads=True,  # eval 阶段总是 True
    )
    
    # 载入训练好的权重
    print(f"[load_reft_model] 2. 从{adv_intervention_dir}加载训练好的Reft权重...")
    reft_model.load_intervention(adv_intervention_dir, include_model=False)    
    
    reft_model.eval()
    print(f"[load_reft_model] 评估模型加载成功")
    
    return reft_model, tokenizer