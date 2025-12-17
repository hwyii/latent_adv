"""
In latent space, do GCG attack or PGD attack
GCG may need less steps to achieve similary performance with PGD
"""

from regex import B
from pyvene import SourcelessIntervention, TrainableIntervention  
import torch
import pdb
import torch.nn.functional as F
from typing import Dict
from src.utils.tools import find_loreft_for_layer, get_outputs_from_reft, inject_hidden_at_layer
from src.utils.compute import incr_count
import pyvene as pv
from dataclasses import dataclass 
from typing import Optional  

def unwrap_base_model(m):
    """
    输入：Trainer 里传进来的 model
    输出：真正的 HF base model 里面有 gpt_neox.layers[...]
    """
    # ReftModel / IntervenableModel: 真正的 HF 模型在 .model 里
    if isinstance(m, pv.IntervenableModel):
        return m.model
    else:
        return m
@dataclass
class AttackConfig:
    """
    配置对抗性攻击的DataClass
    """
    inner_attack: str = "gcg"
    attack_layer: Optional[int] = None
    reft_layer: Optional[int] = None
    eps: float = None               # GCG 和 PGD 共享
    steps: int = None                # GCG 和 PGD 共享
    lr: Optional[float] = None      # PGD 用
    gcg_topk: int = None
    gcg_alpha: Optional[float] = None
    lambda_adv: float = None

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

# class AdversarialIntervention(SourcelessIntervention, TrainableIntervention):  
#     def __init__(self, delta, **kwargs):  
#         super().__init__(**kwargs)  
#         self.delta = delta  
      
#     def forward(self, base, source=None, subspaces=None):  
#         print("[AdvIntervention] base.shape =", base.shape)
#         if self.delta is not None:
#             print("[AdvIntervention] delta.shape =", self.delta.shape)

#         # 直接在 base 上加 delta  
#         return base + self.delta  

def compute_adv_loss(reft_model, inputs, attack_config):  
    """
    使用 PGD 在 latent space 计算对抗性损失 (base_model 直接前向传播)
    1) 在指定层注册 hook，添加扰动 delta
    2) 多步 PGD 优化 delta
    3) 返回最终的 adv_loss
    注意这与reft_model无关，直接调用base_model
    """
    base_model = unwrap_base_model(reft_model)  
    B = inputs["input_ids"].size(0)  
    S = inputs["input_ids"].size(1)  
    device = inputs["input_ids"].device  
    hidden_dim = base_model.config.hidden_size  
    dtype = next(base_model.parameters()).dtype  
      
    # 初始化 delta  
    num_pos = 1
    delta = torch.zeros(B, num_pos, hidden_dim, device=device, dtype=dtype)  
    print("[PGD] 初始化 delta，形状为:", delta.shape)
    delta.uniform_(-attack_config.eps, attack_config.eps)  
    delta.requires_grad_(True)  
      
    # 注册 hook 到基础模型  
    attack_layer_idx = attack_config.attack_layer or attack_config.reft_layer  
    attack_module_name = find_target_component_name(reft_model, attack_layer_idx)  
    attack_module = get_module_by_name(base_model, attack_module_name)  
      
    def attack_hook(module, inp, output):  
        if isinstance(output, (tuple, list)):  
            h = output[0]  
            h_perturbed = h + delta  
            return (h_perturbed,) + tuple(output[1:])  
        else:  
            return output + delta  
      
    handle = attack_module.register_forward_hook(attack_hook)  
      
    # PGD 循环（直接调用基础模型）  
    step_size = attack_config.lr or (attack_config.eps / max(1, attack_config.steps))  
    labels = inputs["labels"].view(-1) if inputs["labels"].dim() > 1 else inputs["labels"]  
      
    # 移除 pyreft 特有的字段  
    clean_inputs = {  
        'input_ids': inputs['input_ids'],  
        'attention_mask': inputs['attention_mask'],  
        'labels': labels  
    }  
      
    for t in range(attack_config.steps):  
        delta.requires_grad_(True)  
        base_model.zero_grad()  
        if delta.grad is not None:  
            delta.grad.zero_()  
          
        # 直接调用基础模型  
        outputs = base_model(**clean_inputs)  
        loss_step = outputs.loss  
          
        print(f"[PGD] step {t} loss_step = {loss_step.item()}")   
        print(f"[PGD] step {t}, delta.shape = {delta.shape}, delta.norm = {delta.norm().item():.4f}")
 
          
        loss_step.backward()  
          
        if delta.grad is None:  
            print("[PGD][ERROR] delta.grad is None!")  
            handle.remove()  
            return torch.tensor(0.0, device=device)  
          
        # PGD 更新  
        with torch.no_grad():  
            delta.add_(step_size * torch.sign(delta.grad))  
            delta.clamp_(-attack_config.eps, attack_config.eps)  
      
    # 计算最终 adv_loss  
    base_model.zero_grad()  
    outputs = base_model(**clean_inputs)  
    adv_loss = outputs.loss  
      
    handle.remove()  
    return adv_loss

