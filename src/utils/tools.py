from genericpath import exists
import torch
import random, numpy as np, json, os, datetime
import pyreft

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# tools
def find_loreft_for_layer(reft_model: torch.nn.Module, key):
    """
    key 可以是：
      - int: 在 ReftConfig.representations 里注册的 "layer" 标签（0-based或1-based，必须和注册一致）
      - str: component 路径子串（例如 "gpt_neox.layers[20].output"）
    返回：对应的 LoreftIntervention 模块
    """
    if not hasattr(reft_model, "representations"):
        raise ValueError("reft_model has no 'representations' attribute. Is this a PyReFT model?")

    # 遍历已注册的 representations
    for rep_name, rep in reft_model.representations.items():
        # 兼容字段来源：rep.layer / rep.config.layer
        rep_layer = getattr(rep, "layer", None)
        if rep_layer is None and hasattr(rep, "config"):
            rep_layer = getattr(rep.config, "layer", None)

        # 兼容字段来源：rep.component / rep.config.component
        rep_component = getattr(rep, "component", None)
        if rep_component is None and hasattr(rep, "config"):
            rep_component = getattr(rep.config, "component", None)

        # 匹配逻辑：key 为 int -> 比较 layer；key 为 str -> 比较 component 包含
        hit = False
        if isinstance(key, int) and rep_layer == key:
            hit = True
        elif isinstance(key, str) and rep_component and key in rep_component:
            hit = True

        if hit:
            # 拿到 intervention 实例：rep.intervention / rep.module.intervention
            interv = getattr(rep, "intervention", None)
            if interv is None and hasattr(rep, "module") and hasattr(rep.module, "intervention"):
                interv = rep.module.intervention
            if interv is not None:
                return interv

    # 兜底：打印已注册的条目信息，帮助你核对索引体系
    info = []
    for rep_name, rep in reft_model.representations.items():
        rl = getattr(rep, "layer", None)
        if rl is None and hasattr(rep, "config"):
            rl = getattr(rep.config, "layer", None)
        rc = getattr(rep, "component", None)
        if rc is None and hasattr(rep, "config"):
            rc = getattr(rep.config, "component", None)
        info.append(f"[{rep_name}] layer={rl}, component={rc}")
    raise ValueError(f"LoReFT module for key={key} not found. Registered reps:\n" + "\n".join(info))

def get_outputs_from_reft(out: dict):
    if isinstance(out, dict):
        for key in ("intervened_outputs", "counterfactual_outputs", "outputs", "model_outputs"):
            if key in out:
                return out[key]
        # 部分版本直接扁平化
        if "loss" in out and "logits" in out:
            return out
    return out

def set_phase_freeze(reft_model: torch.nn.Module, freeze_R: bool):
    for name, param in reft_model.named_parameters():
        if "rotate_layer" in name:
            param.requires_grad = not freeze_R
        elif "learned_source" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
def save_json_report(report: dict, out_dir: str, prefix: str = "report"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    fname = f"{prefix}_{ts}.json"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[save] report -> {out_path}")
    return out_path
            
            
# utils/hooks_inject.py
import contextlib, torch

@contextlib.contextmanager
def inject_hidden_at_layer(model, attack_layer_index, h_adv, mode="replace"):
    """
    临时在 gpt_neox.layers[attack_layer_index].output 注入隐藏状态。
    `h_adv` 形状为 [B, S, D]，设备/类型需正确。
    mode = "replace" 用 h_adv 替换；"add" 表示在原输出上相加。
    """
    block = model.gpt_neox.layers[attack_layer_index].output

    handle = None
    def _hook(_module, _inp, out):
        if mode == "replace":
            return h_adv
        elif mode == "add":
            return out + h_adv
        else:
            raise ValueError(f"Unknown mode: {mode}")

    handle = block.register_forward_hook(lambda m, i, o: _hook(m, i, o))
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()
