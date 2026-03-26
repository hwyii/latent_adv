from genericpath import exists
import torch
import random, numpy as np, json, os, datetime
import pyreft
from typing import Optional, Dict, Any, Tuple
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss


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
            
            
def _cosine(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    u = u.float().view(-1)
    v = v.float().view(-1)
    return float(torch.dot(u, v) / (u.norm() * v.norm() + eps))


def _proj_scalar(h: torch.Tensor, r: torch.Tensor, eps: float = 1e-12) -> float:
    # scalar projection: <h, r_hat>
    h = h.float().view(-1)
    r = r.float().view(-1)
    r_hat = r / (r.norm() + eps)
    return float(torch.dot(h, r_hat))


def _safe_mean(x: torch.Tensor) -> float:
    return float(x.detach().float().mean().item())


def _safe_norm(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().item())


def load_r_attack_any(
    path: str,
    *,
    layer: Optional[int] = None,
    key: Optional[str] = None,
) -> torch.Tensor:
    """
    Load r_attack vector from:
      - .pt: expects tensor or dict containing tensor
      - .npy: numpy array
      - .json/.jsonl: expects dict with keys like "layer_16" or custom `key`
                      and inside contains either:
                        {"r_attack": [...]} or {"r_attack_npy": "..."} etc.
    Returns: torch.FloatTensor [H]
    """
    assert path is not None
    if not os.path.exists(path):
        raise FileNotFoundError(f"r_attack path not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.float().view(-1)
        if isinstance(obj, dict):
            # try typical keys
            for k in ["r_attack", "vector", "v", "data"]:
                if k in obj and isinstance(obj[k], torch.Tensor):
                    return obj[k].float().view(-1)
            # maybe stored by layer key
            if layer is not None:
                lk = f"layer_{layer}"
                if lk in obj and isinstance(obj[lk], torch.Tensor):
                    return obj[lk].float().view(-1)
        raise ValueError(f"Unrecognized .pt format for r_attack: {path}")

    if ext == ".npy":
        import numpy as np
        arr = np.load(path)
        return torch.tensor(arr, dtype=torch.float32).view(-1)

    if ext in [".json", ".jsonl"]:
        def _read_json_or_jsonl(p: str) -> Any:
            if p.endswith(".json"):
                with open(p, "r") as f:
                    return json.load(f)
            # jsonl -> list of dict
            rows = []
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows

        obj = _read_json_or_jsonl(path)

        # If jsonl list, allow key lookup across rows
        if isinstance(obj, list):
            # try to find first row containing the key
            if key is not None:
                for row in obj:
                    if key in row:
                        obj = row[key]
                        break
            elif layer is not None:
                lk = f"layer_{layer}"
                for row in obj:
                    if lk in row:
                        obj = row[lk]
                        break

        # Now obj should be dict
        if isinstance(obj, dict):
            # Resolve layer selection
            if key is not None and key in obj:
                obj = obj[key]
            elif layer is not None:
                lk = f"layer_{layer}"
                if lk in obj:
                    obj = obj[lk]

            # common patterns
            if isinstance(obj, dict):
                if "r_attack" in obj and isinstance(obj["r_attack"], (list, tuple)):
                    return torch.tensor(obj["r_attack"], dtype=torch.float32).view(-1)
                if "r_attack_npy" in obj and isinstance(obj["r_attack_npy"], str):
                    return load_r_attack_any(obj["r_attack_npy"])
                if "vector" in obj and isinstance(obj["vector"], (list, tuple)):
                    return torch.tensor(obj["vector"], dtype=torch.float32).view(-1)

            # maybe the dict itself is list
            if isinstance(obj, (list, tuple)):
                return torch.tensor(obj, dtype=torch.float32).view(-1)

        raise ValueError(f"Unrecognized json/jsonl format for r_attack: {path}")

    raise ValueError(f"Unsupported r_attack file extension: {ext}")


def _ensure_labels_1d(inputs: Dict[str, torch.Tensor]) -> None:
    if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor) and inputs["labels"].dim() == 2:
        inputs["labels"] = inputs["labels"].squeeze(1)


def _infer_problem_type(model, labels: torch.Tensor) -> str:
    cfg = model.model.config
    if cfg.problem_type is not None:
        return cfg.problem_type
    if model.model.num_labels == 1:
        return "regression"
    if model.model.num_labels > 1 and (labels.dtype in (torch.long, torch.int64, torch.int32)):
        return "single_label_classification"
    return "multi_label_classification"


def _compute_clean_loss(model, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pt = _infer_problem_type(model, labels)
    if pt == "regression":
        loss_fct = MSELoss()
        if model.model.num_labels == 1:
            return loss_fct(logits.squeeze(), labels.squeeze().to(torch.bfloat16))
        return loss_fct(logits, labels.to(torch.bfloat16))
    if pt == "single_label_classification":
        return CrossEntropyLoss()(logits.view(-1, model.model.num_labels), labels.view(-1))
    return BCEWithLogitsLoss()(logits, labels)
