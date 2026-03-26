# src/circuit/circuit_mask.py
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple

import torch


@dataclass
class CircuitSpec:
    """
    Holds a head-level circuit mask: shape [num_layers, num_heads]
    mask[l, h] = True means (layer=l, head=h) is in the circuit.
    """
    mask: torch.Tensor  # bool tensor [L, H]
    num_layers: int
    num_heads: int
    meta: Dict[str, Any]


def _empty_mask(num_layers: int, num_heads: int, device=None) -> torch.Tensor:
    return torch.zeros((num_layers, num_heads), dtype=torch.bool, device=device)


def load_circuit_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_mask_from_mapping(
    mapping: Dict[Union[str, int], List[int]],
    num_layers: int,
    num_heads: int,
    device=None,
) -> torch.Tensor:
    """
    mapping example:
      {"13": [0,2,3], "12":[6,13]}
    """
    m = _empty_mask(num_layers, num_heads, device=device)
    for k, heads in mapping.items():
        layer = int(k)
        if layer < 0 or layer >= num_layers:
            continue
        for h in heads:
            h = int(h)
            if 0 <= h < num_heads:
                m[layer, h] = True
    return m


def build_mask_from_list(
    items: List[Dict[str, Any]],
    num_layers: int,
    num_heads: int,
    top_k: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    """
    items example:
      [{"layer":13,"head":6,"score":...}, ...]
    if top_k is not None, take first top_k items (assumed sorted).
    """
    m = _empty_mask(num_layers, num_heads, device=device)
    if top_k is not None:
        items = items[:top_k]
    for it in items:
        layer = int(it["layer"])
        head = int(it["head"])
        if 0 <= layer < num_layers and 0 <= head < num_heads:
            m[layer, head] = True
    return m


def load_circuit_mask(json_path, num_layers, num_heads, device=None):
    """
    加载包含 'mask' 键的 Circuit JSON 文件。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取 mask 列表
    mask_list = data["mask"] if "mask" in data else data
    mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
    
    # 封装成 Spec 对象供 Trainer 使用
    class CircuitSpec:
        def __init__(self, mask_tensor):
            self.mask = mask_tensor
            
    return CircuitSpec(mask_tensor)

def mask_summary(mask: torch.Tensor) -> Tuple[int, List[Tuple[int, List[int]]]]:
    """
    Returns:
      total_heads_selected,
      list of (layer, heads_list) for layers that have any selected head
    """
    mask_cpu = mask.detach().cpu()
    total = int(mask_cpu.sum().item())
    per_layer = []
    for l in range(mask_cpu.shape[0]):
        hs = torch.where(mask_cpu[l])[0].tolist()
        if len(hs) > 0:
            per_layer.append((l, hs))
    return total, per_layer
