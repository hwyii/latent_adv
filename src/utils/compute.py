# ---------- FLOPs 计数器（放 trainer 文件顶部或 src/utils/instrument.py） ----------
GLOBAL_COUNTS = {
    # training-level full forward/back
    "n_train_full_fwd_samples": 0,
    "n_train_full_bwd_samples": 0,
    # training-level submodule (target+head) forward/back
    "n_train_sub_fwd_samples": 0,
    "n_train_sub_bwd_samples": 0,

    # search-level (attack) full forward/back (if attack recomputes full forward)
    "n_search_full_fwd_samples": 0,
    "n_search_full_bwd_samples": 0,
    # search-level submodule forward/back (target/ head & autograd.grad)
    "n_search_sub_fwd_samples": 0,
    "n_search_sub_bwd_samples": 0,
    "n_search_candidate_fwd_samples": 0,  # e.g., token-level candidate evals

    # auxiliary counts (probe / deviation / periodic validation etc.)
    "n_aux_full_fwd_samples": 0,
    "n_aux_sub_fwd_samples": 0,
    "n_aux_sub_bwd_samples": 0,

    # bookkeeping
    "steps_recorded": 0,
}

def reset_counts():
    for k in GLOBAL_COUNTS:
        GLOBAL_COUNTS[k] = 0

def incr_count(name: str, batch_size: int = 1):
    if name not in GLOBAL_COUNTS:
        raise KeyError(name)
    GLOBAL_COUNTS[name] += int(batch_size)

def get_counts():
    return dict(GLOBAL_COUNTS)

# compute FLOPs conversion (param-ratio approx for submodule)
def compute_flops_from_counts(
    counts: dict,
    F_fwd_full: float,
    F_fwd_sub: float = None,
    r_ratio: float = None,   # optional: if you prefer compute sub forward from ratio
    pretraining_N: float = None,
    pretraining_D: float = None,
    attack_layer = None,
    n_total_layers=None
):
    """
    counts: GLOBAL_COUNTS-like dict (sample-level counts)
    F_fwd_full: per-sequence full forward FLOPs (paper unit)
    F_fwd_sub: per-sequence submodule forward FLOPs (if None, will use r_ratio)
    r_ratio: ratio N_sub / N (used if F_fwd_sub is None)
    pretraining_N, pretraining_D: N and D for denominator 6*N*D (optional)
    Returns: stats dict with Ctrain, Csearch, Caux, Cadv_total, plus breakdowns
    """
    c = {k: int(counts.get(k, 0)) for k in counts.keys()}
    steps = c.get("steps_recorded", 0)

    # derive sub forward if needed
    if F_fwd_sub is None:
        if r_ratio is None:
            raise ValueError("Either F_fwd_sub or r_ratio must be provided")
        F_fwd_sub = float(F_fwd_full) * float(r_ratio)

    # backward FLOPs approximations (paper-style)
    F_bwd_full = 2.0 * float(F_fwd_full)
    F_bwd_sub = 2.0 * float(F_fwd_sub)

    # Ctrain: full forward/back + train sub forward/back
    Ctrain = (
        c.get("n_train_full_fwd_samples", 0) * F_fwd_full
        + c.get("n_train_full_bwd_samples", 0) * F_bwd_full
        + c.get("n_train_sub_fwd_samples", 0) * F_fwd_sub
        + c.get("n_train_sub_bwd_samples", 0) * F_bwd_sub
    )

    # 1. Csearch_fwd (不变, 搜索时的前向传播总是完整的)
    Csearch_fwd = c.get("n_search_full_fwd_samples", 0) * F_fwd_full
    
    # 2. Csearch_bwd (使用新公式)
    if attack_layer is not None and n_total_layers is not None and n_total_layers > 0:
        # 使用我们的新“单价”
        bwd_cost_fraction = (n_total_layers - attack_layer + 1) / float(n_total_layers)
        F_bwd_partial = F_bwd_full * bwd_cost_fraction
    else:
        # 降级：如果信息不全，就使用旧的“错误”方法
        F_bwd_partial = F_bwd_full
        
    Csearch_bwd = c.get("n_search_full_bwd_samples",0) * F_bwd_partial
    
    Csearch = Csearch_fwd + Csearch_bwd

    # Caux: auxiliary forward/back (probe, deviation, periodic val)
    Caux = (
        c.get("n_aux_full_fwd_samples", 0) * F_fwd_full
        + c.get("n_aux_sub_fwd_samples", 0) * F_fwd_sub
        + c.get("n_aux_sub_bwd_samples", 0) * F_bwd_sub
    )

    Cadv_total = Ctrain + Csearch + Caux

    stats = {
        "counts": c,
        "F_fwd_full": float(F_fwd_full),
        "F_bwd_full": float(F_bwd_full),
        "F_fwd_sub": float(F_fwd_sub),
        "F_bwd_sub": float(F_bwd_sub),
        "Ctrain": float(Ctrain),
        "Csearch": float(Csearch),
        "Caux": float(Caux),
        "Cadv_total": float(Cadv_total),
        "steps_recorded": steps,
    }

    # optional normalization to pretraining FLOPs
    if pretraining_N is not None and pretraining_D is not None:
        total_pretrain = 6.0 * float(pretraining_N) * float(pretraining_D)
        stats["total_pretrain_flops"] = total_pretrain
        stats["proportion_of_pretraining"] = float(Cadv_total) / total_pretrain

    return stats

# put into src/utils/compute.py (or tools.py)
import torch
import math
from typing import Optional, Tuple

def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p is not None)

def summarize_model_and_submodule_flops(
    model: torch.nn.Module,
    target_module: Optional[torch.nn.Module],
    head_module: Optional[torch.nn.Module],
    seq_len: int,
    pretrain_tokens_D: float = 3e11,
    use_paper_unit: bool = True,
) -> dict:
    """
    Return dict with N, N_sub, ratio, F_fwd_full, F_bwd_full, F_fwd_sub, F_bwd_sub,
    and normalization denominator (6*N*D).
    - model: full HF model (AutoModelForSequenceClassification)
    - target_module: LoReFT module (or None)
    - head_module: classifier/head module (or None)
    - seq_len: L (max tokens per sequence) used for per-seq FLOP estimate
    - pretrain_tokens_D: D used in normalization (paper often uses 3e11)
    """
    # total params
    N = float(sum(p.numel() for p in model.parameters()))
    # submodule params: combine target and head if available, otherwise try to find rotate_layer
    N_sub = 0.0
    if target_module is not None:
        N_sub += float(count_parameters(target_module))
    if head_module is not None:
        N_sub += float(count_parameters(head_module))
    # fallback: small positive if not found
    if N_sub == 0.0:
        # put a small guard to avoid div by zero
        N_sub = max(1.0, 1e-6 * N)

    r_ratio = float(N_sub / max(1.0, N))

    # paper unit approximations
    # per-seq forward (paper uses approx 2*N*L)
    F_fwd_full = 2.0 * N * float(seq_len)
    F_bwd_full = 2.0 * F_fwd_full  # approx
    # submodule forward/back
    F_fwd_sub = F_fwd_full * r_ratio
    F_bwd_sub = 2.0 * F_fwd_sub

    stats = {
        "N_params": float(N),
        "N_sub_params": float(N_sub),
        "sub_to_full_param_ratio": float(r_ratio),
        "seq_len": int(seq_len),
        "F_fwd_full": float(F_fwd_full),
        "F_bwd_full": float(F_bwd_full),
        "F_fwd_sub": float(F_fwd_sub),
        "F_bwd_sub": float(F_bwd_sub),
        "pretrain_tokens_D": float(pretrain_tokens_D),
        "pretrain_denominator_6ND": float(6.0 * N * pretrain_tokens_D),
    }
    return stats
