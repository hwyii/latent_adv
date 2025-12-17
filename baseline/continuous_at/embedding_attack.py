# baseline/continuous_at/embedding_attack.py
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class AttackConfig:
    eps: float = 0.5              # L2 radius (or Linf if norm="linf")
    alpha: float = 0.1            # step size
    steps: int = 3                # PGD steps (1 -> FGSM-style)
    norm: str = "l2"              # "l2" or "linf"
    random_init: bool = True      # start from random perturbation in ball


def _project(delta: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    if norm == "linf":
        return delta.clamp(min=-eps, max=eps)
    elif norm == "l2":
        # per-token L2 (same shape as delta: [B,T,H]) -> project each token vector
        flat = delta.view(-1, delta.size(-1))
        n = flat.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        factor = (eps / n).clamp_max(1.0)
        flat = flat * factor
        return flat.view_as(delta)
    else:
        raise ValueError(f"Unknown norm: {norm}")


@torch.no_grad()
def _rand_init(shape, eps: float, norm: str, device, dtype):
    if norm == "linf":
        return (2 * torch.rand(shape, device=device, dtype=dtype) - 1.0) * eps
    elif norm == "l2":
        # random direction then scale to <= eps per token
        delta = torch.randn(shape, device=device, dtype=dtype)
        delta = _project(delta, eps, "l2")
        # random radius in [0,1]
        r = torch.rand(delta.size(0), delta.size(1), 1, device=device, dtype=dtype)
        return delta * r
    else:
        raise ValueError(f"Unknown norm: {norm}")


def pgd_attack_embeddings(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    attack: AttackConfig,
):
    """
    Returns:
      adv_embeds: adversarial input embeddings, shape [B,T,H]
    """
    model.eval()

    emb_layer = model.get_input_embeddings()
    clean_embeds = emb_layer(input_ids)  # [B,T,H]
    B, T, H = clean_embeds.shape
    device, dtype = clean_embeds.device, clean_embeds.dtype

    if attack.random_init:
        delta = _rand_init((B, T, H), attack.eps, attack.norm, device, dtype)
    else:
        delta = torch.zeros((B, T, H), device=device, dtype=dtype)

    delta.requires_grad_(True)

    for _ in range(attack.steps):
        adv_embeds = clean_embeds + delta

        out = model(inputs_embeds=adv_embeds, attention_mask=attention_mask, labels=labels)
        loss = out.loss if hasattr(out, "loss") and out.loss is not None else F.cross_entropy(out.logits, labels)

        # maximize loss
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]

        with torch.no_grad():
            if attack.norm == "linf":
                delta = delta + attack.alpha * grad.sign()
            elif attack.norm == "l2":
                g = grad.view(-1, H)
                g_norm = g.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
                g = (g / g_norm).view(B, T, H)
                delta = delta + attack.alpha * g
            else:
                raise ValueError(f"Unknown norm: {attack.norm}")

            delta = _project(delta, attack.eps, attack.norm)
            delta.requires_grad_(True)

    return (clean_embeds + delta).detach()
