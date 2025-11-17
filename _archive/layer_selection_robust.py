# layer_selection_robust.py
# 目标：在对抗扰动后的样本上，逐层计算 Logistic / 核SVM / Sliced-Wasserstein / MLP 指标
# 输出结构与 clean 版本对齐，文件名添加 _adv_<mode> 后缀

import os, json, yaml, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data.adv_dataset import AdvDataset
from src.data.Collator import Collator

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

from scipy.stats import wasserstein_distance

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============== 你已提供的 token-level GCG ===============
# 假定在同目录或可导入路径下可用；若路径不同请自行调整 import
from src.attack.eval_spam_gcg import token_level_gcg_single  # ← 请把模块名改成实际存放位置
from src.utils.layer_selection import fit_logistic_probe, fit_kernel_svm, compute_sliced_wasserstein, fit_mlp_probe, balanced_subsample, minmax_scale

# =============== Embedding-level PGD (untargeted) ===============
@torch.no_grad()
def _pred_and_loss(model, ids, mask, label_t):
    out = model(input_ids=ids, attention_mask=mask)
    loss = F.cross_entropy(out.logits, label_t)
    return out.logits, loss

def embedding_pgd_single(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    true_label: int,
    device: torch.device,
    epsilon: float = 0.1,     # 约束半径（相对较小：按你的模型尺度可调）
    alpha: float = 0.1,       # 步长
    steps: int = 5,           # 内层步数
    norm: str = "linf"
):
    """
    对 embedding 输出做 PGD，上升交叉熵，约束在 epsilon 球。
    返回 adv_embeds (1,T,H)
    """
    model.eval()
    ids = input_ids.unsqueeze(0).to(device)
    mask = attention_mask.unsqueeze(0).to(device)
    label_t = torch.tensor([true_label], device=device)

    # 初始嵌入
    emb_layer = model.get_input_embeddings()
    embeds0 = emb_layer(ids).detach()  # [1,T,H]
    embeds = embeds0.clone().detach().requires_grad_(True)

    for _ in range(steps):
        out = model(inputs_embeds=embeds, attention_mask=mask)
        loss = F.cross_entropy(out.logits, label_t)
        grad = torch.autograd.grad(loss, embeds, only_inputs=True)[0]
        with torch.no_grad():
            if norm == "linf":
                embeds = embeds + alpha * torch.sign(grad)
                delta = torch.clamp(embeds - embeds0, min=-epsilon, max=epsilon)
                embeds = (embeds0 + delta).detach().requires_grad_(True)
            else:  # l2
                grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1).view(-1, 1, 1) + 1e-12
                step = alpha * grad / grad_norm
                embeds = embeds + step
                # 投影到 L2 球
                delta = embeds - embeds0
                delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).view(-1, 1, 1) + 1e-12
                factor = torch.clamp(epsilon / delta_norm, max=1.0)
                embeds = (embeds0 + delta * factor).detach().requires_grad_(True)
    return embeds.detach()  # [1,T,H]

# =============== 主流程 ===============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--attack_mode", choices=["token", "embedding"], default="token")
    parser.add_argument("--n_attack_tokens", type=int, default=5)
    parser.add_argument("--attack_start", type=int, default=0)
    parser.add_argument("--beam_k", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--forbid_special", action="store_true", default=True)
    parser.add_argument("--epsilon", type=float, default=0.5)  # embedding-PGD
    parser.add_argument("--alpha", type=float, default=0.1)    # embedding-PGD
    parser.add_argument("--steps", type=int, default=5)        # embedding-PGD
    parser.add_argument("--limit", type=int, default=0, help="仅用于调试，限制样本数；0=不限制")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["data"]["dataset"].replace("AlignmentResearch/","")
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"]["name"]
    ckpt_path = f"{cfg['out']['dir']}/best_{dataset}.pt"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    state = torch.load(ckpt_path, map_location=device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.load_state_dict(state, strict=True)
    if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    model.to(device).eval()

    # 验证/探针 split
    with open(cfg["data"]["split_file"], "r") as f:
        splits = json.load(f)
    ds = AdvDataset(splits["probe"], cfg["data"]["dataset"])
    collate = Collator(model_name=model_name, max_length=512)
    val_loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    # === 逐样本生成对抗输入，并收集每层 hidden ===
    all_hiddens_adv = {}  # layer -> tensor list
    all_labels = []
    n_seen = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].squeeze(0)      # [T]
        attention_mask = batch["attention_mask"].squeeze(0)  # [T]
        label = int(batch["labels"].item())
        all_labels.append(label)

        if args.attack_mode == "token":
            _, _, _, _, _, adv_ids = token_level_gcg_single(
                model, tokenizer,
                input_ids=input_ids, attention_mask=attention_mask,
                true_label=label, device=torch.device(device),
                n_attack_tokens=args.n_attack_tokens,
                attack_start=args.attack_start,
                beam_k=args.beam_k,
                rounds=args.rounds,
                forbid_special=args.forbid_special
            )
            # 用对抗后的 ids 前向，拿 hidden
            adv_ids_b = adv_ids.unsqueeze(0).to(device)
            mask_b = attention_mask.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_ids=adv_ids_b, attention_mask=mask_b, output_hidden_states=True, return_dict=True)
                hiddens = outputs.hidden_states

        else:  # embedding-level
            adv_embeds = embedding_pgd_single(
                model,
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                true_label=label,
                device=torch.device(device),
                epsilon=args.epsilon,
                alpha=args.alpha,
                steps=args.steps,
                norm="linf",
            )
            mask_b = attention_mask.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(inputs_embeds=adv_embeds, attention_mask=mask_b, output_hidden_states=True, return_dict=True)
                hiddens = outputs.hidden_states

        # 取最后非pad token 的句向量
        last_idx = attention_mask.sum() - 1
        idx = torch.arange(1, device=device)  # bsz=1
        for l, h in enumerate(hiddens):
            sent = h[idx-1, last_idx, :].detach().cpu()  # (1,H)
            all_hiddens_adv.setdefault(l, []).append(sent)

        n_seen += 1
        if args.limit and n_seen >= args.limit:
            break

    y = np.array(all_labels, dtype=np.int64).ravel()
    for l in all_hiddens_adv:
        all_hiddens_adv[l] = torch.cat(all_hiddens_adv[l], dim=0).numpy().astype(np.float32)

    latent_out_dir = os.path.join(cfg['out']['dir'], "latents")
    os.makedirs(latent_out_dir, exist_ok=True)

    # === 逐层评估 ===
    results = {}
    print(f"\n=== Dataset: {dataset} | Attack: {args.attack_mode} | Layers: {len(all_hiddens_adv)} ===")
    for l, X in all_hiddens_adv.items():
        print(f"\n[Layer {l:02d}] dim={X.shape[1]} n={X.shape[0]}")
        # 保存 adv latent
        np.savez_compressed(
            os.path.join(latent_out_dir, f"{dataset}_adv_{args.attack_mode}_L{l}.npz"),
            X=X, y=y
        )
        # 1) Logistic
        log_acc, log_auc, u, v = fit_logistic_probe(X, y, seed=42)
        # 2) 核 SVM
        svm_acc, svm_auc, sv_ratio, med_margin, svc = fit_kernel_svm(X, y, seed=42)
        np.savez_compressed(
            os.path.join(latent_out_dir, f"{dataset}_adv_{args.attack_mode}_L{l}_svm_margin.npz"),
            margin=svc.decision_function(X), y=y
        )
        # 3) Sliced-Wasserstein
        sw_mean, sw_std, sw_mean_sq = compute_sliced_wasserstein(X, y, n_proj=128, seed=42, max_per_class=5000)
        # 4) MLP probe
        mlp_acc, mlp_auc = fit_mlp_probe(X, y, seed=42, max_epochs=30, batch_size=256, lr=3e-3, hidden=128, dropout=0.1)

        results[l] = dict(
            log_acc=log_acc, log_auc=log_auc, u_norm=float(np.linalg.norm(u)), v=v,
            svm_acc=svm_acc, svm_auc=svm_auc, sv_ratio=sv_ratio, svm_med_margin=med_margin,
            sw_mean=sw_mean, sw_std=sw_std, sw_mean_sq=sw_mean_sq,
            mlp_acc=mlp_acc, mlp_auc=mlp_auc,
        )

        print(f"  Logistic  : acc={log_acc:.3f} | auc={log_auc:.3f} | ||u||={np.linalg.norm(u):.3f}")
        print(f"  Kernel SVM: acc={svm_acc:.3f} | auc={svm_auc:.3f} | sv_ratio={sv_ratio:.3f} | med|margin|={med_margin:.3f}")
        print(f"  Sliced-W  : mean(W1)={sw_mean:.4f} ± {sw_std:.4f} | mean(W1^2)={sw_mean_sq:.5f}")
        print(f"  MLP-probe : acc={mlp_acc:.3f} | auc={mlp_auc:.3f}")

    # 保存 JSON（结构与 clean 对齐，文件名体现 adv + mode）
    out_dir = cfg['out']['dir']
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"layer_tri_metrics_{dataset}_adv_{args.attack_mode}.json")
    with open(json_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\n>>> Saved: {json_path}")

if __name__ == "__main__":
    main()
    
#     # token-level GCG
# python -m src.utils.layer_selection_robust --config configs/baseline.yaml --attack_mode token --n_attack_tokens 10 --beam_k 20 --rounds 20 --forbid_special

# # embedding-level PGD
# python -m src.utils.layer_selection_robust --config configs/baseline.yaml --attack_mode embedding --epsilon 0.5 --alpha 0.1 --steps 20

