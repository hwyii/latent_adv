#!/usr/bin/env python3
""" Trainer for ReFT with latent adversarial training (GCG on LoReFT) """

from re import A
from sympy import comp
import datetime, json, os, math, argparse, time, numpy as np, yaml, csv
from tqdm.auto import tqdm
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.linear_model import LogisticRegression
import pyreft
from pyreft import ReftConfig
from zmq import DEALER, device

import pdb

# utils
from src.utils.tools import set_seed, get_outputs_from_reft, set_phase_freeze, save_json_report, find_loreft_for_layer
from src.utils.compute import incr_count, GLOBAL_COUNTS, reset_counts, compute_flops_from_counts, summarize_model_and_submodule_flops
# dataset
from src.data.adv_dataset import AdvDataset
from src.data.Collator import Collator
# attack inner
from src.attack.inner_attack import compute_adv_loss_via_hooks

# helper functions


@torch.no_grad()
def calculate_u_new(
    reft_model: torch.nn.Module,
    val_loader: DataLoader, # 使用验证集来计算u，避免过拟合
    layer_idx: int,
    device: str,
    max_batches: int = 2
) -> np.ndarray:
    reft_model.eval() # 切换到评估模式
    
    all_h_last = []
    all_labels = []

    for i, batch in enumerate(val_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        out = reft_model.model(**batch, output_hidden_states=True, return_dict=True)
        incr_count("n_aux_full_fwd_samples", int(batch["input_ids"].size(0)))
        h = out.hidden_states[layer_idx] # [B, S, D]
        
        last_pos = batch["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(h.size(0), device=h.device)
        h_last = h[idx, last_pos, :].cpu().numpy() # [B, D]
        
        all_h_last.append(h_last)
        all_labels.append(batch["labels"].cpu().numpy())
        
        # 通常用一两个batch的数据就足够估计u了，不需要跑完整个验证集
        if i + 1 >= max_batches: 
            break
            
    X = np.concatenate(all_h_last, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # 2. 训练一个线性探针
    probe = LogisticRegression(class_weight="balanced", C=0.1, solver="liblinear")
    probe.fit(X, y)
    
    # 3. 探针的系数就是新的判别方向 u
    u_new = probe.coef_.flatten().astype(np.float32)
    
    reft_model.train() # 恢复到训练模式
    return u_new

@torch.no_grad()
def deviation_phi_from_u(
    reft_model: torch.nn.Module,
    val_loader: DataLoader, # 需要一批数据来计算实际的修正量
    layer_idx: int,
    u_new: np.ndarray,
    device: str
) -> float:
    """
    计算 ReFT 模块的实际作用方向 Φ(h)-h 与判别方向 u 的偏移。
    """
    reft_model.eval()
    
    # 1. 获取一批验证数据
    batch = next(iter(val_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # 2. 计算 h 和 Φ(h)
    out_hf = reft_model.model(**batch, output_hidden_states=True, return_dict=True)
    incr_count("n_aux_full_fwd_samples", batch["input_ids"].size(0))
    h = out_hf.hidden_states[layer_idx] # [B, S, D]
    
    target_loreft = find_loreft_for_layer(reft_model, layer_idx)
    phi_h = target_loreft(h) # [B, S, D]
    incr_count("n_aux_sub_fwd_samples", h.size(0))
    # 3. 计算实际修正量 delta_h = Φ(h) - h
    delta_h = phi_h - h
    
    # 只关心 last token 的修正方向
    last_pos = batch["attention_mask"].sum(dim=1) - 1
    idx = torch.arange(h.size(0), device=h.device)
    delta_h_last = delta_h[idx, last_pos, :] # [B, D]
    
    # 4. 计算与 u_new 的平均余弦相似度
    u_tensor = torch.tensor(u_new, device=device, dtype=delta_h_last.dtype)
    u_tensor = u_tensor / (u_tensor.norm() + 1e-12)
    
    # 归一化每个修正向量
    delta_h_last_norm = delta_h_last / (delta_h_last.norm(dim=1, keepdim=True) + 1e-12)
    
    # 计算平均余弦相似度
    alignment = torch.abs((delta_h_last_norm * u_tensor).sum(dim=1)).mean()
    
    dev = 1 - alignment.item()
    
    reft_model.train()
    return dev


def init_R(reft_model, layer_idx: int, u_np: np.ndarray, mode: str = "probe"):
    """初始化 R 的第一列为 û，其余列做 Gram–Schmidt。mode 可选 "probe" 或 "random"。随后冻结 R（初始阶段）。"""
    target = find_loreft_for_layer(reft_model, layer_idx)
    Rbase = target.rotate_layer.parametrizations.weight[0].base  # [D, r]
    D, r = Rbase.shape
    device, dtype = Rbase.device, Rbase.dtype

    with torch.no_grad():
        if mode == "probe" and u_np is not None:
            u = torch.tensor(u_np, device=device, dtype=dtype)
            u = u / (u.norm() + 1e-12)
            Rnew = torch.empty_like(Rbase)
            Rnew[:, 0] = u
            for j in range(1, r):
                v = torch.randn(D, device=device, dtype=dtype)
                for k in range(j):
                    v = v - (Rnew[:, k] @ v) * Rnew[:, k]
                v = v / (v.norm() + 1e-12)
                Rnew[:, j] = v
        elif mode == "random":
            # 全随机正交初始化
            Rnew = torch.empty_like(Rbase)
            for j in range(r):
                v = torch.randn(D, device=device, dtype=dtype)
                for k in range(j):
                    v = v - (Rnew[:, k] @ v) * Rnew[:, k]
                v = v / (v.norm() + 1e-12)
                Rnew[:, j] = v
        Rbase.copy_(Rnew)
    for n, p in target.named_parameters():
        if "rotate_layer" in n:   
            p.requires_grad = False
        if "learned_source" in n: 
            p.requires_grad = True
        

# ------------------ instrumented training loop (counters added) -----------------
def train_one_epoch_with_attack(
    reft_model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    device: str,
    epoch: int,
    inner_attack: str = "gcg",  
    r_update_strategy: str = "curriculum",  # "freeze_always" | "curriculum" | "always_update"
    layer_idx: int = 0, # Reft 所在层
    attack_layer=None, # inner attack 所在层
    lambda_adv: float = 0.5,
):
    if attack_layer is None:
        attack_layer = layer_idx
    reft_model.train()
    losses = []
    steps = 0
    gcg_alpha = cfg["train"]["eps_r"] / cfg["train"]["gcg_steps"]
    # state for curriculum unfreeze
    unfrozen = False
    unfreeze_steps_remaining = 0
    unfreeze_k = int(cfg["train"].get("unfreeze_k_steps", 150))

    check_every = int(cfg["train"].get("check_every", 50))
    dev_thresh_t = float(cfg["train"].get("dev_thresh_t", 0.865))

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    # try:
    #     param_snapshot = find_loreft_for_layer(reft_model, layer_idx).learned_source.weight.data.clone()
    #     print(f"[VERIFY] 已拍摄 L{layer_idx} 的 learned_source.weight 快照。")
    # except Exception as e:
    #     print(f"[VERIFY ERROR] 拍摄快照失败: {e}")
    #     param_snapshot = None
    # # --- 测试结束 ---
    for step, batch in enumerate(pbar, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        B = int(batch["input_ids"].size(0))

        out_clean = reft_model(base=batch, return_dict=True)
        hf_out_clean = get_outputs_from_reft(out_clean)
        logits_clean = hf_out_clean.logits
        clean_loss = F.cross_entropy(logits_clean, batch["labels"])
        
        incr_count("n_train_full_fwd_samples", B)
        
        attack_mode_to_use = "none"
        if inner_attack == "gcg":
            attack_mode_to_use = "gcg"
        elif inner_attack == "latent_pgd":
            attack_mode_to_use = "pgd_fullspace"
        elif inner_attack == "none":
            attack_mode_to_use = "none"
        else:
            raise ValueError(f"Unknown inner_attack type:{inner_attack}")


        if attack_mode_to_use == "none":
             adv_loss = torch.tensor(0.0, device=device)
        else:
            if attack_mode_to_use == "gcg":
                current_eps = cfg["train"]["eps_r"] # GCG 使用 eps_r
                current_steps = cfg["train"]["gcg_steps"]
                current_lr = None # GCG 使用 alpha
                current_gcg_alpha = gcg_alpha # gcg_alpha 在函数顶部已定义
            else: # pgd_fullspace
                current_eps = cfg["train"].get("pgd_eps", 0.05)
                current_steps = cfg["train"].get("pgd_steps", 10)
                current_lr = cfg["train"].get("pgd_lr", None)
                current_gcg_alpha = None # PGD 不使用 alpha

            adv_loss = compute_adv_loss_via_hooks(
                reft_model=reft_model,
                batch=batch,
                attack_layer=attack_layer,
                reft_layer=layer_idx,
                attack_mode=attack_mode_to_use,
                eps=current_eps,
                steps=current_steps,
                lr=current_lr,
                gcg_topk=cfg["train"].get("gcg_topk", 2),
                gcg_alpha=current_gcg_alpha # 传入正确的 alpha
            )
        total_loss = clean_loss + lambda_adv * adv_loss

        optimizer.zero_grad()
        total_loss.backward()

        # counting backward passes for training
        incr_count("n_train_full_bwd_samples", B)

        optimizer.step()

        losses.append({
            "total_loss": total_loss.item(),
            "clean_loss": clean_loss.item(),
            "adv_loss": adv_loss.item()
        })
        steps += 1
        GLOBAL_COUNTS["steps_recorded"] += 1
        
        # # --- 在这里加入“冒烟测试”：打印损失并提前退出 ---
        
        # # 1. 打印损失对比
        # print(f"[VERIFY] Batch {step}: Clean Loss = {clean_loss.item():.4f}, Adv Loss = {adv_loss.item():.4f}")

        # # 2. 运行 5 步后自动停止
        # if step >= 5:
        #     print("[VERIFY] 冒烟测试完成，提前停止 epoch。")
        #     break 
        # # --- 测试结束 ---
        
        
        
        # 提供更详细的进度条
        pbar.set_description(f"Epoch {epoch} | Total: {total_loss.item():.4f} (Clean: {clean_loss.item():.4f}, Adv: {adv_loss.item():.4f})")
        # --- curriculum logic (unfreeze R temporarily when dev > thresh) ---
        if r_update_strategy == "always_update":
            set_phase_freeze(reft_model, freeze_R=False)
        elif r_update_strategy == "freeze_always":
            set_phase_freeze(reft_model, freeze_R=True)
        else:
            # curriculum: periodically compute dev and if dev > thresh then unfreeze for K steps
            if step % check_every == 0:
                u_new = calculate_u_new(reft_model, val_loader, layer_idx, device)
                dev = deviation_phi_from_u(reft_model, val_loader, layer_idx, u_new, device)
                print(f"[curriculum] step {step} deviation={dev:.6f}")
                if (not unfrozen) and (dev > dev_thresh_t):
                    print(f"[curriculum] dev {dev:.6f} > {dev_thresh_t}, unfreezing R for {unfreeze_k} steps")
                    set_phase_freeze(reft_model, freeze_R=False)
                    optimizer = build_optimizer(
                        reft_model, lr_R=cfg["train"].get("lr_R", 1e-5),
                        lr_Wb=cfg["train"].get("lr_wb", 5e-4),
                        weight_decay_R=0.0,
                        weight_decay_Wb=cfg["train"].get("weight_decay", 0.0)
                    )
                    unfrozen = True
                    unfreeze_steps_remaining = unfreeze_k

        # decrement unfreeze counter and re-freeze when exhausted
        if unfrozen:
            unfreeze_steps_remaining -= 1
            if unfreeze_steps_remaining <= 0:
                print(f"[curriculum] re-freezing R after {unfreeze_k} steps")
                set_phase_freeze(reft_model, freeze_R=True)
                optimizer = build_optimizer(
                    reft_model, lr_R=cfg["train"].get("lr_R", 1e-5),
                    lr_Wb=cfg["train"].get("lr_wb", 5e-4),
                    weight_decay_R=0.0,
                    weight_decay_Wb=cfg["train"].get("weight_decay", 0.0)
                )
                unfrozen = False
    # # --- 在这里加入“冒烟测试”：检查参数是否更新 ---
    # if param_snapshot is not None:
    #     try:
    #         param_after = find_loreft_for_layer(reft_model, layer_idx).learned_source.weight.data
    #         diff = (param_snapshot - param_after).abs().sum().item()
    #         print(f"[VERIFY] 权重更新检查 (参数差异总和): {diff}")
    #         if diff == 0:
    #             print("[VERIFY ERROR] 权重没有更新！你的优化器或 requires_grad 设置可能有问题。")
    #         else:
    #             print("[VERIFY OK] 权重已成功更新！")
    #     except Exception as e:
    #         print(f"[VERIFY ERROR] 检查权重更新时出错: {e}")
    # # --- 测试结束 ---
    return losses, optimizer


 
@torch.no_grad()
def evaluate(reft_model, val_loader: DataLoader, device:str) -> float:
    reft_model.eval()
    correct = 0
    total = 0
    for step, batch in enumerate(tqdm(val_loader, desc="Evaluating"), 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = reft_model(base=batch, return_dict=True)
        hf_out = get_outputs_from_reft(out)
        logits = hf_out.logits
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return float(correct) / max(1, total)
            
def build_optimizer(
    reft_model: torch.nn.Module, 
    lr_R: float,
    lr_Wb: float,
    weight_decay_R: float,
    weight_decay_Wb: float
):
    param_groups = [
        {"params": [], "lr": lr_R, "weight_decay": weight_decay_R},   # LoReFT 的 R
        {"params": [], "lr": lr_Wb, "weight_decay": weight_decay_Wb}  # LoReFT 的 b 和 W，以及其他可训练参数
    ]
    for name, param in reft_model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "rotate_layer" in name:
            param_groups[0]["params"].append(param)
        else: # "learned_source" 等
            param_groups[1]["params"].append(param)
            
    # 确保没有空的参数组
    final_groups = [pg for pg in param_groups if pg["params"]]
    
    optim = AdamW(final_groups)
    return optim

# main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/train_reft.yaml")
    parser.add_argument("--seed", type=int, default=None, help="override seed from config")
    parser.add_argument("--R_init_mode", type=str, default=None, help="override model.R_init_mode from config")
    parser.add_argument("--inner_attack", type=str, default=None, help="override train.inner_attack from config")
    parser.add_argument("--out_dir", type=str, default=None, help="override output directory for this run")
    parser.add_argument("--pretrain_tokens", type=float, default=3e11, help="optional normalization denom")
    parser.add_argument("--param_grid", type=str, default=None, help="optional param grid TSV/CSV path")
    parser.add_argument("--array_idx", type=int, default=None, help="1-based index into param_grid (if used)")
    args = parser.parse_args()
    # paramer setting
    with open(args.config, "r") as f: # check
        cfg = yaml.safe_load(f)
    
    array_idx = args.array_idx
    if array_idx is None:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if slurm_id is not None:
            try:
                array_idx = int(slurm_id)
            except Exception:
                array_idx = None
    if args.param_grid and array_idx is not None:
        param_grid_path = args.param_grid
        # read small chunk and sniff delimiter
        with open(param_grid_path, "r", newline="") as fpg:
            head = fpg.read(4096)
            fpg.seek(0)
            delim = '\t' if '\t' in head else ','
            reader = csv.DictReader(fpg, delimiter=delim)
            rows = list(reader)
        if len(rows) == 0:
            raise RuntimeError(f"param_grid {param_grid_path} seems empty or no header/rows found.")
        # array_idx is 1-based (SLURM convention). Convert to 0-based index:
        if not (1 <= array_idx <= len(rows)):
            raise IndexError(f"array index {array_idx} out of range for param_grid with {len(rows)} rows (1-based).")
        row = rows[array_idx - 1]
        # apply param-grid values only if not empty
        if "seed" in row and row["seed"].strip() != "":
            cfg.setdefault("train", {})["seed"] = int(row["seed"])
        if "r_init_mode" in row and row["r_init_mode"].strip() != "":
            cfg.setdefault("model", {})["R_init_mode"] = row["r_init_mode"].strip()
        if "layer_idx" in row and row["layer_idx"].strip() != "":
            L = int(row["layer_idx"])
            cfg.setdefault("model", {})["layer_idx"] = L
            cfg["model"]["hidden_state_index"] = L
            cfg["model"]["component_block_index"] = L - 1  # 注意 off-by-one
        if "attack_layer" in row and row["attack_layer"].strip() != "":
            A = int(row["attack_layer"])
            cfg.setdefault("train", {})["attack_layer"] = A
        
        if "out_dir_tag" in row and row["out_dir_tag"].strip() != "":
            tag = row["out_dir_tag"].strip()
            base_dir = cfg.get("out", {}).get("dir", "out")
            cfg.setdefault("out", {})["dir"] = os.path.join(base_dir, tag)
            
        print(f"[param_grid] row {array_idx}: seed={cfg['train'].get('seed')}, "
              f"R_init_mode={cfg['model'].get('R_init_mode')}, "
              f"layer={cfg['model'].get('layer_idx')}, "
              f"out_dir={cfg.get('out',{}).get('dir')}")
    # --- CLI overrides have highest priority ---
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = int(args.seed)
    if args.R_init_mode is not None:
        cfg.setdefault("model", {})["R_init_mode"] = args.R_init_mode
    if args.inner_attack is not None:
        cfg.setdefault("train", {})["inner_attack"] = args.inner_attack
    if args.out_dir is not None:
        cfg.setdefault("out", {})["dir"] = args.out_dir
    
    # ----------------- END PATCH -----------------
    run_seed = cfg.get("train", {}).get("seed", None)
    if run_seed is None:
        run_seed = 42
    run_seed = int(run_seed)
    cfg.setdefault("train", {})["seed"] = run_seed  # persist back to cfg for consistency

    DEVICES = cfg["device"]
    
    reset_counts()

    # load dataset splits
    with open(cfg["data"]["split_file"], "r") as f:
        splits = json.load(f)
    # determine array id for tagging output
    L_hidden = int(cfg["model"].get("layer_idx",1))
    if L_hidden <= 0:
        raise ValueError(f"layer_idx must be >=1 (0 is embeddings). Got {L_hidden}.")
    
    cfg["model"]["hidden_state_index"] = L_hidden
    cfg["model"]["component_block_index"] = L_hidden - 1  # off-by-one 
    array_id = "local"
    if args.array_idx is not None:
        array_id = str(args.array_idx)
    else:
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")

    print(f"[main] using array_id = {array_id}")
        
    out_root = cfg.get("out", {}).get("dir", "out/spam_pythia410m")
    r_mode = cfg["model"].get("R_init_mode", "random")
    inner_attack = cfg["train"].get("inner_attack", "gcg")
    layer_idx = cfg["model"].get("layer_idx", cfg["model"].get("layer_idx", 14))
    rank_r = cfg["model"].get("rank_r", 4)
    att_layer = cfg["train"].get("attack_layer", None)

    unique_tag = f"seed{run_seed}_R{r_mode}_L{layer_idx}_A{att_layer}_r{rank_r}_atk{inner_attack}_task{array_id}"
    out_dir = os.path.join(out_root, unique_tag)
    os.makedirs(out_dir, exist_ok=True)
    cfg.setdefault("out", {})["dir"] = out_dir
    print(f"[main] outputs -> {out_dir}")
    
    run_start =time.time()
    print(f"\n=== Starting seed {run_seed} at {datetime.datetime.now()} ===")
    set_seed(run_seed)
    model_name = cfg["model"]["model_name"]
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    # 加载之前的 full-FT 权重
    state = torch.load(cfg["model"]["load_baseline_ckpt"], map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.to(DEVICES)
    MODEL_DTYPE = next(model.parameters()).dtype
    layer_idx = cfg["model"]["layer_idx"]   
    comp_idx = layer_idx - 1  # component_block_index
    # ReFT config
    reft_config = ReftConfig(representations={
        "layer": layer_idx,
        "component": f"gpt_neox.layers[{comp_idx}].output",
        "low_rank_dimension": cfg["model"]["rank_r"],
        "intervention": pyreft.LoreftIntervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=cfg["model"]["rank_r"],
            dtype=MODEL_DTYPE,
            dropout=0.0,
            act_fn=None
        )
    })
    reft_model = pyreft.get_reft_model(model, reft_config)

    reft_model.set_device(DEVICES)


    # init R
    u_np = None
    if cfg["model"].get("probe_uv_path") and cfg["model"]["R_init_mode"] == "probe":
        uv = np.load(cfg["model"]["probe_uv_path"])
        u_np = uv.get(f"L{cfg['model']['layer_idx']}", None)[:-1]  # 取出 u，忽略 v
    init_R(reft_model, layer_idx=cfg["model"]["layer_idx"], u_np=u_np,
            mode=cfg["model"].get("R_init_mode", "random"))

    # Dataset & Loader
    train_dataset = AdvDataset(splits[cfg["data"]["train_split"]], dataset_name=cfg["data"]["dataset"])
    val_dataset = AdvDataset(None, dataset_name=cfg["data"]["dataset"], split="validation")
    collator = Collator(model_name, max_length=cfg["model"]["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["train_bsz"], shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["eval_bsz"], shuffle=False, collate_fn=collator)

    # freeze R initially depending on strategy
    r_update_strategy = cfg["train"].get("r_update_strategy", "curriculum")
    if r_update_strategy == "freeze_always":
        set_phase_freeze(reft_model, freeze_R=True)
    elif r_update_strategy == "always_update":
        set_phase_freeze(reft_model, freeze_R=False)
    else:
        set_phase_freeze(reft_model, freeze_R=True)
    optimizer = build_optimizer(
        reft_model,
        lr_R=1e-5, # R被冻结，学习率无所谓
        lr_Wb=cfg["train"]["lr_wb"],
        weight_decay_R=0.0,
        weight_decay_Wb=cfg["train"]["weight_decay"]
    ) 

    run_losses = []
    eval_accs = []
    inner_attack = cfg["train"]["inner_attack"]
    att_layer = cfg["train"].get("attack_layer", None)
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        print(f"\n=== epoch {epoch} ===")
        losses_epoch, optimizer = train_one_epoch_with_attack(
            reft_model, train_loader, val_loader, optimizer,
            cfg=cfg, device=DEVICES,
            epoch=epoch,
            inner_attack=inner_attack,
            r_update_strategy=r_update_strategy,
            layer_idx=cfg["model"]["layer_idx"],
            attack_layer=att_layer, 
            lambda_adv=cfg["train"]["lambda_adv"],
        )
        run_losses.extend(losses_epoch)
        acc = evaluate(reft_model, val_loader, DEVICES)

        eval_accs.append(acc)
        print(f"*** epoch {epoch} eval acc: {acc:.4f} ***")

    elapsed = time.time() - run_start
    print(f"Total elapsed (s): {elapsed:.1f}")
    try:
        target_module = find_loreft_for_layer(reft_model, cfg["model"]["layer_idx"])
    except Exception:
        target_module = None
    head_module = getattr(reft_model.model, "classifier", None) or getattr(reft_model.model, "score", None)

    # seq_len: 使用 cfg 中的 max_length（或者用实际 batch avg）
    seq_len = cfg["model"].get("max_length", 512)
    model_flop_info = summarize_model_and_submodule_flops(
        model=reft_model.model,
        target_module=target_module,
        head_module=head_module,
        seq_len=seq_len,
        pretrain_tokens_D=args.pretrain_tokens if 'args' in globals() else cfg["train"].get("pretrain_tokens", 3e11)
    )
    F_fwd_full = model_flop_info["F_fwd_full"]
    r_ratio = model_flop_info["sub_to_full_param_ratio"]
    n_layers = model.config.num_hidden_layers
    #stats = compute_flops_from_counts(reft_model.model, target_module, head_module, seq_len=MAX_LEN, pretrain_tokens_D=args.pretrain_tokens)
    stats = compute_flops_from_counts(
        GLOBAL_COUNTS,
        F_fwd_full=F_fwd_full,
        r_ratio=r_ratio,
        pretraining_N=model_flop_info["N_params"],
        pretraining_D=model_flop_info["pretrain_tokens_D"],
        attack_layer=att_layer,
        n_total_layers=n_layers
    )
    stats.update(model_flop_info)
    # add meta and elapsed seconds for traceability
    stats["meta"] = {"seed": run_seed, "array_id": array_id, "out_dir": out_dir, "elapsed_seconds": elapsed}
    # include losses and evals & cfg summary
    stats["train_losses"] = run_losses
    stats["eval_accs"] = eval_accs
    stats["cfg_summary"] = {
        "model_name": model_name,
        "layer_idx": cfg["model"]["layer_idx"],
        "R_init_mode": cfg["model"]["R_init_mode"],
        "attack_layer": att_layer,
        "rank_r": cfg["model"]["rank_r"],
        "inner_attack": inner_attack,
        "r_update_strategy": r_update_strategy
    }

    save_json_report(stats, out_dir=out_dir, prefix=f"Rinit{cfg['model']['R_init_mode']}_seed{run_seed}_L{cfg['model']['layer_idx']}_A{att_layer}_r{cfg['model']['rank_r']}")

    # save model checkpoint
    save_path = os.path.join(out_dir, f"reft_lat_A{att_layer}_Rinit{cfg['model']['R_init_mode']}_{cfg['model']['model_name'].replace('/','_')}_rank{cfg['model']['rank_r']}_L{cfg['model']['layer_idx']}_seed{run_seed}.pt")
    torch.save(reft_model.state_dict(), save_path)
    print("saved:", save_path)

if __name__ == "__main__":
    main()
    
# python -m src.training.trainer_reft_lat2  --pretrain_tokens 3e11 --seed 42