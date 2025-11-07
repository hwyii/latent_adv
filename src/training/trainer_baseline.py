from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from zmq import device
from ..data.adv_dataset import AdvDataset
from ..data.Collator import Collator
from torch.utils.data import DataLoader
import os, time, math, torch, numpy as np, random, json, time
from src.utils.tools import set_seed
import wandb
# ------------------ 全局计数器 ------------------
GLOBAL_COUNTS = {
    "n_train_full_fwd_samples": 0,
    "n_train_full_bwd_samples": 0,
    "n_train_tokens": 0,
    "steps_recorded": 0,
}

def incr_count(key: str, n: int = 1):
    if key not in GLOBAL_COUNTS:
        GLOBAL_COUNTS[key] = 0
    GLOBAL_COUNTS[key] += int(n)

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def _wandb_make_run_name(cfg: dict) -> str:
    ds = cfg["data"]["dataset"].split("/")[-1]
    model_last = cfg["model"]["name"].split("/")[-1].replace("/", "-")
    seed = cfg.get("seed", "NA")
    suffix = cfg.get("wandb", {}).get("suffix", "")
    base = f"{ds}/{model_last}/seed{seed}"
    return f"{base}-{suffix}" if suffix else base

# dataset and dataloader and collator
def build_dataloaders(cfg):
    with open(cfg["data"]["split_file"], "r") as f:
        splits = json.load(f)
    train_ds = AdvDataset(splits["ft_train"], cfg["data"]["dataset"], split="train")
    print(f"{cfg['data']['dataset']} Train samples:", len(train_ds))
    val_ds   = AdvDataset(None, cfg["data"]["dataset"], split="validation")  # 全量 val
    print(f"{cfg['data']['dataset']} Validation samples:", len(val_ds))
    collate = Collator(model_name=cfg["model"]["name"], max_length=cfg["data"]["max_length"])
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["train_bsz"], shuffle=True, collate_fn=collate)
    #print("Train batches:", len(train_loader), "Batch size:", cfg["data"]["train_bsz"], "Total train samples:", len(train_ds))
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["eval_bsz"], shuffle=False, collate_fn=collate)
    return train_loader, val_loader
    
# model and device
def build_model(cfg, num_labels=2):
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"]["name"], num_labels=num_labels)
    
    model.config.pad_token_id = tok.pad_token_id
    if model.config.eos_token_id is None and tok.eos_token_id is not None:
        model.config.eos_token_id = tok.eos_token_id
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

# optimizer and scheduler
def build_opt_sched(model, cfg, train_loader):
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    total_steps = len(train_loader) * int(cfg["train"]["epochs"])
    warmup_steps = 0
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    return optim, sched
    
# training loop
def train_one_epoch(model, loader, device, optim, sched, scaler=None, log_every=50):
    model.train()
    running_loss = 0.0
    t0 = time.time()
    for step, batch in enumerate(loader, 1):
        # move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["input_ids"].size(0)

        # 统计 token 数（考虑 variable length）
        # 假设你的 collator 提供 attention_mask
        tokens_in_batch = int(batch["attention_mask"].sum().item())  # 总 token 数（batch 内所有序列加和）
        incr_count("n_train_tokens", tokens_in_batch)
        incr_count("n_train_full_fwd_samples", B)
        incr_count("steps_recorded", 1)

        use_amp = (scaler is not None)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(**batch)
            loss = out.loss

        # backward
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        # 统计 backward（按样本计）
        incr_count("n_train_full_bwd_samples", B)

        optim.zero_grad(set_to_none=True)
        sched.step()

        running_loss += loss.item()
        if step % log_every == 0:
            avg = running_loss / log_every
            spd = log_every / max(time.time() - t0, 1e-9)
            print(f"[train] step={step:04d} loss={avg:.4f} steps/s={spd:.2f}")
            # === wandb: train/loss、学习率、吞吐 ===
            if wandb is not None and wandb.run is not None:
                lr = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else optim.param_groups[0]["lr"]
                wandb.log({
                    "train/loss": avg,
                    "train/steps_per_sec": spd,
                    "train/tokens_cum": int(GLOBAL_COUNTS.get("n_train_tokens", 0)),
                    "optimization/lr": float(lr),
                })
            running_loss = 0.0
            t0 = time.time()
            
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    loss_sum, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        correct += (preds == batch['labels']).sum().item()
        total += batch['labels'].size(0)
        loss_sum += float(out.loss.item())
        n_batches += 1
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(n_batches, 1)
    return acc, avg_loss
def compute_flops_baseline(global_counts, model_num_params, pretrain_total_tokens=None):
    """
    global_counts: GLOBAL_COUNTS
    model_num_params: N (整数)
    pretrain_total_tokens: optional, 用于归一化到论文的 '6*N*D' 分母（D = pretraining tokens）
    """
    c = {k: int(global_counts.get(k, 0)) for k in global_counts}
    N = float(model_num_params)
    T = float(c.get("n_train_tokens", 0))

    # Kaplan 近似：每 token 训练 (fwd + bwd) ≈ 6 * N FLOPs
    Ctrain = 6.0 * N * T

    stats = {
        "counts": c,
        "model_params_N": N,
        "total_train_tokens": T,
        "Ctrain_FLOPs": Ctrain,
    }

    if pretrain_total_tokens is not None:
        denom = 6.0 * N * float(pretrain_total_tokens)
        stats["pretrain_total_tokens"] = pretrain_total_tokens
        stats["proportion_of_pretraining"] = float(Ctrain) / denom

    return stats

def save_checkpoint(model, out_dir, name="best.pt"):
    ensure_outdir(out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, name))
    
def train_baseline(cfg: dict):
    t0_wall = time.time()
    data = cfg["data"]["dataset"].split("/")[-1]  # e.g., "EnronSpam"
    
    # === wandb init ===
    if cfg.get("wandb", {}).get("enable", False) and wandb is not None:
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        run_name = cfg["wandb"].get("run_name") or _wandb_make_run_name(cfg)
        wandb.init(
            project=cfg["wandb"].get("project", "latent_adv_baseline"),
            name=run_name,
            config=cfg,
            group=cfg["wandb"].get("group", "baseline"),
            tags=[
                data,
                cfg["model"]["name"].split("/")[-1].replace("/", "-"),
                f"seed{cfg.get('seed','NA')}",
                cfg.get("wandb", {}).get("suffix", ""),
            ]
        )
    
    set_seed(cfg["seed"])
    ensure_outdir(cfg["out"]["dir"])
    
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token   
    
    train_loader, val_loader = build_dataloaders(cfg)
    model, device = build_model(cfg)
    optim, sched = build_opt_sched(model, cfg, train_loader)
    
    use_fp16 = bool(cfg["train"]["fp16"]) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    best_acc = 0.0
    best_eval_loss = float("inf")
    
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        print(f"\n=== Epoch {epoch}/{cfg['train']['epochs']} ===")
        train_one_epoch(model, train_loader, device, optim, sched, scaler, log_every=cfg["train"]["log_every"])
        acc, eval_loss = evaluate(model, val_loader, device)
        print(f"[eval] accuracy: {acc:.4f} | loss: {eval_loss:.4f}")

        # wandb: eval
        if wandb is not None and wandb.run is not None:
            wandb.log({"eval/acc": acc, "eval/loss": eval_loss, "epoch": epoch})
        if acc > best_acc:               
            best_acc = acc
            save_checkpoint(model, cfg["out"]["dir"], name=f"best_{data}.pt")
            print(f"New best accuracy: {best_acc:.4f}. Model checkpoint saved.")
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_checkpoint(model, cfg["out"]["dir"], name=f"best_loss_{data}.pt")
    # 在 train_baseline 最后，打印 / 保存 stats
    N = sum(p.numel() for p in model.parameters())
    stats = compute_flops_baseline(GLOBAL_COUNTS, model_num_params=N, pretrain_total_tokens=3e11)
    duration = time.time() - t0_wall
    stats["training_duration_seconds"] = float(duration)
    if stats.get("total_train_tokens", 0) > 0:
        stats["training_tokens_per_second"] = float(stats["total_train_tokens"]) / max(duration, 1e-9)

    stats["best_validation_accuracy"] = best_acc
    stats["best_validation_loss"] = best_eval_loss
    stats["seed"] = cfg["seed"]
    stats["model_name"] = cfg["model"]["name"]
    stats["dataset"] = cfg["data"]["dataset"]
    stats["out_dir"] = cfg["out"]["dir"]
    stats["device"] = str(device)

    # wandb summary
    if wandb is not None and wandb.run is not None:
        wandb.summary["best_acc"] = best_acc
        wandb.summary["best_eval_loss"] = best_eval_loss
        wandb.summary["train_time_sec"] = duration
        wandb.summary["model_n_params"] = int(N)
        wandb.finish()

    with open(os.path.join(cfg["out"]["dir"], f"flop_stats_{data}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats
