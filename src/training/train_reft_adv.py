import os, argparse, yaml, torch, json, csv, sys, logging
from transformers import TrainingArguments
from transformers.utils import logging as hf_logging
from transformers import EarlyStoppingCallback
from src.models.reft_latent import build_reft_model
from src.data.datasets_reft import (
    build_reft_classification_datasets,
    build_reft_data_collator
)
from src.training.reft_adv_trainer import (
    ReftAdversarialTrainerForSequenceClassification,
)
from src.attack.inner_attack import AttackConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  
from src.utils.tools import set_seed

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    # 转 numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # 分类 argmax
    if preds.ndim > 1:
        preds = preds.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="指向 .yaml 配置文件")
    parser.add_argument("--param_grid", type=str, default=None, help="指向 param grid CSV/TSV")
    parser.add_argument("--array_idx", type=int, default=None, help="SLURM array task ID (1-based)")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    train_cfg = cfg.get("train", {})
    use_wandb = train_cfg.get("use_wandb", False)
    wandb_project = train_cfg.get("wandb_project", None)
    wandb_entity = train_cfg.get("wandb_entity", None)
    wandb_run_name = train_cfg.get("wandb_run_name", None)

    array_idx = args.array_idx
    if array_idx is None:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if slurm_id is not None: array_idx = int(slurm_id)
            
    if args.param_grid and array_idx is not None:
        print(f"[Train] 正在从 {args.param_grid} 加载第 {array_idx} 行参数...")
        with open(args.param_grid, "r", newline="") as fpg:
            head = fpg.read(4096); fpg.seek(0)
            delim = '\t' if '\t' in head else ','
            reader = csv.DictReader(fpg, delimiter=delim)
            rows = list(reader)
        if not (1 <= array_idx <= len(rows)):
            raise IndexError(f"array index {array_idx} out of range for {len(rows)} rows.")
        row = rows[array_idx - 1]
        
        # 动态覆盖 cfg
        if "seed" in row and row["seed"].strip() != "": cfg.setdefault("train", {})["seed"] = int(row["seed"])
        if "r_init_mode" in row and row["r_init_mode"].strip() != "": cfg.setdefault("model", {})["R_init_mode"] = row["r_init_mode"].strip()
        if "layer_idx" in row and row["layer_idx"].strip() != "":
            L = int(row["layer_idx"]); cfg.setdefault("model", {})["layer_idx"] = L
        if "attack_layer" in row and row["attack_layer"].strip() != "":
            A = int(row["attack_layer"]); cfg.setdefault("train", {})["attack_layer"] = A
        if "out_dir_tag" in row and row["out_dir_tag"].strip() != "":
            tag = row["out_dir_tag"].strip(); base_dir = cfg.get("out", {}).get("dir", "out")
            cfg.setdefault("out", {})["dir"] = os.path.join(base_dir, tag)
        print(f"[param_grid] row {array_idx}: Applied overrides from grid.")
    
    run_seed = cfg.get("train", {}).get("seed", 42)
    set_seed(run_seed)
    
    # --- 2. 设置输出目录 
    out_root = cfg.get("out", {}).get("dir", "out/pythia410m")
    r_mode = cfg["model"].get("R_init_mode", "random")
    inner_attack = cfg["train"].get("inner_attack", "gcg")
    layer_idx = cfg["model"]["layer_idx"]
    rank_r = cfg["model"]["rank_r"]
    att_layer = cfg["train"].get("attack_layer", layer_idx)
    
    # [关键] 目录名必须包含 L 和 r, 供评估器解析
    unique_tag = f"seed{run_seed}_R{r_mode}_L{layer_idx}_A{att_layer}_r{rank_r}_atk{inner_attack}_task{array_idx or 'local'}"
    out_dir = os.path.join(out_root, unique_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Train] 输出目录: {out_dir}")
    
    log_file = os.path.join(out_dir, "train.log")
    handlers = [
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
    hf_logging.set_verbosity_info()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    logger = hf_logging.get_logger(__name__)
    logger.info(f"Logging to {log_file}")
    
    # 训练时构建模型，和eval做区别
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    reft_model, tokenizer = build_reft_model(
        model_name=cfg["model"]["model_name"],
        baseline_ckpt=cfg["model"].get("load_baseline_ckpt", None),
        reft_layer=layer_idx,
        rank_r=rank_r,
        device=device,
        attack_layer=cfg["train"].get("attack_layer", layer_idx),
        disable_model_grads=True, 
    )
    
    max_len = cfg["model"].get("max_length", 512)
    tokenizer.model_max_length = max_len
    
    with open(cfg["data"]["split_file"], "r") as f:
        splits = json.load(f)
    train_indices = splits.get(cfg["data"]["train_split"])
    if train_indices is None:
        print(f"警告: 在 {cfg['data']['split_file']} 中未找到 split '{cfg['data']['train_split']}'。使用完整训练集。")

    train_dataset, eval_dataset = build_reft_classification_datasets(
        tokenizer=tokenizer,
        data_path=cfg["data"]["dataset"],
        train_split="train", 
        eval_split="validation",
        train_indices=train_indices, # 用索引筛选原始训练集的一部分
        input_field=cfg["data"].get("input_field", "content"),
        label_field=cfg["data"].get("label_field", "clf_label"),
        position=cfg["model"].get("position", "l1"),
    )

    data_collator = build_reft_data_collator(
        tokenizer, max_length=cfg["model"]["max_length"]
    )
    
    # --- 5. TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=out_dir, # 检查点会保存到 {out_dir}/checkpoint-xxx
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["train_bsz"],
        per_device_eval_batch_size=cfg["train"]["eval_bsz"],
        learning_rate=cfg["train"]["lr_wb"],
        weight_decay=cfg["train"]["weight_decay"],
        logging_steps=cfg["train"].get("logging_steps", 20),
        evaluation_strategy=cfg["train"].get("evaluation_strategy", "epoch"),
        save_strategy=cfg["train"].get("save_strategy", "epoch"),
        load_best_model_at_end=True, # <-- 确保结束时是最佳模型
        metric_for_best_model="eval_accuracy", 
        save_total_limit=cfg["train"].get("save_total_limit", 2),
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
    )
    
    if use_wandb:
        import wandb
        wandb_kwargs = {}
        if wandb_project is not None:
            wandb_kwargs["project"] = wandb_project
        if wandb_entity is not None:
            wandb_kwargs["entity"] = wandb_entity
        if wandb_run_name is not None:
            wandb_kwargs["name"] = wandb_run_name
        else:
            wandb_kwargs["name"] = unique_tag  # 用输出目录里的 tag 做 run name

        wandb.init(**wandb_kwargs, config=cfg)
        print(f"[Train] wandb enabled. Project={wandb_project}, run={wandb_kwargs['name']}")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("[Train] wandb disabled.")

    # --- 6. AttackConfig (Part C) ---
    attack_cfg = AttackConfig(
        inner_attack=cfg["train"].get("inner_attack", "gcg"),
        attack_layer=cfg["train"].get("attack_layer", None), 
        reft_layer=layer_idx,
        eps=cfg["train"]["eps_r"],
        steps=cfg["train"]["gcg_steps"],
        lr=cfg["train"].get("pgd_lr", None),
        gcg_topk=cfg["train"].get("gcg_topk", 2),
        gcg_alpha=cfg["train"].get("gcg_alpha", None), 
        lambda_adv=cfg["train"]["lambda_adv"],
    )

    # --- 7. Trainer (Part C) ---
    trainer = ReftAdversarialTrainerForSequenceClassification(
        model=reft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        attack_config=attack_cfg,
    )
    
    patience = cfg["train"].get("early_stopping_patience", None)
    if patience is not None and patience > 0:
        threshold = cfg["train"].get("early_stopping_threshold", 0.0)
        es_callback = EarlyStoppingCallback(
            early_stopping_patience=patience,
            early_stopping_threshold=threshold,
        )
        trainer.add_callback(es_callback)
        print(f"[Train] Early stopping enabled: patience={patience}, threshold={threshold}")
    else:
        print("[Train] Early stopping disabled.")
    
    # --- 8. 训练 ---
    print("[Train] 开始训练...")
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    trainer.train()
    
    final_save_dir = os.path.join(out_dir, "final_intervention")
    print(f"[Train] 训练完成。保存 ReFT 模块到: {final_save_dir}")
    
    # 仅保存 ReFT 插件
    trainer.model.save_intervention(
        save_directory=final_save_dir, 
        include_model=False 
    )
    
    # 保存 tokenizer 以便评估器使用
    tokenizer.save_pretrained(final_save_dir)
    print(f"[Train] Tokenizer 已保存。训练流程结束。")

if __name__ == "__main__":
    main()

# python -m src.training.train_reft_adv --config configs/train_reft.yaml --param_grid param_grid.tsv --array_idx 1