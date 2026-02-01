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
from src.training.reft_adv_trainer_new import (
    ReftAdversarialTrainerForSequenceClassificationNew,
)
from src.attack.inner_attack import AttackConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  
from src.utils.tools import set_seed
from omegaconf import OmegaConf

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

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
    # 允许命令行传入 key=value 来覆盖 yaml 配置，例如: python train.py config=base.yaml train.lr=0.01
    cli_conf = OmegaConf.from_cli()
    
    # 获取基础配置文件路径 (默认 configs/train_reft.yaml)
    config_path = cli_conf.get("config", "configs/train_reft.yaml")
    base_conf = OmegaConf.load(config_path)
    
    # 合并配置：CLI 参数 > YAML 参数
    cfg = OmegaConf.merge(base_conf, cli_conf)
    
    # --- 2. 准备输出目录 ---
    # 如果 CLI 没有指定 out.dir，这里会用 yaml 里的默认值，建议在 Launcher 里指定
    out_dir = cfg.out.dir
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存最终使用的配置，方便复现
    OmegaConf.save(cfg, os.path.join(out_dir, "config_final.yaml"))
    
    # 设置日志
    log_file = os.path.join(out_dir, "train.log")
    file_handler = logging.FileHandler(log_file, mode="w")
    logger.addHandler(file_handler)
    logger.info(f"Output Directory: {out_dir}")

    set_seed(cfg.train.seed)
    
    # --- 3. 构建模型 ---
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    reft_model, tokenizer = build_reft_model(
        model_name=cfg.model.model_name,
        baseline_ckpt=cfg.model.get("load_baseline_ckpt", None),
        reft_layer=cfg.model.layer_idx,
        rank_r=cfg.model.rank_r,
        device=device,
        attack_layer=cfg.train.get("attack_layer", cfg.model.layer_idx),
        disable_model_grads=True, 
    )
    
    tokenizer.model_max_length = cfg.model.max_length
    
    with open(cfg.data.split_file, "r") as f:
        splits = json.load(f)
    train_indices = splits.get(cfg.data.train_split)
    
    if train_indices is None:
        logger.warning(f"Split '{cfg.data.train_split}' not found. Using full dataset.")

    train_dataset, eval_dataset = build_reft_classification_datasets(
        tokenizer=tokenizer,
        data_path=cfg.data.dataset,
        train_split="train", 
        eval_split="validation",
        train_indices=train_indices,
        input_field=cfg.data.get("input_field", "content"),
        label_field=cfg.data.get("label_field", "clf_label"),
        position=cfg.model.get("position", "l1"),
    )

    # --- 5. 训练参数 ---
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.train_bsz,
        per_device_eval_batch_size=cfg.train.eval_bsz,
        learning_rate=cfg.train.lr_wb,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.get("logging_steps", 20),
        evaluation_strategy=cfg.train.get("evaluation_strategy", "epoch"),
        save_strategy=cfg.train.get("save_strategy", "epoch"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy", 
        save_total_limit=cfg.train.get("save_total_limit", 2),
        report_to="none", # 在这里关掉 wandb，如果需要可以在 launcher 里控制环境变量
        remove_unused_columns=False,
    )

    # --- 6. 攻击配置 ---
    # 确保 cfg.train 里的 key 和 AttackConfig 需要的一致
    attack_cfg = AttackConfig(
        inner_attack=cfg.train.get("inner_attack", "gcg"),
        attack_layer=cfg.train.get("attack_layer", None), 
        reft_layer=cfg.model.layer_idx,
        eps=cfg.train.eps_r,
        steps=cfg.train.gcg_steps,
        lr=cfg.train.get("pgd_lr", None),
        gcg_topk=cfg.train.get("gcg_topk", 2),
        gcg_alpha=cfg.train.get("gcg_alpha", None), 
        lambda_adv=cfg.train.lambda_adv,
        use_circuit_gate=cfg.train.get("use_circuit_gate", False),
        circuit_path=cfg.train.get("circuit_path", None),
        gate_mode=cfg.train.get("gate_mode", "inner_only"),
        circuit_top_k=cfg.train.get("circuit_top_k", None),
    )

    # --- 7. Trainer & Run ---
    trainer = ReftAdversarialTrainerForSequenceClassification(
        model=reft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=build_reft_data_collator(tokenizer, cfg.model.max_length),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        attack_config=attack_cfg,
    )
    
    # Early Stopping
    patience = cfg.train.get("early_stopping_patience", None)
    if patience is not None and patience > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

    logger.info("Starting training...")
    trainer.train()
    
    # 保存结果
    final_save_dir = os.path.join(out_dir, "final_intervention")
    trainer.model.save_intervention(save_directory=final_save_dir, include_model=False)
    tokenizer.save_pretrained(final_save_dir)
    logger.info(f"Done. Saved to {final_save_dir}")

if __name__ == "__main__":
    main()