import os
import re
import json
import math
import time
import shutil
import subprocess
from dataclasses import dataclass, asdict
from multiprocessing import Pool, Manager
from typing import Dict, List, Tuple, Optional

# =========================
# Config
# =========================

def pick_free_gpus(k: int = 1, min_free_mb: int = 8000, max_util: int = 30):
    """
    从 nvidia-smi 自动挑 k 张“最空闲”的 GPU。
    - min_free_mb: 至少要有多少空闲显存（避免 OOM）
    - max_util: GPU 利用率超过这个就认为忙
    返回 GPU index 列表。
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    infos = []
    for line in out:
        # line example: "0, 12, 1024, 24576"
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx = int(parts[0])
        util = int(parts[1])
        mem_used = int(parts[2])
        mem_total = int(parts[3])
        mem_free = mem_total - mem_used
        infos.append((idx, util, mem_free, mem_used, mem_total))

    # 过滤忙卡
    candidates = [x for x in infos if x[2] >= min_free_mb and x[1] <= max_util]
    if not candidates:
        # 如果没有满足条件的，就退化：按空闲显存排序选
        candidates = infos

    # 排序：先按 mem_free 降序，再按 util 升序
    candidates.sort(key=lambda x: (-x[2], x[1]))

    chosen = [x[0] for x in candidates[:k]]
    return chosen

GPUS = pick_free_gpus(k=2, min_free_mb=40000, max_util=100)
MAX_WORKERS = len(GPUS)
print("[GPU] chosen:", GPUS)

# 训练的 position 列表：l1, l5, l10, l15, l20
POSITIONS = [1, 5, 10, 15, 20]

# eps 缩放：给定 l20 eps=0.5
BASE_TOKENS = 20
BASE_EPS = 0.5

# 数据集配置（你先只开 Helpful，其他自行打开）
DATASETS = {
    # "Helpful": {
    #     "path": "AlignmentResearch/Helpful",
    #     "split_file": "src/data/helpful_splits.json",
    #     "train_split": "ft_train",
    #     "ckpt": "out/pythia410m/helpful/best_Helpful.pt",
    # },
    "Harmless": {
        "path": "AlignmentResearch/Harmless",
        "split_file": "src/data/harmless_splits.json",
        "train_split": "ft_train",
        "ckpt": "out/pythia410m/harmless/best_Harmless.pt",
    },
    "IMDB": {
        "path": "AlignmentResearch/IMDB",
        "split_file": "src/data/imdb_splits.json",
        "train_split": "ft_train",
        "ckpt": "out/pythia410m/imdb/best_IMDB.pt",
    },
}

# Layer grid: (reft_layer, attack_layer)
LAYER_PAIRS = [
    (5, 4),
]

# 训练相关固定参数
TRAIN_CFG = {
    "model_name": "EleutherAI/pythia-410m",
    "rank_r": 64,
    "epochs": 10,
    "lambda_adv": 10.0,
    "inner_attack": "latent_pgd",
    "gcg_steps": 10,
    "gcg_topk": 64,
    "seed": 42,
    "max_length": 512,
    # 如果还有 stepsize 等参数，在 train_reft.yaml 或 dotlist 覆盖都行
}

# 评测相关固定参数（token gcg）
EVAL_CFG = {
    "max_eval_samples": 100,
    "seed": 42,
    "attack_start": 10,
    "n_attack_tokens": 10,
    "beam_k": 256,
    "rounds": 20,
    "attack_mode": "suffix",
    "n_candidates_per_it": 128,
    "reft_loc_mode": "last_n",
}

# model name and dataset name
if "gpt2" in TRAIN_CFG["model_name"].lower():
    MODEL_NAME = "gpt2"
    model_tag = MODEL_NAME
else:
    MODEL_NAME = TRAIN_CFG["model_name"].split("/")[-1]
    model_tag = MODEL_NAME.replace("-", "")

# 输出根目录
OUT_ROOT = f"outputs_{MODEL_NAME}_auto"          # 训练输出统一放这里
LOG_ROOT = f"logs_auto_{MODEL_NAME}"                 # 所有日志统一放这里



# =========================
# Helpers
# =========================


def eps_for_k(k: int) -> float:
    # eps(k) = BASE_EPS * sqrt(BASE_TOKENS / k)
    return BASE_EPS * math.sqrt(BASE_TOKENS / float(k))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def run_shell(cmd_str: str, log_file: str) -> int:
    ensure_dir(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        p = subprocess.run(cmd_str, shell=True, stdout=f, stderr=f)
    return p.returncode

def find_final_intervention(run_dir: str) -> str:
    # 训练脚本保存到 out_dir/final_intervention
    return os.path.join(run_dir, "final_intervention")

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except:
        return None

def parse_eval_json(json_path: str):
    """
    返回一个 dict，包含你关心的所有 summary 指标。
    如果文件不存在/解析失败，返回空 dict。
    """
    if not os.path.isfile(json_path):
        return {}
    try:
        with open(json_path, "r") as f:
            obj = json.load(f)
        summ = (obj.get("summary", {}) or {})
        m = (summ.get("summary_metrics", {}) or {})
        c = (summ.get("raw_counts", {}) or {})

        def getf(k):
            v = m.get(k, None)
            return float(v) if v is not None else None

        def geti(k):
            v = c.get(k, None)
            return int(v) if v is not None else None

        return {
            "acc": getf("Clean Accuracy (ACC)"),
            "rob_acc": getf("Robust Accuracy (Rob-Acc)"),
            "asr_cond": getf("ASR_cond_on_correct"),
            "asr_overall": getf("ASR_overall"),
            "n_initially_correct": geti("n_initially_correct"),
            "n_correct_after_attack": geti("n_correct_after_attack"),
            "n_flipped_C_to_W": geti("n_flipped_to_wrong_C_to_W"),
            "n_flipped_W_to_C": geti("n_flipped_to_correct_W_to_C"),
            "n_correct_after_attack_C_to_C": geti("n_correct_after_attack_C_to_C"),
            "n_stayed_wrong_W_to_W": geti("n_stayed_wrong_W_to_W"),
        }
    except Exception:
        return {}


@dataclass
class Task:
    dataset_name: str
    dataset_path: str
    split_file: str
    train_split: str
    baseline_ckpt: str
    reft_layer: int
    attack_layer: int
    position_k: int  # e.g., 1/5/10/15/20
    eps_r: float

    @property
    def position_str(self) -> str:
        return f"l{self.position_k}"

    @property
    def tag(self) -> str:
        # 你可以按喜好改命名规则，这里保证唯一且包含关键信息
        # 注意 eps 用简化格式避免目录名太长
        eps_txt = f"{self.eps_r:.6g}".replace(".", "p")
        return f"{self.position_str}_{self.dataset_name}_{model_tag}_L{self.reft_layer}_A{self.attack_layer}_pgd{TRAIN_CFG['gcg_steps']}_eps{eps_txt}_ep{TRAIN_CFG['epochs']}_r{TRAIN_CFG['rank_r']}_lam{TRAIN_CFG['lambda_adv']}"

    @property
    def out_dir(self) -> str:
        return os.path.join(OUT_ROOT, self.dataset_name, self.position_str, self.tag)

    @property
    def train_log(self) -> str:
        return os.path.join(LOG_ROOT, "train", self.dataset_name, self.position_str, f"{self.tag}.log")

    @property
    def eval_log(self) -> str:
        return os.path.join(LOG_ROOT, "eval", self.dataset_name, self.position_str, f"{self.tag}.log")

    @property
    def eval_json(self) -> str:
        # 输出 json 放在 run_dir 下，方便你 find/grep
        return os.path.join(self.out_dir, f"eval_token_gcg_{self.position_str}.json")


def build_all_tasks() -> List[Task]:
    tasks: List[Task] = []
    for d_name, d_cfg in DATASETS.items():
        for (reft_l, att_l) in LAYER_PAIRS:
            for k in POSITIONS:
                tasks.append(Task(
                    dataset_name=d_name,
                    dataset_path=d_cfg["path"],
                    split_file=d_cfg["split_file"],
                    train_split=d_cfg["train_split"],
                    baseline_ckpt=d_cfg["ckpt"],
                    reft_layer=reft_l,
                    attack_layer=att_l,
                    position_k=k,
                    eps_r=eps_for_k(k),
                ))
    return tasks


# =========================
# Train & Eval Commands
# =========================

def make_train_cmd(task: Task, gpu_id: int) -> str:
    # OmegaConf dotlist
    cmd = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python -m src.training.train_reft_adv",
        "config=configs/train_reft.yaml",

        f"data.dataset={task.dataset_path}",
        f"data.split_file={task.split_file}",
        f"data.train_split={task.train_split}",

        f"model.model_name={TRAIN_CFG['model_name']}",
        f"model.load_baseline_ckpt={task.baseline_ckpt}",
        f"model.layer_idx={task.reft_layer}",
        f"model.rank_r={TRAIN_CFG['rank_r']}",
        f"model.position={task.position_str}",
        f"model.max_length={TRAIN_CFG['max_length']}",

        f"train.attack_layer={task.attack_layer}",
        f"train.lambda_adv={TRAIN_CFG['lambda_adv']}",
        f"train.epochs={TRAIN_CFG['epochs']}",
        f"train.gcg_steps={TRAIN_CFG['gcg_steps']}",
        f"train.inner_attack={TRAIN_CFG['inner_attack']}",
        f"train.eps_r={task.eps_r}",
        f"train.gcg_topk={TRAIN_CFG['gcg_topk']}",
        f"train.seed={TRAIN_CFG['seed']}",

        f"wandb_name={task.tag}",
        f"out.dir={task.out_dir}",
    ]
    return " ".join(cmd)

def make_eval_cmd(task: Task, gpu_id: int) -> str:
    # 你 eval 脚本用的是 argparse 参数形式
    cmd = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python -m src.attack.eval_token_gcg",
        f"--model_name {TRAIN_CFG['model_name']}",
        f"--baseline_ckpt {task.baseline_ckpt}",
        f"--data_dir {task.split_file}",
        f"--layer_idx {task.reft_layer}",
        f"--rank_r {TRAIN_CFG['rank_r']}",
        f"--run_dir {task.out_dir}",
        f"--dataset {task.dataset_path}",
        f"--max_eval_samples {EVAL_CFG['max_eval_samples']}",
        f"--seed {EVAL_CFG['seed']}",
        f"--attack_start {EVAL_CFG['attack_start']}",
        f"--n_attack_tokens {EVAL_CFG['n_attack_tokens']}",
        f"--beam_k {EVAL_CFG['beam_k']}",
        f"--rounds {EVAL_CFG['rounds']}",
        f"--output_json {task.eval_json}",
        f"--attack_mode {EVAL_CFG['attack_mode']}",
        f"--n_candidates_per_it {EVAL_CFG['n_candidates_per_it']}",
        f"--reft_loc_mode {EVAL_CFG['reft_loc_mode']}",
        f"--n_reft_positions {task.position_k}",
        f"--position {task.position_str}",
    ]
    return " ".join(cmd)


# =========================
# Worker logic
# =========================

def train_one(task: Task, gpu_id: int) -> Tuple[Task, int]:
    ensure_dir(task.out_dir)

    final_dir = find_final_intervention(task.out_dir)
    # 断点续跑：如果 final_intervention 存在就跳过训练
    if os.path.isdir(final_dir):
        return task, 0

    cmd = make_train_cmd(task, gpu_id)
    rc = run_shell(cmd, task.train_log)
    return task, rc

def eval_one(task: Task, gpu_id: int) -> Tuple[Task, int]:
    final_dir = find_final_intervention(task.out_dir)
    if not os.path.isdir(final_dir):
        # 训练没成功就不 eval
        return task, 2

    # 断点续跑：如果 eval json 已存在就跳过
    if os.path.isfile(task.eval_json):
        return task, 0

    cmd = make_eval_cmd(task, gpu_id)
    rc = run_shell(cmd, task.eval_log)
    return task, rc

def worker_train(gpu_queue, task: Task):
    gpu_id = gpu_queue.get()
    try:
        t, rc = train_one(task, gpu_id)
        if rc != 0:
            print(f"[TRAIN FAIL] GPU{gpu_id} {t.tag} rc={rc} (see {t.train_log})")
        else:
            print(f"[TRAIN OK]  GPU{gpu_id} {t.tag}")
    finally:
        gpu_queue.put(gpu_id)

def worker_eval(gpu_queue, task: Task):
    gpu_id = gpu_queue.get()
    try:
        t, rc = eval_one(task, gpu_id)
        if rc != 0:
            print(f"[EVAL FAIL] GPU{gpu_id} {t.tag} rc={rc} (see {t.eval_log})")
        else:
            print(f"[EVAL OK]  GPU{gpu_id} {t.tag}")
    finally:
        gpu_queue.put(gpu_id)


# =========================
# Summary writer
# =========================

def write_summary_per_dataset(tasks: List[Task]):
    # 分组：dataset -> tasks
    by_ds: Dict[str, List[Task]] = {}
    for t in tasks:
        by_ds.setdefault(t.dataset_name, []).append(t)

    header = [
        "dataset","position","reft_layer","attack_layer","eps_r_train","out_dir",
        "acc","rob_acc","asr_cond","asr_overall",
        "n_initially_correct","n_correct_after_attack","n_flipped_C_to_W","n_flipped_W_to_C",
        "n_correct_after_attack_C_to_C","n_stayed_wrong_W_to_W",
        "train_log","eval_log","eval_json"
    ]

    for ds_name, ds_tasks in by_ds.items():
        dataset_out = os.path.join(OUT_ROOT, ds_name)
        ensure_dir(dataset_out)

        summary_csv = os.path.join(dataset_out, "summary.csv")
        with open(summary_csv, "w") as f:
            f.write(",".join(header) + "\n")
            for t in ds_tasks:
                d = parse_eval_json(t.eval_json)
                row = [
                    t.dataset_name, t.position_str, str(t.reft_layer), str(t.attack_layer),
                    f"{t.eps_r:.8f}", t.out_dir,
                    "" if d.get("acc") is None else f"{d['acc']:.6f}",
                    "" if d.get("rob_acc") is None else f"{d['rob_acc']:.6f}",
                    "" if d.get("asr_cond") is None else f"{d['asr_cond']:.6f}",
                    "" if d.get("asr_overall") is None else f"{d['asr_overall']:.6f}",
                    "" if d.get("n_initially_correct") is None else str(d["n_initially_correct"]),
                    "" if d.get("n_correct_after_attack") is None else str(d["n_correct_after_attack"]),
                    "" if d.get("n_flipped_C_to_W") is None else str(d["n_flipped_C_to_W"]),
                    "" if d.get("n_flipped_W_to_C") is None else str(d["n_flipped_W_to_C"]),
                    "" if d.get("n_correct_after_attack_C_to_C") is None else str(d["n_correct_after_attack_C_to_C"]),
                    "" if d.get("n_stayed_wrong_W_to_W") is None else str(d["n_stayed_wrong_W_to_W"]),
                    t.train_log, t.eval_log, t.eval_json
                ]
                f.write(",".join(row) + "\n")
        print(f"[Summary] wrote: {summary_csv} (rows={len(ds_tasks)})")


# =========================
# Main
# =========================

def main():
    tasks = build_all_tasks()
    print(f"[Launcher] total tasks = {len(tasks)} (datasets={len(DATASETS)}, layers={len(LAYER_PAIRS)}, positions={len(POSITIONS)})")
    print("[eps check]", {f"l{k}": eps_for_k(k) for k in POSITIONS})

    ensure_dir(LOG_ROOT)
    ensure_dir(OUT_ROOT)

    manager = Manager()
    gpu_queue = manager.Queue()
    for g in GPUS:
        gpu_queue.put(g)

    # -------- 1) Train stage --------
    print("\n=== Stage 1: TRAIN ===")
    pool = Pool(processes=MAX_WORKERS)
    for t in tasks:
        pool.apply_async(worker_train, args=(gpu_queue, t))
    pool.close()
    pool.join()

    # -------- 2) Eval stage --------
    print("\n=== Stage 2: EVAL ===")
    pool = Pool(processes=MAX_WORKERS)
    for t in tasks:
        pool.apply_async(worker_eval, args=(gpu_queue, t))
    pool.close()
    pool.join()

    # -------- 3) Summary --------
    write_summary_per_dataset(tasks)
    print("[Done] all finished.")

if __name__ == "__main__":
    main()