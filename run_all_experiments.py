import os
import subprocess
import time
from multiprocessing import Pool, Manager

# ================= 配置区 =================

# 1. 可用 GPU
GPUS = [2, 2, 2, 2]
MAX_WORKERS = len(GPUS) # 并行任务数

# 2. 定义你要跑的 3 个数据集
DATASETS = {
    "Helpful": {
        "path": "AlignmentResearch/Helpful",
        "split_file": "src/data/helpful_splits.json",
        "train_split": "ft_train",
        "ckpt": "out/pythia410m/helpful/best_Helpful.pt",
        "circuit_base": "analysis/helpful_train_circuit" 
    },
    # "Harmless": {
    #     "path": "AlignmentResearch/Harmless",
    #     "split_file": "src/data/harmless_splits.json",
    #     "train_split": "ft_train",
    #     "ckpt": "out/pythia410m/harmless/best_Harmless.pt",
    #     "circuit_base": "analysis/harmless_train_circuit"
    # },
    "IMDB": {
        "path": "AlignmentResearch/IMDB",
        "split_file": "src/data/imdb_splits.json",
        "train_split": "ft_train",
        "ckpt": "out/pythia410m/imdb/best_IMDB.pt",
        "circuit_base": "analysis/imdb_train_circuit"
    }
}

# 3. 定义 Layer Grid (攻击层 vs ReFT 层)
# (Layer Idx, Attack Layer)
LAYER_PAIRS = [
    (1, 1),
    (6, 1), (6, 6),
    (11, 1), (11, 6), (11, 11),
    (16, 1), (16, 6), (16, 11), (16, 16),
    (21, 1), (21, 6), (21, 11), (21, 16), (21, 21)
]
TASKS = []

for d_name, d_cfg in DATASETS.items():
    for reft_l, attack_l in LAYER_PAIRS:
        
        task = {
            "dataset_name": d_name,
            "dataset_cfg": d_cfg,
            "reft_layer": reft_l,
            "attack_layer": attack_l
        }
        TASKS.append(task)

print(f"[Launcher] Generated {len(TASKS)} tasks across {len(DATASETS)} datasets.")

# ================= 执行引擎 =================

def run_task(args):
    """
    单个任务的执行逻辑
    """
    gpu_id, task_info = args
    d_name = task_info["dataset_name"]
    d_cfg = task_info["dataset_cfg"]
    reft_l = task_info["reft_layer"]
    att_l = task_info["attack_layer"]
    
    # 构造唯一 Tag (目录名)
    # 格式: Helpful_L16_A6_gcg1_top20
    tag = f"{d_name}_L{reft_l}_A{att_l}_gcg1_top20"
    out_dir = f"output_20circuits/{d_name}/{tag}"
    
    # 如果跑过了，跳过
    if os.path.exists(os.path.join(out_dir, "final_intervention")):
        print(f"[Skip] {tag} already exists.")
        return

    # 构造 Circuit Path
    # 你的需求：analysis/helpful_train_circuit/circuit_20.json
    # 所以这里固定拼接 "circuit_20.json"
    circuit_file = os.path.join(d_cfg["circuit_base"], "circuit_20.json")
    
    # 检查 circuit 文件是否存在，防止跑空
    if not os.path.exists(circuit_file):
        print(f"[Error] Circuit file not found: {circuit_file}")
        return

    # 构造命令 (使用 OmegaConf dotlist 语法: key=value)
    # 必须匹配 train_reft_adv.py 里的 cfg 结构
    cmd = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python", "-m", "src.training.train_reft_adv",
        "config=configs/train_reft.yaml",  # 基础配置
        
        # 数据集覆盖
        f"data.dataset={d_cfg['path']}",
        f"data.split_file={d_cfg['split_file']}",
        f"data.train_split={d_cfg['train_split']}",
        # f"data.input_field=content", # 如果三个数据集字段不同，要在 DATASETS 里定义并在这里引用
        # f"data.label_field=clf_label",
        
        # 模型覆盖
        f"model.load_baseline_ckpt={d_cfg['ckpt']}",
        f"model.layer_idx={reft_l}",
        f"train.attack_layer={att_l}",
        
        # 训练参数覆盖 (如果 yaml 里不是这些值，需要在这里强制覆盖)
        f"train.gcg_steps=1",
        f"train.circuit_top_k=20",
        
        # Circuit 相关覆盖
        f"train.circuit_path={circuit_file}",
        
        # 输出目录覆盖
        f"out.dir={out_dir}"
    ]
    
    cmd_str = " ".join(cmd)
    print(f"[Start GPU{gpu_id}] {tag}")
    
    # 这里的 log 是 launcher 的调度日志
    os.makedirs("logs_launcher", exist_ok=True)
    log_file = f"logs_launcher/{tag}.log"
    
    with open(log_file, "w") as f:
        subprocess.run(cmd_str, shell=True, stdout=f, stderr=f)
        
    print(f"[Finish GPU{gpu_id}] {tag}")

def worker_wrapper(queue, task_info):
    """
    从 Queue 获取 GPU ID，跑完归还
    """
    gpu_id = queue.get()
    try:
        run_task((gpu_id, task_info))
    except Exception as e:
        print(f"Task failed: {e}")
    finally:
        queue.put(gpu_id)

if __name__ == "__main__":
    # 使用 Manager Queue 管理 GPU 资源池
    manager = Manager()
    gpu_queue = manager.Queue()
    for g in GPUS:
        gpu_queue.put(g)
        
    # 启动进程池
    pool = Pool(processes=MAX_WORKERS)
    
    for task in TASKS:
        pool.apply_async(worker_wrapper, args=(gpu_queue, task))
        
    pool.close()
    pool.join()
    print("All experiments finished.")