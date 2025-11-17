#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, time, subprocess, glob

def newest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="out1", help="训练输出的根目录（含 param_grid_task_*）")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--baseline_ckpt", type=str, required=True,
                        help="基线模型 ckpt 路径（例如 full FT 的 best.pt）")
    parser.add_argument("--dataset", type=str, default="AlignmentResearch/Harmless")
    parser.add_argument("--splits_json", type=str, default="src/data/enron_splits.json")
    parser.add_argument("--split", type=str, default="attack")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_attack_tokens", type=int, default=5)
    parser.add_argument("--attack_start", type=int, default=0)
    parser.add_argument("--beam_k", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_concurrent", type=int, default=8)
    parser.add_argument("--start_task", type=int, default=1, help="起始 param_grid_task_ 序号（含）")
    parser.add_argument("--end_task", type=int, default=10**9, help="结束 param_grid_task_ 序号（含）")
    
    parser.add_argument("--eval_seeds", type=str, default="42", 
                        help="用于评估的随机种子列表, 逗号分隔 (例如: '1234,2000,3000')")
    parser.add_argument("--eval_script", type=str, default="src.attack.eval_spam_gcg",
                        help="评估脚本的 Python 模块路径")
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()!=""]
    if not gpus:
        print("No GPUs provided via --gpus", file=sys.stderr); sys.exit(1)
        
    eval_seeds = [s.strip() for s in args.eval_seeds.split(",") if s.strip()]
    if not eval_seeds:
        print("No evaluation seeds provided via --eval_seeds", file=sys.stderr); sys.exit(1)

    # 构建所有待攻击子任务：扫描 param_grid_task_*/子目录/ 里的 reft_lat_*.pt
    tasks = []
    # 路径匹配 '.../param_grid_task_*/seed*.../'
    task_dirs_pattern = os.path.join(args.out_root, "param_grid_task_*", "seed*")
    task_dirs = sorted(glob.glob(task_dirs_pattern))
    
    for sdir in task_dirs: # sdir 现在是 '.../seed42_R..._L1_A1_.../'
        if not os.path.isdir(sdir):
            continue
            
        # 过滤 param_grid_task_N 范围
        tdir = os.path.dirname(sdir) # '.../param_grid_task_N'
        base = os.path.basename(tdir)
        try:
            n = int(base.rsplit("_", 1)[-1])
        except Exception:
            print(f"警告: 无法从 {base} 解析 task 编号, 跳过")
            continue
        if not (args.start_task <= n <= args.end_task):
            continue

        # 在 sdir 中找到 adv_ckpt
        adv_ckpt = newest_file(os.path.join(sdir, "reft_lat_*.pt"))
        if adv_ckpt is None:
            # 没有训练产物，跳过
            print(f"警告: 在 {sdir} 中未找到 reft_lat_*.pt, 跳过")
            continue

        # 攻击输出目录（与训练目录并列放一个 attacks 子目录）
        atk_out = os.path.join(sdir, "attacks_check")
        os.makedirs(atk_out, exist_ok=True)
        
        # --- (新) 为每一个 seed 创建一个任务 ---
        for eval_seed in eval_seeds:
            
            # (新) 每个 seed 都有唯一的 JSON 输出路径
            out_json_path = os.path.join(atk_out, f"attack_results_seed{eval_seed}.json")

            cmd = [
                "python", "-u", "-m", args.eval_script,
                "--model_name", args.model_name,
                "--baseline_ckpt", args.baseline_ckpt,
                "--adv_ckpt", adv_ckpt,
                "--splits_json", args.splits_json,
                "--dataset_name", args.dataset, # <-- 修复: 使用 --dataset_name
                "--split", args.split,
                "--n_samples", str(args.n_samples),
                "--n_attack_tokens", str(args.n_attack_tokens),
                "--attack_start", str(args.attack_start),
                "--beam_k", str(args.beam_k),
                "--rounds", str(args.rounds),
                "--max_length", str(args.max_length),
                "--device", "cuda",
                "--seed", str(eval_seed), # <-- (新) 传递 seed
                "--out_json_path", out_json_path # <-- (新) 传递 JSON 输出路径
            ]
            tasks.append((n, sdir, atk_out, cmd, eval_seed))

    if not tasks:
        print("No attack tasks found. Check --out_root and directory layout.", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(tasks)} attack tasks ({len(task_dirs)} models x {len(eval_seeds)} seeds).")
    
    # 简单并发调度
    running = {}
    next_gpu = 0
    max_conc = min(args.max_concurrent, len(gpus), len(tasks))

    for (n, sdir, atk_out, cmd, eval_seed) in tasks:
        # 限流
        while len(running) >= max_conc:
            time.sleep(5)
            for pid, info in list(running.items()):
                p = info["proc"]
                if p.poll() is not None:
                    print(f"[DONE] task_dir={info['sdir']} (seed {info['seed']}) (exit {p.returncode})")
                    del running[pid]

        # 轮转分配 GPU
        gpu = gpus[next_gpu % len(gpus)]; next_gpu += 1
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
        # (新) 每个 seed 都有唯一的日志
        stdout_log = os.path.join(atk_out, f"stdout_seed{eval_seed}.log")
        stderr_log = os.path.join(atk_out, f"stderr_seed{eval_seed}.log")
        stdout = open(stdout_log, "w") # 'w' 覆盖旧日志
        stderr = open(stderr_log, "w")

        print(f"LAUNCH (GPU {gpu}) {sdir} (Seed: {eval_seed})")
        print("  CMD:", " ".join(cmd))
        p = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
        running[p.pid] = {"proc": p, "sdir": sdir, "seed": eval_seed}

        time.sleep(1)

    # 等所有结束
    while running:
        time.sleep(10)
        for pid, info in list(running.items()):
            if info["proc"].poll() is not None:
                print(f"[DONE] task_dir={info['sdir']} (seed {info['seed']}) (exit {info['proc'].returncode})")
                del running[pid]

    print("All attacks done.")

if __name__ == "__main__":
    main()
"""
python run_attacks_local.py \
  --out_root out/pythia410m/harmless/reft_att_heatmap \
  --baseline_ckpt out/pythia410m/harmless/best_Harmless.pt \
  --model_name EleutherAI/pythia-410m \
  --splits_json src/data/harmless_splits.json \
  --dataset AlignmentResearch/Harmless \
  --split attack \
  --n_samples 40 \
  --n_attack_tokens 10 \
  --beam_k 20 \
  --rounds 20 \
  --gpus "1,2,3" \
  --max_concurrent 4 \
  --start_task 1 \
  --end_task 2 \
  --eval_seeds "42"
"""