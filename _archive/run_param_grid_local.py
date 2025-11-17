#!/usr/bin/env python3
import os, subprocess, csv, time, argparse, math

parser = argparse.ArgumentParser()
parser.add_argument("--param_grid", type=str, default="param_grid.tsv")
parser.add_argument("--start", type=int, default=2)   # skip header
parser.add_argument("--end", type=int, default=0)
parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
parser.add_argument("--max_concurrent", type=int, default=8)
args = parser.parse_args()

gpus = [int(x) for x in args.gpus.split(",")]
n_gpus = len(gpus)

# read param grid rows
with open(args.param_grid, "r") as f:
    first = f.read(4096); f.seek(0)
    delim = '\t' if '\t' in first else ','
    reader = list(csv.DictReader(f, delimiter=delim))
rows = reader

start = args.start
end = args.end if args.end>0 else (len(rows)+1)  # rows are header removed
# rows in file correspond to line numbers 2.. so array_idx = i+1

# build commands
tasks = []
for i, row in enumerate(rows, start=1):  # i is 1-based for rows list
    lineno = i+1  # actual file line number
    if lineno < start or lineno > end:
        continue
    array_idx = lineno - 1
    outdir = f"out/pythia410m/harmless/reft_att_heatmap/param_grid_task_{array_idx}"
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        "python", "-m", "src.training.trainer_reft_lat2",
        "--config", "configs/train_reft.yaml",
        "--param_grid", args.param_grid,
        "--array_idx", str(array_idx),
        "--out_dir", outdir
    ]
    tasks.append((array_idx, cmd, outdir))

# simple scheduler: launch tasks and cap concurrency
running = {}
next_gpu = 0
max_concurrent = args.max_concurrent
for task in tasks:
    while len(running) >= max_concurrent:
        # poll running processes
        time.sleep(5)
        for pid, info in list(running.items()):
            p = info["proc"]
            if p.poll() is not None:
                print(f"Task {info['array_idx']} finished (exit {p.returncode})")
                del running[pid]

    array_idx, cmd, outdir = task
    gpu = gpus[next_gpu % n_gpus]; next_gpu += 1
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    stdout = open(os.path.join(outdir, "stdout.log"), "ab")
    stderr = open(os.path.join(outdir, "stderr.log"), "ab")
    print(f"LAUNCH array_idx={array_idx} -> GPU {gpu}: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
    running[p.pid] = {"proc": p, "array_idx": array_idx}
    time.sleep(1)

# wait for all to finish
while running:
    time.sleep(5)
    for pid, info in list(running.items()):
        if info["proc"].poll() is not None:
            print(f"Task {info['array_idx']} finished (exit {info['proc'].returncode})")
            del running[pid]
print("All tasks done.")

# python run_param_grid_local.py --param_grid param_grid.tsv --start 2 --end 2 --gpus "3" --max_concurrent 4
