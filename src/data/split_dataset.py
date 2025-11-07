import json, os
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def build_splits(dataset, seed, out_path, ratios=(0.90,0.05,0.05)):
    assert abs(sum(ratios)-1.0) < 1e-6, "Ratios must sum to 1.0"
    ds_all = load_dataset(dataset)['train']
    N = len(ds_all)
    
    labels = np.array([int(ds_all[i]["clf_label"]) for i in range(N)])
    all_idx = np.arange(N)
    
    # 先切出 Attack
    rest_idx, attack_idx, y_rest, y_attack = train_test_split(
        all_idx, labels, test_size=ratios[2], random_state=seed, stratify=labels
    )
    # 再切出 Probe
    # probe 占总体 ratios[2]，在剩余 rest 里比例应为 ratios[1]/(1-ratios[2])
    probe_ratio_in_rest = ratios[1] / (1 - ratios[2])
    ft_train_idx, probe_idx, y_train, y_probe = train_test_split(
        rest_idx, y_rest, test_size=probe_ratio_in_rest, random_state=seed+1, stratify=y_rest
    )
    
    splits = {
        "ft_train": ft_train_idx.tolist(),
        "probe":    probe_idx.tolist(),
        "attack":   attack_idx.tolist(),
        "seed":     seed,
        "ratios":   ratios
    }
    with open(out_path, "w") as f:
        json.dump(splits, f)
    print(f"Saved splits to {out_path} | sizes:",
          {k: len(v) for k, v in splits.items() if isinstance(v, list)})

if __name__ == "__main__":
    dataset = "AlignmentResearch/EnronSpam"
    seed = 42
    out_path = "src/data/enron_splits.json"
    build_splits(dataset, seed, out_path)