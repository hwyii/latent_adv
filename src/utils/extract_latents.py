import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from src.data.adv_dataset import AdvDataset
from src.data.Collator import Collator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

import numpy as np, json, yaml
import os

from src.training.trainer_reft_lat import DEVICE; os.environ["TOKENIZERS_PARALLELISM"] = "false"
with open("configs/baseline.yaml") as f:
    cfg = yaml.safe_load(f)
    
dataset = cfg["data"]["dataset"]
dataset = dataset.replace("AlignmentResearch/","")
device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
# 1. load model and tokenizer
model_name = cfg["model"]["name"]
ckpt_path = f"{cfg['out']['dir']}/best_{dataset}.pt"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

state = torch.load(ckpt_path, map_location=device)
model.config.pad_token_id = tokenizer.pad_token_id
model.load_state_dict(state, strict=True)
if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
    model.config.eos_token_id = tokenizer.eos_token_id

model.to(device)
model.eval()

# 2. load validation data
with open(cfg["data"]["split_file"], "r") as f:
    splits = json.load(f) 
ds = AdvDataset(splits["probe"], cfg["data"]["dataset"])
collate = Collator(model_name=model_name, max_length=512)
val_loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)

# 3. collect hidden states
all_hiddens = { }
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
        hiddens = outputs.hidden_states # tuple of (batch_size, seq_len, hidden_size)
        # len=num_layers+1, including input embeddings
        
        attn = batch['attention_mask'] # (bsz, seq_len)
        last_idx = attn.sum(dim=1)-1 # (bsz,) index of last non-padded token
        bsz = last_idx.size(0)
        arange = torch.arange(bsz, device=device)
        
        for l, h in enumerate(hiddens): # l is the layer index, h is (bsz, seq_len, hidden_size)
            
            sent = h[arange, last_idx, :] # (bsz, hidden_size) take the last token's hidden state
            
            all_hiddens.setdefault(l, []).append(sent.detach().cpu())
        all_labels.append(batch['labels'].detach().cpu())
        
        
# 4. concatenate and save
y = torch.cat(all_labels,dim=0).numpy().astype(np.int64).ravel()

for l in all_hiddens:
    all_hiddens[l] = torch.cat(all_hiddens[l], dim=0).numpy().astype(np.float32)
    #print(f"Layer {l}: shape={all_hiddens[l].shape}")
    

    
layer_scores = {}
layer_aucs = {}
us_vs = {}
for l, X in all_hiddens.items():
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            solver="saga", penalty="l2", C=1.0, max_iter=5000, n_jobs=-1
        )
    )
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_va)
    y_proba = pipe.predict_proba(X_va)[:, 1]
    acc = accuracy_score(y_va, y_pred)
    try:
        auc = roc_auc_score(y_va, y_proba)
    except ValueError:
        auc = np.nan

    # 取出最后一层 Logistic 的权重作为 (u, v)
    clf = pipe.named_steps["logisticregression"]
    u = clf.coef_.ravel().copy()
    v = float(clf.intercept_[0])

    layer_scores[l] = acc
    layer_aucs[l]   = auc
    us_vs[l] = (u, v)

    print(f"Layer {l:02d}: acc={acc:.3f}, auc={auc:.3f}, ||u||={np.linalg.norm(u):.3f}, v={v:.3f}")

best_layer = max(layer_scores, key=layer_scores.get)
print(f"\nBest layer by acc: {best_layer} (acc={layer_scores[best_layer]:.3f}, auc={layer_aucs[best_layer]:.3f})")
print(f"Result of dataset {dataset}.")
# 可选：保存 (u, v) 以在 LAT/对齐R时使用
np.savez(f"probe_uv_ft_val_{dataset}.npz", **{f"L{l}": np.concatenate([us_vs[l][0], np.array([us_vs[l][1]])]) for l in us_vs})