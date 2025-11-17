# train_gnet.py
import os, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
import re, random, yaml 
from pyexpat import model
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from zmq import device

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
class GNet(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): 
        return self.net(x).squeeze(-1)
    
def load_gnet(path: str, GNetCls, dim_in: int, hidden: int = 128, device: str = "cuda"):
    """
    读取你训练好的 gnet 权重，并冻结为评估模式。
    GNetCls 是你已有的 class GNet(in_dim, hidden=128)。
    """
    gnet = GNetCls(in_dim=dim_in, hidden=hidden)
    sd = torch.load(path, map_location="cpu")
    # 兼容不同保存方式
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        gnet.load_state_dict(sd, strict=True)
    except Exception:
        # 有些脚本会包一层 {"model": ...}
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            gnet.load_state_dict(sd["model"], strict=False)
        else:
            gnet.load_state_dict(sd, strict=False)

    gnet.to(device)
    gnet.eval()
    for p in gnet.parameters():
        p.requires_grad = False
    return gnet
    
def load_latent(base_dir, dataset, layer):
    lat = np.load(os.path.join(base_dir, f"{dataset}_L{layer}.npz"))
    return lat["X"].astype(np.float32), lat["y"].astype(np.float32)

def train_gnet_from_svm(X, margin, epochs = 30, lr=1e-3):
    # 拟合SVM margin作为标签，训练GNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = StandardScaler().fit(X)
    Xs = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
    ys = torch.tensor(margin, dtype=torch.float32, device=device)
    model = GNet(in_dim=X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        preds = model(Xs)
        loss = loss_fn(preds, ys)
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model, scaler

def fine_tune_with_labels(model, scaler, X, y, epochs=5, lr=1e-4):
    """
    使用真实标签微调GNet模型
    """
    device = next(model.parameters()).device
    Xs = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
    ys = torch.tensor(y, dtype=torch.float32, device=device)
    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xs)
        loss = bce(logits, ys)
        loss.backward()
        opt.step()
    return model

def evaluate(model, scaler, X, y):
    device = next(model.parameters()).device
    Xs = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
    ys = torch.tensor(y, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(Xs)
        preds = torch.sigmoid(logits).cpu().numpy()
    auc = roc_auc_score(y, preds)
    acc = accuracy_score(y, (preds > 0.5).astype(int))
    return acc, auc

if __name__ == "__main__":
    # === 1. 加载配置 ===
    with open("configs/baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    dataset = cfg["data"]["dataset"].replace("AlignmentResearch/", "")
    base_dir = os.path.join(cfg["out"]["dir"], "latents")

    # === 2. 自动检测有哪些层 ===
    layer_ids = []
    for fn in os.listdir(base_dir):
        if fn.startswith(dataset) and re.match(rf"{dataset}_L(\d+)\.npz", fn):
            layer_ids.append(int(re.findall(r"L(\d+)", fn)[0]))
    layer_ids = sorted(set(layer_ids))
    print(f"Detected {len(layer_ids)} layers: {layer_ids}")

    # === 3. 逐层训练 g-net ===
    all_results = {}
    for l in layer_ids:
        print(f"\n=== [Layer {l}] ===")
        X, y = load_latent(base_dir, dataset, l)
        if X is None:
            print(f"  Skip L{l}: latent file not found.")
            continue
        svm_path = os.path.join(base_dir, f"{dataset}_L{l}_svm_margin.npz")
        model, scaler = None, None

        # ---- Stage 1: SVM蒸馏 or 直接训练 ----
        if os.path.exists(svm_path):
            print("  Stage 1: distill from SVM margin")
            margin = np.load(svm_path)["margin"].astype(np.float32)
            model, scaler = train_gnet_from_svm(X, margin)
        else:
            print("  Stage 1: train from labels (no SVM margin)")
            scaler = StandardScaler().fit(X)
            Xs = torch.tensor(scaler.transform(X), dtype=torch.float32)
            ys = torch.tensor(y, dtype=torch.float32)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = GNet(X.shape[1]).to(device)
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            bce = nn.BCEWithLogitsLoss()
            for ep in range(30):
                opt.zero_grad()
                loss = bce(model(Xs.to(device)), ys.to(device))
                loss.backward()
                opt.step()
                if (ep + 1) % 10 == 0:
                    print(f"    [Direct] Epoch {ep+1}/30 loss={loss.item():.4f}")

        # ---- Stage 2: Label fine-tune ----
        print("  Stage 2: fine-tuning on true labels")
        model = fine_tune_with_labels(model, scaler, X, y, epochs=5)

        # ---- Evaluate & Save ----
        acc, auc = evaluate(model, scaler, X, y)
        print(f"  Final -> AUC={auc:.3f}, ACC={acc:.3f}")
        outp = os.path.join(base_dir, f"{dataset}_L{l}_gnet.pt")
        torch.save({
            "model": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        }, outp)
        print(f"  [Saved] {outp}")

        all_results[l] = {"acc": float(acc), "auc": float(auc)}

    # === 4. 汇总结果 ===
    out_json = os.path.join(base_dir, f"{dataset}_gnet_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== Done: saved summary to {out_json} ===")
    