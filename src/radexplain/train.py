"""Train RadExplain (Stage 1). Examples:

  python -m radexplain.train --subset 2000 --epochs 1      # quick smoke test (writes to 'smoke')
  python -m radexplain.train                               # full run (writes to 'phase1')
  python -m radexplain.train --resume                      # continue an interrupted run
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # must be set before importing torch

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from radexplain.config import ROOT, load_config, path
from radexplain.data.dataset import CXRDataset
from radexplain.labels import active_organ_labels
from radexplain.models.losses import compute_pos_weight, masked_bce
from radexplain.models.radexplain_net import RadExplainNet


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def auroc_per_column(p, y, m):
    """AUROC for each column using only annotated rows; NaN if only one class present."""
    out = np.full(p.shape[1], np.nan)
    for j in range(p.shape[1]):
        keep = m[:, j] > 0
        if keep.sum() > 1 and len(np.unique(y[keep, j])) == 2:
            out[j] = roc_auc_score(y[keep, j], p[keep, j])
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P = {"organ": [], "disease": []}; Y = {"organ": [], "disease": []}; M = {"organ": [], "disease": []}
    for b in tqdm(loader, desc="val", leave=False):
        out = model(b["image"].to(device))
        for k in P:
            P[k].append(torch.sigmoid(out[k]).cpu().numpy())
            Y[k].append(b[f"{k}_y"].numpy()); M[k].append(b[f"{k}_mask"].numpy())
    return {k: auroc_per_column(np.concatenate(P[k]), np.concatenate(Y[k]), np.concatenate(M[k])) for k in P}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train.yaml")
    ap.add_argument("--subset", type=int, help="use only N training images (quick test)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--name", help="run name (default: 'smoke' with --subset, else 'phase1')")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr, mcfg, dcfg = cfg["train"], cfg["model"], cfg["data"]
    epochs = args.epochs or tr["epochs"]
    name = args.name or ("smoke" if args.subset else "phase1")
    ckpt_dir, out_dir = ROOT / "checkpoints" / name, ROOT / "outputs" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(tr["seed"]); np.random.seed(tr["seed"])
    device = pick_device()
    print(f"device: {device} | run: {name}")

    master = pd.read_csv(path(load_config(), "master_csv"))
    organ_labels = active_organ_labels(master)
    print("active labels:", {o: len(l) for o, l in organ_labels.items()})

    train_ds = CXRDataset(master, "train", organ_labels, train=True, size=dcfg["image_size"], limit=args.subset)
    val_ds = CXRDataset(master, "val", organ_labels, train=False, size=dcfg["image_size"],
                        limit=max(256, args.subset // 4) if args.subset else None)
    nw = dcfg["num_workers"]
    kw = dict(num_workers=nw, persistent_workers=nw > 0, pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=tr["batch_size"] * 2, shuffle=False, **kw)
    print(f"train {len(train_ds):,} | val {len(val_ds):,} images")

    init_path = ROOT / mcfg["init_path"] if mcfg.get("init_path") else None
    model = RadExplainNet(organ_labels, mcfg["backbone"], mcfg["pretrained"], mcfg["dropout"], init_path).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw_dis = compute_pos_weight(train_ds.disease_y, train_ds.disease_mask, tr["pos_weight_max"]).to(device)
    pw_org = compute_pos_weight(train_ds.organ_y, train_ds.organ_mask, tr["pos_weight_max"]).to(device)

    start_epoch, best, history = 0, -1.0, []
    if args.resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        print(f"resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for b in tqdm(train_dl, desc=f"epoch {epoch + 1}/{epochs}"):
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b["image"])
            loss = (tr["organ_loss_weight"] * masked_bce(out["organ"], b["organ_y"], b["organ_mask"], pw_org)
                    + tr["disease_loss_weight"] * masked_bce(out["disease"], b["disease_y"], b["disease_mask"], pw_dis))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item(); n += 1
        sched.step()
        speed = len(train_ds) / (time.time() - t0)

        auc = evaluate(model, val_dl, device)
        organ_auc, dis_auc = float(np.nanmean(auc["organ"])), float(np.nanmean(auc["disease"]))
        rec = {"epoch": epoch + 1, "train_loss": total / max(n, 1), "val_organ_auroc": organ_auc,
               "val_disease_auroc": dis_auc, "img_per_sec": speed,
               "per_label_auroc": {l: (None if np.isnan(a) else round(float(a), 4))
                                   for l, a in zip(train_ds.labels, auc["disease"])},
               "per_organ_auroc": {o: (None if np.isnan(a) else round(float(a), 4))
                                   for o, a in zip(model.organs, auc["organ"])}}
        history.append(rec)
        print(f"epoch {epoch + 1}: loss {rec['train_loss']:.4f} | organ AUROC {organ_auc:.3f} | "
              f"disease AUROC {dis_auc:.3f} | {speed:.0f} img/s")

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "epoch": epoch, "best": best, "history": history, "organ_labels": organ_labels,
              "backbone": mcfg["backbone"], "image_size": dcfg["image_size"]}
        if dis_auc > best:
            best = ck["best"] = dis_auc
            torch.save(ck, ckpt_dir / "best.pt")
        torch.save(ck, ckpt_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print("\nPer-label validation AUROC (last epoch):")
    for l, a in history[-1]["per_label_auroc"].items():
        print(f"  {l:22s} {a}")
    print(f"\nBest checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
