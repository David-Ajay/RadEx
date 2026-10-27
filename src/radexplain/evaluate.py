"""Stage 2: evaluate a trained model on the TEST set.

Steps (all choices are made on the VALIDATION set, then applied untouched to test):
  1. Calibration: fit a per-label sigmoid (Platt scaling) so probabilities are honest.
     (Training used class weights, which inflates raw probabilities for rare findings.)
  2. Threshold per label: the cut-off maximising sensitivity + specificity - 1 (Youden's J).
  3. Report on test: AUROC (+95% bootstrap CI), AUPRC, sensitivity, specificity, Brier score.

Outputs:
  checkpoints/<name>/postprocess.json   calibration + thresholds (used by predict.py)
  outputs/<name>/test_metrics.json/.csv and roc_test.png

Usage:  python -m radexplain.evaluate [--name phase1] [--bootstrap 200]
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from radexplain.config import ROOT, load_config, path
from radexplain.data.dataset import CXRDataset
from radexplain.models.io import load_model
from radexplain.train import pick_device


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@torch.no_grad()
def collect(model, ds, device, batch_size=64, workers=4):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    persistent_workers=workers > 0)
    out = {k: {"logit": [], "y": [], "m": []} for k in ("disease", "organ")}
    for b in tqdm(dl, desc=f"predicting {len(ds):,} images"):
        o = model(b["image"].to(device))
        for k in out:
            out[k]["logit"].append(o[k].float().cpu().numpy())
            out[k]["y"].append(b[f"{k}_y"].numpy())
            out[k]["m"].append(b[f"{k}_mask"].numpy())
    return {k: {n: np.concatenate(v) for n, v in d.items()} for k, d in out.items()}


def fit_calibration(logit, y):
    """Platt scaling on one column. Returns (a, b): calibrated p = sigmoid(a*logit + b)."""
    if len(np.unique(y)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e4).fit(logit.reshape(-1, 1), y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return (a, b) if a > 0 else (1.0, 0.0)          # never flip the ranking


def youden_threshold(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    t = float(thr[np.argmax(tpr - fpr)])
    return min(max(t, 0.0), 1.0)


def high_threshold(y, p, thr, target_spec=0.90):
    """Stricter cut-off (~90% specificity on the val set). Used for the 'likely' tier in reports."""
    if len(np.unique(y)) < 2:
        return min(1.0, 2 * thr)
    t = float(np.quantile(p[y == 0], target_spec))
    return min(max(t, thr), 1.0)


def auroc_ci(y, p, n_boot, seed=0):
    if n_boot <= 0 or len(np.unique(y)) < 2:
        return None
    rng, n, vals = np.random.default_rng(seed), len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None


def evaluate_kind(kind, names, V, T, n_boot):
    """Calibrate + threshold on val, then score on test, for every column of one output kind."""
    rows, post, test_probs = [], {}, {}
    for j, name in enumerate(names):
        vk, tk = V[kind]["m"][:, j] > 0, T[kind]["m"][:, j] > 0
        vl, vy = V[kind]["logit"][vk, j], V[kind]["y"][vk, j]
        a, b = fit_calibration(vl, vy)
        thr = youden_threshold(vy, sigmoid(a * vl + b))
        hi = high_threshold(vy, sigmoid(a * vl + b), thr)
        post[name] = {"a": a, "b": b, "threshold": thr, "high_threshold": hi}

        tl, ty = T[kind]["logit"][tk, j], T[kind]["y"][tk, j]
        p_raw, p_cal = sigmoid(tl), sigmoid(a * tl + b)
        test_probs[name] = (ty, p_cal)
        row = {"name": name, "n_test": int(tk.sum()), "n_pos": int(ty.sum()), "threshold": round(thr, 4)}
        if len(np.unique(ty)) == 2:
            pred = p_cal >= thr
            tp, fn = int((pred & (ty == 1)).sum()), int((~pred & (ty == 1)).sum())
            tn, fp = int((~pred & (ty == 0)).sum()), int((pred & (ty == 0)).sum())
            row.update(auroc=round(float(roc_auc_score(ty, p_cal)), 4), auroc_ci=auroc_ci(ty, p_cal, n_boot),
                       auprc=round(float(average_precision_score(ty, p_cal)), 4),
                       sensitivity=round(tp / max(tp + fn, 1), 4), specificity=round(tn / max(tn + fp, 1), 4),
                       brier_raw=round(float(brier_score_loss(ty, p_raw)), 4),
                       brier_calibrated=round(float(brier_score_loss(ty, p_cal)), 4))
        rows.append(row)
    return rows, post, test_probs


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'':22s}{'n_pos':>7s}{'AUROC [95% CI]':>24s}{'AUPRC':>8s}{'thr':>7s}{'sens':>7s}{'spec':>7s}{'Brier raw->cal':>18s}")
    for r in rows:
        if "auroc" not in r:
            print(f"{r['name']:22s}{r['n_pos']:>7d}   (needs both classes in test set)")
            continue
        ci = f"[{r['auroc_ci'][0]:.3f}-{r['auroc_ci'][1]:.3f}]" if r["auroc_ci"] else ""
        print(f"{r['name']:22s}{r['n_pos']:>7d}{r['auroc']:>10.3f} {ci:>13s}{r['auprc']:>8.3f}{r['threshold']:>7.2f}"
              f"{r['sensitivity']:>7.2f}{r['specificity']:>7.2f}{r['brier_raw']:>10.3f} ->{r['brier_calibrated']:.3f}")
    aucs = [r["auroc"] for r in rows if "auroc" in r]
    if aucs:
        print(f"{'MEAN':22s}{'':>7s}{np.mean(aucs):>10.3f}")


def plot_roc(test_probs, out_file):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping ROC plot: pip install matplotlib)")
        return
    items = [(n, y, p) for n, (y, p) in test_probs.items() if len(np.unique(y)) == 2]
    if not items:
        return
    cols = 4
    rows = -(-len(items) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (n, y, p) in zip(axes.ravel(), items):
        fpr, tpr, _ = roc_curve(y, p)
        ax.axis("on"); ax.plot(fpr, tpr, lw=1.8); ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{n} (AUC {roc_auc_score(y, p):.2f})", fontsize=9)
        ax.set_xlabel("False positive rate", fontsize=7); ax.set_ylabel("True positive rate", fontsize=7)
    fig.tight_layout(); fig.savefig(out_file, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="phase1")
    ap.add_argument("--checkpoint", help="default: checkpoints/<name>/best.pt")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples for AUROC CI (0 = off)")
    ap.add_argument("--limit", type=int, help="evaluate on only N val/test images (quick test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ckpt_path = ROOT / (args.checkpoint or f"checkpoints/{args.name}/best.pt")
    out_dir = ROOT / "outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device: {device} | checkpoint: {ckpt_path}")

    model, ck = load_model(ckpt_path, device)
    master = pd.read_csv(path(load_config(), "master_csv"))
    mk = lambda split: CXRDataset(master, split, ck["organ_labels"], train=False,
                                  size=ck["image_size"], limit=args.limit)
    val_ds, test_ds = mk("val"), mk("test")
    V = collect(model, val_ds, device, workers=args.workers)
    T = collect(model, test_ds, device, workers=args.workers)

    organs = list(ck["organ_labels"])
    dis_rows, dis_post, dis_probs = evaluate_kind("disease", val_ds.labels, V, T, args.bootstrap)
    org_rows, org_post, _ = evaluate_kind("organ", organs, V, T, args.bootstrap)

    print_table("ORGAN-LEVEL ('is this organ affected?') - test set", org_rows)
    print_table("DISEASE-LEVEL - test set", dis_rows)

    post = {"method": "platt calibration + youden threshold (+ 90%-specificity high threshold), fit on validation set",
            "organ": org_post, "disease": dis_post}
    (ckpt_path.parent / "postprocess.json").write_text(json.dumps(post, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps({"organ": org_rows, "disease": dis_rows}, indent=2))
    pd.DataFrame(dis_rows).to_csv(out_dir / "test_metrics.csv", index=False)
    plot_roc(dis_probs, out_dir / "roc_test.png")
    print(f"\nSaved: {ckpt_path.parent / 'postprocess.json'}\n       {out_dir}/test_metrics.json, test_metrics.csv, roc_test.png")


if __name__ == "__main__":
    main()
