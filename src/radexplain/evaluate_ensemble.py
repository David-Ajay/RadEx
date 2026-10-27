"""Do several trained models together beat the best single model? No training needed.

  python -m radexplain.evaluate_ensemble --runs pleura phase1            # also try:  --tta   or add  xray_chex
  --tta   average each model over 5 views (centre + 4 corner crops): slower, often a little better

For every non-empty subset of the runs, each finding's raw score (logit) is averaged over the models in the subset.
We print mean AUROC on VALIDATION and on TEST. Pick the subset by VALIDATION (choosing on test would flatter the
result): the one marked CHOSEN is that subset, and its test numbers are the honest ones.
"""
import argparse
import itertools
import json
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from radexplain.config import ROOT, load_config, path
from radexplain.data.dataset import CXRDataset
from radexplain.models.io import load_model
from radexplain.train import pick_device


def views(x, crop=224):
    """x: (B, 3, S, S) -> centre crop + 4 corner crops, each (B, 3, crop, crop)."""
    S = x.shape[-1]
    c = (S - crop) // 2
    out = [x[:, :, c:c + crop, c:c + crop]]
    out += [x[:, :, t:t + crop, l:l + crop] for t, l in ((0, 0), (0, S - crop), (S - crop, 0), (S - crop, S - crop))]
    return out


def mean_auroc(logit, y, m):
    aucs = []
    for j in range(logit.shape[1]):
        keep = (m[:, j] > 0) & ~np.isnan(logit[:, j])
        if keep.sum() > 1 and len(np.unique(y[keep, j])) == 2:
            aucs.append(roc_auc_score(y[keep, j], logit[keep, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def combo_logits(arrays, idx):
    """Average the chosen models' logits per finding (a model without that finding is simply skipped)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack([arrays[i] for i in idx]), axis=0)


def rank_combos(logits: dict, yv, mv, yt, mt):
    """logits: {run: (val_logits, test_logits)}. Returns every subset, best validation AUROC first."""
    names, rows = list(logits), []
    for r in range(1, len(names) + 1):
        for sub in itertools.combinations(range(len(names)), r):
            rows.append({"models": tuple(names[i] for i in sub),
                         "val": mean_auroc(combo_logits([logits[names[i]][0] for i in sub], range(len(sub))), yv, mv),
                         "test": mean_auroc(combo_logits([logits[names[i]][1] for i in sub], range(len(sub))), yt, mt)})
    return sorted(rows, key=lambda d: -d["val"])


@torch.no_grad()
def collect(models, labels, ds, device, tta, workers):
    """-> {run: (N, n_labels) logits aligned to `labels` (NaN where the model lacks a label)}"""
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=workers, persistent_workers=workers > 0)
    cols = {r: [m.organ_labels_flat.index(l) if l in m.organ_labels_flat else -1 for l in labels] for r, m in models.items()}
    out = {r: [] for r in models}
    for b in tqdm(dl, desc=f"{len(ds):,} images x {len(models)} models" + (" x 5 views" if tta else "")):
        x = b["image"].to(device)
        for r, m in models.items():
            vs = views(x) if tta else [x]
            lg = torch.stack([m(v)["disease"].float() for v in vs]).mean(0).cpu().numpy()
            arr = np.full((lg.shape[0], len(labels)), np.nan, np.float32)
            for j, c in enumerate(cols[r]):
                if c >= 0:
                    arr[:, j] = lg[:, c]
            out[r].append(arr)
    return {r: np.concatenate(v) for r, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names, e.g. pleura phase1 (the first one defines the finding list)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--limit", type=int, help="use only N val/test images (quick test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = pick_device()
    models, cks = {}, {}
    for r in args.runs:
        models[r], cks[r] = load_model(ROOT / "checkpoints" / r / "best.pt", device)
        models[r].organ_labels_flat = [l for labs in cks[r]["organ_labels"].values() for l in labs]
    ref = cks[args.runs[0]]
    master = pd.read_csv(path(load_config(), "master_csv"))
    size = 256 if args.tta else ref["image_size"]
    data = {sp: CXRDataset(master, sp, ref["organ_labels"], train=False, size=size, limit=args.limit) for sp in ("val", "test")}
    labels = data["val"].labels
    L = {sp: collect(models, labels, data[sp], device, args.tta, args.workers) for sp in data}
    ranked = rank_combos({r: (L["val"][r], L["test"][r]) for r in models}, data["val"].disease_y, data["val"].disease_mask,
                         data["test"].disease_y, data["test"].disease_mask)

    print(f"\nENSEMBLE CHECK ({'5-view TTA' if args.tta else 'single view'}), mean AUROC over {len(labels)} findings\n")
    print(f"{'models':40s}{'validation':>12s}{'test':>9s}")
    for i, row in enumerate(ranked):
        print(f"{' + '.join(row['models']):40s}{row['val']:>12.4f}{row['test']:>9.4f}" + ("   <-- CHOSEN (best on validation)" if i == 0 else ""))
    best = ranked[0]
    single = max((r for r in ranked if len(r["models"]) == 1), key=lambda d: d["val"])
    print(f"\nchosen test AUROC {best['test']:.4f} vs best single model ({single['models'][0]}, by validation) {single['test']:.4f}"
          f"  ->  {best['test'] - single['test']:+.4f}")
    idx = [list(models).index(n) for n in best["models"]]
    t = combo_logits([L["test"][r] for r in models], idx)
    print("\nper-finding TEST AUROC of the chosen combination:")
    per = {}
    for j, lab in enumerate(labels):
        keep = (data["test"].disease_mask[:, j] > 0) & ~np.isnan(t[:, j])
        if len(np.unique(data["test"].disease_y[keep, j])) == 2:
            per[lab] = float(roc_auc_score(data["test"].disease_y[keep, j], t[keep, j]))
            print(f"  {lab:22s}{per[lab]:.4f}")
    out = ROOT / "outputs" / "ensemble"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ensemble_results.json").write_text(json.dumps({"tta": args.tta, "ranked": ranked, "chosen_per_finding_test_auroc": per}, indent=2))
    print(f"\nSaved {out / 'ensemble_results.json'}")


if __name__ == "__main__":
    main()
