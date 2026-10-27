"""Build ONE master CSV from the raw NIH ChestX-ray14 and CheXpert label files.

Output columns:
  image_id, dataset, patient_id, split, src_path, processed_path, <one column per unified label>
Label values: 1.0 positive, 0.0 negative, NaN = not annotated / uncertain-ignored.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from radexplain.config import ROOT, load_config, path
from radexplain.labels import (CHEXPERT_ANNOTATED, NIH_ANNOTATED, NIH_RENAME,
                               UNIFIED_LABELS)


def patient_split(df: pd.DataFrame, val_fraction: float, seed: int) -> pd.Series:
    """'train'/'val' assignment with no patient appearing in both."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    _, val_idx = next(gss.split(df, groups=df["patient_id"]))
    out = pd.Series("train", index=df.index)
    out.iloc[val_idx] = "val"
    return out


def build_nih(raw: Path, cfg: dict) -> pd.DataFrame:
    entry = next(raw.rglob("Data_Entry_2017*.csv"))
    df = pd.read_csv(entry)
    # filename -> full path (images live in images_001/images/, ... or a flat folder)
    index = {p.name: p for p in raw.rglob("*.png")}
    df = df[df["Image Index"].isin(index)].copy()
    if df.empty:
        raise SystemExit(f"No NIH images found under {raw}. Did the download/unzip finish?")

    out = pd.DataFrame({
        "image_id": "nih_" + df["Image Index"].str.replace(".png", "", regex=False),
        "dataset": "nih",
        "patient_id": "nih_" + df["Patient ID"].astype(str),
        "src_path": df["Image Index"].map(index).astype(str),
    })
    findings = df["Finding Labels"].str.split("|")
    findings = findings.apply(lambda xs: {NIH_RENAME.get(x, x) for x in xs})
    for lab in UNIFIED_LABELS:
        out[lab] = (findings.apply(lambda s, l=lab: float(l in s)).values
                    if lab in NIH_ANNOTATED else np.nan)

    # Official test_list.txt is the test set; split val off train_val by patient
    test_names = set(next(raw.rglob("test_list.txt")).read_text().split())
    is_test = df["Image Index"].isin(test_names).values
    out["split"] = "test"
    tv = out[~is_test]
    out.loc[tv.index, "split"] = patient_split(tv, cfg["split"]["val_fraction"], cfg["split"]["seed"])
    return out.reset_index(drop=True)


def _resolve_chexpert(root: Path, rel: str) -> Path:
    """Kaggle/Stanford copies differ in how the 'Path' column maps to disk."""
    parts = Path(rel).parts
    for cand in (root / rel, root / Path(*parts[1:]), root.parent / rel):
        if cand.exists():
            return cand
    return root / rel  # not found; filtered out by caller


def build_chexpert(raw: Path, cfg: dict) -> pd.DataFrame:
    train_csv = next(raw.rglob("train.csv"))
    valid_csv = next(raw.rglob("valid.csv"))
    root = train_csv.parent
    tr = pd.read_csv(train_csv).assign(_orig="train")
    va = pd.read_csv(valid_csv).assign(_orig="valid")
    df = pd.concat([tr, va], ignore_index=True)
    df = df[df["Frontal/Lateral"] == "Frontal"].copy()        # frontal views only

    df["src"] = df["Path"].map(lambda p: _resolve_chexpert(root, p))
    df = df[df["src"].map(lambda p: p.exists())].copy()
    if df.empty:
        raise SystemExit(f"No CheXpert images found under {raw}.")

    pid = df["Path"].str.extract(r"(patient\d+)")[0]
    study = df["Path"].str.extract(r"(study\d+)")[0]
    view = df["Path"].str.extract(r"(view\d+)")[0]
    out = pd.DataFrame({
        "image_id": "chexpert_" + pid + "_" + study + "_" + view,
        "dataset": "chexpert",
        "patient_id": "chexpert_" + pid,
        "src_path": df["src"].astype(str),
    }, index=df.index)

    unc = cfg["uncertain"]
    for lab in UNIFIED_LABELS:
        if lab not in CHEXPERT_ANNOTATED:
            out[lab] = np.nan
            continue
        col = df[lab].astype(float)                    # blank -> NaN already
        policy = unc.get(lab, unc["default"])
        col = col.where(col != -1.0, np.nan if policy == "ignore" else float(policy))
        out[lab] = col

    # official 'valid' (234 studies) -> test; split val off 'train' by patient
    is_valid = (df["_orig"] == "valid").values
    out["split"] = "test"
    tr_part = out[~is_valid]
    out.loc[tr_part.index, "split"] = patient_split(tr_part, cfg["split"]["val_fraction"], cfg["split"]["seed"])
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-root", help="override raw NIH folder")
    ap.add_argument("--chexpert-root", help="override raw CheXpert folder")
    ap.add_argument("--only", choices=["nih", "chexpert"], help="build a single dataset")
    args = ap.parse_args()

    cfg = load_config()
    parts = []
    if args.only in (None, "nih"):
        parts.append(build_nih(Path(args.nih_root) if args.nih_root else path(cfg, "raw_nih"), cfg))
    if args.only in (None, "chexpert"):
        parts.append(build_chexpert(Path(args.chexpert_root) if args.chexpert_root else path(cfg, "raw_chexpert"), cfg))

    master = pd.concat(parts, ignore_index=True)
    proc = path(cfg, "processed_images")
    master["processed_path"] = [
        str((proc / d / f"{i}.jpg").relative_to(ROOT)) for d, i in zip(master["dataset"], master["image_id"])]

    out = path(cfg, "master_csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out, index=False)

    print(f"Wrote {out}  ({len(master):,} images)")
    print(master.groupby(["dataset", "split"]).size().unstack(fill_value=0), "\n")
    pos = master.groupby("dataset")[UNIFIED_LABELS].mean().T.round(3)
    print("Positive rate per label (NaN = not annotated by that dataset):\n", pos)


if __name__ == "__main__":
    main()
