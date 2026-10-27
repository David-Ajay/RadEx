"""How well does the shading land where doctors drew boxes?

NIH ships BBox_List_2017.csv: radiologist-drawn boxes for ~900 images (Atelectasis, Cardiomegaly, Effusion,
Infiltrate, Mass, Nodule, Pneumonia, Pneumothorax). For every boxed image in the TEST split we compute the same
highlight the app would draw and report, per finding:
  pointing accuracy : does the strongest point of the highlight fall inside the box?   (chance level shown too)
  mask hit          : does the highlighted area touch the box at all?
  recall / precision / IoU : how much of the box is covered / how much of the highlight is inside the box.
Usage:  python -m radexplain.evaluate_localization --name pleura
"""
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from radexplain.config import ROOT, load_config, path
from radexplain.models.cam import class_activation_maps
from radexplain.predict import Predictor
from radexplain.report.overlay import highlight_region, thorax_roi

LABEL_MAP = {"Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Effusion": "Pleural Effusion",
             "Infiltrate": "Infiltration", "Infiltration": "Infiltration", "Mass": "Mass", "Nodule": "Nodule",
             "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax"}


def box_mask(box, size):
    gt = np.zeros((size, size), bool)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    gt[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return gt


def box_metrics(mask: np.ndarray, peak, box) -> dict:
    """mask: bool (S,S) highlight; peak: (row, col); box: (x0, y0, x1, y1) in the same coordinates."""
    gt = box_mask(box, mask.shape[0])
    inter, union = (mask & gt).sum(), (mask | gt).sum()
    x0, y0, x1, y1 = box
    return {"point_hit": float(x0 <= peak[1] <= x1 and y0 <= peak[0] <= y1), "mask_hit": float(inter > 0),
            "iou": float(inter / union) if union else 0.0,
            "precision": float(inter / mask.sum()) if mask.sum() else 0.0,
            "recall": float(inter / gt.sum()) if gt.sum() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="phase1")
    args = ap.parse_args()
    cfg = load_config()
    csv = next((ROOT / cfg["paths"]["raw_nih"]).rglob("BBox_List_2017.csv"), None)
    if csv is None:
        raise SystemExit("BBox_List_2017.csv not found in data/raw/nih")
    bb = pd.read_csv(csv).iloc[:, :6]
    bb.columns = ["image", "label", "x", "y", "w", "h"]
    master = pd.read_csv(path(cfg, "master_csv"))
    master = master[(master["dataset"] == "nih") & (master["split"] == "test")].copy()
    master["key"] = master["image_id"].str.replace("nih_", "", n=1)
    bb["key"] = bb["image"].str.replace(".png", "", regex=False)
    rows = bb.merge(master[["key", "processed_path", "src_path"]], on="key")
    print(f"{len(rows)} boxes on {rows['key'].nunique()} test images (of {len(bb)} boxes in the NIH file)")

    pred = Predictor(f"checkpoints/{args.name}/best.pt")
    store, crop = pred.store_size, pred.crop
    roi_area = thorax_roi(store).sum()
    results = []
    for key, grp in tqdm(rows.groupby("key"), desc="images"):
        r0 = grp.iloc[0]
        maps = class_activation_maps(pred.model, pred.preprocess(Image.open(ROOT / r0["processed_path"])).to(pred.device))
        ow, oh = Image.open(r0["src_path"]).size
        for _, r in grp.iterrows():
            lab = LABEL_MAP.get(r["label"])
            if lab not in maps:
                continue
            region = highlight_region(maps[lab], store, crop)
            if region is None:
                continue
            box = (r["x"] * store / ow, r["y"] * store / oh, (r["x"] + r["w"]) * store / ow, (r["y"] + r["h"]) * store / oh)
            m = box_metrics(region["mask"], region["peak"], box)
            m.update(label=lab, area_pct=region["area_pct"],
                     chance=float((box_mask(box, store) & thorax_roi(store)).sum() / roi_area))
            results.append(m)
    df = pd.DataFrame(results)
    if df.empty:
        raise SystemExit("No boxes could be evaluated (does the model include these findings?)")
    agg = df.groupby("label").agg(n=("iou", "size"), pointing=("point_hit", "mean"), chance=("chance", "mean"),
                                  mask_hit=("mask_hit", "mean"), recall=("recall", "mean"),
                                  precision=("precision", "mean"), iou=("iou", "mean"),
                                  shaded_pct=("area_pct", "mean")).round(3)
    print("\nLOCALIZATION vs doctor-drawn boxes (test images only)\n")
    print(agg.to_string())
    print(f"\nOVERALL ({len(df)} boxes): pointing accuracy {df['point_hit'].mean():.3f} "
          f"(chance {df['chance'].mean():.3f}) | mask hit {df['mask_hit'].mean():.3f} | "
          f"recall {df['recall'].mean():.3f} | precision {df['precision'].mean():.3f} | IoU {df['iou'].mean():.3f}")
    out = ROOT / "outputs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "localization.json").write_text(json.dumps({"per_finding": agg.reset_index().to_dict("records"),
                                                       "overall_pointing": float(df["point_hit"].mean()),
                                                       "overall_chance": float(df["chance"].mean())}, indent=2))
    print(f"\nSaved {out / 'localization.json'}")


if __name__ == "__main__":
    main()
