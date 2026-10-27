"""Stage 2: run the trained model on ONE chest X-ray -> organs + ranked disease probabilities.

  python -m radexplain.predict path/to/xray.png [--name phase1] [--json out.json]

In code (this is what the report generator and the web UI will call):
  p = Predictor();  result = p.predict("xray.png")        # path, PIL image, or uploaded file
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from radexplain.config import ROOT, load_config
from radexplain.data.dataset import build_transform
from radexplain.models.io import load_model
from radexplain.train import pick_device


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Predictor:
    def __init__(self, checkpoint="checkpoints/phase1/best.pt", device=None):
        ckpt = Path(checkpoint)
        ckpt = ckpt if ckpt.is_absolute() else ROOT / ckpt
        self.device = device or pick_device()
        self.model, ck = load_model(ckpt, self.device)
        self.organ_labels = ck["organ_labels"]
        self.crop = ck["image_size"]
        self.tf = build_transform(train=False, size=ck["image_size"])
        self.store_size = load_config()["preprocess"]["image_size"]   # same resize as training data
        pp = ckpt.parent / "postprocess.json"
        self.post = json.loads(pp.read_text()) if pp.exists() else None
        if self.post is None:
            print("note: postprocess.json not found - run `python -m radexplain.evaluate` for "
                  "calibrated probabilities and tuned thresholds (using raw probabilities, threshold 0.5)")

    def _prob_thr(self, kind, name, logit):
        cfg = (self.post or {}).get(kind, {}).get(name, {"a": 1.0, "b": 0.0, "threshold": 0.5})
        thr = float(cfg["threshold"])
        hi = float(cfg.get("high_threshold", min(1.0, 2 * thr)))
        return float(_sigmoid(cfg["a"] * logit + cfg["b"])), thr, hi

    def preprocess(self, image):
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = img.convert("L").resize((self.store_size, self.store_size), Image.LANCZOS).convert("RGB")
        return self.tf(img).unsqueeze(0)

    @torch.no_grad()
    def predict(self, image) -> dict:
        out = self.model(self.preprocess(image).to(self.device))
        organ_logits = out["organ"][0].float().cpu().numpy()
        dis_logits = out["disease"][0].float().cpu().numpy()

        result = {"organs": {}, "findings": {}}
        k = 0
        for i, (organ, labels) in enumerate(self.organ_labels.items()):
            p, thr, hi = self._prob_thr("organ", organ, organ_logits[i])
            result["organs"][organ] = {"probability": round(p, 4), "threshold": round(thr, 4),
                                       "high_threshold": round(hi, 4), "affected": p >= thr}
            items = []
            for lab in labels:
                p, thr, hi = self._prob_thr("disease", lab, dis_logits[k]); k += 1
                items.append({"finding": lab, "probability": round(p, 4), "threshold": round(thr, 4),
                              "high_threshold": round(hi, 4), "flagged": p >= thr})
            result["findings"][organ] = sorted(items, key=lambda d: d["probability"], reverse=True)
        return result


    def explain(self, image, result: dict, max_findings: int = 3):
        """Soft shading over the X-ray for the flagged findings. -> {"jpeg": bytes, "legend": [...]} or None."""
        from radexplain.models.cam import class_activation_maps
        from radexplain.report.overlay import flagged_findings, render_overlay
        items = flagged_findings(result, max_findings)
        if not items:
            return None
        img = image if isinstance(image, Image.Image) else Image.open(image)
        maps = class_activation_maps(self.model, self.preprocess(img).to(self.device))
        for it in items:
            it["cam"] = maps[it["finding"]]
        base = img.convert("L")
        base.thumbnail((900, 900))
        return render_overlay(base, items, store=self.store_size, crop=self.crop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--name", default="phase1")
    ap.add_argument("--top", type=int, default=5, help="findings to show per organ")
    ap.add_argument("--json", help="also save the full result to this file")
    args = ap.parse_args()

    res = Predictor(f"checkpoints/{args.name}/best.pt").predict(args.image)
    for organ, info in res["organs"].items():
        flag = "AFFECTED" if info["affected"] else "no clear abnormality"
        print(f"\n{organ.upper()}: {info['probability']:.1%} organ-level score -> {flag}")
        for r, f in enumerate(res["findings"][organ][:args.top], 1):
            mark = "  <-- flagged" if f["flagged"] else ""
            print(f"  {r}. {f['finding']:20s} {f['probability']:6.1%}{mark}")
    print("\nResearch prototype - not a medical device.")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2)); print(f"saved {args.json}")


if __name__ == "__main__":
    main()
