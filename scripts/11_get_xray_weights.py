"""One-time download of chest-X-ray starting weights (needs: pip install torchxrayvision).

Saves ONLY the feature extractor (no 18-class head) to checkpoints/pretrained/<name>_features.pt.

Which weights are safe for an honest NIH test score?  The weights must never have seen NIH images:
    densenet121-res224-chex      CheXpert only            (default, recommended)
    densenet121-res224-mimic_ch  MIMIC-CXR only
    densenet121-res224-pc        PadChest only
    NOT 'nih' or 'all': they were trained on NIH ChestX-ray14, so testing on NIH would look better than it is.
Usage:  python scripts/11_get_xray_weights.py [weights-name]
"""
import sys
from pathlib import Path

import torch

name = sys.argv[1] if len(sys.argv) > 1 else "densenet121-res224-chex"
if name.endswith("-nih") or name.endswith("-all"):
    print(f"WARNING: '{name}' saw NIH images in training; test scores on NIH would be inflated.")
try:
    import torchxrayvision as xrv
except ImportError:
    raise SystemExit("Please run first:  pip install torchxrayvision")

model = xrv.models.DenseNet(weights=name)          # downloads ~30 MB to ~/.torchxrayvision on first use
feats = {k[len("features."):]: v for k, v in model.state_dict().items() if k.startswith("features.")}
out = Path(__file__).resolve().parents[1] / "checkpoints" / "pretrained" / f"{name}_features.pt"
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(feats, out)
print(f"Saved {len(feats)} tensors to {out}")
