"""Sanity check: Python, PyTorch + Apple GPU (MPS), disk space, Kaggle credentials."""
import shutil
import sys
from pathlib import Path

import torch

print("Python  :", sys.version.split()[0])
print("PyTorch :", torch.__version__)
print("MPS (Apple GPU) available:", torch.backends.mps.is_available())
free = shutil.disk_usage(Path.home()).free / 1e9
print(f"Free disk: {free:.0f} GB", "(OK)" if free > 120 else "(low - need ~120 GB for Phase 1 downloads)")
kj = Path.home() / ".kaggle" / "kaggle.json"
print("Kaggle token:", "found" if kj.exists() else "MISSING (see README step 3)")
if torch.backends.mps.is_available():
    x = torch.randn(2000, 2000, device="mps")
    print("MPS matmul test OK:", (x @ x).shape)
