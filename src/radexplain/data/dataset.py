"""PyTorch Dataset over master.csv.

Each item gives:
  image         (3,224,224) float tensor (grayscale repeated to 3 channels, ImageNet-normalised)
  disease_y     (L,) 0/1 targets for every active disease label (NaN -> 0, ignored via mask)
  disease_mask  (L,) 1 where the label was really annotated, 0 where it was NaN
  organ_y       (O,) 1 if ANY label of that organ is positive ("organ affected")
  organ_mask    (O,) 1 if at least one label of that organ was annotated
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from radexplain.config import ROOT

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_transform(train: bool, size: int = 224):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if train:
        # NO horizontal flip: it would put the heart on the wrong side.
        return T.Compose([T.RandomCrop(size), T.RandomRotation(5),
                          T.ColorJitter(brightness=0.15, contrast=0.15), *norm])
    return T.Compose([T.CenterCrop(size), *norm])


def organ_targets(arr: np.ndarray, organ_labels: dict):
    """arr: (N, L) labels in organ order (1 / 0 / NaN). Returns organ_y, organ_mask, both (N, O)."""
    ys, ms, start = [], [], 0
    for labs in organ_labels.values():
        block = arr[:, start:start + len(labs)]
        start += len(labs)
        ys.append((block == 1).any(axis=1).astype(np.float32))
        ms.append((~np.isnan(block)).any(axis=1).astype(np.float32))
    return np.stack(ys, axis=1), np.stack(ms, axis=1)


class CXRDataset(Dataset):
    def __init__(self, master, split, organ_labels, train=False, size=224, limit=None, seed=0):
        df = master[master["split"] == split]
        if limit:
            df = df.sample(min(limit, len(df)), random_state=seed)
        self.organ_labels = organ_labels
        self.labels = [l for labs in organ_labels.values() for l in labs]
        arr = df[self.labels].to_numpy(dtype=np.float32)
        self.disease_mask = (~np.isnan(arr)).astype(np.float32)
        self.disease_y = np.nan_to_num(arr, nan=0.0).astype(np.float32)
        self.organ_y, self.organ_mask = organ_targets(arr, organ_labels)
        self.paths = [ROOT / p for p in df["processed_path"]]
        self.image_ids = df["image_id"].tolist()
        self.tf = build_transform(train, size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return {
            "image": self.tf(img),
            "disease_y": torch.from_numpy(self.disease_y[i]),
            "disease_mask": torch.from_numpy(self.disease_mask[i]),
            "organ_y": torch.from_numpy(self.organ_y[i]),
            "organ_mask": torch.from_numpy(self.organ_mask[i]),
        }
