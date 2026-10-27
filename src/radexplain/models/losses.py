import numpy as np
import torch
import torch.nn.functional as F


def masked_bce(logits, targets, mask, pos_weight=None):
    """BCE that ignores entries where mask == 0 (labels the dataset never annotated)."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def compute_pos_weight(y: np.ndarray, mask: np.ndarray, max_weight: float = 20.0) -> torch.Tensor:
    """Per-label weight = negatives/positives (among annotated rows), clipped. Fights class imbalance."""
    pos = (y * mask).sum(0)
    neg = ((1 - y) * mask).sum(0)
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(w, 1.0, max_weight), dtype=torch.float32)
