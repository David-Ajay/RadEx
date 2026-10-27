"""Class Activation Maps (CAM): which image regions pushed a finding's score up.

Our network ends in  features -> global average pool -> linear head, so the CAM is EXACT (no gradients needed):
    logit_k = bias_k + mean over the feature map of  ( sum_c W[k, c] * A_c(x, y) )
The inner part, sum_c W[k, c] * A_c(x, y), is the map returned here, one value per feature-map cell
(7x7 for a 224-pixel input). Positive values = evidence FOR the finding.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def class_activation_maps(model, x) -> dict:
    """x: (1, 3, H, W). Returns {finding_name: (h, w) numpy array, "organ:<organ>": (h, w) array}."""
    model.eval()
    f = model.backbone(x)
    if model._relu:
        f = F.relu(f)
    a = f[0]                                                    # (C, h, w)
    maps = {}
    for organ, labels in model.organ_labels.items():
        cams = torch.einsum("nc,chw->nhw", model.disease_heads[organ].weight, a).float().cpu().numpy()
        for lab, cam in zip(labels, cams):
            maps[lab] = cam
        maps[f"organ:{organ}"] = torch.einsum("c,chw->hw", model.organ_heads[organ].weight[0], a).float().cpu().numpy()
    return maps
