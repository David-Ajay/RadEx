"""Stage 1 model: ONE shared backbone + organ-specific heads.

  image -> backbone -> pooled features (dim)
             |-> organ_heads[organ]   : 1 logit  = "is this organ affected?"
             |-> disease_heads[organ] : n logits = which diseases of that organ

forward() returns {"organ": (B,O), "disease": (B,L)} (raw logits; apply sigmoid for probabilities).
Disease columns are ordered organ by organ, matching CXRDataset.labels.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class XrayBackbone(nn.Module):
    """DenseNet121 features that expect what TorchXRayVision networks expect: ONE channel scaled to [-1024, 1024].

    Our dataset yields 3-channel ImageNet-normalised tensors (gray repeated). Channel 0 is recovered exactly with
    g = x0 * 0.229 + 0.485 (back to [0, 1]) and then (2g - 1) * 1024, so the same data pipeline feeds both kinds
    of backbone.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.register_buffer("scale", torch.tensor(0.229 * 2048.0))
        self.register_buffer("shift", torch.tensor((2 * 0.485 - 1) * 1024.0))

    def forward(self, x):
        return self.features(x[:, :1] * self.scale + self.shift)


def make_backbone(name: str, pretrained: bool, init_path=None):
    if name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        return m.features, 1024, True            # True -> needs ReLU after features
    if name == "densenet121_xray":                # same network, started from chest-X-ray weights (see scripts/11)
        feats = models.densenet121(weights=None).features
        feats.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            if not init_path or not Path(init_path).exists():
                raise FileNotFoundError(f"X-ray starting weights not found at '{init_path}'. "
                                        "Run: python scripts/11_get_xray_weights.py")
            feats.load_state_dict(torch.load(init_path, map_location="cpu"), strict=True)
        return XrayBackbone(feats), 1024, True
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(*list(m.children())[:-2]), 2048, False
    if name == "resnet18":                       # tiny, only for fast debugging
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(*list(m.children())[:-2]), 512, False
    raise ValueError(f"unknown backbone: {name}")


class RadExplainNet(nn.Module):
    def __init__(self, organ_labels: dict, backbone="densenet121", pretrained=True, dropout=0.2, init_path=None):
        super().__init__()
        self.organs = list(organ_labels)
        self.organ_labels = organ_labels
        self.backbone, dim, self._relu = make_backbone(backbone, pretrained, init_path)
        self.drop = nn.Dropout(dropout)
        self.organ_heads = nn.ModuleDict({o: nn.Linear(dim, 1) for o in self.organs})
        self.disease_heads = nn.ModuleDict({o: nn.Linear(dim, len(l)) for o, l in organ_labels.items()})

    def features(self, x):
        f = self.backbone(x)
        if self._relu:
            f = F.relu(f)
        return self.drop(F.adaptive_avg_pool2d(f, 1).flatten(1))

    def forward(self, x):
        f = self.features(x)
        organ = torch.cat([self.organ_heads[o](f) for o in self.organs], dim=1)
        disease = torch.cat([self.disease_heads[o](f) for o in self.organs], dim=1)
        return {"organ": organ, "disease": disease}
