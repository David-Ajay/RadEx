"""Run:  python tests/test_stage1.py     (or: pytest tests)"""
import numpy as np
import torch

from radexplain.data.dataset import organ_targets
from radexplain.models.losses import compute_pos_weight, masked_bce
from radexplain.models.radexplain_net import RadExplainNet

ORGANS = {"lungs": ["Atelectasis", "Edema"], "heart": ["Cardiomegaly"]}
NAN = np.nan


def test_organ_targets_and_masking():
    #                  Atel  Edema  Cardio
    arr = np.array([[1,    0,     0],      # lungs affected, heart clear
                    [0,    0,     1],      # lungs clear, heart affected
                    [NAN,  NAN,   0],      # lungs never annotated -> must be masked
                    [0,    NAN,   NAN]],   # lungs: one known negative -> clear; heart masked
                   dtype=np.float32)
    y, m = organ_targets(arr, ORGANS)
    assert y.tolist() == [[1, 0], [0, 1], [0, 0], [0, 0]]
    assert m.tolist() == [[1, 1], [1, 1], [0, 1], [1, 0]]


def test_masked_loss_ignores_nan_labels():
    logits = torch.tensor([[5.0, -5.0]])
    targets = torch.tensor([[1.0, 1.0]])           # 2nd label is "wrong" ...
    assert masked_bce(logits, targets, torch.tensor([[1.0, 0.0]])) < 0.01   # ... but masked out
    assert masked_bce(logits, targets, torch.tensor([[1.0, 1.0]])) > 1.0    # unmasked it hurts


def test_pos_weight():
    y = np.array([[1], [0], [0], [0]], dtype=np.float32)
    m = np.ones_like(y)
    assert abs(float(compute_pos_weight(y, m)[0]) - 3.0) < 1e-6


def test_model_shapes():
    net = RadExplainNet(ORGANS, backbone="resnet18", pretrained=False)
    out = net(torch.randn(2, 3, 224, 224))
    assert out["organ"].shape == (2, 2) and out["disease"].shape == (2, 3)


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    net = RadExplainNet(ORGANS, backbone="resnet18", pretrained=False, dropout=0.0)
    x = torch.randn(8, 3, 224, 224)
    yd = torch.randint(0, 2, (8, 3)).float(); yo = torch.randint(0, 2, (8, 2)).float()
    ones_d, ones_o = torch.ones_like(yd), torch.ones_like(yo)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    first = None
    for _ in range(30):
        out = net(x)
        loss = masked_bce(out["organ"], yo, ones_o) + masked_bce(out["disease"], yd, ones_d)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * first, f"loss {first:.3f} -> {loss.item():.3f} (model can't learn?)"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
