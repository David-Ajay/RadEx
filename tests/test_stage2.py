"""Run:  python tests/test_stage2.py"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from radexplain.evaluate import fit_calibration, sigmoid, to_logit, youden_threshold
from radexplain.models.radexplain_net import RadExplainNet
from radexplain.predict import Predictor

ORGANS = {"lungs": ["Atelectasis", "Edema"], "heart": ["Cardiomegaly"]}


def test_threshold_separates_classes():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([.1, .2, .3, .4, .6, .7, .8, .9])
    assert 0.4 <= youden_threshold(y, p) <= 0.6
    assert youden_threshold(np.zeros(5), np.linspace(0, 1, 5)) == 0.5        # single class -> default


def test_calibration_shrinks_inflated_probabilities():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.05).astype(float)                              # rare finding
    logit = np.where(y == 1, 1.5, -0.5) + rng.normal(0, 1, 4000) + 2.0       # model over-predicts (+2 shift)
    a, b = fit_calibration(logit, y)
    assert a > 0
    assert sigmoid(a * logit + b).mean() < sigmoid(logit).mean()             # calibrated mean prob is lower
    assert abs(sigmoid(a * logit + b).mean() - y.mean()) < 0.01              # and close to the true rate
    assert fit_calibration(logit[:5], np.zeros(5)) == (1.0, 0.0)


def test_predictor_output_structure():
    with tempfile.TemporaryDirectory() as d:
        net = RadExplainNet(ORGANS, backbone="resnet18", pretrained=False)
        ck = Path(d) / "best.pt"
        torch.save({"model": net.state_dict(), "organ_labels": ORGANS, "backbone": "resnet18", "image_size": 224}, ck)
        (Path(d) / "postprocess.json").write_text(json.dumps({
            "organ": {"lungs": {"a": 1.0, "b": 0.0, "threshold": 0.3}, "heart": {"a": 1.0, "b": 0.0, "threshold": 0.3}},
            "disease": {l: {"a": 1.0, "b": 0.0, "threshold": 0.3} for ls in ORGANS.values() for l in ls}}))
        img = Image.fromarray(np.random.randint(0, 255, (700, 640), dtype=np.uint8))   # arbitrary size, grayscale
        res = Predictor(str(ck), device=torch.device("cpu")).predict(img)
        assert set(res["organs"]) == {"lungs", "heart"}
        assert [f["finding"] for f in res["findings"]["heart"]] == ["Cardiomegaly"]
        probs = [f["probability"] for f in res["findings"]["lungs"]]
        assert probs == sorted(probs, reverse=True) and all(0 <= p <= 1 for p in probs)   # ranked
        assert res["organs"]["lungs"]["affected"] == (res["organs"]["lungs"]["probability"] >= 0.3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
