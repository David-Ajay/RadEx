"""Run:  python tests/test_stage6.py   - soft shading of affected areas (CAM, overlay, PDF, DB, localization metrics)"""
import io
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from radexplain.app import db
from radexplain.evaluate_localization import box_metrics
from radexplain.models.cam import class_activation_maps
from radexplain.models.radexplain_net import RadExplainNet
from radexplain.predict import Predictor
from radexplain.report.generator import build_report
from radexplain.report.overlay import flagged_findings, highlight_region, render_overlay
from radexplain.report.pdf import build_pdf

ORGANS = {"lungs": ["Atelectasis", "Edema"], "pleura": ["Pneumothorax"], "heart": ["Cardiomegaly"]}
F = lambda n, p, t=0.10, h=0.30: {"finding": n, "probability": p, "threshold": t, "high_threshold": h, "flagged": p >= t}
RES = {"organs": {o: {"probability": .3, "threshold": .4, "affected": False} for o in ORGANS},
       "findings": {"lungs": [F("Edema", 0.40), F("Atelectasis", 0.15)], "pleura": [F("Pneumothorax", 0.12)],
                    "heart": [F("Cardiomegaly", 0.01)]}}


def hot(cells, shape=(7, 7)):
    c = np.zeros(shape, np.float32)
    for (r, k), v in cells.items():
        c[r, k] = v
    return c


def test_cam_is_exact():
    """mean(CAM) + bias must equal the model's logit, for both backbones: the maps are the model's own evidence."""
    torch.manual_seed(0)
    for backbone in ("resnet18", "densenet121"):
        net = RadExplainNet(ORGANS, backbone=backbone, pretrained=False).eval()
        x = torch.randn(1, 3, 224, 224)
        out, maps = net(x), class_activation_maps(net, x)
        k = 0
        for organ, labels in ORGANS.items():
            for lab in labels:
                logit = out["disease"][0, k].item(); k += 1
                bias = net.disease_heads[organ].bias[labels.index(lab)].item()
                assert abs(maps[lab].mean() + bias - logit) < 1e-3 * max(1.0, abs(logit)), (backbone, lab)
        assert maps["Edema"].shape == (7, 7) and "organ:heart" in maps


def test_shading_stays_local_and_in_the_right_place():
    r = highlight_region(hot({(3, 1): 5.0, (3, 2): 2.0}))                    # evidence at row 3, columns 1-2
    row, col = r["peak"]
    assert abs(row - (16 + 3.5 * 32)) < 30 and abs(col - (16 + 1.5 * 32)) < 45
    assert r["area_pct"] < 25 and not r["mask"][220, 220] and not r["mask"][20, 128]   # far corners stay clear
    assert r["alpha"].max() > 0.5 and r["alpha"][220, 220] == 0
    assert highlight_region(np.zeros((7, 7), np.float32)) is None            # no evidence -> no shading
    assert highlight_region(-np.ones((7, 7), np.float32)) is None            # only negative evidence -> none
    assert highlight_region(np.ones((7, 7), np.float32))["area_pct"] <= 26   # diffuse map is hard-capped


def test_overlay_is_transparent_coloured_and_numbered():
    base = Image.fromarray(np.tile(np.linspace(40, 200, 400, dtype=np.uint8), (400, 1)))
    items = [{"finding": "Edema", "term": "pulmonary edema", "tier": "likely", "cam": hot({(3, 1): 5.0})},
             {"finding": "Pneumothorax", "term": "pneumothorax", "tier": "possible", "cam": hot({(2, 5): 4.0})}]
    out = render_overlay(base, items)
    img = Image.open(io.BytesIO(out["jpeg"])).convert("RGB")
    assert img.size == base.size and [e["n"] for e in out["legend"]] == [1, 2]
    assert out["legend"][0]["color"] == "#ef4444" and out["legend"][1]["color"] == "#f59e0b"
    px, orig = np.asarray(img).astype(int), np.asarray(base.convert("RGB")).astype(int)
    r0, c0 = int((16 + 3.5 * 32) / 256 * 400) + 22, int((16 + 1.5 * 32) / 256 * 400)   # inside the shade, below the number tag
    assert px[r0, c0, 0] - px[r0, c0, 2] > 22 and px[r0, c0, 0] > orig[r0, c0, 0]        # red tint where the finding is
    assert abs(px[r0, c0].mean() - orig[r0, c0].mean()) < 60                              # but see-through, not painted over
    assert np.abs(px[380, 380] - orig[380, 380]).max() < 12                               # untouched far away (JPEG noise only)
    assert render_overlay(base, [{"finding": "x", "term": "x", "tier": "likely", "cam": np.zeros((7, 7))}]) is None


def test_flagged_findings_order_and_limit():
    items = flagged_findings(RES, max_findings=2)
    assert [i["finding"] for i in items] == ["Edema", "Atelectasis"] and items[0]["tier"] == "likely"
    assert all(i["tier"] != "not_detected" for i in flagged_findings(RES, 9)) and len(flagged_findings(RES, 9)) == 3


def test_predictor_explain_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        net = RadExplainNet(ORGANS, backbone="resnet18", pretrained=False)
        ck = Path(d) / "best.pt"
        torch.save({"model": net.state_dict(), "organ_labels": ORGANS, "backbone": "resnet18", "image_size": 224}, ck)
        zero = {"a": 1.0, "b": 0.0, "threshold": 0.0, "high_threshold": 0.99}      # flag everything
        (Path(d) / "postprocess.json").write_text(json.dumps({"organ": {o: zero for o in ORGANS},
                                                              "disease": {l: zero for ls in ORGANS.values() for l in ls}}))
        p = Predictor(str(ck), device=torch.device("cpu"))
        img = Image.fromarray(np.random.randint(0, 255, (700, 500), dtype=np.uint8))
        out = p.explain(img, p.predict(img))
        shown = Image.open(io.BytesIO(out["jpeg"]))
        assert shown.size[0] <= 900 and abs(shown.size[0] / shown.size[1] - 500 / 700) < 0.01   # keeps the aspect ratio
        assert 1 <= len(out["legend"]) <= 3 and all(e["finding"] in {"Atelectasis", "Edema", "Pneumothorax", "Cardiomegaly"} for e in out["legend"])
        nothing = {"organs": RES["organs"], "findings": {k: [F(i["finding"], 0.0) for i in v] for k, v in RES["findings"].items()}}
        assert p.explain(img, nothing) is None                                  # nothing flagged -> nothing to shade


def test_pdf_uses_the_shaded_image_and_legend():
    base = Image.new("L", (300, 340), 120)
    out = render_overlay(base, [{"finding": "Edema", "term": "pulmonary edema", "tier": "likely", "cam": hot({(3, 1): 5.0})}])
    plain = build_pdf(build_report(RES), b"", "x.png")
    shaded = build_pdf(build_report(RES), None, "x.png", overlay=out)
    assert shaded[:5] == b"%PDF-" and len(shaded) > len(plain) + 2000          # the image is embedded


def test_database_upgrade_and_overlay_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        old_root, db.ROOT = db.ROOT, Path(d)
        try:
            (Path(d) / "data").mkdir()
            con = sqlite3.connect(Path(d) / "data" / "app.db")           # a database made BEFORE overlays existed
            con.executescript(db.SCHEMA.replace("    overlay BLOB,\n    overlay_legend TEXT,\n", ""))
            con.execute("INSERT INTO users(username, username_lower, password_hash, salt, created_at) VALUES ('a','a','x','y','t')")
            con.commit(); con.close()
            db.init()                                                     # must add the columns without losing data
            kept = sqlite3.connect(Path(d) / "data" / "app.db").execute("SELECT COUNT(*) FROM users").fetchone()[0]
            assert kept == 1                                              # the existing user survived the upgrade
            rep = build_report(RES)
            old_case = db.create_case(1, "t", "x", b"i", RES, rep, "p")
            new_case = db.create_case(1, "t", "x", b"i", RES, rep, "p", b"jpegbytes", [{"n": 1, "term": "t"}])
            assert db.get_case(1, old_case)["overlay"] is None and db.get_case(1, old_case)["overlay_legend"] == []
            c = db.get_case(1, new_case)
            assert c["overlay"] == b"jpegbytes" and c["overlay_legend"] == [{"n": 1, "term": "t"}]
        finally:
            db.ROOT = old_root


def test_localization_metrics():
    mask = np.zeros((256, 256), bool); mask[50:100, 50:100] = True
    perfect = box_metrics(mask, (60, 60), (50, 50, 100, 100))
    assert perfect["iou"] == 1.0 and perfect["point_hit"] == 1.0 and perfect["precision"] == perfect["recall"] == 1.0
    miss = box_metrics(mask, (60, 60), (150, 150, 200, 200))
    assert miss["iou"] == 0.0 and miss["mask_hit"] == 0.0 and miss["point_hit"] == 0.0
    half = box_metrics(mask, (60, 60), (75, 50, 125, 100))                 # overlaps half of each
    assert abs(half["recall"] - 0.5) < 0.02 and abs(half["precision"] - 0.5) < 0.02 and 0.3 < half["iou"] < 0.4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
