"""Run:  python tests/test_stage5.py   (needs: pip install streamlit)"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from radexplain.app import db
from radexplain.app.helpers import (available_runs, case_title, chart_rows, chat_pairs, findings_rows, md_escape,
                                    model_card_rows, organ_summary, pick_sample, shrink_image)
from radexplain.report.generator import build_report, render_markdown
from radexplain.report.pdf import build_pdf

F = lambda n, p, t=0.10, h=0.30: {"finding": n, "probability": p, "threshold": t, "high_threshold": h, "flagged": p >= t}
RES = {"organs": {"lungs": {"probability": .3, "threshold": .4, "affected": False},
                  "heart": {"probability": .01, "threshold": .4, "affected": False}},
       "findings": {"lungs": [F("Edema", 0.40), F("Atelectasis", 0.151), F("Mass", 0.01)],
                    "heart": [F("Cardiomegaly", 0.003)]}}


def test_findings_rows():
    rows = findings_rows(RES, {"Edema": "moderate"})
    by = {r["Finding"]: r for r in rows}
    assert by["Edema"]["Assessment"].endswith("likely") and by["Edema"]["Model reliability"] == "moderate"
    assert by["Atelectasis"]["Assessment"].endswith("possible") and by["Atelectasis"]["Probability"] == "15.1%"
    assert by["Mass"]["Assessment"].endswith("not flagged") and by["Mass"]["Model reliability"] == "unknown"


def test_helpers_for_the_new_ui():
    rep = build_report(RES)
    assert case_title(rep, "x.png") == "Pulmonary edema +1"                  # most important finding + count
    assert case_title(build_report({"organs": RES["organs"], "findings": {"lungs": [F("Mass", 0.01)], "heart": []}}), None) == "No findings flagged"
    rows = chart_rows(RES)
    assert rows[0]["Finding"] == "Edema" and rows[0]["Assessment"] == "likely" and rows[-1]["Assessment"] == "not flagged"
    assert chat_pairs([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                       {"role": "user", "content": "q2"}])[1][1] is None
    assert md_escape("*bold* :red[x]") == "\\*bold\\* \\:red\\[x\\]"
    big = Image.new("RGB", (3000, 2000)); import io; b = io.BytesIO(); big.save(b, "PNG")
    assert max(Image.open(io.BytesIO(shrink_image(b.getvalue()))).size) == 900


def test_database_users_cases_and_privacy():
    with tempfile.TemporaryDirectory() as d:
        old, db.ROOT = db.ROOT, Path(d)
        try:
            db.init()
            a, err = db.register("  Alice ", "pw"); assert a and a["username"] == "Alice" and not err
            assert db.register("alice", "other")[1]                          # case-insensitive duplicate
            assert db.register("", "pw")[1] and db.register("bob", "")[1]
            assert db.login("ALICE", "pw")[0]["id"] == a["id"] and db.login("alice", "nope")[0] is None
            import sqlite3
            raw = sqlite3.connect(Path(d) / "data" / "app.db").execute("SELECT password_hash, salt FROM users").fetchone()
            assert "pw" != raw[0] and len(raw[0]) == 64 and raw[1]           # hashed + salted, never plain text
            rep = build_report(RES)
            cid = db.create_case(a["id"], "t", "x.png", b"img", RES, rep, "phase1")
            assert db.add_message(a["id"], cid, "user", "hi", []) and db.add_message(a["id"], cid, "assistant", "yo", ["s"])
            c = db.get_case(a["id"], cid)
            assert c["image"] == b"img" and c["report"] == rep and [m["content"] for m in c["messages"]] == ["hi", "yo"]
            b, _ = db.register("bob", "pw")                                  # bob can't see or touch alice's data
            assert db.get_case(b["id"], cid) is None and db.list_cases(b["id"]) == []
            assert db.get_messages(b["id"], cid) == [] and db.add_message(b["id"], cid, "user", "x") is False
            db.delete_case(b["id"], cid); assert db.get_case(a["id"], cid) is not None
            db.delete_case(a["id"], cid); assert db.get_case(a["id"], cid) is None and db.get_messages(a["id"], cid) == []
            assert db.register("o'brien; DROP TABLE users;--", "p")[0]       # odd names are just data
            assert db.login("o'brien; DROP TABLE users;--", "p")[0]
        finally:
            db.ROOT = old


def test_pdf_report():
    rep = build_report(RES, {"Atelectasis": "lower"})
    im = Image.new("L", (200, 240)); import io; b = io.BytesIO(); im.save(b, "JPEG")
    pdf = build_pdf(rep, b.getvalue(), "scan <1>&.png")                      # awkward characters must not break it
    assert pdf[:5] == b"%PDF-" and len(pdf) > 3000
    assert build_pdf(build_report({"organs": RES["organs"], "findings": {"lungs": [F("Mass", 0.01)], "heart": []}}))[:5] == b"%PDF-"


def test_art_is_valid_svg():
    import xml.dom.minidom as md
    from radexplain.app.art import data_uri, xray_svg
    md.parseString(xray_svg()); md.parseString(xray_svg(overlay=False, scan=False))
    assert data_uri(xray_svg()).startswith("data:image/svg+xml;base64,")


def test_chart_has_spaced_bars_bold_ticks_and_clear_labels():
    from radexplain.app.charts import findings_chart
    spec = findings_chart(chart_rows(RES)).to_dict()
    bars, ticks, labels = spec["layer"]
    assert bars["encoding"]["y"]["scale"]["paddingInner"] >= 0.3          # gaps between bars
    assert ticks["mark"]["type"] == "tick" and ticks["mark"]["thickness"] >= 4
    assert labels["encoding"]["x"]["field"] == "LabelX"                    # label sits clear of the tick
    assert spec["height"] == {"step": 36}                                  # every row gets its own space


def test_pleura_split_and_report_titles():
    from radexplain.labels import ORGAN_LABELS, UNIFIED_LABELS, active_organ_labels
    assert list(ORGAN_LABELS) == ["lungs", "pleura", "heart"] and len(UNIFIED_LABELS) == len(set(UNIFIED_LABELS)) == 16
    assert "Pleural Effusion" in ORGAN_LABELS["pleura"] and "Pleural Effusion" not in ORGAN_LABELS["lungs"]
    import pandas as pd
    master = pd.DataFrame({"split": ["train", "train"], **{l: [1.0, 0.0] if l != "Lung Opacity" else [float("nan")] * 2 for l in UNIFIED_LABELS}})
    assert "Lung Opacity" not in active_organ_labels(master)["lungs"] and "pleura" in active_organ_labels(master)
    old = build_report(RES)                                                  # model without a pleura head
    assert old["organs"]["lungs"]["title"] == "Lungs and pleura"
    new_res = {"organs": {**RES["organs"], "pleura": {"probability": .1, "threshold": .4, "affected": False}},
               "findings": {"lungs": RES["findings"]["lungs"], "pleura": [F("Pneumothorax", 0.2)], "heart": RES["findings"]["heart"]}}
    new = build_report(new_res)
    assert new["organs"]["lungs"]["title"] == "Lungs" and new["organs"]["pleura"]["title"].startswith("Pleura")


def test_organ_summary():
    lines = organ_summary(build_report(RES))
    assert "findings flagged" in lines[0] and "nothing flagged" in lines[1]


def test_model_card_and_runs_and_sample():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        assert model_card_rows(root, "phase1") == [] and pick_sample(root) is None
        (root / "outputs/phase1").mkdir(parents=True)
        (root / "outputs/phase1/test_metrics.json").write_text(json.dumps({"disease": [
            {"name": "Edema", "n_pos": 5, "auroc": 0.9, "auroc_ci": [0.8, 0.95]}, {"name": "Rare", "n_pos": 0}]}))
        rows = model_card_rows(root, "phase1")
        assert len(rows) == 1 and rows[0]["95% CI"] == "0.800-0.950"        # label without AUROC is skipped
        for r in ("smoke", "phase1", "abc"):
            (root / f"checkpoints/{r}").mkdir(parents=True); (root / f"checkpoints/{r}/best.pt").write_bytes(b"")
        assert available_runs(root) == ["phase1", "abc", "smoke"]
        folder = root / "data/raw/nih/images_002/images"; folder.mkdir(parents=True)
        Image.new("L", (8, 8)).save(folder / "x.png")
        data, name = pick_sample(root, random.Random(0))
        assert name == "x.png" and len(data) > 0


def test_app_headless():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, str(root / "tests" / "apptest_stage5.py")], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "APPTEST OK" in out.stdout, out.stdout[-1500:] + out.stderr[-2500:]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
