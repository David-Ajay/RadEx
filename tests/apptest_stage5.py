"""Headless run of the real Streamlit app against a TEMP folder (never touches your data/checkpoints).
Called by test_stage5.py; can also be run directly:  python tests/apptest_stage5.py"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp())
shutil.copytree(SRC / "configs", tmp / "configs")
shutil.copytree(SRC / "knowledge", tmp / "knowledge")
(tmp / "checkpoints/phase1").mkdir(parents=True)
(tmp / "checkpoints/phase1/best.pt").write_bytes(b"")
img_dir = tmp / "data/raw/nih/images_001/images"; img_dir.mkdir(parents=True)
Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8)).save(img_dir / "a.png")
(tmp / "outputs/phase1").mkdir(parents=True)
(tmp / "outputs/phase1/test_metrics.json").write_text(json.dumps({"organ": [], "disease": [
    {"name": "Atelectasis", "n_pos": 10, "auroc": 0.8, "auroc_ci": [0.7, 0.9]},
    {"name": "Edema", "n_pos": 5, "auroc": 0.9, "auroc_ci": [0.8, 0.95]}]}))
os.environ["RADEXPLAIN_ROOT"] = str(tmp)
os.environ.pop("ANTHROPIC_API_KEY", None)

F = lambda n, p, t=0.10, h=0.30: {"finding": n, "probability": p, "threshold": t, "high_threshold": h, "flagged": p >= t}
RESULT = {"organs": {"lungs": {"probability": .3, "threshold": .4, "affected": False},
                     "heart": {"probability": .01, "threshold": .4, "affected": False}},
          "findings": {"lungs": [F("Edema", 0.40), F("Atelectasis", 0.151), F("Mass", 0.01)],
                       "heart": [F("Cardiomegaly", 0.003)]}}

import radexplain.predict as P                      # noqa: E402  (after RADEXPLAIN_ROOT is set)
from radexplain.app import db                        # noqa: E402


import io as _io
_buf = _io.BytesIO(); Image.new("RGB", (300, 300), (90, 90, 90)).save(_buf, "JPEG")
OVERLAY = {"jpeg": _buf.getvalue(), "legend": [
    {"n": 1, "finding": "Edema", "term": "pulmonary edema", "tier": "likely", "color": "#ef4444", "area_pct": 6.0},
    {"n": 2, "finding": "Atelectasis", "term": "atelectasis", "tier": "possible", "color": "#f59e0b", "area_pct": 4.0}]}


class FakePredictor:
    def __init__(self, *a, **k): pass
    def predict(self, image):
        assert image.size == (300, 300)
        return RESULT
    def explain(self, image, result):
        return OVERLAY


P.Predictor = FakePredictor
from streamlit.testing.v1 import AppTest            # noqa: E402

at = AppTest.from_file(str(SRC / "src/radexplain/app/main.py"), default_timeout=90)
btn = lambda label: next(b for b in at.button if b.label == label)
texts = lambda: " ".join(m.value for m in at.markdown)
chat_texts = lambda: [m.markdown[0].value for m in at.chat_message]


def ok():
    assert not at.exception, [e.value for e in at.exception]


# ---- 1. logged-out visitors only see the login / register page
at.run(); ok()
assert "rx-bl" in texts() and "Radiology&nbsp;Explained" in texts() and "Conversational&nbsp;AI&nbsp;Assistant" in texts() and any(b.label == "Login" for b in at.button)
assert not at.sidebar.button, "sidebar must be empty before login"

# ---- 2. register alice -> lands on a fresh upload page with an empty history
at.text_input(key="reg_user").set_value("alice"); at.text_input(key="reg_pw").set_value("pw1")
btn("Create account").click().run(); ok()
assert at.session_state["user"]["username"] == "alice" and "New X-ray" in texts() and "rx-topbar" in texts()
assert at.sidebar.button(key="new_case") and not [b for b in at.sidebar.button if b.key.startswith("case_")]

# ---- 3. analyse a sample -> case page with 4 tabs, findings table, chart, colour cards
at.button(key="sample").click().run(); ok()
alice_case = at.session_state["case_id"]
assert len(at.tabs) == 4 and any("Atelectasis" in str(d.value) for d in at.dataframe)
assert "rx-topbar" in texts() and "rx-scan" in texts()                           # animated top bar + scan line inside the app
assert "rx-organ" in texts()
legend_cards = lambda: sum("Highlighted areas" in m.value for m in at.markdown)
assert legend_cards() == 2 and "Pulmonary edema" in texts()                      # legend beside the image + in the Report tab
assert at.toggle(key="show_hl").value is True
assert db.get_case(at.session_state["user"]["id"], at.session_state["case_id"])["overlay"] == OVERLAY["jpeg"]
at.toggle(key="show_hl").set_value(False).run(); ok()
assert legend_cards() == 1                                                       # plain X-ray view: left legend gone
at.toggle(key="show_hl").set_value(True).run(); ok()
assert [b for b in at.sidebar.button if b.key == f"case_{alice_case}"]

# ---- 4. chat: newest exchange is shown FIRST; everything is saved to the database
at.button(key="sug1").click().run(); ok()                                   # "What does 'possible' mean?"
at.chat_input[0].set_value("what is edema").run(); ok()
shown = chat_texts()
assert len(shown) == 4 and shown[0] == "what is edema", shown              # newest question on top
assert "possible" in shown[3].lower()                                       # older exchange below it
saved = db.get_messages(at.session_state["user"]["id"], alice_case)
assert [m["role"] for m in saved] == ["user", "assistant", "user", "assistant"]
at.chat_input[0].set_value("I have chest pain").run()
assert "urgent medical care" in chat_texts()[1]

# ---- 5. "New X-ray" opens a fresh upload page; the old chat is one click away in the history
at.sidebar.button(key="new_case").click().run(); ok()
assert at.session_state["case_id"] is None and "New X-ray" in texts() and not chat_texts()
at.sidebar.button(key=f"case_{alice_case}").click().run(); ok()
assert at.session_state["case_id"] == alice_case and len(chat_texts()) == 6   # 3 questions + 3 answers restored

# ---- 6. logout, wrong password, right password -> history is still there
at.sidebar.button(key="logout").click().run(); ok()
assert "user" not in at.session_state and any(b.label == "Login" for b in at.button)
at.text_input(key="login_user").set_value("alice"); at.text_input(key="login_pw").set_value("WRONG")
btn("Login").click().run(); ok()
assert any("Wrong username or password" in e.value for e in at.error)
assert not at.get("help_info"), "a wrong password must show ONLY the error, no stray debug output"
at.text_input(key="login_pw").set_value("pw1"); btn("Login").click().run(); ok()
assert at.session_state["case_id"] is None and [b for b in at.sidebar.button if b.key == f"case_{alice_case}"]

# ---- 7. a second user starts empty and can't open alice's case, even by forcing its id
at.sidebar.button(key="logout").click().run()
at.text_input(key="reg_user").set_value("ALICE"); at.text_input(key="reg_pw").set_value("x")
btn("Create account").click().run(); ok()
assert any("already taken" in e.value for e in at.error)                    # names are case-insensitive
at.text_input(key="reg_user").set_value("bob"); at.text_input(key="reg_pw").set_value("anything")
btn("Create account").click().run(); ok()
assert at.session_state["user"]["username"] == "bob" and not [b for b in at.sidebar.button if b.key.startswith("case_")]
at.session_state["case_id"] = alice_case; at.run(); ok()
assert "New X-ray" in texts() and not chat_texts() and not at.tabs        # upload page, no leak

# ---- 8. engine without a key -> friendly warning + automatic fallback to the basic engine, no crash
at.button(key="sample").click().run(); ok()
at.sidebar.selectbox(key="provider").select("anthropic").run(); ok()
assert any("ANTHROPIC_API_KEY" in w.value and "basic engine" in w.value for w in at.warning), [w.value for w in at.warning]
print("APPTEST OK")
