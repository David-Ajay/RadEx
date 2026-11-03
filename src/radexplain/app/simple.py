"""RadEx - simple version: upload a chest X-ray, get the findings and the coloured shading. Nothing else.
No login, no chatbot, no report, no saved history, no heart prediction.
Run:  bash scripts/10_app.sh
"""
import hashlib
import io

import pandas as pd
import streamlit as st
from PIL import Image

from radexplain.app.helpers import available_runs, findings_rows
from radexplain.config import ROOT
from radexplain.predict import Predictor
from radexplain.report.generator import load_reliability

st.set_page_config(page_title="RadEx", page_icon="🩻", layout="wide")

EMOJI = {"likely": "🔴", "possible": "🟡"}


@st.cache_resource(show_spinner="Loading the model...")
def get_predictor(run: str):
    return Predictor(f"checkpoints/{run}/best.pt")


st.title("🩻 RadEx - Chest X-ray Findings")
st.caption("Research prototype. Not a medical device and not a substitute for a radiologist.")

runs = available_runs(ROOT)
if not runs:
    st.error("No trained model found in `checkpoints/`.")
    st.stop()
run = st.selectbox("Model", runs, index=runs.index("pleura") if "pleura" in runs else 0)

uploaded = st.file_uploader("Upload a chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload a chest X-ray to see the findings.")
    st.stop()

data = uploaded.getvalue()
key = (hashlib.sha1(data).hexdigest(), run)
if st.session_state.get("rx_key") != key:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        st.error("That file couldn't be opened as an image. Please upload a PNG or JPG chest X-ray.")
        st.stop()
    with st.spinner("Analysing the X-ray..."):
        predictor = get_predictor(run)
        result = predictor.predict(image)
        # heart is left out on purpose: drop it from both the table and the shading
        shown = {"organs": {k: v for k, v in result["organs"].items() if k != "heart"},
                 "findings": {k: v for k, v in result["findings"].items() if k != "heart"}}
        try:
            overlay = predictor.explain(image, shown)
        except Exception:
            overlay = None
    st.session_state.update(rx_key=key, rx_bytes=data, rx_shown=shown, rx_overlay=overlay)

shown, overlay = st.session_state["rx_shown"], st.session_state["rx_overlay"]
left, right = st.columns([1, 1.4], gap="large")

with left:
    show = bool(overlay) and st.toggle("Show highlighted areas", value=True)
    st.image(overlay["jpeg"] if show else st.session_state["rx_bytes"])
    if show:
        for e in overlay["legend"]:
            st.markdown(f"**{e['n']}.** {EMOJI[e['tier']]} {e['term'].capitalize()} ({e['tier']})")
        st.caption("Shaded areas show where the model found the strongest evidence for each flagged "
                   "finding. This is an approximation, not a precise outline.")
    elif overlay is None:
        st.caption("Nothing was flagged strongly enough to highlight.")

with right:
    st.subheader("Findings")
    rows = findings_rows(shown, load_reliability(ROOT, run))
    rows.sort(key=lambda r: -float(r["Probability"].rstrip("%")))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Probabilities are calibrated, so rare findings can be flagged at small percentages. "
               "'Possible' is a low-confidence flag; false alarms are common.")