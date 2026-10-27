"""RadEx web app.   Run:  streamlit run src/radexplain/app/main.py   (opens http://localhost:8501)

Flow: login / create account -> a fresh "upload an X-ray" page -> results, report (PDF), chat.
Every analysed X-ray is saved as a "case" in the sidebar history, together with its chat.
"""
import html
import io
import json

import pandas as pd
import streamlit as st
from PIL import Image

from radexplain.app import db
from radexplain.app.charts import findings_chart
from radexplain.app.helpers import (available_runs, case_title, chart_rows, chat_pairs, findings_rows, md_escape,
                                    model_card_rows, pick_sample, short_date, shrink_image)
from radexplain.app.brand import topbar_html
from radexplain.app.style import hero, inject_css, legend_card, organ_cards, overlay_legend, steps_strip
from radexplain.config import ROOT, load_config
from radexplain.predict import Predictor
from radexplain.rag.chatbot import ReportChatbot
from radexplain.report.generator import build_report, load_reliability, render_markdown
from radexplain.report.pdf import build_pdf

st.set_page_config(page_title="RadEx", page_icon="🩻", layout="wide")
inject_css()
db.init()
ss = st.session_state
rag_cfg = load_config("rag.yaml")
ENGINES = ["gemini", "extractive", "ollama", "anthropic"]
SUGGESTIONS = ["Explain my report in simple words", "What does 'possible' mean?", "How sure is the software?",
               "Should I be worried?"]


@st.cache_resource(show_spinner="Loading the model...")
def get_predictor(run: str):
    return Predictor(f"checkpoints/{run}/best.pt")


# ----------------------------------------------------------------------------- callbacks
def start_new():
    ss["case_id"] = None
    ss["upload_n"] = ss.get("upload_n", 0) + 1


def open_case(case_id):
    ss["case_id"] = case_id


def logout():
    ss.clear()


def delete_case(case_id):
    db.delete_case(ss["user"]["id"], case_id)
    start_new()


def finish_login(user):
    ss["user"], ss["case_id"] = user, None
    st.rerun()


# ----------------------------------------------------------------------------- login / register
def render_auth():
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown(hero("RadEx", "Upload a chest X-ray, get a clear colour-coded report, and ask questions about "
                                  "it in plain language.",
                         ["Chest X-ray AI", "Colour-coded report", "PDF download", "Chat assistant"], animated_brand=True),
                    unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown("#### Welcome")
            t_login, t_reg = st.tabs(["Login", "Create account"])
            with t_login:
                with st.form("login_form"):
                    u = st.text_input("Username", key="login_user")
                    p = st.text_input("Password", type="password", key="login_pw")
                    go = st.form_submit_button("Login", type="primary")
                if go:
                    user, err = db.login(u, p)
                    if err:
                        st.error(err)
                    else:
                        finish_login(user)
            with t_reg:
                with st.form("register_form"):
                    u = st.text_input("Choose a username", key="reg_user", help="Anything you like. No email needed.")
                    p = st.text_input("Choose a password", type="password", key="reg_pw", help="Anything you like.")
                    go = st.form_submit_button("Create account", type="primary")
                if go:
                    user, err = db.register(u, p)
                    if err:
                        st.error(err)
                    else:
                        finish_login(user)
            st.caption("Accounts and reports are stored only on this computer (data/app.db).")
    st.markdown(steps_strip(), unsafe_allow_html=True)
    st.markdown(legend_card(), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar
def render_sidebar(user):
    with st.sidebar:
        st.markdown("<div class='rx-brand'>🩻 RadEx</div>", unsafe_allow_html=True)
        st.button("➕  New X-ray", key="new_case", on_click=start_new, type="primary")
        st.markdown("<div class='rx-side-label'>History</div>", unsafe_allow_html=True)
        cases = db.list_cases(user["id"])
        if not cases:
            st.caption("Your analysed X-rays and their chats will appear here.")
        for c in cases:
            st.button(f"{c['title'][:26]} · {short_date(c['created_at'])}", key=f"case_{c['id']}",
                      on_click=open_case, args=(c["id"],),
                      type="primary" if c["id"] == ss.get("case_id") else "secondary")
        st.divider()
        with st.expander("Settings"):
            provider = st.selectbox("Chatbot engine", ENGINES, index=ENGINES.index(rag_cfg["provider"]), key="provider",
                                    help="gemini = Google Gemini (key in the .env file), understands normal English. "
                                         "extractive = no setup, basic answers. ollama = local model. "
                                         "anthropic = Claude API (needs ANTHROPIC_API_KEY).")
            runs = available_runs(ROOT)
            run = st.selectbox("Model", runs, key="run") if runs else None
        st.caption(f"Signed in as **{md_escape(user['username'])}**")
        st.button("Log out", key="logout", on_click=logout)
    return provider, run


# ----------------------------------------------------------------------------- upload page
def render_upload(user, run):
    st.markdown(topbar_html(), unsafe_allow_html=True)
    st.markdown(hero("New X-ray", "Upload a chest X-ray to start a new analysis. It is saved to your history "
                                  "automatically.", ["Frontal chest X-ray", "PNG or JPG"]), unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop a chest X-ray here", type=["png", "jpg", "jpeg"],
                                key=f"uploader_{ss.get('upload_n', 0)}")
    sample = st.button("✨ Try a sample X-ray", key="sample",
                       help="A random image from your downloaded NIH data. It may have been used in training, "
                            "so use it only to see how the app works.")
    data = name = None
    if uploaded is not None:
        data, name = uploaded.getvalue(), uploaded.name
    elif sample:
        picked = pick_sample(ROOT)
        if picked is None:
            st.error("No NIH images found in data/raw/nih.")
        else:
            data, name = picked
    if data is None:
        st.markdown(steps_strip(), unsafe_allow_html=True)
        st.markdown(legend_card(), unsafe_allow_html=True)
        return
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        st.error("That file couldn't be opened as an image. Please upload a PNG or JPG chest X-ray.")
        return
    with st.spinner("Analysing the X-ray..."):
        result = get_predictor(run).predict(image)
    report = build_report(result, load_reliability(ROOT, run))
    try:                                              # soft shading of the affected areas (optional extra)
        overlay = get_predictor(run).explain(image, result)
    except Exception:
        overlay = None
    ss["case_id"] = db.create_case(user["id"], case_title(report, name), name, shrink_image(data), result, report, run,
                                   overlay["jpeg"] if overlay else None, overlay["legend"] if overlay else None)
    st.rerun()


# ----------------------------------------------------------------------------- case page
def get_bot(case, provider):
    key = (case["id"], provider)
    if ss.get("bot_key") != key:
        notice = None
        try:
            bot = ReportChatbot(case["report"], cfg=dict(rag_cfg, provider=provider))
        except RuntimeError as e:                  # e.g. no API key yet: fall back to the basic engine and say so
            notice = f"{e}\n\nUsing the built-in basic engine for now."
            bot = ReportChatbot(case["report"], cfg=dict(rag_cfg, provider="extractive"))
        bot.history = [{"role": m["role"], "content": m["content"]} for m in case["messages"]]
        ss["bot"], ss["bot_notice"], ss["bot_key"] = bot, notice, key
    return ss["bot"]


def render_case(user, case, provider, run):
    report, result = case["report"], case["result"]
    reliability = load_reliability(ROOT, case["model_run"] or run)
    st.markdown(topbar_html(), unsafe_allow_html=True)
    st.markdown(hero(case["title"], f"{case['source_name'] or 'X-ray'}  ·  {short_date(case['created_at'])}",
                     small=True), unsafe_allow_html=True)
    left, right = st.columns([1, 2])
    overlay = {"jpeg": case["overlay"], "legend": case["overlay_legend"]} if case["overlay"] else None
    with left:
        show = bool(overlay) and st.toggle("Show highlighted areas", value=True, key="show_hl")
        st.image(overlay["jpeg"] if show else case["image"])
        if show:
            st.markdown(overlay_legend(overlay["legend"]), unsafe_allow_html=True)
            st.caption("Shaded areas show where the model found the strongest evidence for each flagged finding. "
                       "This is an approximation, not a precise outline.")
        st.markdown(organ_cards(report), unsafe_allow_html=True)
        st.caption("Flags are screening signals, not diagnoses.")
        with st.expander("Manage"):
            st.button("🗑 Delete this X-ray and chat", key="delete_case", on_click=delete_case, args=(case["id"],))

    with right:
        t_find, t_report, t_chat, t_model = st.tabs(["Findings", "Report", "Ask about your report", "About the model"])

        with t_find:
            st.altair_chart(findings_chart(chart_rows(result)))
            st.dataframe(pd.DataFrame(findings_rows(result, reliability)), hide_index=True)
            st.caption("Probabilities are calibrated, so rare findings can be flagged at small percentages. "
                       "'Possible' is a low-confidence flag; false alarms are common.")

        with t_report:
            md = render_markdown(report)
            if ss.get("pdf_key") != case["id"]:
                ss["pdf"], ss["pdf_key"] = build_pdf(report, case["image"], case["source_name"],
                                                     short_date(case["created_at"]), overlay), case["id"]
            c1, c2, c3 = st.columns(3)
            c1.download_button("⬇ Download PDF", ss["pdf"], f"radexplain_report_{case['id']}.pdf", "application/pdf",
                               type="primary")
            c2.download_button("Download Markdown", md, f"radexplain_report_{case['id']}.md")
            c3.download_button("Download data (JSON)", json.dumps({"report": report, "model_output": result}, indent=2),
                               f"radexplain_report_{case['id']}.json")
            if overlay:
                c_img, c_leg = st.columns([1, 1])
                c_img.image(overlay["jpeg"])
                c_leg.markdown(overlay_legend(overlay["legend"]), unsafe_allow_html=True)
                c_leg.caption("Shaded areas are approximate: they show where the model found the strongest evidence.")
            st.markdown(render_markdown(report, rich=True))

        with t_chat:
            bot = get_bot(case, provider)
            if ss.get("bot_notice"):
                st.warning(ss["bot_notice"])
            elif provider == "gemini":
                st.caption("🔒 Gemini engine: your question and the report text (never the X-ray image) are sent to "
                           "Google. On Google's free tier this content may be used to improve its products and read by "
                           "human reviewers, so please don't type names or other personal details.")
            cols = st.columns(len(SUGGESTIONS))
            for i, (col, s) in enumerate(zip(cols, SUGGESTIONS)):
                col.button(s, key=f"sug{i}", on_click=lambda q=s: ss.update(pending_q=q))
            prompt = st.chat_input("Ask a question about your report") or ss.pop("pending_q", None)
            if prompt:
                try:
                    with st.spinner("Thinking..."):
                        r = bot.ask(prompt)
                    if r.get("note"):
                        st.warning(f"{r['note']}\n\nShowing the built-in basic engine's answer instead.")
                    db.add_message(user["id"], case["id"], "user", prompt)
                    db.add_message(user["id"], case["id"], "assistant", r["answer"], r["sources"])
                except RuntimeError as e:
                    st.error(str(e))
            pairs = chat_pairs(db.get_messages(user["id"], case["id"]))
            if not pairs:
                st.caption("Pick a question above or type your own. The assistant only uses your report and a "
                           "medical glossary, and cannot give medical advice.")
            else:
                st.caption("Newest question and answer first.")
            for q, a in reversed(pairs):
                with st.chat_message("user"):
                    st.markdown(md_escape(q["content"]))
                if a:
                    with st.chat_message("assistant"):
                        st.markdown(a["content"])
                        if a["sources"]:
                            st.caption("Based on: " + "; ".join(a["sources"]))

        with t_model:
            rows = model_card_rows(ROOT, case["model_run"] or run)
            if rows:
                st.markdown("**How well does the model work?** Results on the held-out NIH ChestX-ray14 test set "
                            "(AUROC: 0.5 = guessing, 1.0 = perfect).")
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            else:
                st.info("Run `bash scripts/05_evaluate.sh` to see test results here.")
            st.markdown("**Limitations**\n"
                        "- Trained on one public dataset (NIH ChestX-ray14) whose labels were extracted automatically "
                        "from reports, so they contain errors.\n"
                        "- Looks at a single frontal image, with no symptoms, history or earlier scans.\n"
                        "- Covers lungs and heart only (no ribs, diaphragm, trachea or esophagus yet).\n"
                        "- Tuned to flag generously, so false alarms are common.\n"
                        "- Not clinically validated and not a medical device.")


# ----------------------------------------------------------------------------- router
user = ss.get("user")
if not user:
    render_auth()
    st.stop()

provider, run = render_sidebar(user)
if run is None:
    st.error("No trained model found in `checkpoints/`. Train one first: `bash scripts/04_train.sh`.")
    st.stop()

case = db.get_case(user["id"], ss["case_id"]) if ss.get("case_id") else None
if case is None:
    render_upload(user, run)
else:
    render_case(user, case, provider, run)
