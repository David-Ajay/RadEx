"""Look and feel: a light, projector-friendly theme with an X-ray-blue palette, a faint blurred X-ray
background, X-ray artwork in the banners, and small HTML builders (all dynamic text is escaped)."""
import html

import streamlit as st

from radexplain.app import brand
from radexplain.app.art import bg_art_uri, hero_art_uri

ART_URI = hero_art_uri()      # your own photo at assets/xray_hero.* if present, else the drawn artwork
BG_URI = bg_art_uri()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {--navy:#0a2540; --ink:#12304f; --blue:#0b6bcb; --sky:#12a4d6; --line:#d7e5f5; --wash:244,248,253;}

/* page: pale radiology light-box blue with a faint, soft X-ray in the corner */
.stApp {background-color: #f4f8fd;
  background-image: linear-gradient(rgba(var(--wash), .87), rgba(var(--wash), .87)), url("__BG__");
  background-repeat: no-repeat, no-repeat; background-attachment: fixed, fixed;
  background-position: 0 0, right -60px top 70px; background-size: auto, auto 820px;}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2rem; max-width: 1280px;}
h1, h2, h3, .rx-hero, .rx-card, .rx-step, .rx-brand {font-family: 'Inter', system-ui, sans-serif;}
h1, h2, h3 {color: var(--navy); letter-spacing: -.01em;}

/* sidebar */
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #ffffff 0%, #eaf3fc 100%); border-right: 1px solid var(--line);}
section[data-testid="stSidebar"] .stButton > button {width: 100%; justify-content: flex-start; text-align: left;}
.rx-brand {font-size: 1.3rem; font-weight: 800; letter-spacing: -.01em; color: var(--navy); margin: 4px 0 12px;}
.rx-side-label {font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: #5b7692; margin: 16px 0 6px;}

/* hero banner with X-ray artwork */
.rx-hero {position: relative; overflow: hidden; border-radius: 24px; padding: 36px 42px; margin-bottom: 22px; min-height: 215px; color: #fff;
  background: linear-gradient(115deg, #05213f 0%, #0a4a85 55%, #1486d1 100%);
  box-shadow: 0 24px 50px -28px rgba(5, 33, 63, .85); animation: rx-fade .55s ease both;}
.rx-hero .rx-art {position: absolute; top: 0; right: 0; bottom: 0; width: 50%; background-size: cover; background-position: center 28%;
  -webkit-mask-image: linear-gradient(to left, #000 42%, transparent 100%); mask-image: linear-gradient(to left, #000 42%, transparent 100%);}
.rx-hero .rx-body {position: relative; z-index: 1; max-width: 58%;}
.rx-hero h1 {margin: 0; padding: 0; font-size: 2.4rem; font-weight: 800; line-height: 1.1; letter-spacing: -.02em; color: #fff;}
.rx-hero p {margin: .6rem 0 0; font-size: 1.05rem; line-height: 1.5; color: #dbeafe;}
.rx-hero.rx-small {min-height: 120px; padding: 22px 32px;} .rx-hero.rx-small h1 {font-size: 1.55rem;}
.rx-hero.rx-small .rx-art {width: 36%;}
.rx-chips {margin-top: 16px;}
.rx-chips span {display: inline-block; margin: 0 8px 6px 0; padding: 4px 13px; border-radius: 999px; font-size: .8rem; font-weight: 600; color: #fff;
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.34);}

/* cards */
.rx-card {border-radius: 18px; padding: 14px 18px; margin: 0 0 12px; background: #fff; border: 1px solid var(--line);
  box-shadow: 0 10px 28px -20px rgba(11, 95, 165, .55); animation: rx-fade .5s ease both; color: var(--ink);}
.rx-organ {border-left: 6px solid var(--c); display: flex; gap: 12px; align-items: center;}
.rx-organ .ic {font-size: 1.7rem; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center; border-radius: 12px; background: #eef5fc;}
.rx-organ .tt {font-weight: 700; color: var(--navy);} .rx-organ .rx-sub {font-size: .85rem; color: #5b7692; margin-top: 3px;}
.rx-pill {display: inline-block; margin-left: 6px; padding: 2px 11px; border-radius: 999px; font-size: .74rem; font-weight: 700; color: #fff; background: var(--c);}
.rx-steps {display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0 4px;}
.rx-step {flex: 1 1 190px; border-radius: 18px; padding: 18px; background: #fff; border: 1px solid var(--line); box-shadow: 0 10px 28px -22px rgba(11,95,165,.6);}
.rx-step .n {display: inline-flex; width: 42px; height: 42px; align-items: center; justify-content: center; border-radius: 13px; margin-bottom: 10px;
  background: linear-gradient(135deg, #0b6bcb, #12a4d6);}
.rx-step .n svg {width: 22px; height: 22px; stroke: #fff; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;}
.rx-step div.t {font-weight: 700; color: var(--navy);} .rx-step div.d {font-size: .86rem; color: #5b7692; margin-top: 3px; line-height: 1.4;}
.rx-dot {display: inline-flex; width: 22px; height: 22px; border-radius: 50%; background: var(--c); color: #fff; align-items: center; justify-content: center; font-size: .75rem; font-weight: 700; margin-right: 8px;}
.rx-leg {margin-top: 7px; color: var(--ink);} .rx-leg i {font-style: normal; color: #5b7692; font-size: .85rem;}
.rx-legend span.rx-pill {margin: 0 6px 0 14px;} .rx-legend {line-height: 2.2; font-size: .9rem;}

/* controls */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {border-radius: 12px; font-weight: 600; border: 1px solid var(--line); background: #fff; color: var(--navy); transition: all .15s ease;}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {border-color: var(--blue); color: var(--blue); transform: translateY(-1px); box-shadow: 0 10px 22px -12px rgba(11,107,203,.7);}
button[kind="primary"], button[kind="primaryFormSubmit"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, #0b6bcb, #12a4d6) !important; border: none !important; color: #fff !important;}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {color: #fff !important; filter: brightness(1.06);}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 8px 16px; font-weight: 600;}
[data-testid="stFileUploader"] section {border: 2px dashed #7db7ea; border-radius: 20px; background: #f3f9ff; padding: 34px;}
[data-testid="stFileUploader"] section:hover {border-color: var(--blue); background: #eaf4ff;}
[data-testid="stImage"] img {border-radius: 18px; border: 1px solid var(--line); box-shadow: 0 14px 34px -20px rgba(5,33,63,.7);}
[data-testid="stChatMessage"] {border-radius: 16px; border: 1px solid var(--line); background: #fff;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 20px;}

/* scanning line over the X-ray picture: works with ANY image (your photo or the drawn artwork), sweeps up and down */
.rx-hero .rx-scan {position: absolute; top: 0; right: 0; bottom: 0; width: 50%; overflow: hidden; pointer-events: none;
  -webkit-mask-image: linear-gradient(to left, #000 42%, transparent 100%); mask-image: linear-gradient(to left, #000 42%, transparent 100%);}
.rx-hero .rx-scan:before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; border-radius: 3px;
  background: linear-gradient(90deg, transparent, #a5f3fc 25%, #ecfeff 50%, #a5f3fc 75%, transparent);
  box-shadow: 0 0 14px 6px rgba(103,232,249,.55), 0 0 44px 20px rgba(103,232,249,.20); animation: rx-scan 4.2s ease-in-out infinite alternate;}
.rx-hero .rx-scan:after {content: ""; position: absolute; inset: 0; background: repeating-linear-gradient(0deg, rgba(165,243,252,.05) 0 1px, transparent 1px 7px);}
@keyframes rx-scan {from {top: 0;} to {top: calc(100% - 3px);}}
.rx-hero h1.rx-brandtitle {margin: 0; padding: 0; line-height: 1.15;}
@keyframes rx-fade {from {opacity: 0; transform: translateY(8px);} to {opacity: 1; transform: none;}}
</style>
""".replace("__BG__", BG_URI).replace("</style>", brand.css() + "</style>")

TIER_COLOR = {"likely": "#ef4444", "possible": "#f59e0b", None: "#7a8ea5"}
ORGAN_ICON = {"lungs": "🫁", "pleura": "🫧", "heart": "🫀"}

ICONS = {
    "upload": "<svg viewBox='0 0 24 24'><path d='M12 16V4m0 0l-4 4m4-4l4 4M4 20h16'/></svg>",
    "scan": "<svg viewBox='0 0 24 24'><path d='M4 8V5a1 1 0 011-1h3M16 4h3a1 1 0 011 1v3M20 16v3a1 1 0 01-1 1h-3M8 20H5a1 1 0 01-1-1v-3M4 12h16'/></svg>",
    "doc": "<svg viewBox='0 0 24 24'><path d='M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6'/></svg>",
    "chat": "<svg viewBox='0 0 24 24'><path d='M4 5h16v11H9l-5 4z'/></svg>",
}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", chips: list[str] | None = None, small: bool = False, animated_brand: bool = False) -> str:
    chips_html = "<div class='rx-chips'>" + "".join(f"<span>{html.escape(c)}</span>" for c in chips) + "</div>" if chips else ""
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    head = f"<h1 class='rx-brandtitle'>{brand.brand_html('lg')}</h1>" if animated_brand else f"<h1>{html.escape(title)}</h1>"
    return (f"<div class='rx-hero{' rx-small' if small else ''}'><div class='rx-art' style=\"background-image:url('{ART_URI}')\"></div>"
            f"<div class='rx-scan'></div><div class='rx-body'>{head}{sub}{chips_html}</div></div>")


def organ_cards(report: dict) -> str:
    cards = []
    for organ, o in report["organs"].items():
        top = "likely" if any(e["tier"] == "likely" for e in o["findings"]) else ("possible" if o["findings"] else None)
        if top:
            label, names = f"{len(o['findings'])} flagged", ", ".join(e["term"] for e in o["findings"][:4])
        elif o["status"] == "nonspecific":
            label, names, top = "non-specific", "organ-level signal only", "possible"
        else:
            label, names = "nothing flagged", "no finding passed the thresholds"
        c = TIER_COLOR[top]
        cards.append(f"<div class='rx-card rx-organ' style='--c:{c}'><div class='ic'>{ORGAN_ICON.get(organ, '🩻')}</div>"
                     f"<div><div class='tt'>{html.escape(o['title'])}<span class='rx-pill' style='--c:{c}'>{html.escape(label)}</span></div>"
                     f"<div class='rx-sub'>{html.escape(names)}</div></div></div>")
    return "".join(cards)


def steps_strip() -> str:
    steps = [("upload", "Upload", "Drop in a chest X-ray (PNG or JPG)"), ("scan", "Analyse", "The AI screens the lungs, pleura and heart"),
             ("doc", "Read", "Get a colour-coded report, downloadable as PDF"), ("chat", "Ask", "Chat about any finding in plain language")]
    return "<div class='rx-steps'>" + "".join(
        f"<div class='rx-step'><span class='n'>{ICONS[k]}</span><div class='t'>{t}</div><div class='d'>{d}</div></div>"
        for k, t, d in steps) + "</div>"


def legend_card() -> str:
    pill = lambda c, t: f"<span class='rx-pill' style='--c:{c}'>{t}</span>"
    return ("<div class='rx-card rx-legend'><b>How to read the colours</b>"
            f"{pill('#ef4444', 'likely')} more confident flag"
            f"{pill('#f59e0b', 'possible')} low-confidence flag, worth a second look"
            f"{pill('#7a8ea5', 'not flagged')} nothing passed the threshold</div>")


def overlay_legend(legend: list[dict]) -> str:
    """Numbered key for the shaded areas on the X-ray."""
    rows = "".join(f"<div class='rx-leg'><span class='rx-dot' style='--c:{e['color']}'>{e['n']}</span>"
                   f"<b>{html.escape(e['term'].capitalize())}</b> <i>({html.escape(e['tier'])})</i></div>" for e in legend)
    return f"<div class='rx-card'><b>Highlighted areas</b>{rows}</div>"
