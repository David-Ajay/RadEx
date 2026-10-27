"""UI-independent helpers for the web app (no streamlit import, so they are easy to test)."""
import io
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

from radexplain.report.generator import tier

TIER_BADGE = {"likely": "🔴 likely", "possible": "🟡 possible", "not_detected": "⚪ not flagged"}
ORGAN_STATUS = {"findings": "🟡 findings flagged", "nonspecific": "🟡 non-specific signal",
                "no_findings": "⚪ nothing flagged"}


def findings_rows(result: dict, reliability: dict | None = None) -> list[dict]:
    """Table rows for the Findings tab: each organ's findings, highest probability first."""
    reliability = reliability or {}
    rows = []
    for organ, items in result["findings"].items():
        for it in items:
            rows.append({"Organ": organ.title(), "Finding": it["finding"],
                         "Assessment": TIER_BADGE[tier(it)], "Probability": f"{it['probability']:.1%}",
                         "Flag threshold": f"{it['threshold']:.1%}",
                         "Model reliability": reliability.get(it["finding"], "unknown")})
    return rows


def organ_summary(report: dict) -> list[str]:
    return [f"**{o['title']}**: {ORGAN_STATUS[o['status']]}" for o in report["organs"].values()]


def model_card_rows(root, name: str) -> list[dict]:
    """Per-finding test performance from `evaluate` (outputs/<name>/test_metrics.json)."""
    f = Path(root) / "outputs" / name / "test_metrics.json"
    if not f.exists():
        return []
    rows = []
    for r in json.loads(f.read_text())["disease"]:
        if "auroc" in r:
            ci = f"{r['auroc_ci'][0]:.3f}-{r['auroc_ci'][1]:.3f}" if r.get("auroc_ci") else ""
            rows.append({"Finding": r["name"], "AUROC": r["auroc"], "95% CI": ci,
                         "Test images with finding": r["n_pos"]})
    return rows


def pick_sample(root, rng=random):
    """A random NIH X-ray from the raw data folder, for a one-click demo. Returns (bytes, filename) or None."""
    base = Path(root) / "data" / "raw" / "nih"
    folders = [p for p in base.glob("images_*") if p.is_dir()] if base.exists() else []
    for folder in rng.sample(folders, len(folders)):
        img_dir = folder / "images" if (folder / "images").is_dir() else folder
        names = [n for n in os.listdir(img_dir) if n.lower().endswith(".png")]
        if names:
            n = rng.choice(names)
            return (img_dir / n).read_bytes(), n
    return None


def available_runs(root) -> list[str]:
    runs = sorted(p.parent.name for p in (Path(root) / "checkpoints").glob("*/best.pt"))
    return sorted(runs, key=lambda r: (r != "phase1", r))        # phase1 first


def md_escape(text: str) -> str:
    """Show user-typed text literally inside Streamlit markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$:])", r"\\\1", text or "")


def shrink_image(data: bytes, max_side: int = 900) -> bytes:
    """Smaller grayscale JPEG copy of the upload, to keep the database light."""
    im = Image.open(io.BytesIO(data)).convert("L")
    im.thumbnail((max_side, max_side))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=88)
    return out.getvalue()


def case_title(report: dict, source_name: str | None) -> str:
    """Short history label: the most important flagged finding (or 'No findings flagged')."""
    flagged = [(e["tier"] != "likely", -e["probability"], e["term"]) for o in report["organs"].values()
               for e in o["findings"]]
    head = sorted(flagged)[0][2].capitalize() if flagged else "No findings flagged"
    if flagged and len(flagged) > 1:
        head += f" +{len(flagged) - 1}"
    return head


def short_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %H:%M")
    except ValueError:
        return iso


def chat_pairs(messages: list[dict]) -> list[tuple[dict, dict | None]]:
    """[(question, answer), ...] in the order asked (answer is None if it failed)."""
    pairs, i = [], 0
    while i < len(messages):
        if messages[i]["role"] == "user":
            nxt = messages[i + 1] if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant" else None
            pairs.append((messages[i], nxt))
            i += 2 if nxt else 1
        else:
            i += 1
    return pairs


def chart_rows(result: dict, top: int = 10) -> list[dict]:
    """Bars for the Findings chart: the highest-probability findings with their flag threshold."""
    rows = []
    for items in result["findings"].values():
        for it in items:
            t = tier(it)
            rows.append({"Finding": it["finding"], "Probability": round(it["probability"] * 100, 2),
                         "Threshold": round(it["threshold"] * 100, 2),
                         "Assessment": "not flagged" if t == "not_detected" else t})
    return sorted(rows, key=lambda r: -r["Probability"])[:top]
