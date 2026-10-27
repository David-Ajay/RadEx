"""Stage 3: turn Predictor output into a structured radiology-style report.

No torch needed here, so the web UI and tests can import it cheaply.

  report = build_report(predictor_result, reliability)   # dict (JSON-serialisable, feeds the chatbot)
  text   = render_markdown(report)                       # human-readable report

Design rules (deliberate):
  * Each finding gets a tier from its CALIBRATED probability:
        likely    : probability >= high_threshold  (~90% specificity on validation data)
        possible  : probability >= threshold       (sensitivity/specificity balanced)
        not_detected otherwise
  * The wording never shows raw percentages in Findings/Impression (they are small and confusing
    for rare findings); numbers appear only in the 'Model confidence' table.
  * Organ status is reconciled: specific findings win; if only the organ-level screen fired we say
    "non-specific"; otherwise "no findings". The report never claims the absence of disease, only
    that nothing crossed the model's detection thresholds.
  * Findings the model is weak at (low test AUROC) carry an explicit caution.
"""
import json
from pathlib import Path

from radexplain.report.templates import (CLOSING_IMPRESSION, DISCLAIMER, LOW_RELIABILITY_NOTE,
                                         ORGAN_TITLES, PHRASES, TECHNIQUE)

TIER_ORDER = {"likely": 0, "possible": 1}


def tier(item: dict) -> str:
    p, t = item["probability"], item["threshold"]
    hi = item.get("high_threshold", min(1.0, 2 * t))
    if p >= hi:
        return "likely"
    if p >= t:
        return "possible"
    return "not_detected"


def reliability_from_auroc(auroc):
    if auroc is None:
        return "unknown"
    return "higher" if auroc >= 0.85 else "moderate" if auroc >= 0.75 else "lower"


def load_reliability(root, name="phase1") -> dict:
    """finding -> reliability class, from the test-set evaluation (outputs/<name>/test_metrics.json)."""
    f = Path(root) / "outputs" / name / "test_metrics.json"
    if not f.exists():
        return {}
    rows = json.loads(f.read_text())["disease"]
    return {r["name"]: reliability_from_auroc(r.get("auroc")) for r in rows}


def _phrase(label):
    return PHRASES.get(label, (label.lower(), ""))


def build_report(result: dict, reliability: dict | None = None) -> dict:
    reliability = reliability or {}
    organs, flagged = {}, []
    for organ, items in result["findings"].items():
        found, quiet = [], []
        for it in items:
            t = tier(it)
            term, meaning = _phrase(it["finding"])
            entry = {"finding": it["finding"], "term": term, "meaning": meaning, "tier": t,
                     "probability": it["probability"], "threshold": it["threshold"],
                     "reliability": reliability.get(it["finding"], "unknown")}
            (found if t != "not_detected" else quiet).append(entry)
        found.sort(key=lambda e: (TIER_ORDER[e["tier"]], -e["probability"]))
        organ_screen = result["organs"][organ]["affected"]
        status = "findings" if found else ("nonspecific" if organ_screen else "no_findings")
        title = ORGAN_TITLES.get(organ, organ.title())
        if organ == "lungs" and "pleura" not in result["findings"]:      # older model: pleural findings sit under lungs
            title = "Lungs and pleura"
        organs[organ] = {"title": title, "status": status,
                         "organ_screen_positive": organ_screen, "findings": found,
                         "not_flagged": [q["finding"] for q in quiet]}
        flagged += [dict(e, organ=organ) for e in found]

    flagged.sort(key=lambda e: (TIER_ORDER[e["tier"]], -e["probability"]))
    impression = [f"{'Likely' if e['tier'] == 'likely' else 'Possible'} {e['term']} ({e['organ']})."
                  for e in flagged[:5]]
    if not flagged:
        if any(o["status"] == "nonspecific" for o in organs.values()):
            impression.append("A non-specific abnormality is suggested by the organ-level screen; "
                              "no individual finding reached the model's confidence thresholds.")
        else:
            impression.append("No abnormality identified above the model's detection thresholds.")
    impression.append(CLOSING_IMPRESSION)

    return {"title": "AI-assisted preliminary chest X-ray report", "technique": TECHNIQUE,
            "organs": organs, "impression": impression, "disclaimer": DISCLAIMER}


COLOR = {"likely": "red", "possible": "orange"}
CONFIDENCE_NOTE = ("Probabilities are calibrated on validation data. Low values can still be flagged for "
                   "rare findings.")


def _note(text, rich):
    """A highlighted callout that starts with a bold 'Note:'."""
    return f":orange-background[**Note:** {text}]" if rich else f"> **Note:** {text}"


def _h(text, rich):
    return f"## :blue[{text}]" if rich else f"## {text}"


def _finding_sentence(e, rich=False):
    label = f"{'Likely' if e['tier'] == 'likely' else 'Possible'} {e['term']}"
    head = f":{COLOR[e['tier']]}[**{label}**]" if rich else label
    s = head + (f" - {e['meaning']}." if e["meaning"] else ".")
    if e["reliability"] == "lower":
        s += f" (Caution: {LOW_RELIABILITY_NOTE}.)"
    return s


def render_markdown(report: dict, rich: bool = False) -> str:
    """rich=False -> portable Markdown (for the .md download).
    rich=True  -> Streamlit-flavoured Markdown with soft colours (blue headings, red 'likely', orange 'possible')."""
    L = [f"# {report['title']}", "", f"**Technique:** {report['technique']}", "", _h("Findings", rich), ""]
    for organ, o in report["organs"].items():
        L.append(f"**{o['title']}**")
        if o["status"] == "findings":
            L += [f"- {_finding_sentence(e, rich)}" for e in o["findings"]]
        elif o["status"] == "nonspecific":
            L.append(f"- The organ-level screen suggests a possible abnormality in the {organ}, but no specific "
                     "finding was identified with enough confidence to name.")
        else:
            L.append(f"- No abnormality identified in the {organ} above the model's detection thresholds.")
        if o["not_flagged"]:
            L.append(f"- Not flagged by the model: {', '.join(n.lower() for n in o['not_flagged'])}.")
        L.append("")
    imp = []
    for i, line in enumerate(report["impression"], 1):
        word = line.split(" ", 1)[0].lower()
        imp.append(f"{i}. :{COLOR[word]}[**{line}**]" if rich and word in COLOR else f"{i}. {line}")
    L += [_h("Impression", rich), ""] + imp + [""]

    rows = [e for o in report["organs"].values() for e in o["findings"]]
    if rows:
        L += [_h("Model confidence (technical)", rich), "",
              "| Finding | Assessment | Calibrated probability | Model reliability |", "|---|---|---|---|"]
        for e in rows:
            tier_txt = f":{COLOR[e['tier']]}[{e['tier']}]" if rich else e["tier"]
            L.append(f"| {e['finding']} | {tier_txt} | {e['probability']:.1%} | {e['reliability']} |")
        L += ["", _note(CONFIDENCE_NOTE, rich), ""]
    L += ["---", "", _note(report["disclaimer"], rich)]
    return "\n".join(L)
