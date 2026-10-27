"""Turn the report JSON and the glossary into small text chunks that can be retrieved."""
import json
from pathlib import Path

from radexplain.report.generator import build_report

TIER_TEXT = {"likely": "likely", "possible": "possible"}


def load_glossary(path) -> list[dict]:
    return json.loads(Path(path).read_text())


def report_chunks(report: dict) -> list[dict]:
    """One chunk per idea in the report. Each: {id, title, source, text, keywords, always}."""
    ch = []
    imp = " ".join(report["impression"][:-1]) or report["impression"][0]
    ch.append({"id": "R-summary", "title": "Report summary", "source": "report", "always": True,
               "keywords": "summary overview report impression explain whole report main result",
               "text": f"The report's impression: {imp}"})
    for organ, o in report["organs"].items():
        if o["status"] == "findings":
            names = ", ".join(f"{e['tier']} {e['term']}" for e in o["findings"])
            txt = f"For the {o['title'].lower()}, the software flagged: {names}."
        elif o["status"] == "nonspecific":
            txt = (f"For the {o['title'].lower()}, the organ-level screen suggested a possible abnormality, "
                   "but no specific finding was confident enough to name.")
        else:
            txt = f"For the {o['title'].lower()}, nothing passed the software's detection thresholds."
        ch.append({"id": f"R-organ-{organ}", "title": f"Your report: {o['title']}", "source": "report",
                   "keywords": f"{organ} {o['title']}", "text": txt})
        for e in o["findings"]:
            txt = (f"Your report lists {e['tier']} {e['term']} ({organ}). In plain words, that means {e['meaning']}. "
                   f"The software's calibrated probability was {e['probability']:.1%}, which passed its "
                   f"'{e['tier']}' threshold. Model reliability for this finding: {e['reliability']}.")
            ch.append({"id": f"R-find-{e['finding']}", "title": f"Your report: {e['term']}", "source": "report",
                       "tier": e["tier"],
                       "keywords": f"{e['finding']} {e['term']} {organ} flagged in my report",
                       "text": txt})
        if o["not_flagged"]:
            ch.append({"id": f"R-clear-{organ}", "title": f"Not flagged ({organ})", "source": "report",
                       "keywords": "not flagged normal clear nothing found " + " ".join(o["not_flagged"]),
                       "text": "Not flagged by the software: " + ", ".join(n.lower() for n in o["not_flagged"]) +
                               ". This does not prove these are absent."})
    ch.append({"id": "R-disclaimer", "title": "Report disclaimer", "source": "report",
               "keywords": "disclaimer trust validated medical device safe diagnosis",
               "text": report["disclaimer"]})
    return ch


def glossary_chunks(glossary: list[dict]) -> list[dict]:
    return [{"id": f"G-{g['id']}", "title": g["term"], "source": "glossary",
             "keywords": " ".join(g["keywords"]) + " " + g["term"], "text": f"{g['term']}: {g['text']}"}
            for g in glossary]
