"""Coloured PDF version of the report (pure Python, needs `pip install reportlab`).

Colour scheme (soft, not harsh): deep-teal headings, red = 'likely', amber = 'possible',
grey = not flagged, light-amber callout boxes starting with a bold 'Note:'.
"""
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY, TEAL, SLATE = colors.HexColor("#0f2a4a"), colors.HexColor("#0f766e"), colors.HexColor("#64748b")
RED, AMBER = colors.HexColor("#dc2626"), colors.HexColor("#d97706")
RED_BG, AMBER_BG, GREY_BG = colors.HexColor("#fef2f2"), colors.HexColor("#fffbeb"), colors.HexColor("#f1f5f9")
TIER_COLOR = {"likely": RED, "possible": AMBER}
TIER_BG = {"likely": RED_BG, "possible": AMBER_BG}
CONFIDENCE_NOTE = ("Probabilities are calibrated on validation data. Low values can still be flagged for "
                   "rare findings.")

_REPL = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2192": "->",
         "\u2022": "-", "\u00a0": " "}


def _t(text) -> str:
    """Escape for reportlab markup and keep to characters the built-in Helvetica font can draw."""
    text = "".join(_REPL.get(c, c) for c in str(text))
    return escape(text.encode("latin-1", "replace").decode("latin-1"))


def _hex(c) -> str:
    return "#%02x%02x%02x" % (int(c.red * 255), int(c.green * 255), int(c.blue * 255))


def _styles():
    base = dict(fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1e293b"), alignment=TA_LEFT)
    return {
        "body": ParagraphStyle("body", **base),
        "small": ParagraphStyle("small", **{**base, "fontSize": 8.5, "leading": 11.5, "textColor": SLATE}),
        "h2": ParagraphStyle("h2", **{**base, "fontName": "Helvetica-Bold", "fontSize": 13, "leading": 16,
                                       "textColor": TEAL, "spaceBefore": 10, "spaceAfter": 3}),
        "organ": ParagraphStyle("organ", **{**base, "fontName": "Helvetica-Bold", "fontSize": 10.5,
                                             "textColor": NAVY, "spaceBefore": 6, "spaceAfter": 3}),
        "title": ParagraphStyle("title", **{**base, "fontName": "Helvetica-Bold", "fontSize": 20, "leading": 24,
                                             "textColor": colors.white}),
        "sub": ParagraphStyle("sub", **{**base, "fontSize": 10.5, "textColor": colors.HexColor("#cffafe")}),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 9, "leading": 12}),
        "head": ParagraphStyle("head", **{**base, "fontName": "Helvetica-Bold", "fontSize": 9,
                                           "textColor": colors.white}),
    }


def _rule(color=TEAL):
    t = Table([[""]], colWidths=[17 * cm], rowHeights=[2])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.2, color)]))
    return t


def _note_box(text, st):
    box = Table([[Paragraph(f"<b>Note:</b> {_t(text)}", st["body"])]], colWidths=[17 * cm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), AMBER_BG), ("LINEBEFORE", (0, 0), (0, -1), 3, AMBER),
                             ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#fde68a")),
                             ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                             ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return box


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)
    canvas.drawString(2 * cm, 1.1 * cm, "RadEx - research prototype. Not a medical device.")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(report: dict, image_bytes: bytes | None = None, source_name: str | None = None,
              created_at: str | None = None, overlay: dict | None = None) -> bytes:
    """overlay = {"jpeg": bytes, "legend": [...]} from Predictor.explain: the shaded image replaces the plain one."""
    st = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.8 * cm, title=report["title"], author="RadEx")
    when = created_at or datetime.now().strftime("%d %b %Y, %H:%M")
    story = []

    # ---- banner
    banner = Table([[Paragraph("RadEx", st["title"])], [Paragraph(_t(report["title"]), st["sub"])],
                    [Paragraph(_t(f"{when}" + (f"  |  {source_name}" if source_name else "")), st["sub"])]],
                   colWidths=[17 * cm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0e7490")),
                                ("LEFTPADDING", (0, 0), (-1, -1), 14), ("TOPPADDING", (0, 0), (-1, 0), 12),
                                ("BOTTOMPADDING", (0, -1), (-1, -1), 12)]))
    story += [banner, Spacer(1, 10)]

    # ---- image + at a glance
    glance = [Paragraph("<b>At a glance</b>", st["organ"])]
    for o in report["organs"].values():
        top = o["findings"][0]["tier"] if o["findings"] else None
        if top:
            c, txt = TIER_COLOR[top], f"{len(o['findings'])} finding(s) flagged"
        elif o["status"] == "nonspecific":
            c, txt = AMBER, "non-specific signal"
        else:
            c, txt = SLATE, "nothing flagged"
        glance.append(Paragraph(f"<b>{_t(o['title'])}</b>: <font color='{_hex(c)}'><b>{txt}</b></font>", st["body"]))
        glance.append(Spacer(1, 3))
    glance.append(Paragraph(_t("Flags are screening signals, not diagnoses."), st["small"]))
    if overlay and overlay.get("legend"):
        glance += [Spacer(1, 6), Paragraph("<b>Highlighted areas</b>", st["organ"])]
        for e in overlay["legend"]:
            c = TIER_COLOR[e["tier"]]
            glance.append(Paragraph(f"<font color='{_hex(c)}'><b>{e['n']}</b></font>  {_t(e['term'].capitalize())} "
                                    f"<font color='{_hex(c)}'>({e['tier']})</font>", st["body"]))
        glance.append(Spacer(1, 3))
        glance.append(Paragraph(_t("Shaded areas show where the model found the strongest evidence for each flagged "
                                   "finding. This is an approximation, not a precise outline."), st["small"]))
    shown = overlay["jpeg"] if overlay and overlay.get("jpeg") else image_bytes
    if shown:
        try:
            im = RLImage(BytesIO(shown))
            ratio = im.imageHeight / im.imageWidth
            wide = 6.4 if overlay else 5.2
            im.drawWidth, im.drawHeight = wide * cm, wide * cm * ratio
            row = Table([[im, glance]], colWidths=[(wide + 0.6) * cm, (17 - wide - 0.6) * cm])
            row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            story.append(row)
        except Exception:
            story += glance
    else:
        story += glance
    story += [Spacer(1, 4), Paragraph(f"<b>Technique:</b> {_t(report['technique'])}", st["small"])]

    # ---- findings
    story += [Paragraph("Findings", st["h2"]), _rule()]
    for o in report["organs"].values():
        story.append(Paragraph(_t(o["title"]), st["organ"]))
        if o["status"] == "findings":
            for e in o["findings"]:
                c, bg = TIER_COLOR[e["tier"]], TIER_BG[e["tier"]]
                label = f"{'Likely' if e['tier'] == 'likely' else 'Possible'} {e['term']}"
                txt = f"<font color='{_hex(c)}'><b>{_t(label)}</b></font>"
                if e["meaning"]:
                    txt += f" - {_t(e['meaning'])}."
                if e["reliability"] == "lower":
                    txt += f" <i><font color='{_hex(SLATE)}'>(Caution: the model is less reliable at detecting this finding.)</font></i>"
                row = Table([["", Paragraph(txt, st["cell"])]], colWidths=[0.25 * cm, 16.75 * cm])
                row.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), c), ("BACKGROUND", (1, 0), (1, 0), bg),
                                         ("LEFTPADDING", (1, 0), (1, 0), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                         ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                story += [row, Spacer(1, 3)]
        elif o["status"] == "nonspecific":
            story.append(Paragraph(_t("The organ-level screen suggests a possible abnormality, but no specific "
                                      "finding was identified with enough confidence to name."), st["body"]))
        else:
            story.append(Paragraph(_t("No abnormality identified above the model's detection thresholds."), st["body"]))
        if o["not_flagged"]:
            story.append(Paragraph(_t("Not flagged by the model: " + ", ".join(n.lower() for n in o["not_flagged"]) + "."),
                                   st["small"]))

    # ---- impression
    story += [Paragraph("Impression", st["h2"]), _rule()]
    for i, line in enumerate(report["impression"], 1):
        word = line.split(" ", 1)[0].lower()
        if word in TIER_COLOR:
            story.append(Paragraph(f"{i}. <font color='{_hex(TIER_COLOR[word])}'><b>{_t(line)}</b></font>", st["body"]))
        else:
            story.append(Paragraph(f"{i}. {_t(line)}", st["body"]))
        story.append(Spacer(1, 2))

    # ---- confidence table
    rows = [e for o in report["organs"].values() for e in o["findings"]]
    if rows:
        story += [Paragraph("Model confidence (technical)", st["h2"]), _rule()]
        data = [[Paragraph(h, st["head"]) for h in ("Finding", "Assessment", "Calibrated probability", "Model reliability")]]
        for e in rows:
            data.append([Paragraph(_t(e["finding"]), st["cell"]),
                         Paragraph(f"<font color='{_hex(TIER_COLOR[e['tier']])}'><b>{e['tier']}</b></font>", st["cell"]),
                         Paragraph(f"{e['probability']:.1%}", st["cell"]), Paragraph(_t(e["reliability"]), st["cell"])])
        t = Table(data, colWidths=[5 * cm, 3.5 * cm, 4.5 * cm, 4 * cm], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), TEAL),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY_BG]),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
                               ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story += [t, Spacer(1, 8), _note_box(CONFIDENCE_NOTE, st)]

    story += [Spacer(1, 12), _note_box(report["disclaimer"], st)]
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
