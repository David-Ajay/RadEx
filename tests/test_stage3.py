"""Run:  python tests/test_stage3.py"""
import re

from radexplain.report.generator import build_report, reliability_from_auroc, render_markdown, tier


def F(name, p, thr=0.10, hi=0.30):
    return {"finding": name, "probability": p, "threshold": thr, "high_threshold": hi, "flagged": p >= thr}


def make(lung_items, heart_items, lungs_screen, heart_screen):
    org = lambda a: {"probability": 0.5, "threshold": 0.4, "affected": a}
    return {"organs": {"lungs": org(lungs_screen), "heart": org(heart_screen)},
            "findings": {"lungs": lung_items, "heart": heart_items}}


def test_tiers():
    assert tier(F("Edema", 0.05)) == "not_detected"
    assert tier(F("Edema", 0.10)) == "possible"
    assert tier(F("Edema", 0.30)) == "likely"
    assert tier({"probability": 0.5, "threshold": 0.2}) == "likely"        # fallback high = 2 x threshold


def test_normal_study():
    r = build_report(make([F("Edema", 0.01)], [F("Cardiomegaly", 0.01)], False, False))
    assert all(o["status"] == "no_findings" for o in r["organs"].values())
    assert "No abnormality identified above" in r["impression"][0]
    assert "Likely" not in render_markdown(r)


def test_findings_override_organ_screen_and_are_ordered():
    # organ screen says "no" but findings fired (the mismatch seen on real output) -> findings win
    r = build_report(make([F("Atelectasis", 0.15), F("Edema", 0.40), F("Mass", 0.01)],
                          [F("Cardiomegaly", 0.01)], False, False))
    assert r["organs"]["lungs"]["status"] == "findings"
    assert r["impression"][0].startswith("Likely pulmonary edema")        # 'likely' ranks first
    assert r["impression"][1].startswith("Possible atelectasis")


def test_nonspecific_when_only_organ_screen_fires():
    r = build_report(make([F("Edema", 0.01)], [F("Cardiomegaly", 0.01)], True, False))
    assert r["organs"]["lungs"]["status"] == "nonspecific"
    assert "non-specific" in r["impression"][0]


def test_no_percentages_in_findings_or_impression_and_low_reliability_caution():
    r = build_report(make([F("Pneumonia", 0.12)], [F("Cardiomegaly", 0.01)], True, False),
                     reliability={"Pneumonia": reliability_from_auroc(0.709)})
    md = render_markdown(r)
    body = md.split("## Model confidence")[0]
    assert not re.search(r"\d\s*%", body)                                  # numbers only in the table
    assert "Caution" in body and "12.0%" in md


def test_note_callouts_and_rich_mode():
    r = build_report(make([F("Edema", 0.40), F("Atelectasis", 0.15)], [F("Cardiomegaly", 0.01)], True, False))
    plain, rich = render_markdown(r), render_markdown(r, rich=True)
    assert plain.count("> **Note:**") == 2 and plain.rstrip().endswith(r["disclaimer"])
    assert ":red[**Likely pulmonary edema**]" in rich and ":orange[**Possible atelectasis**]" in rich
    assert "## :blue[Findings]" in rich and ":orange-background[**Note:** This report was generated" in rich
    assert ":red[" not in plain and "## Findings" in plain


def test_reliability_classes():
    assert [reliability_from_auroc(x) for x in (0.9, 0.8, 0.7, None)] == ["higher", "moderate", "lower", "unknown"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
