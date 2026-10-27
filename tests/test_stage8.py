"""Run:  python tests/test_stage8.py  - animated RadEx banner, scan line, top bar, model ensemble tool"""
import base64
import html as htmllib
import re

import numpy as np
import torch

from radexplain.app import brand
from radexplain.app.art import hero_art_uri
from radexplain.app.style import CSS, hero


def test_banner_schedule_is_ordered_and_keyframes_are_valid():
    s = brand.schedule()
    # the spark and the text overlap a little on purpose (typing starts as the spark vanishes; the spark grows
    # back as the text finishes erasing), so check the real order of events
    stops = [s["morph_start"], s["word_start"], s["morph_end"], s["word_end"]]
    for p in s["phrases"]:
        stops += [p["a"], p["b"], p["c"], p["d"]]
    stops += [s["word_hold_end"], s["back_start"], s["word_erase_end"], s["back_end"], s["total"]]
    assert stops == sorted(stops) and stops[0] > 0                           # every event happens in order
    kf = brand.keyframes()
    assert set(kf) >= {"rx-sparkbox", "rx-typed", "rx-word", "rx-s0", "rx-s1", "rx-s2", "rx-s3"}
    for name, blocks in kf.items():
        flat = [p for pcts, _ in blocks for p in pcts]
        assert flat == sorted(flat) and flat[0] == 0 and flat[-1] == 100 and all(0 <= p <= 100 for p in flat), name
    for i, ph in enumerate(brand.PHRASES):                                   # typing steps = number of characters
        assert f"steps({len(brand.DASH + ph)},end)" in brand.keyframes_css()
    css = brand.css()
    assert css.count("{") == css.count("}") and CSS.count("{") == CSS.count("}")


def test_banner_plays_the_requested_sequence():
    s = brand.schedule()
    assert brand.spark_visible(0.4) and brand.visible_text(0.4) == ""                    # starts as the orange spark
    assert not brand.spark_visible(s["word_end"] + 0.1) and brand.visible_text(s["word_end"] + 0.1) == "RadEx"
    for p, want in zip(s["phrases"], ["RadEx — Radiology Explained", "RadEx — Radiology Interpreted",
                                      "RadEx — Automated Report Generation", "RadEx — Conversational AI Assistant"]):
        assert brand.visible_text(p["b"] + 0.3) == want                                   # fully typed during the hold
        assert brand.visible_text(p["d"] + 0.01) == "RadEx"                               # erased back to RadEx
        mid = brand.visible_text(p["a"] + 0.5 * (p["b"] - p["a"]))
        assert want.startswith(mid) and 5 < len(mid) < len(want)                          # typed letter by letter
    assert brand.visible_text(s["word_hold_end"] - 0.1) == "RadEx"                        # ends on RadEx ...
    assert brand.spark_visible(s["back_end"] + 0.1) and brand.visible_text(s["back_end"] + 0.1) == ""   # ... then the spark
    assert brand.visible_text(s["total"] + 0.4) == ""                                     # and the loop restarts


def test_banner_html_is_safe_and_complete():
    for size in ("lg", "md"):
        h = brand.brand_html(size)
        assert "<script" not in h.lower() and "onload" not in h.lower() and f"rx-{size}" in h and "rx-spark" in h
        text = htmllib.unescape(h).replace("\xa0", " ")
        assert brand.WORD in text and all(brand.DASH + ph in text for ph in brand.PHRASES)
    assert "rx-bl" in brand.topbar_html() and "not a medical device" in brand.topbar_html()
    assert "\n" not in brand.brand_html()                                                  # one line: markdown can't mangle it


def test_scan_line_and_brand_hero_work_with_any_image():
    plain, branded = hero("Title", "sub", ["chip"]), hero("RadEx", "sub", animated_brand=True)
    assert "rx-scan" in plain and "rx-scan" in branded                                     # scan div is independent of the picture
    assert "rx-bl" in branded and "<h1>RadEx</h1>" not in branded and "<h1>Title</h1>" in plain
    assert "rx-scan:before" in CSS and "alternate" in CSS and "rx-scan" in CSS             # sweeps up and down continuously
    from radexplain.app.art import data_uri, xray_svg
    svg = base64.b64decode(data_uri(xray_svg(overlay=True, scan=False)).split(",", 1)[1]).decode()
    assert "<animate" not in svg                                                           # the drawn art no longer scans by itself
    import tempfile
    from pathlib import Path
    from radexplain.app import art
    with tempfile.TemporaryDirectory() as d:                                               # no photo in assets/ -> drawn artwork
        old_root, art.ROOT = art.ROOT, Path(d)
        try:
            assert art.hero_art_uri().startswith("data:image/svg+xml")
            assert "<animate" not in base64.b64decode(art.hero_art_uri().split(",", 1)[1]).decode()
        finally:
            art.ROOT = old_root
    assert hero_art_uri().startswith("data:image/")                                        # whatever is in YOUR assets/ folder works


# ---------------------------------------------------------------- ensemble tool
def test_ensemble_views_and_combination_logic():
    from radexplain.evaluate_ensemble import combo_logits, mean_auroc, rank_combos, views
    x = torch.arange(2 * 3 * 256 * 256, dtype=torch.float32).reshape(2, 3, 256, 256)
    v = views(x, 224)
    assert len(v) == 5 and all(t.shape == (2, 3, 224, 224) for t in v)
    assert torch.equal(v[0], x[:, :, 16:240, 16:240]) and torch.equal(v[4], x[:, :, 32:, 32:])   # centre + 4 corners

    rng = np.random.default_rng(0)
    y = (rng.random((4000, 2)) < 0.3).astype(np.float32); m = np.ones_like(y)
    noisy = lambda s: np.where(y == 1, 1.0, 0.0) + rng.normal(0, s, y.shape)            # two independent noisy scorers
    a, b = noisy(1.4), noisy(1.4)
    a_only = a.copy(); b_missing = b.copy(); b_missing[:, 1] = np.nan                  # model b lacks label 1
    single, both = mean_auroc(a, y, m), mean_auroc(combo_logits([a, b], [0, 1]), y, m)
    assert both > single + 0.01                                                         # averaging independent errors helps
    assert np.allclose(combo_logits([a, b_missing], [0, 1])[:, 1], a[:, 1])             # a missing label uses the other model
    ranked = rank_combos({"a": (a, a), "b": (b, b)}, y, m, y, m)                        # (val logits, test logits) per run
    assert len(ranked) == 3 and ranked[0]["models"] == ("a", "b")                       # 2^2 - 1 subsets, best first


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
