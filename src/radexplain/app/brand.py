"""The animated RadEx banner: an orange spark morphs into typed text and back again (pure CSS, no JavaScript).

Loop (about 24 s):  spark  ->  "RadEx"  ->  "RadEx — Radiology Explained"  ->  "RadEx — Radiology Interpreted"
                    ->  "RadEx — Automated Report Generation"  ->  "RadEx — Conversational AI Assistant"
                    ->  "RadEx"  ->  spark  ->  (repeat)
One schedule() drives both the generated CSS keyframes and visible_text() (used by the tests), so what the
tests check is what the browser plays.
"""
import html

WORD = "RadEx"
DASH = " — "
PHRASES = ("Radiology Explained", "Radiology Interpreted", "Automated Report Generation", "Conversational AI Assistant")
SPARK_COLOR = "#f97316"
SPARK_SVG = ("<svg class='rx-spark' viewBox='0 0 24 24'><g stroke='%s' stroke-width='2.5' stroke-linecap='round'>"
             "<line x1='12' y1='2' x2='12' y2='22'/><line x1='2' y1='12' x2='22' y2='12'/>"
             "<line x1='5' y1='5' x2='19' y2='19'/><line x1='19' y1='5' x2='5' y2='19'/></g></svg>" % SPARK_COLOR)

# seconds
T_ALONE, T_MORPH, T_WORD, T_TYPE, T_HOLD, T_ERASE = 1.4, 0.6, 0.7, 1.6, 1.8, 0.9
T_WORD_HOLD, T_WORD_ERASE, T_BACK, T_REST = 1.0, 0.6, 0.6, 1.8


def schedule() -> dict:
    s = {"morph_start": T_ALONE}
    s["morph_end"] = s["morph_start"] + T_MORPH                      # the spark has shrunk away
    s["word_start"] = s["morph_end"] - 0.1                           # "RadEx" starts typing as it vanishes
    s["word_end"] = s["word_start"] + T_WORD
    t, phrases = s["word_end"] + 0.3, []
    for ph in PHRASES:
        a, b, c, d = t, t + T_TYPE, t + T_TYPE + T_HOLD, t + T_TYPE + T_HOLD + T_ERASE
        phrases.append({"text": DASH + ph, "n": len(DASH + ph), "a": a, "b": b, "c": c, "d": d})
        t = d
    s["phrases"] = phrases
    s["word_hold_end"] = t + T_WORD_HOLD
    s["word_erase_end"] = s["word_hold_end"] + T_WORD_ERASE
    s["back_start"] = s["word_erase_end"] - 0.2                      # the spark grows back
    s["back_end"] = s["back_start"] + T_BACK
    s["total"] = s["back_end"] + T_REST
    return s


def _prog(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


def visible_text(t: float) -> str:
    """What the typed line shows at second t of the loop (the spark is separate)."""
    s, t = schedule(), t % schedule()["total"]
    word = int(len(WORD) * min(_prog(t, s["word_start"], s["word_end"]), 1 - _prog(t, s["word_hold_end"], s["word_erase_end"])) + 1e-9)
    suffix = ""
    for p in s["phrases"]:
        k = int(p["n"] * min(_prog(t, p["a"], p["b"]), 1 - _prog(t, p["c"], p["d"])) + 1e-9)
        if k:
            suffix = p["text"][:k]
    return WORD[:word] + (suffix if word == len(WORD) else "")


def spark_visible(t: float) -> bool:
    s, t = schedule(), t % schedule()["total"]
    return t < s["morph_start"] + T_MORPH / 2 or t >= s["back_start"] + T_BACK / 2


def keyframes() -> dict:
    """name -> [(list of percentages, declarations)], percentages ascending."""
    s = schedule()
    pc = lambda t: round(100 * t / s["total"], 3)
    sp = "width:var(--sp);opacity:1;transform:scale(1);"
    gone = "width:0;opacity:0;transform:scale(.2);"
    kf = {
        "rx-sparkbox": [([0, pc(s["morph_start"])], sp), ([pc(s["morph_end"]), pc(s["back_start"])], gone), ([pc(s["back_end"]), 100], sp)],
        "rx-typed": [([0, pc(s["word_start"] - 0.01)], "opacity:0;"), ([pc(s["word_start"]), pc(s["word_erase_end"] - 0.05)], "opacity:1;"),
                     ([pc(s["word_erase_end"]), 100], "opacity:0;")],
        "rx-word": [([0, pc(s["word_start"])], f"width:0ch;animation-timing-function:steps({len(WORD)},end);"),
                    ([pc(s["word_end"]), pc(s["word_hold_end"])], f"width:{len(WORD)}ch;animation-timing-function:steps({len(WORD)},end);"),
                    ([pc(s["word_erase_end"]), 100], "width:0ch;")],
    }
    for i, p in enumerate(s["phrases"]):
        kf[f"rx-s{i}"] = [([0, pc(p["a"])], f"width:0ch;animation-timing-function:steps({p['n']},end);"),
                          ([pc(p["b"]), pc(p["c"])], f"width:{p['n']}ch;animation-timing-function:steps({p['n']},end);"),
                          ([pc(p["d"]), 100], "width:0ch;")]
    return kf


def keyframes_css() -> str:
    out = []
    for name, blocks in keyframes().items():
        body = "".join(f"{','.join(f'{p:g}%' for p in pcts)}{{{decl}}}" for pcts, decl in blocks)
        out.append(f"@keyframes {name}{{{body}}}")
    out.append("@keyframes rx-spin{to{transform:rotate(360deg);}}@keyframes rx-blink{50%{border-right-color:transparent;}}")
    return "".join(out)


def css() -> str:
    s, total = schedule(), schedule()["total"]
    longest = max(p["n"] for p in s["phrases"])
    anims = "".join(f".rx-bl .rx-s{i}{{animation:rx-s{i} {total}s linear infinite,rx-blink .9s steps(1,end) infinite;}}" for i in range(len(PHRASES)))
    return (f".rx-bl{{display:inline-flex;align-items:center;font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,Consolas,monospace;"
            f"line-height:1.15;white-space:pre;--w:2.3rem;--s:1.05rem;--sp:2.1rem;--spark:{SPARK_COLOR};}}"
            ".rx-bl.rx-md{--w:1.45rem;--s:.92rem;--sp:1.5rem;}"
            f".rx-bl .rx-sparkbox{{display:inline-block;width:var(--sp);height:var(--sp);animation:rx-sparkbox {total}s ease-in-out infinite;transform-origin:center;}}"
            ".rx-bl .rx-spark{display:block;width:100%;height:100%;animation:rx-spin 7s linear infinite;}"
            f".rx-bl .rx-typed{{display:inline-flex;align-items:flex-end;opacity:0;animation:rx-typed {total}s linear infinite;}}"
            f".rx-bl .rx-word{{display:inline-block;overflow:hidden;white-space:pre;width:0ch;font-size:var(--w);font-weight:800;animation:rx-word {total}s linear infinite;}}"
            f".rx-bl .rx-suffix{{position:relative;display:inline-block;width:{longest}ch;height:calc(var(--s)*1.5);font-size:var(--s);}}"
            ".rx-bl .rx-s{position:absolute;left:0;bottom:.1em;overflow:hidden;white-space:pre;width:0ch;height:1.5em;line-height:1.5em;"
            "font-weight:500;opacity:.95;border-right:.14em solid var(--spark);}"
            + anims + keyframes_css() +
            ".rx-topbar{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:0 4px 12px;margin:-10px 0 18px;"
            "border-bottom:1px solid var(--line);color:var(--navy);}"
            ".rx-topbar .rx-tag{font-size:.78rem;color:#5b7692;border:1px solid var(--line);padding:3px 12px;border-radius:999px;background:#fff;white-space:nowrap;}"
            ".rx-hero .rx-bl{color:#fff;}"
            "@media (prefers-reduced-motion:reduce){.rx-bl .rx-sparkbox,.rx-bl .rx-spark,.rx-bl .rx-typed,.rx-bl .rx-word,.rx-bl .rx-s,"
            ".rx-hero .rx-scan:before{animation:none !important;}.rx-bl .rx-sparkbox{display:none;}.rx-bl .rx-typed{opacity:1;}"
            f".rx-bl .rx-word{{width:{len(WORD)}ch;}}.rx-bl .rx-s0{{width:{s['phrases'][0]['n']}ch;border-right:0;}}}}")


def brand_html(size: str = "lg") -> str:
    """size: 'lg' (login hero) or 'md' (top bar). One line, no scripts."""
    spans = "".join(f"<span class='rx-s rx-s{i}'>{html.escape(DASH + ph).replace(' ', '&nbsp;')}</span>" for i, ph in enumerate(PHRASES))
    return (f"<span class='rx-bl rx-{size}'><span class='rx-sparkbox'>{SPARK_SVG}</span>"
            f"<span class='rx-typed'><span class='rx-word'>{WORD}</span><span class='rx-suffix'>{spans}</span></span></span>")


def topbar_html() -> str:
    return f"<div class='rx-topbar'>{brand_html('md')}<span class='rx-tag'>Research prototype · not a medical device</span></div>"
