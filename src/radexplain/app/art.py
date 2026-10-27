"""Procedurally drawn chest X-ray artwork (pure SVG, no image files needed, works offline).

xray_svg() is a stylised radiograph: ribs, spine, clavicles, lungs, heart and diaphragm in an X-ray blue
palette, optionally with AI-style outlines and a scanning line. It is used as the blurred background
and as the hero illustration.
"""
import base64
import io

from PIL import Image

from radexplain.config import ROOT


def _ribs(side: int) -> str:
    """side = -1 (left of image) or +1 (right). Ten curved ribs sweeping down and outwards."""
    out = []
    for i in range(10):
        y = 118 + i * 30
        x_out = 200 + side * (112 + min(i, 6) * 8 - max(0, i - 6) * 10)
        y_out = y + 58 + i * 3
        mid_x = 200 + side * 70
        d = f"M{200 + side * 14},{y} C{mid_x},{y - 22} {x_out - side * 8},{y - 4} {x_out},{y_out}"
        out.append(f"<path d='{d}' stroke='#cfe8ff' stroke-opacity='.30' stroke-width='9' fill='none' stroke-linecap='round'/>")
        out.append(f"<path d='{d}' stroke='#ffffff' stroke-opacity='.34' stroke-width='2.2' fill='none' stroke-linecap='round'/>")
    return "".join(out)


def xray_svg(overlay: bool = True, scan: bool = True) -> str:
    vertebrae = "".join(
        f"<rect x='188' y='{70 + i * 19}' width='24' height='13' rx='4' fill='#d7ecff' fill-opacity='.38'/>"
        for i in range(19))
    clav = ("<path d='M196,96 C160,80 118,82 66,104' stroke='#e5f3ff' stroke-opacity='.5' stroke-width='7' fill='none' stroke-linecap='round'/>"
            "<path d='M204,96 C240,80 282,82 334,104' stroke='#e5f3ff' stroke-opacity='.5' stroke-width='7' fill='none' stroke-linecap='round'/>")
    lung_l = "M178,112 C130,108 78,150 70,250 C64,330 92,378 150,384 C176,386 184,350 182,300 Z"
    lung_r = "M222,112 C270,108 322,150 330,250 C336,330 308,378 250,384 C224,386 216,350 218,300 Z"
    heart = "M204,262 C250,238 300,268 292,330 C286,374 240,396 214,372 C196,356 192,290 204,262 Z"
    ai = ""
    if overlay:
        ai = (f"<path d='{lung_l}' fill='none' stroke='#22d3ee' stroke-width='3' stroke-dasharray='9 6'/>"
              f"<path d='{lung_r}' fill='none' stroke='#22d3ee' stroke-width='3' stroke-dasharray='9 6'/>"
              f"<path d='{heart}' fill='rgba(52,211,153,.10)' stroke='#34d399' stroke-width='3'/>"
              "<circle cx='112' cy='318' r='26' fill='rgba(245,158,11,.14)' stroke='#f59e0b' stroke-width='3.5'/>"
              "<circle cx='296' cy='196' r='20' fill='rgba(239,68,68,.14)' stroke='#ef4444' stroke-width='3.5'/>")
    scan_line = ""
    if scan:
        scan_line = ("<rect x='0' y='-30' width='400' height='30' fill='url(#scan)'>"
                     "<animate attributeName='y' values='-30;470' dur='4.5s' repeatCount='indefinite'/></rect>")
    return f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 460' preserveAspectRatio='xMidYMid slice'>
<defs>
<radialGradient id='body' cx='50%' cy='45%' r='70%'><stop offset='0' stop-color='#3b7fb8'/><stop offset='.55' stop-color='#14416f'/><stop offset='1' stop-color='#051a33'/></radialGradient>
<linearGradient id='scan' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#67e8f9' stop-opacity='0'/><stop offset='.85' stop-color='#67e8f9' stop-opacity='.55'/><stop offset='1' stop-color='#ecfeff' stop-opacity='.95'/></linearGradient>
<filter id='soft' x='-20%' y='-20%' width='140%' height='140%'><feGaussianBlur stdDeviation='5'/></filter>
<filter id='soft2'><feGaussianBlur stdDeviation='1.1'/></filter>
</defs>
<rect width='400' height='460' fill='url(#body)'/>
<path d='M40,460 C60,400 120,372 200,372 C280,372 340,400 360,460 Z' fill='#9fd0f5' fill-opacity='.20' filter='url(#soft)'/>
<g filter='url(#soft)'><path d='{lung_l}' fill='#020d1c' fill-opacity='.62'/><path d='{lung_r}' fill='#020d1c' fill-opacity='.62'/></g>
<path d='{heart}' fill='#b9dcff' fill-opacity='.30' filter='url(#soft)'/>
<g filter='url(#soft2)'>{_ribs(-1)}{_ribs(1)}{clav}{vertebrae}</g>
<path d='M70,384 C120,352 170,356 190,392 M330,384 C280,350 232,352 212,392' stroke='#e5f3ff' stroke-opacity='.35' stroke-width='5' fill='none'/>
{ai}
{scan_line}
</svg>"""


def data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


CUSTOM_NAMES = ("xray_hero.png", "xray_hero.jpg", "xray_hero.jpeg", "xray_hero.webp")


def find_custom_art():
    """Your own realistic X-ray photo: put it at <project>/assets/xray_hero.png (or .jpg/.jpeg/.webp)."""
    for name in CUSTOM_NAMES:
        f = ROOT / "assets" / name
        if f.exists():
            return f
    return None


def photo_uri(path, max_side: int):
    """Photo -> small JPEG data URI (so it can be used as a CSS background without any web server)."""
    im = Image.open(path)
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def hero_art_uri() -> str:
    f = find_custom_art()
    try:
        return photo_uri(f, 1400) if f else data_uri(xray_svg(overlay=True, scan=False))
    except Exception:                                   # unreadable file: fall back to the drawn artwork
        return data_uri(xray_svg(overlay=True, scan=False))


def bg_art_uri() -> str:
    f = find_custom_art()
    try:
        return photo_uri(f, 1000) if f else data_uri(xray_svg(overlay=False, scan=False))
    except Exception:
        return data_uri(xray_svg(overlay=False, scan=False))
