"""Soft 'highlighter' shading over the X-ray, only where the model found evidence for a flagged finding.

For every flagged finding (max 3) we take its CAM and:
  1. smooth it and keep only the strongest part (top of the map, never more than ~25% of the chest area),
  2. drop tiny specks, shade the rest in a transparent red (likely) or amber (possible) with a thin outline,
  3. put a small numbered tag on each region (the legend says which finding each number is).
The shading is semi-transparent, so the X-ray underneath stays visible. It shows where the model LOOKED:
an approximation, not a precise outline of a lesion.
"""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from radexplain.report.generator import TIER_ORDER, tier
from radexplain.report.templates import PHRASES

TIER_RGB = {"likely": (239, 68, 68), "possible": (245, 158, 11)}
TIER_HEX = {"likely": "#ef4444", "possible": "#f59e0b"}


def thorax_roi(size: int) -> np.ndarray:
    """Rough chest area (an ellipse) so shading never lands on the image border, corners or text markers."""
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    return (((xx - 0.5) / 0.47) ** 2 + ((yy - 0.53) / 0.45) ** 2) <= 1


def cam_to_canvas(cam: np.ndarray, store: int = 256, crop: int = 224) -> np.ndarray:
    """The model sees the central `crop` x `crop` part of the `store` x `store` image. Put its (h, w) CAM back
    into full-image coordinates (zero outside the crop)."""
    up = np.asarray(Image.fromarray(np.maximum(cam, 0).astype(np.float32)).resize((crop, crop), Image.BILINEAR))
    pad = (store - crop) // 2
    out = np.zeros((store, store), np.float32)
    out[pad:pad + crop, pad:pad + crop] = up
    return out


def highlight_region(cam, store=256, crop=224, thr=0.55, max_area=0.25, min_blob=0.004):
    """-> dict(mask, alpha, threshold, area_pct, peak) in store x store coordinates, or None if there is no evidence."""
    roi = thorax_roi(store)
    m = ndi.gaussian_filter(cam_to_canvas(cam, store, crop), store / 50) * roi
    if m.max() <= 1e-8:
        return None
    m = m / m.max()
    roi_area = roi.sum()
    peak = np.unravel_index(np.argmax(m), m.shape)
    cap = int(max_area * roi_area)
    t = thr
    while t < 0.95 and (m >= t).sum() > cap:
        t += 0.05
    binary = m >= t
    if binary.sum() > cap:                                         # very diffuse map: hard cap, ties broken by distance to the peak
        dist = np.hypot(*(np.indices(m.shape) - np.array(peak).reshape(2, 1, 1)))
        score = np.where(roi, m - 1e-5 * dist / store, -1.0)
        binary = score >= np.partition(score.ravel(), -cap)[-cap]
        t = float(m[binary].min())
    labels, n = ndi.label(binary)
    if n == 0:
        return None
    keep = np.zeros_like(binary)
    for i in range(1, n + 1):
        comp = labels == i
        if comp.sum() >= min_blob * m.size or labels[peak] == i:    # drop specks, always keep the strongest blob
            keep |= comp
    soft = np.clip((m - 0.85 * t) / (1 - 0.85 * t), 0, 1) ** 0.75 * ndi.binary_dilation(keep, iterations=max(2, store // 50))
    alpha = ndi.gaussian_filter(soft, store / 120).astype(np.float32)
    return {"mask": keep, "alpha": alpha, "threshold": float(t), "area_pct": float(100 * keep.sum() / roi_area),
            "peak": (int(peak[0]), int(peak[1]))}


def flagged_findings(result: dict, max_findings: int = 3) -> list[dict]:
    """The findings worth shading: 'likely' first, then 'possible', each by probability."""
    items = []
    for organ, findings in result["findings"].items():
        for it in findings:
            t = tier(it)
            if t != "not_detected":
                items.append({"finding": it["finding"], "term": PHRASES.get(it["finding"], (it["finding"].lower(),))[0],
                              "tier": t, "probability": it["probability"], "organ": organ})
    items.sort(key=lambda d: (TIER_ORDER[d["tier"]], -d["probability"]))
    return items[:max_findings]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                              # very old Pillow
        return ImageFont.load_default()


def render_overlay(base: Image.Image, items: list[dict], store=256, crop=224, opacity=0.58, quality=90):
    """items: [{finding, term, tier, cam, ...}] most important first. -> {"jpeg": bytes, "legend": [...]} or None."""
    rgb = base.convert("RGB")
    W, H = rgb.size
    arr = np.asarray(rgb).astype(np.float32)
    regions = []
    for it in items:
        r = highlight_region(it["cam"], store, crop)
        if r is not None:
            regions.append((it, r))
    if not regions:
        return None
    resize = lambda a, mode, res: np.asarray(Image.fromarray(a).resize((W, H), res)).astype(np.float32)
    centroids = []
    for it, r in reversed(regions):                                # paint the most important finding last (on top)
        a = resize((r["alpha"] * 255).astype(np.uint8), "L", Image.BILINEAR)[..., None] / 255
        color = np.array(TIER_RGB[it["tier"]], np.float32)
        arr = arr * (1 - a * opacity) + color * a * opacity
        mask = resize((r["mask"] * 255).astype(np.uint8), "L", Image.NEAREST) > 127
        edge = ndi.binary_dilation(mask, iterations=max(1, W // 300)) & ~mask
        arr[edge] = 0.3 * arr[edge] + 0.7 * color
        cy, cx = ndi.center_of_mass(mask)
        centroids.append((it, cx, cy))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw, rad, font = ImageDraw.Draw(img), max(10, W // 45), _font(max(12, W // 40))
    order = {id(it): n for n, (it, _) in enumerate(regions, 1)}
    for it, cx, cy in centroids:
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=TIER_RGB[it["tier"]], outline="white", width=2)
        draw.text((cx, cy), str(order[id(it)]), fill="white", font=font, anchor="mm")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    legend = [{"n": order[id(it)], "finding": it["finding"], "term": it["term"], "tier": it["tier"],
               "color": TIER_HEX[it["tier"]], "area_pct": round(r["area_pct"], 1)} for it, r in regions]
    return {"jpeg": buf.getvalue(), "legend": legend}
