# -*- coding: utf-8 -*-
"""Inversion with fixed opacities at the best-correlating spot, for hand-picking.

Usage: python fixed_k.py <img> <sheet.png> <k,sigma> [<k,sigma> ...]
Writes <sheet.png> (original first, then one tile per variant) and saves each
variant next to it as <sheet>_<k>_<sigma>.<ext of img>.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gwclean as g  # noqa: E402

T = 200


def locate(gray, w, h):
    e = g.expected_size(w, h)
    best = None
    for f in np.arange(0.7, 1.31, 0.05):
        s = e * f
        for dx in np.arange(-0.6, 0.61, 0.1):
            for dy in np.arange(-0.6, 0.61, 0.1):
                cx, cy = w - 1.1667 * e + dx * e, h - 1.1667 * e + dy * e
                x0, y0, x1, y1 = g._window(gray, cx, cy, s, 0.3)
                v = g.ncc_at(gray[y0:y1, x0:x1], g.alpha_map((y1 - y0, x1 - x0), x0, y0, cx, cy, s, 1.0))
                if best is None or v > best[0]:
                    best = (v, cx, cy, s)
    # sub-pixel refinement
    v, cx, cy, s = best
    for ds in np.arange(-0.04, 0.041, 0.01):
        for dx in np.arange(-1.5, 1.51, 0.25):
            for dy in np.arange(-1.5, 1.51, 0.25):
                ss = s * (1 + ds)
                x0, y0, x1, y1 = g._window(gray, cx + dx, cy + dy, ss, 0.3)
                vv = g.ncc_at(gray[y0:y1, x0:x1],
                              g.alpha_map((y1 - y0, x1 - x0), x0, y0, cx + dx, cy + dy, ss, 1.0))
                if vv > best[0]:
                    best = (vv, cx + dx, cy + dy, ss)
    return best, e


def main(img, sheet, *variants):
    im = Image.open(img).convert("RGB")
    rgb = np.asarray(im).astype(np.float64)
    h, w = rgb.shape[:2]
    (v, cx, cy, s), e = locate(g.luminance(rgb), w, h)
    r = 1.9 * e
    box = (int(cx - r), int(cy - r), int(cx + r), int(cy + r))
    tiles = [(im.crop(box).resize((T, T)), "orig")]
    stem, ext = os.path.splitext(sheet)[0], os.path.splitext(img)[1]
    for spec in variants:
        k, sg = map(float, spec.split(","))
        out = np.clip(np.rint(g.apply_inversion(rgb, [cx, cy, s, k, sg])), 0, 255).astype(np.uint8)
        g.save_like(out, None, f"{stem}_{k}_{sg}{ext}")
        tiles.append((Image.fromarray(out).crop(box).resize((T, T)), spec))
    S = Image.new("RGB", (len(tiles) * (T + 3), T), (255, 0, 255))
    for i, (t, lab) in enumerate(tiles):
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 70, 11], fill=(0, 0, 0))
        d.text((2, 0), lab, fill=(255, 255, 0))
        S.paste(t, (i * (T + 3), 0))
    S.save(sheet)
    print(f"spot ncc={v:.2f} c=({cx:.1f},{cy:.1f}) s={s:.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
