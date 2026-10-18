# -*- coding: utf-8 -*-
"""Hand-fixing toolkit for single images (used per hard case).

  locate <img>                         sparkle candidates near the canonical spot
  invert <in> <out> cx,cy,s,k[,sigma]  un-blend a sparkle (k<0 undoes a dark ghost)
  clone  <in> <out> cx,cy,s[,grow] [nosrc x0,y0,x1,y1 ...]
                                       clone-fill the sparkle shape (+grow px)
  rect   <in> <out> x0,y0,x1,y1 [nosrc x0,y0,x1,y1 ...]
                                       clone-fill a rectangle
  blob   <in> <out> x0,y0,x1,y1 lum_thr[,grow] [nosrc ...]
                                       clone-fill the largest region brighter
                                       than lum_thr inside the box
  zoom   <sheet.png> cx,cy,r <img> [<img> ...]
                                       side-by-side 4x zoom of the same square
Inputs may be outputs of earlier steps, so fixes chain. nosrc boxes are never
used as clone sources (keeps HUD/text from being copied into the fill).
"""
import os
import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gwclean as g  # noqa: E402


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def nosrc_mask(shape, boxes):
    m = np.zeros(shape[:2], bool)
    for b in boxes:
        x0, y0, x1, y1 = (int(v) for v in b.split(","))
        m[max(0, y0):y1, max(0, x0):x1] = True
    return m


def cmd_locate(img):
    rgb = load(img).astype(np.float64)
    h, w = rgb.shape[:2]
    e = g.expected_size(w, h)
    gray = g.luminance(rgb)
    found = []
    for sign in (1, -1):
        for f in np.arange(0.5, 1.45, 0.05):
            s = e * f
            for dx in np.arange(-1.2, 1.21, 0.1):
                for dy in np.arange(-1.2, 1.21, 0.1):
                    cx, cy = w - 1.1667 * e + dx * e, h - 1.1667 * e + dy * e
                    if cx + s / 2 > w or cy + s / 2 > h:
                        continue
                    x0, y0, x1, y1 = g._window(gray, cx, cy, s, 0.3)
                    v = sign * g.ncc_at(gray[y0:y1, x0:x1],
                                        g.alpha_map((y1 - y0, x1 - x0), x0, y0, cx, cy, s, 1.0))
                    found.append((v, sign, cx, cy, s))
    found.sort(reverse=True)
    kept = []
    for v, sign, cx, cy, s in found:
        if all(np.hypot(cx - k[2], cy - k[3]) > 0.4 * max(s, k[4]) for k in kept):
            kept.append((v, sign, cx, cy, s))
        if len(kept) == 6:
            break
    print(f"image {w}x{h}, expected size {e:.1f}, canonical centre ({w - 1.1667 * e:.1f},{h - 1.1667 * e:.1f})")
    for v, sign, cx, cy, s in kept:
        print(f"  {'bright' if sign > 0 else 'dark  '} ncc={v:.2f}  cx,cy,s = {cx:.1f},{cy:.1f},{s:.1f}")


def cmd_invert(src, dst, spec):
    p = [float(v) for v in spec.split(",")]
    if len(p) == 4:
        p.append(0.0)
    out = g.apply_inversion(load(src).astype(np.float64), p)
    g.save_like(np.clip(np.rint(out), 0, 255).astype(np.uint8), None, dst)


def cmd_clone(src, dst, spec, *rest):
    p = [float(v) for v in spec.split(",")]
    cx, cy, s = p[:3]
    grow = int(p[3]) if len(p) > 3 else 0
    rgb = load(src)
    mask = g.star_mask(rgb.shape, [(int(round(cx - s / 2)), int(round(cy - s / 2)), int(round(s)), 1.0)])
    if grow:
        mask = ndi.binary_dilation(mask, iterations=grow)
    boxes = list(rest[1:]) if rest and rest[0] == "nosrc" else []
    g.save_like(g.fill(rgb, mask, nosrc_mask(rgb.shape, boxes) if boxes else None), None, dst)


def cmd_rect(src, dst, box, *rest):
    rgb = load(src)
    mask = nosrc_mask(rgb.shape, [box])
    boxes = list(rest[1:]) if rest and rest[0] == "nosrc" else []
    g.save_like(g.fill(rgb, mask, nosrc_mask(rgb.shape, boxes) if boxes else None), None, dst)


def cmd_blob(src, dst, box, thr, *rest):
    rgb = load(src)
    x0, y0, x1, y1 = (int(v) for v in box.split(","))
    t = [float(v) for v in thr.split(",")]
    lum = g.luminance(rgb[y0:y1, x0:x1].astype(np.float64))
    lab, k = ndi.label(lum > t[0])
    if k == 0:
        raise SystemExit("no pixel above threshold in box")
    blob = lab == (1 + int(np.argmax(ndi.sum(lum > t[0], lab, range(1, k + 1)))))
    blob = ndi.binary_dilation(ndi.binary_fill_holes(blob), iterations=int(t[1]) if len(t) > 1 else 3)
    mask = np.zeros(rgb.shape[:2], bool)
    mask[y0:y1, x0:x1] = blob
    boxes = list(rest[1:]) if rest and rest[0] == "nosrc" else []
    g.save_like(g.fill(rgb, mask, nosrc_mask(rgb.shape, boxes) if boxes else None), None, dst)


def cmd_zoom(sheet, spec, *imgs):
    cx, cy, r = (float(v) for v in spec.split(","))
    box = (int(cx - r), int(cy - r), int(cx + r), int(cy + r))
    T = 320
    S = Image.new("RGB", (len(imgs) * (T + 4), T + 14), (255, 0, 255))
    d = ImageDraw.Draw(S)
    for i, p in enumerate(imgs):
        S.paste(Image.open(p).convert("RGB").crop(box).resize((T, T), Image.LANCZOS), (i * (T + 4), 14))
        d.text((i * (T + 4) + 2, 1), f"{i}: {os.path.basename(os.path.dirname(p))}/"[:52], fill=(0, 0, 0))
    S.save(sheet)
    print(sheet)


if __name__ == "__main__":
    {"locate": cmd_locate, "invert": cmd_invert, "clone": cmd_clone, "rect": cmd_rect,
     "blob": cmd_blob, "zoom": cmd_zoom}[sys.argv[1]](*sys.argv[2:])
