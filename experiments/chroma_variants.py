# -*- coding: utf-8 -*-
"""Render luma/chroma opacity variants on a video's temporal mean, in colour.

Usage: python chroma_variants.py <video> <sheet.png> cx cy s ky,kc [ky,kc ...]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gwvideo  # noqa: E402
from gwclean import alpha_map  # noqa: E402

T = 220


def to_rgb(y, u, v):
    u = np.repeat(np.repeat(u, 2, 0), 2, 1)
    v = np.repeat(np.repeat(v, 2, 0), 2, 1)
    yy = (y - 16) * 255 / 219
    uu, vv = (u - 128) * 255 / 224, (v - 128) * 255 / 224
    rgb = np.stack([yy + 1.5748 * vv, yy - 0.1873 * uu - 0.4681 * vv, yy + 1.8556 * uu], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main(video, sheet, cx, cy, s, *variants):
    cx, cy, s = float(cx), float(cy), float(s)
    w, h, fps = gwvideo.probe(video)
    r = int(1.6 * s) // 2 * 2
    x0, y0 = int(cx) - r, int(cy) - r
    acc = None
    for k, f in enumerate(gwvideo.read_frames(video, w, h)):
        y, u, v = gwvideo.planes(f, w, h)
        parts = [y[y0:y0 + 2 * r, x0:x0 + 2 * r].astype(np.float64),
                 u[y0 // 2:y0 // 2 + r, x0 // 2:x0 // 2 + r].astype(np.float64),
                 v[y0 // 2:y0 // 2 + r, x0 // 2:x0 // 2 + r].astype(np.float64)]
        acc = parts if acc is None else [a + p for a, p in zip(acc, parts)]
    my, mu, mv = [a / (k + 1) for a in acc]
    unit = alpha_map(my.shape, x0, y0, cx, cy, s, 1.0)
    unit_c = unit.reshape(r, 2, r, 2).mean(axis=(1, 3))
    tiles = []
    for spec in ("0,0",) + variants:
        ky, kc = map(float, spec.split(","))
        ay, ac = ky * unit, kc * unit_c
        img = to_rgb((my - 235 * ay) / (1 - ay), (mu - 128 * ac) / (1 - ac), (mv - 128 * ac) / (1 - ac))
        t = Image.fromarray(img).resize((T, T), Image.NEAREST)
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 80, 11], fill=(0, 0, 0))
        d.text((2, 0), spec, fill=(255, 255, 0))
        tiles.append(t)
    S = Image.new("RGB", (len(tiles) * (T + 3), T), (255, 0, 255))
    for i, t in enumerate(tiles):
        S.paste(t, (i * (T + 3), 0))
    S.save(sheet)
    print(sheet)


if __name__ == "__main__":
    main(*sys.argv[1:])
