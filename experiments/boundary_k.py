# -*- coding: utf-8 -*-
"""Estimate overlay opacity from the step across the sparkle boundary.

With the right opacity the un-blended frame is continuous across the sparkle
edge; texture adds zero-mean noise to the inside/outside difference, so the
median over the boundary is a robust, texture-insensitive criterion (unlike
total variation, which penalises the 1/(1-a) texture amplification and so
biases opacity low on static textured backgrounds).

Usage: python boundary_k.py <video> <cx> <cy> <size>
"""
import os
import sys

import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gwclean  # noqa: E402
import gwvideo  # noqa: E402


def boundary_k(mean_y, ox, oy, cx, cy, s, white=235.0):
    unit = gwclean.alpha_map(mean_y.shape, ox, oy, cx, cy, s, 1.0)
    star = unit > 0.5
    inner = star & ~ndi.binary_erosion(star, iterations=3)
    inner &= ndi.binary_erosion(star, iterations=1)       # skip the soft edge pixel
    outer = ndi.binary_dilation(star, iterations=4) & ~ndi.binary_dilation(star, iterations=1)
    edge = ndi.binary_dilation(star, iterations=1) & ~ndi.binary_erosion(star, iterations=1)
    g = 2.0
    best = None
    for k in np.arange(0.0, 0.9, 0.005):
        a = k * unit
        r = (mean_y - white * a) / (1 - a)
        mi = ndi.gaussian_filter(r * inner, g) / np.maximum(ndi.gaussian_filter(inner * 1.0, g), 1e-6)
        mo = ndi.gaussian_filter(r * outer, g) / np.maximum(ndi.gaussian_filter(outer * 1.0, g), 1e-6)
        sel = edge & (ndi.gaussian_filter(inner * 1.0, g) > 0.05) & (ndi.gaussian_filter(outer * 1.0, g) > 0.05)
        d = np.median((mi - mo)[sel])
        if best is None or abs(d) < abs(best[1]):
            best = (k, d)
    return best


def main(video, cx, cy, s):
    cx, cy, s = float(cx), float(cy), float(s)
    w, h, fps = gwvideo.probe(video)
    pad = int(s)
    x0, y0 = int(cx - s / 2) - pad, int(cy - s / 2) - pad
    x1, y1 = min(w, int(cx + s / 2) + pad), min(h, int(cy + s / 2) + pad)
    ys = np.stack([f[:w * h].reshape(h, w)[y0:y1, x0:x1] for f in gwvideo.read_frames(video, w, h)])
    m = ys.astype(np.float64).mean(axis=0)
    k, d = boundary_k(m, x0, y0, cx, cy, s)
    thirds = [boundary_k(ys[i::3].astype(np.float64).mean(axis=0), x0, y0, cx, cy, s)[0] for i in range(3)]
    print(f"k={k:.3f} resid={d:+.2f} thirds={[round(t, 3) for t in thirds]}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
