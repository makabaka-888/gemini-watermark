# -*- coding: utf-8 -*-
"""Detection-only sweep with loose thresholds, to choose size/position priors.

Usage: python detect_stats.py <src_dir> <pass1_dir> <out.tsv>
Columns: name, base, score, size/E, dx/E, dy/E  where E is the expected
sparkle size (96 px on a 2816-px long side) and dx, dy are measured from the
canonical sparkle centre.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gwclean  # noqa: E402

gwclean.THR_BRIGHT = 0.45
gwclean.THR_DARK = 0.50


def main(src_dir, pass1_dir, out):
    names = sorted(n for n in os.listdir(src_dir)
                   if os.path.splitext(n)[1].lower() in (".png", ".jpg", ".jpeg"))
    with open(out, "w", encoding="utf-8") as fh:
        for n in names:
            p = os.path.join(pass1_dir, n)
            base = "pass1" if os.path.exists(p) else "orig"
            if base == "orig":
                p = os.path.join(src_dir, n)
            rgb = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
            h, w = rgb.shape[:2]
            e = 96.0 * max(w, h) / 2816.0
            cx0, cy0 = w - 1.1667 * e, h - 1.1667 * e
            for x, y, s, sc in gwclean.detect(rgb):
                fh.write(f"{n}\t{base}\t{sc:.3f}\t{s / e:.2f}\t"
                         f"{(x + s / 2 - cx0) / e:.2f}\t{(y + s / 2 - cy0) / e:.2f}\n")
    print("done")


if __name__ == "__main__":
    main(*sys.argv[1:4])
