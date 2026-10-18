# -*- coding: utf-8 -*-
"""Try inversion parameter variants on a video's temporal mean (and one frame).

Usage: python variants_video.py <video> <sheet.png> cx,cy,s,k,sigma [...]
Row per variant: un-blended mean, un-blended frame n/2 (no band fill), with
the NCC of the result against the sparkle printed in the label.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gwvideo  # noqa: E402
from gwclean import alpha_map, ncc_at  # noqa: E402

T = 180


def main(video, sheet, *variants):
    w, h, fps = gwvideo.probe(video)
    ys = np.stack([f[:w * h].reshape(h, w) for f in gwvideo.read_frames(video, w, h)])
    mean, mid = ys.astype(np.float64).mean(axis=0), ys[len(ys) // 2].astype(np.float64)
    rows = []
    for spec in variants:
        cx, cy, s, k, sg = map(float, spec.split(","))
        r = int(1.6 * s)
        x0, y0 = int(cx) - r, int(cy) - r
        tiles = []
        for src in (mean, mid):
            win = src[y0:y0 + 2 * r, x0:x0 + 2 * r]
            a = alpha_map(win.shape, x0, y0, cx, cy, s, k, sg)
            out = (win - 235.0 * a) / (1 - a)
            v = ncc_at(out, alpha_map(win.shape, x0, y0, cx, cy, s, 1.0))
            t = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((T, T), Image.NEAREST)
            d = ImageDraw.Draw(t)
            d.rectangle([0, 0, T, 11], fill=0)
            d.text((2, 0), f"{spec} ncc{v:+.2f}", fill=255)
            tiles.append(t)
        rows.append(tiles)
    S = Image.new("L", (2 * (T + 3), len(rows) * (T + 3)), 255)
    for i, tiles in enumerate(rows):
        for j, t in enumerate(tiles):
            S.paste(t, (j * (T + 3), i * (T + 3)))
    S.save(sheet)
    print(sheet)


if __name__ == "__main__":
    main(*sys.argv[1:])
