# -*- coding: utf-8 -*-
"""Crop sheet of detections from detect_stats.py output matching a filter.

Usage: python det_crops.py <stats.tsv> <src_dir> <pass1_dir> <sheet.png> <expr>
<expr> is a Python expression over score, size, dx, dy, dist (in E units).
"""
import os
import sys

from PIL import Image, ImageDraw

T = 150


def main(stats, src_dir, pass1_dir, sheet_path, expr):
    tiles = []
    with open(stats, encoding="utf-8") as fh:
        for line in fh:
            n, base, score, size, dx, dy = line.rstrip("\n").split("\t")
            score, size, dx, dy = float(score), float(size), float(dx), float(dy)
            dist = (dx * dx + dy * dy) ** 0.5
            if not (0.6 <= size <= 1.6 and eval(expr)):
                continue
            p = os.path.join(pass1_dir if base == "pass1" else src_dir, n)
            im = Image.open(p).convert("RGB")
            w, h = im.size
            e = 96.0 * max(w, h) / 2816.0
            cx, cy = w - 1.1667 * e + dx * e, h - 1.1667 * e + dy * e
            r = 1.5 * size * e
            t = im.crop((int(cx - r), int(cy - r), int(cx + r), int(cy + r))).resize((T, T))
            d = ImageDraw.Draw(t)
            d.rectangle([0, 0, T, 12], fill=(0, 0, 0))
            d.text((2, 0), f"{len(tiles)} {score:+.2f} d{dist:.1f}", fill=(255, 255, 0))
            tiles.append(t)
            print(len(tiles) - 1, n, f"{score:+.2f}", f"dist={dist:.2f}")
    cols = 8
    sheet = Image.new("RGB", (cols * (T + 4), ((len(tiles) + cols - 1) // cols) * (T + 4)), (40, 40, 40))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (T + 4), (i // cols) * (T + 4)))
    sheet.save(sheet_path)


if __name__ == "__main__":
    main(*sys.argv[1:6])
