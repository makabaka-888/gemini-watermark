# -*- coding: utf-8 -*-
"""Large before/after zoom of the sparkle area for selected images.

Usage: python pair_zoom.py <index.tsv> <src_dir> <final_dir> <sheet.png> <num> [num ...]
"""
import os
import sys

from PIL import Image, ImageDraw

T = int(os.environ.get("PZ_TILE", 210))
COLS = int(os.environ.get("PZ_COLS", 3))


def crop(im, scale=1.9):
    w, h = im.size
    e = 96.0 * max(w, h) / 2816.0
    cx, cy, r = w - 1.1667 * e, h - 1.1667 * e, scale * e
    return im.crop((int(cx - r), int(cy - r), int(cx + r), int(cy + r))).resize((T, T))


def main(index, src_dir, final_dir, sheet_path, *nums):
    names = dict(l.rstrip("\n").split("\t")[:2] for l in open(index, encoding="utf-8-sig"))
    pairs = []
    for num in nums:
        n = names[num]
        a = crop(Image.open(os.path.join(src_dir, n)).convert("RGB"))
        b = crop(Image.open(os.path.join(final_dir, n)).convert("RGB"))
        p = Image.new("RGB", (2 * T + 3, T), (255, 0, 255))
        p.paste(a, (0, 0))
        p.paste(b, (T + 3, 0))
        d = ImageDraw.Draw(p)
        d.rectangle([0, 0, 28, 11], fill=(0, 0, 0))
        d.text((2, 0), num, fill=(255, 255, 0))
        pairs.append(p)
    pw = 2 * T + 3 + 8
    sheet = Image.new("RGB", (COLS * pw, ((len(pairs) + COLS - 1) // COLS) * (T + 6)), (40, 40, 40))
    for i, p in enumerate(pairs):
        sheet.paste(p, ((i % COLS) * pw, (i // COLS) * (T + 6)))
    sheet.save(sheet_path)
    print(sheet_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
