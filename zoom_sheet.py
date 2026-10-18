# -*- coding: utf-8 -*-
"""One-page zoom of the sparkle area of many images, for residue checks.

Usage: python zoom_sheet.py <index.tsv> <img_dir> <sheet.png>
index.tsv: lines "<number>\t<name>..." (verify_sheet.py's *_index.tsv).
Each tile: the square of +-1.7 expected sparkle sizes around the canonical
sparkle centre, 120 px, labelled with the index number.
"""
import os
import sys

from PIL import Image, ImageDraw

T, COLS = 120, 14


def main(index, img_dir, sheet_path):
    entries = [l.rstrip("\n").split("\t")[:2] for l in open(index, encoding="utf-8-sig")]
    tiles = []
    for num, name in entries:
        im = Image.open(os.path.join(img_dir, name)).convert("RGB")
        w, h = im.size
        e = 96.0 * max(w, h) / 2816.0
        cx, cy, r = w - 1.1667 * e, h - 1.1667 * e, 1.7 * e
        t = im.crop((int(cx - r), int(cy - r), int(cx + r), int(cy + r))).resize((T, T))
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 24, 11], fill=(0, 0, 0))
        d.text((2, 0), num, fill=(255, 255, 0))
        tiles.append(t)
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * (T + 3), rows * (T + 3)), (255, 0, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % COLS) * (T + 3), (i // COLS) * (T + 3)))
    sheet.save(sheet_path)
    print(sheet_path, len(tiles))


if __name__ == "__main__":
    main(*sys.argv[1:4])
