# -*- coding: utf-8 -*-
"""Side-by-side crops of the sparkle area from several versions of each image.

Usage: python compare_dirs.py <sheet_prefix> <per_page> <name_list.txt> <dir> [<dir> ...]
One row group per image: one tile per dir (a missing file shows black),
cropped to +-1.9 expected sparkle sizes around the canonical spot.
Writes <prefix>_pN.png and <prefix>_index.tsv (number -> name).
"""
import os
import sys

from PIL import Image, ImageDraw

T = 130
PER_ROW = 2


def crop_box(w, h):
    e = 96.0 * max(w, h) / 2816.0
    cx, cy, r = w - 1.1667 * e, h - 1.1667 * e, 1.9 * e
    return int(cx - r), int(cy - r), int(cx + r), int(cy + r)


def main(prefix, per_page, names_file, *dirs):
    per_page = int(per_page)
    names = [l.strip() for l in open(names_file, encoding="utf-8-sig") if l.strip()]
    with open(prefix + "_index.tsv", "w", encoding="utf-8") as fh:
        for k, n in enumerate(names):
            fh.write(f"{k}\t{n}\n")
    gw = len(dirs) * (T + 2) + 10
    for page in range((len(names) + per_page - 1) // per_page):
        chunk = names[page * per_page:(page + 1) * per_page]
        rows = (len(chunk) + PER_ROW - 1) // PER_ROW
        sheet = Image.new("RGB", (PER_ROW * gw, rows * (T + 4)), (255, 0, 255))
        for i, n in enumerate(chunk):
            k = page * per_page + i
            with Image.open(os.path.join(dirs[0], n)) as im0:
                box = crop_box(*im0.size)
            for j, d in enumerate(dirs):
                p = os.path.join(d, n)
                t = (Image.open(p).convert("RGB").crop(box).resize((T, T)) if os.path.exists(p)
                     else Image.new("RGB", (T, T)))
                if j == 0:
                    dr = ImageDraw.Draw(t)
                    dr.rectangle([0, 0, 22, 11], fill=(0, 0, 0))
                    dr.text((2, 0), str(k), fill=(255, 255, 0))
                sheet.paste(t, ((i % PER_ROW) * gw + j * (T + 2), (i // PER_ROW) * (T + 4)))
        out = f"{prefix}_p{page}.png"
        sheet.save(out)
        print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
