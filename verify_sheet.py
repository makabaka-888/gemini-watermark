# -*- coding: utf-8 -*-
"""Before/after contact sheets of the bottom-right search region.

Usage: python verify_sheet.py <report.tsv> <src_dir> <final_dir> <prefix> [name ...]

Left half of each tile: original corner with detections boxed (green = bright
sparkle, red = dark ghost). Right half: cleaned result. Tiles are numbered;
<prefix>_index.tsv maps numbers to file names.
"""
import os
import sys

from PIL import Image, ImageDraw

TW = 200
COLS, ROWS = 4, 8
CORNER = 0.3   # square crop, side = CORNER * long side, anchored bottom-right


def read_report(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            name, base, dets = line.rstrip("\n").split("\t")
            parsed = []
            for d in filter(None, dets.split(";")):
                try:
                    x, y, s, sc = d.split(",")
                    parsed.append((int(x), int(y), int(s), float(sc)))
                except ValueError:      # newer reports log actions, not boxes
                    break
            rows.append((name, base, parsed))
    return rows


def pair(src, final, dets):
    a = Image.open(src).convert("RGB")
    b = Image.open(final).convert("RGB")
    w, h = a.size
    side = int(CORNER * max(w, h))
    x0, y0 = max(0, w - side), max(0, h - side)
    ca, cb = a.crop((x0, y0, w, h)), b.crop((x0, y0, w, h))
    d = ImageDraw.Draw(ca)
    lw = max(2, side // 150)
    for x, y, s, sc in dets:
        d.rectangle([x - x0 - lw, y - y0 - lw, x - x0 + s + lw, y - y0 + s + lw],
                    outline=(0, 255, 0) if sc > 0 else (255, 0, 0), width=lw)
    out = Image.new("RGB", (2 * TW + 3, TW), (255, 0, 255))
    out.paste(ca.resize((TW, TW)), (0, 0))
    out.paste(cb.resize((TW, TW)), (TW + 3, 0))
    return out


def main(report, src_dir, final_dir, prefix, *only):
    rows = [r for r in read_report(report) if not only or r[0] in only]
    per = COLS * ROWS
    with open(prefix + "_index.tsv", "w", encoding="utf-8") as idx:
        for k, (n, base, dets) in enumerate(rows):
            idx.write(f"{k}\t{n}\t{base}\t{len(dets)}\n")
    for page in range((len(rows) + per - 1) // per):
        tiles = []
        for k, (n, base, dets) in enumerate(rows[page * per:(page + 1) * per], start=page * per):
            t = pair(os.path.join(src_dir, n), os.path.join(final_dir, n), dets)
            dr = ImageDraw.Draw(t)
            dr.rectangle([0, 0, 52, 13], fill=(0, 0, 0))
            dr.text((3, 1), f"#{k} {len(dets)}d", fill=(255, 255, 0))
            tiles.append(t)
        cw = max(t.size[0] for t in tiles) + 6
        chh = max(t.size[1] for t in tiles) + 6
        sheet = Image.new("RGB", (COLS * cw, ((len(tiles) + COLS - 1) // COLS) * chh), (40, 40, 40))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % COLS) * cw, (i // COLS) * chh))
        path = f"{prefix}_p{page}.png"
        sheet.save(path)
        print(path)


if __name__ == "__main__":
    main(*sys.argv[1:])
