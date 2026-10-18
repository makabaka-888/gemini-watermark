# -*- coding: utf-8 -*-
"""Visual verification sheets for GeminiWatermarkTool image runs.

Parses a `GeminiWatermarkTool -v` batch log, then builds labeled contact
sheets so every result can be checked by eye:

  done     before|after pairs cropped around the removal position
  skipped  bottom-right corners of images the tool did not touch

Usage:
  python montage_images.py <log> <src_dir> <out_dir> done|skipped <sheet_prefix>

Writes <sheet_prefix>_pN.png sheets plus <sheet_prefix>_index.tsv mapping
tile numbers to file names (names may be non-ASCII, so tiles show numbers).
"""
import os
import re
import sys

from PIL import Image, ImageDraw

TILE = 160
COLS_DONE, COLS_SKIP = 4, 6
ROWS = 8

RE_PROC = re.compile(r"\[info\] Processing: (.+?) \((\d+)x(\d+)\)$")
RE_REMOVE = re.compile(r"Removing watermark at \((\d+), (\d+)\) with (\d+)x\d+")
RE_SAVED = re.compile(r"\[info\] Saved: (.+)$")


def parse_log(path):
    """Return {name: {"w","h","pos":(x,y,s) or None,"saved":bool}}."""
    files, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = RE_PROC.search(line)
            if m:
                cur = m.group(1)
                files.setdefault(cur, {"w": int(m.group(2)), "h": int(m.group(3)),
                                       "pos": None, "saved": False})
                continue
            m = RE_REMOVE.search(line)
            if m and cur:
                files[cur]["pos"] = tuple(int(g) for g in m.groups())
                continue
            m = RE_SAVED.search(line)
            if m and m.group(1) in files:
                files[m.group(1)]["saved"] = True
    return files


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 7 * len(text) + 6, 13], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 0))


def done_tile(src, out, info):
    x, y, s = info["pos"]
    box = (max(0, x - s), max(0, y - s), min(info["w"], x + 2 * s), min(info["h"], y + 2 * s))
    pair = Image.new("RGB", (2 * TILE + 4, TILE), (255, 0, 255))
    with Image.open(src) as a, Image.open(out) as b:
        pair.paste(a.convert("RGB").crop(box).resize((TILE, TILE)), (0, 0))
        pair.paste(b.convert("RGB").crop(box).resize((TILE, TILE)), (TILE + 4, 0))
    return pair


def skip_tile(src, info):
    w, h = info["w"], info["h"]
    side = int(0.18 * max(w, h))
    box = (max(0, w - side), max(0, h - side), w, h)
    with Image.open(src) as a:
        return a.convert("RGB").crop(box).resize((TILE, TILE))


def main(log, src_dir, out_dir, mode, prefix):
    files = parse_log(log)
    if mode == "done":
        names = sorted(n for n, i in files.items() if i["saved"] and i["pos"])
        cols = COLS_DONE
    else:
        names = sorted(n for n, i in files.items() if not i["saved"])
        cols = COLS_SKIP
    per = cols * ROWS
    with open(prefix + "_index.tsv", "w", encoding="utf-8") as idx:
        for k, n in enumerate(names):
            idx.write(f"{k}\t{n}\t{files[n]['pos']}\n")
    for page in range((len(names) + per - 1) // per):
        chunk = names[page * per:(page + 1) * per]
        tiles = []
        for k, n in enumerate(chunk, start=page * per):
            src = os.path.join(src_dir, n)
            if mode == "done":
                t = done_tile(src, os.path.join(out_dir, n), files[n])
            else:
                t = skip_tile(src, files[n])
            label(t, f"#{k}")
            tiles.append(t)
        tw, th = tiles[0].size
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (tw + 6), rows * (th + 6)), (40, 40, 40))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * (tw + 6), (i // cols) * (th + 6)))
        path = f"{prefix}_p{page}.png"
        sheet.save(path)
        print(path)
    print(f"{len(names)} tiles")


if __name__ == "__main__":
    main(*sys.argv[1:6])
