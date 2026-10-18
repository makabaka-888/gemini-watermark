# -*- coding: utf-8 -*-
"""Side-by-side close-up of one image before and after, for reviewers.

Usage: python zoom_pair.py <original> <cleaned> <out.png> [radius]
Crops +-radius expected sparkle sizes (default 2.5) around the canonical
Gemini sparkle spot from both files and writes them side by side at 3x.
"""
import sys

from PIL import Image


def main(a, b, out, radius="2.5"):
    ims = [Image.open(p).convert("RGB") for p in (a, b)]
    w, h = ims[0].size
    e = 96.0 * max(w, h) / 2816.0
    cx, cy, r = w - 1.1667 * e, h - 1.1667 * e, float(radius) * e
    box = (max(0, int(cx - r)), max(0, int(cy - r)), min(w, int(cx + r)), min(h, int(cy + r)))
    tiles = [im.crop(box).resize(((box[2] - box[0]) * 3, (box[3] - box[1]) * 3), Image.LANCZOS) for im in ims]
    sheet = Image.new("RGB", (tiles[0].width * 2 + 8, tiles[0].height), (255, 0, 255))
    sheet.paste(tiles[0], (0, 0))
    sheet.paste(tiles[1], (tiles[0].width + 8, 0))
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
