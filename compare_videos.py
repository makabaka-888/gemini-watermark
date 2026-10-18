# -*- coding: utf-8 -*-
"""Before/after crops of two versions of the same videos.

Usage: python compare_videos.py <sheet_prefix> <dirA> <dirB> x0,y0,w,h <name> [<name> ...]
       (a name of the form "a.mp4|b.mp4" compares differently named files)
Per video one row: temporal mean A | mean B | frames at 1/3 and 2/3 for A and B.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

T = 200


def crop_frames(path, box):
    x, y, w, h = box
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"crop={w}:{h}:{x}:{y}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def tile(arr, size, text=None):
    t = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize(size, Image.LANCZOS)
    if text:
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 7 * len(text) + 4, 11], fill=(0, 0, 0))
        d.text((2, 0), text, fill=(255, 255, 0))
    return t


def main(prefix, dir_a, dir_b, box, *names):
    box = tuple(int(v) for v in box.split(","))
    th = int(T * box[3] / box[2])
    rows = []
    for k, spec in enumerate(names):
        na, nb = spec.split("|") if "|" in spec else (spec, spec)
        a, b = crop_frames(os.path.join(dir_a, na), box), crop_frames(os.path.join(dir_b, nb), box)
        n = min(len(a), len(b))
        rows.append([tile(a.astype(np.float32).mean(0), (T, th), f"{k} A mean"),
                     tile(b.astype(np.float32).mean(0), (T, th), "B mean"),
                     tile(a[n // 3], (T, th), "A"), tile(b[n // 3], (T, th), "B"),
                     tile(a[2 * n // 3], (T, th), "A"), tile(b[2 * n // 3], (T, th), "B")])
    per = 8
    for page in range(0, len(rows), per):
        chunk = rows[page:page + per]
        S = Image.new("RGB", (6 * (T + 3), len(chunk) * (th + 3)), (255, 0, 255))
        for i, r in enumerate(chunk):
            for j, t in enumerate(r):
                S.paste(t, (j * (T + 3), i * (th + 3)))
        S.save(f"{prefix}_p{page // per}.png")
        print(f"{prefix}_p{page // per}.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
