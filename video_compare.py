# -*- coding: utf-8 -*-
"""Zoomed before/after sheet for processed videos.

Usage: python video_compare.py <results.jsonl> <out_dir> <sheet_prefix>
results.jsonl holds gwvideo.py's JSON lines; processed files are
<out_dir>/<same name>. Per video: temporal mean before | mean after | a
middle frame after | a late frame after, cropped around each fitted sparkle.
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

TW, TH = 200, 150
PER_ROW, ROWS = 2, 10


def crop_frames(path, x, y, cw, ch):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"crop={cw}:{ch}:{x}:{y}",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(r, np.uint8).reshape(-1, ch, cw, 3)


def tile(arr, text=None):
    t = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize((TW, TH), Image.NEAREST)
    if text:
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 7 * len(text) + 4, 12], fill=(0, 0, 0))
        d.text((2, 0), text, fill=(255, 255, 0))
    return t


def main(results, out_dir, prefix):
    rows = []
    with open(results, encoding="utf-8-sig") as fh:
        for k, line in enumerate(l for l in fh if l.strip().startswith("{")):
            r = json.loads(line)
            fits = [f for f in r.get("fits", []) if f] or [None]
            for j, f in enumerate(fits):
                cx, cy = (f["cx"], f["cy"]) if f else (1184, 624)
                x, y = int(cx - 80), int(cy - 60)
                src = r["src"]
                dst = os.path.join(out_dir, os.path.basename(src))
                a = crop_frames(src, x, y, 160, 120)
                b = crop_frames(dst, x, y, 160, 120) if os.path.exists(dst) else a * 0
                n = len(b)
                rows.append([tile(a.mean(0), f"#{k}{'abcdefgh'[j] if len(fits) > 1 else ''}"),
                             tile(b.mean(0)), tile(b[n // 2]), tile(b[(5 * n) // 6])])
                print(f"#{k} {os.path.basename(src)} {r.get('status')}", flush=True)
    per = PER_ROW * ROWS
    for page in range((len(rows) + per - 1) // per):
        chunk = rows[page * per:(page + 1) * per]
        rw = 4 * (TW + 3) + 12
        sheet = Image.new("RGB", (PER_ROW * rw, ((len(chunk) + PER_ROW - 1) // PER_ROW) * (TH + 4)),
                          (255, 0, 255))
        for i, tiles in enumerate(chunk):
            for j, t in enumerate(tiles):
                sheet.paste(t, ((i % PER_ROW) * rw + j * (TW + 3), (i // PER_ROW) * (TH + 4)))
        sheet.save(f"{prefix}_p{page}.png")
        print(f"{prefix}_p{page}.png")


if __name__ == "__main__":
    main(*sys.argv[1:4])
