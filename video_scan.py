# -*- coding: utf-8 -*-
"""Survey the Veo sparkle in a folder of videos.

For every video: decode the bottom-right corner of each frame, average it
over time (a static overlay stays sharp while moving content blurs), and
find the strongest sparkle in the average and in each 1-second window, so a
watermark that moves between segments shows up as more than one position.

Usage: python video_scan.py <video_dir> <out_prefix>
Writes <out_prefix>.tsv and <out_prefix>_sheet.png (numbered mean corners).
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

import gwclean

CORNER = 0.3


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_frames", "-of", "json", path],
                       capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den)


def corner_frames(path, w, h):
    side = int(CORNER * max(w, h))
    x0, y0 = max(0, w - side), max(0, h - side)
    cw, ch = w - x0, h - y0
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"crop={cw}:{ch}:{x0}:{y0}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    raw = p.stdout.read()
    p.wait()
    frames = np.frombuffer(raw, np.uint8).reshape(-1, ch, cw, 3)
    return frames, x0, y0


def best_star(mean_rgb, e):
    """Strongest bright sparkle anywhere in the corner crop."""
    sizes = sorted({max(8, int(round(e * f))) for f in np.arange(0.6, 1.65, 0.05)})
    peaks = gwclean._peaks(mean_rgb.astype(np.float32), (0, 0, mean_rgb.shape[1], mean_rgb.shape[0]),
                           sizes, 1, 0.3)
    return max(peaks, key=lambda p: p[3]) if peaks else None


def main(video_dir, prefix):
    names = sorted(n for n in os.listdir(video_dir) if n.lower().endswith(".mp4"))
    tiles = []
    with open(prefix + ".tsv", "w", encoding="utf-8") as fh:
        fh.write("idx\tname\tw\th\tfps\tframes\tmean_star(x,y,s,score)\twindows\n")
        for k, n in enumerate(names):
            path = os.path.join(video_dir, n)
            w, h, fps = probe(path)
            frames, x0, y0 = corner_frames(path, w, h)
            e = gwclean.expected_size(w, h)
            mean = frames.astype(np.float32).mean(axis=0)
            star = best_star(mean, e)
            wins = []
            step = max(1, int(round(fps)))
            for t0 in range(0, len(frames), step):
                s = best_star(frames[t0:t0 + step].astype(np.float32).mean(axis=0), e)
                wins.append("-" if s is None else f"{x0 + s[0]},{y0 + s[1]},{s[2]},{s[3]:.2f}")
            ms = "-" if star is None else f"{x0 + star[0]},{y0 + star[1]},{star[2]},{star[3]:.2f}"
            fh.write(f"{k}\t{n}\t{w}\t{h}\t{fps:g}\t{len(frames)}\t{ms}\t{' '.join(wins)}\n")
            fh.flush()
            t = Image.fromarray(np.clip(mean, 0, 255).astype(np.uint8)).resize((160, 160))
            d = ImageDraw.Draw(t)
            d.rectangle([0, 0, 30, 12], fill=(0, 0, 0))
            d.text((2, 0), f"#{k}", fill=(255, 255, 0))
            tiles.append(t)
            print(k, flush=True)
    cols = 8
    sheet = Image.new("RGB", (cols * 164, ((len(tiles) + cols - 1) // cols) * 164), (40, 40, 40))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 164, (i // cols) * 164))
    sheet.save(prefix + "_sheet.png")


if __name__ == "__main__":
    main(*sys.argv[1:3])
