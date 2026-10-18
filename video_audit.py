# -*- coding: utf-8 -*-
"""Audit videos for remaining watermarks (no changes are made).

For every .mp4 in the given folders: stream info, and on the temporal mean of
the bottom-right corner
  - opacity of a sparkle at each Veo grid spot (boundary-step estimate; ~0.37
    for a present overlay, ~0 when absent or removed),
  - in-scene sparkles found by correlation (same acceptance rules as gwvideo),
  - static bright overlay pixels in the bottom-right text area (the "Veo"
    text mark), counted as pixels well above the local background,
plus a small thumbnail signature for matching processed files to originals.

Usage: python video_audit.py <out.jsonl> <dir> [<dir> ...]
"""
import json
import os
import subprocess
import sys

import numpy as np
import scipy.ndimage as ndi

import gwclean
import gwvideo as g
from video_scan import probe


def info(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name,nb_read_packets:format=duration",
                        "-of", "json", path], capture_output=True, text=True)
    j = json.loads(r.stdout)
    return j["streams"][0]["codec_name"], int(j["streams"][0]["nb_read_packets"]), float(j["format"]["duration"])


def signature(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=32:18,format=gray",
                        "-f", "rawvideo", "-"], capture_output=True).stdout
    a = np.frombuffer(r, np.uint8).reshape(-1, 18, 32).astype(np.float32)
    idx = np.linspace(0, len(a) - 1, 8).astype(int)
    return a[idx].round(1).tolist()


def audit(path):
    w, h, fps = probe(path)
    S = g.VEO_SIZE * w
    side = int(6 * S)
    ox, oy = max(0, (w - side) // 2 * 2), max(0, (h - side) // 2 * 2)
    ys = np.stack([g.planes(f, w, h)[0][oy:, ox:].copy() for f in g.read_frames(path, w, h)])
    mean = ys.astype(np.float64).mean(axis=0)
    out = {"w": w, "h": h, "fps": round(fps, 3)}
    out["codec"], out["frames"], out["dur"] = info(path)
    grid = []
    for gg in g.GRID:
        cx, cy = w - gg * S, h - gg * S
        k = g.boundary_opacity(mean, ox, oy, cx, cy, S)
        win, wx, wy = g._crop(mean, ox, oy, cx, cy, S)
        v = gwclean.ncc_at(win, gwclean.alpha_map(win.shape, wx, wy, cx, cy, S, 1.0))
        grid.append({"inset": gg, "opacity": round(k, 3), "ncc": round(v, 2)})
    out["grid"] = grid
    sizes = sorted({max(10, int(round(S * f))) for f in np.arange(0.3, 1.06, 0.05)})
    peaks = gwclean._peaks(np.repeat(mean[..., None], 3, axis=2).astype(np.float32),
                           (0, 0, mean.shape[1], mean.shape[0]), sizes, 1, g.EXTRA_NCC)
    extras = []
    for x, y, s, sc in sorted(peaks, key=lambda p: -p[3]):
        cx, cy = ox + x + s / 2, oy + y + s / 2
        if not (g.EXTRA_BOX[0] * S <= w - cx <= g.EXTRA_REACH * S and
                g.EXTRA_BOX[0] * S <= h - cy <= g.EXTRA_REACH * S):
            continue
        if any(np.hypot(cx - (w - gg * S), cy - (h - gg * S)) < 0.4 * S for gg in g.GRID):
            continue
        if any(np.hypot(cx - e["cx"], cy - e["cy"]) < 0.4 * max(s, e["s"]) for e in extras):
            continue
        v, rx, ry, rs = g.refine(mean, ox, oy, cx, cy, s)
        if g.extra_ok(w, h, rx, ry, rs, S, v):
            extras.append({"cx": round(rx, 1), "cy": round(ry, 1), "s": round(rs, 1), "ncc": round(v, 2)})
    out["extras"] = extras
    # static text overlay: bright, sharp pixels in the bottom-right text box
    tx0, ty0 = int(w * 0.88) - ox, int(h * 0.88) - oy
    box = mean[max(0, ty0):, max(0, tx0):]
    bright = (box - ndi.median_filter(box, size=15)) > 8
    out["text_px"] = int(ndi.binary_opening(bright, iterations=1).sum())
    out["sig"] = signature(path)
    return out


def main(out_path, *dirs):
    with open(out_path, "w", encoding="utf-8") as fh:
        for d in dirs:
            for n in sorted(os.listdir(d)):
                if n.lower().endswith(".mp4"):
                    p = os.path.join(d, n)
                    try:
                        r = audit(p)
                    except Exception as e:  # report and continue
                        r = {"error": str(e)[:200]}
                    r["path"] = p
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fh.flush()
                    print(n, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
