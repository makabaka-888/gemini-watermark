# -*- coding: utf-8 -*-
"""Review sheets for delogo repairs: the old video and one or more repaired
versions side by side over time, plus temporal-stability numbers for the box.

Usage: python review_sheets.py <boxes.tsv> <old_dir> <versions> <out_dir> [only]
  versions: one directory (rows old | repaired | |repaired - old| x4), or
            "label=dir|label=dir..." (rows old | each version, no diff row)
  only:     comma-separated row numbers of boxes.tsv (e.g. 4,7)

For the k-th video in boxes.tsv (id oNN) it writes
  oNN_a.png, oNN_b.png  12 evenly spaced frames, 6 per sheet
  oNN_c.png             three consecutive frames around the largest
                        frame-to-frame jump inside the box of each version
and appends one line per video to <out_dir>/metrics.jsonl.
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

PAD = 56
MARGIN = 2  # same dilation as repair_delogo.py


def parse_boxes(spec):
    return [tuple(int(v) for v in b.split(",")) for b in spec.split(";") if b.strip()]


def crop_frames(path, box):
    x, y, w, h = box
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"crop={w}:{h}:{x}:{y}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def luma(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 6 * len(text) + 4, 11], fill=(0, 0, 0))
    d.text((2, 0), text, fill=(255, 255, 0))
    return img


def sheet(old, versions, ts, path, diff_row):
    """versions: [(label, frames)]; one column per frame index in ts."""
    h, w = old.shape[1:3]
    rows = 1 + len(versions) + (1 if diff_row else 0)
    S = Image.new("RGB", (len(ts) * (w + 4), rows * (h + 4)), (255, 0, 255))
    for j, t in enumerate(ts):
        tiles = [label(Image.fromarray(old[t]), f"t={t} old")]
        tiles += [label(Image.fromarray(fr[t]), name) for name, fr in versions]
        if diff_row:
            new = versions[0][1]
            diff = np.clip(np.abs(new[t].astype(np.int16) - old[t].astype(np.int16)).max(-1) * 4, 0, 255)
            tiles.append(label(Image.fromarray(diff.astype(np.uint8)).convert("RGB"), "diff x4"))
        for i, tile in enumerate(tiles):
            S.paste(tile, (j * (w + 4), i * (h + 4)))
    S.save(path)


def main(boxes_tsv, old_dir, versions, out_dir, only=""):
    os.makedirs(out_dir, exist_ok=True)
    rows = [l.rstrip("\r\n").split("\t")[:2] for l in open(boxes_tsv, encoding="utf-8-sig")
            if l.strip() and not l.startswith("#")]
    if "=" in versions:
        vdirs = [tuple(v.split("=", 1)) for v in versions.split("|")]
        diff_row = False
    else:
        vdirs, diff_row = [("new", versions)], True
    keep = {int(v) for v in only.split(",") if v.strip()}
    out = open(os.path.join(out_dir, "metrics.jsonl"), "a" if keep else "w", encoding="utf-8")
    for k, (name, spec) in enumerate(rows):
        if keep and k not in keep:
            continue
        boxes = parse_boxes(spec)
        x0 = max(0, min(b[0] for b in boxes) - PAD) // 2 * 2
        y0 = max(0, min(b[1] for b in boxes) - PAD) // 2 * 2
        x1 = min(1280, max(b[0] + b[2] for b in boxes) + PAD) // 2 * 2
        y1 = min(720, max(b[1] + b[3] for b in boxes) + PAD) // 2 * 2
        crop = (x0, y0, x1 - x0, y1 - y0)
        old = crop_frames(os.path.join(old_dir, name), crop)
        vers = [(lab, crop_frames(os.path.join(d, name), crop)) for lab, d in vdirs]
        n = min([len(old)] + [len(fr) for _, fr in vers])
        hole = np.zeros(old.shape[1:3], bool)
        for bx, by, bw, bh in boxes:
            hole[by - y0:by - y0 + bh, bx - x0:bx - x0 + bw] = True
        hole = binary_dilation(hole, iterations=MARGIN)
        ring = binary_dilation(hole, iterations=10) & ~binary_dilation(hole, iterations=3)
        outside = ~binary_dilation(hole, iterations=6)
        yo = luma(old[:n].astype(np.float32))
        d_ring = np.abs(np.diff(yo, axis=0))[:, ring].mean(1)
        m = {"id": f"o{k:02d}", "name": name, "frames": {"old": len(old)}, "crop": crop,
             "box_in_crop": [[b[0] - x0, b[1] - y0, b[2], b[3]] for b in boxes],
             "ring_motion": round(float(d_ring.mean()), 2),
             "box_motion": {"old": round(float(np.abs(np.diff(yo, axis=0))[:, hole].mean()), 2)},
             "jump": {}, "outside_change": {}}
        jumps = []
        for lab, fr in vers:
            yn = luma(fr[:n].astype(np.float32))
            d_in = np.abs(np.diff(yn, axis=0))[:, hole].mean(1)
            excess = d_in - d_ring
            t = int(np.argmax(excess))
            jumps.append(t)
            m["frames"][lab] = len(fr)
            m["box_motion"][lab] = round(float(d_in.mean()), 2)
            m["jump"][lab] = [t, round(float(excess[t]), 2)]
            m["outside_change"][lab] = round(float(np.abs(yn - yo)[:, outside].mean()), 2)
        ts = [int(t) for t in np.linspace(0, n - 1, 12).round()]
        sid = m["id"]
        sheet(old, vers, ts[:6], os.path.join(out_dir, f"{sid}_a.png"), diff_row)
        sheet(old, vers, ts[6:], os.path.join(out_dir, f"{sid}_b.png"), diff_row)
        cols = [min(t + d, n - 1) for t in sorted(set(jumps))[:2] for d in (0, 1, 2)]
        sheet(old, vers, cols, os.path.join(out_dir, f"{sid}_c.png"), diff_row)
        out.write(json.dumps(m, ensure_ascii=False) + "\n")
        out.flush()
        print(sid, m["frames"], m["ring_motion"], m["box_motion"], m["jump"], m["outside_change"], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
