# -*- coding: utf-8 -*-
"""Remove any static white overlay (e.g. the "Veo" text mark) from a video,
learning its alpha from the clip itself.

A static overlay gives every frame  O_t = a*W + (1-a)*B_t.  The overlay's
pixels are found on the temporal mean (they stay sharp while content moves);
for every frame the background under them, B_t, is interpolated from the
surrounding pixels (normalised Gaussian convolution), and per pixel

    a = sum_t (O_t - B_t)(W - B_t) / sum_t (W - B_t)^2

(least squares of O - B = a (W - B) over all frames). Each frame is then
un-blended, B = (O - a W) / (1 - a), in yuv420p (white = 235 / 128,128), and
the 1 px rim where encoding rings differently per frame is refilled.

Usage: python gwstatic.py <in.mp4> <out.mp4> x0,y0,x1,y1 [mask_threshold] [frame_fixes.tsv]
x0..y1: a box around the overlay (a margin of ~10 px is added).
frame_fixes.tsv: lines "frame<TAB>cx,cy,s" -- a moving sparkle (e.g. an
image-to-video source image's watermark on a moving object) clone-filled in
just those frames, after the static overlay is removed.
Prints one JSON line.
"""
import json
import subprocess
import sys

import numpy as np
import scipy.ndimage as ndi

from gwvideo import planes, read_frames
from video_scan import probe

Y_WHITE, C_WHITE = 235.0, 128.0


def fill_under(img, hole, sigma, usable=None):
    """Replace hole pixels by a normalised Gaussian average of usable pixels."""
    keep = (~hole if usable is None else (~hole & usable)).astype(np.float64)
    num = ndi.gaussian_filter(img * keep, sigma)
    den = ndi.gaussian_filter(keep, sigma)
    out = img.copy()
    ok = hole & (den > 1e-4)
    out[ok] = num[ok] / den[ok]
    return out


def read_frame_fixes(path):
    fixes = {}
    if path:
        for line in open(path, encoding="utf-8-sig"):
            if line.strip() and not line.startswith("#"):
                t, spec = line.rstrip("\r\n").split("\t")[:2]
                fixes.setdefault(int(t), []).append(tuple(float(v) for v in spec.split(",")))
    return fixes


def clone_sparkle(frame, w, h, cx, cy, s):
    """Clone-fill a sparkle in one yuv420p frame (luma: best-offset clone +
    Poisson blend; chroma: smooth interpolation under the hole)."""
    import gwclean
    yp, up, vp = planes(frame, w, h)
    pad = int(3 * s) + 8
    x0, y0 = max(0, int(cx - pad)) // 2 * 2, max(0, int(cy - pad)) // 2 * 2
    x1, y1 = min(w, int(cx + pad)) // 2 * 2, min(h, int(cy + pad)) // 2 * 2
    win = yp[y0:y1, x0:x1]
    mask = gwclean.star_mask(win.shape + (3,), [(int(round(cx - s / 2)) - x0, int(round(cy - s / 2)) - y0,
                                                   int(round(s)), 1.0)])
    filled = gwclean.fill(np.repeat(win[..., None], 3, axis=2), mask)[..., 0]
    yp[y0:y1, x0:x1] = filled
    hole_c = ndi.binary_dilation(mask.reshape(mask.shape[0] // 2, 2, mask.shape[1] // 2, 2).any(axis=(1, 3)),
                                 iterations=1)
    for plane in (up, vp):
        c = plane[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
        plane[y0 // 2:y1 // 2, x0 // 2:x1 // 2] = np.clip(
            np.rint(fill_under(c.astype(np.float64), hole_c, 2.0)), 0, 255).astype(np.uint8)


def main(src, dst, box, thr="5", fixes_path=None):
    fixes = read_frame_fixes(fixes_path)
    w, h, fps = probe(src)
    rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=r_frame_rate", "-of", "csv=p=0", src],
                          capture_output=True, text=True, check=True).stdout.strip()
    x0, y0, x1, y1 = (int(v) for v in box.split(","))
    m = 10
    x0, y0 = max(0, x0 - m) // 2 * 2, max(0, y0 - m) // 2 * 2
    x1, y1 = min(w, x1 + m) // 2 * 2, min(h, y1 + m) // 2 * 2
    ys = np.stack([planes(f, w, h)[0][y0:y1, x0:x1].astype(np.float64)
                   for f in read_frames(src, w, h)])
    n = len(ys)
    mean = ys.mean(axis=0)

    # overlay pixels: brighter than the local background of the mean
    bg = ndi.median_filter(mean, size=15)
    core = (mean - bg) > float(thr)
    core = ndi.binary_opening(core, iterations=1) | core & ndi.binary_dilation(
        ndi.binary_opening(core, iterations=1), iterations=1)
    hole = ndi.binary_dilation(core, iterations=2)
    # letterbox rows (black across the box) are not background for the
    # interpolation; they dragged the estimate dark at the letters' feet
    bar_rows = mean.max(axis=1) < 25
    usable = np.repeat(~bar_rows[:, None], mean.shape[1], axis=1)

    # per-pixel opacity by least squares over frames
    num = np.zeros_like(mean)
    den = np.zeros_like(mean)
    for t in range(n):
        b = fill_under(ys[t], hole, 3.0, usable)
        num += (ys[t] - b) * (Y_WHITE - b)
        den += (Y_WHITE - b) ** 2
    a = np.where(hole, np.clip(num / np.maximum(den, 1e-6), 0, 0.9), 0.0)
    a = np.where(a < 0.02, 0.0, a)
    rim = ndi.binary_dilation(a > 0.05, iterations=1) & ~ndi.binary_erosion(a > 0.05, iterations=1)
    damp = np.clip((1 / (1 - a) - 1) / 1.5, 0, 0.8)
    # chroma: inverting with a luma-derived alpha over-saturated (orange
    # specks); chroma is smooth, so it is simply interpolated under the mark
    hole_c = ndi.binary_dilation(hole.reshape(hole.shape[0] // 2, 2, hole.shape[1] // 2, 2)
                                 .any(axis=(1, 3)), iterations=1)
    usable_c = usable[::2, ::2]

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                            "-s", f"{w}x{h}", "-r", rate, "-i", "-", "-i", src,
                            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
                            "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy",
                            "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    sl = (slice(y0, y1), slice(x0, x1))
    slc = (slice(y0 // 2, y1 // 2), slice(x0 // 2, x1 // 2))
    for t, frame in enumerate(read_frames(src, w, h)):
        f = frame.copy()
        yp, up, vp = planes(f, w, h)
        win = yp[sl].astype(np.float64)
        win = (win - Y_WHITE * a) / (1 - a)
        # inversion multiplies compression noise by 1/(1-a) (~2x at a=0.5),
        # which showed as specks on bright cloth: damp it in proportion
        win += damp * (ndi.gaussian_filter(win, 0.8) - win)
        if rim.any():
            win = fill_under(win, rim, 1.0, usable)
        yp[sl] = np.clip(np.rint(win), 0, 255).astype(np.uint8)
        for plane in (up, vp):
            c = fill_under(plane[slc].astype(np.float64), hole_c, 2.0, usable_c)
            plane[slc] = np.clip(np.rint(c), 0, 255).astype(np.uint8)
        for cx, cy, s in fixes.get(t, []):
            clone_sparkle(f, w, h, cx, cy, s)
        enc.stdin.write(f.tobytes())
    enc.stdin.close()
    rc = enc.wait()
    print(json.dumps({"src": src, "status": "ok" if rc == 0 else f"encode-failed:{rc}", "frames": n,
                      "box": [x0, y0, x1, y1], "overlay_px": int((a > 0.02).sum()),
                      "frame_fixes": sum(len(v) for v in fixes.values()),
                      "alpha_max": round(float(a.max()), 3),
                      "alpha_median": round(float(np.median(a[a > 0.02])) if (a > 0.02).any() else 0, 3)}))
    return rc


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
