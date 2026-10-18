# -*- coding: utf-8 -*-
"""Fix a watermark in a static video region by editing one still "plate".

  extract <video> x0,y0,w,h <plate.png>
      temporal median of the region (in yuv420p) saved as an RGB PNG
  apply   <video> x0,y0,w,h <plate.png> <plate_fixed.png> <out.mp4>
      the edit (fixed minus plate, converted back to yuv) is added to every
      frame; pixels the edit did not touch are left exactly as they are

Meant for regions where the background does not move (checked by the
caller): the hand edit then applies to every frame, and HUD text or fine
stripes that a per-frame fill would smear are kept. x0, y0, w, h must be even.
"""
import subprocess
import sys

import numpy as np
from PIL import Image

from gwvideo import planes, read_frames
from video_scan import probe


def yuv_to_rgb(y, u, v):
    u = np.repeat(np.repeat(u, 2, 0), 2, 1) - 128.0
    v = np.repeat(np.repeat(v, 2, 0), 2, 1) - 128.0
    yy = (y - 16.0) * 255 / 219
    uu, vv = u * 255 / 224, v * 255 / 224
    return np.stack([yy + 1.402 * vv, yy - 0.344136 * uu - 0.714136 * vv, yy + 1.772 * uu], -1)


def rgb_to_yuv(rgb):
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (b - y) / 1.772
    v = (r - y) / 1.402
    return 16 + y * 219 / 255, 128 + u * 224 / 255, 128 + v * 224 / 255


def region(box):
    x0, y0, w, h = (int(v) for v in box.split(","))
    assert x0 % 2 == 0 and y0 % 2 == 0 and w % 2 == 0 and h % 2 == 0, "box must be even"
    return x0, y0, w, h


def extract(video, box, plate):
    x0, y0, bw, bh = region(box)
    w, h, _ = probe(video)
    Y, U, V = [], [], []
    for f in read_frames(video, w, h):
        y, u, v = planes(f, w, h)
        Y.append(y[y0:y0 + bh, x0:x0 + bw].copy())
        U.append(u[y0 // 2:(y0 + bh) // 2, x0 // 2:(x0 + bw) // 2].copy())
        V.append(v[y0 // 2:(y0 + bh) // 2, x0 // 2:(x0 + bw) // 2].copy())
    med = [np.median(np.stack(c), axis=0) for c in (Y, U, V)]
    Image.fromarray(np.clip(np.rint(yuv_to_rgb(*med)), 0, 255).astype(np.uint8)).save(plate)
    print(plate)


def apply(video, box, plate, fixed, out):
    x0, y0, bw, bh = region(box)
    w, h, _ = probe(video)
    rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=r_frame_rate", "-of", "csv=p=0", video],
                          capture_output=True, text=True, check=True).stdout.strip()
    p = np.asarray(Image.open(plate).convert("RGB")).astype(np.float64)
    q = np.asarray(Image.open(fixed).convert("RGB")).astype(np.float64)
    assert p.shape == q.shape == (bh, bw, 3), (p.shape, q.shape)
    py, pu, pv = rgb_to_yuv(p)
    qy, qu, qv = rgb_to_yuv(q)
    touched = np.abs(q - p).max(-1) > 0
    dy = np.where(touched, qy - py, 0.0)
    pool = lambda a: a.reshape(bh // 2, 2, bw // 2, 2).mean(axis=(1, 3))
    du, dv = pool(np.where(touched, qu - pu, 0.0)), pool(np.where(touched, qv - pv, 0.0))
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                            "-s", f"{w}x{h}", "-r", rate, "-i", "-", "-i", video,
                            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
                            "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy",
                            "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = 0
    for f in read_frames(video, w, h):
        f = f.copy()
        y, u, v = planes(f, w, h)
        for plane, d, sl in ((y, dy, (slice(y0, y0 + bh), slice(x0, x0 + bw))),
                             (u, du, (slice(y0 // 2, (y0 + bh) // 2), slice(x0 // 2, (x0 + bw) // 2))),
                             (v, dv, (slice(y0 // 2, (y0 + bh) // 2), slice(x0 // 2, (x0 + bw) // 2)))):
            plane[sl] = np.clip(np.rint(plane[sl] + d), 0, 255).astype(np.uint8)
        enc.stdin.write(f.tobytes())
        n += 1
    enc.stdin.close()
    rc = enc.wait()
    print({"status": "ok" if rc == 0 else f"encode-failed:{rc}", "frames": n,
           "touched_px": int(touched.sum())})
    return rc


if __name__ == "__main__":
    {"extract": extract, "apply": apply}[sys.argv[1]](*sys.argv[2:])
