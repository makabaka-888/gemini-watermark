# -*- coding: utf-8 -*-
"""Repair delogo boxes with a hand-inpainted plate that follows the camera.

plate mode in repair_delogo.py pastes one still into every frame, so when the
camera shakes the fill stays put while the scene around it moves (a "sticker"
with a seam at the box edge), and flow propagation pulled garbage into the box
during fast motion. Here, per frame:

  1. the plate's context (everything outside the box, which is the temporal
     median of the video) is registered to the frame: a few translation
     candidates per frame (zero-mean NCC, coarse-to-fine, parabolic subpixel
     step), then one path through them over the whole clip that trades match
     quality against shift changes (track());
  2. the fixed plate, shifted by that translation, is the Poisson guide inside
     the box, scaled by how well the context matched (NCC). Frames where the
     scene does not match the plate (a cut, a limb crossing) fade to the smooth
     membrane instead of showing the wrong texture. No flow propagation.

Usage: python mplate_fix.py <in.mp4> <out.mp4> x,y,w,h[;...] <plate_fixed.png>@x0,y0 [max_shift]
Prints one JSON line with the registration statistics.
"""
import json
import os
import subprocess
import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

from gwvideo import planes, read_frames
from plate_fix import rgb_to_yuv
from repair_delogo import MARGIN, Poisson
from video_scan import probe

Q_LO, Q_HI = 0.6, 0.85      # context NCC -> plate weight 0..1 (a plate built from a
                            # scene that changes mid-clip only reaches ~0.6-0.75)
TOP_K = 4                   # registration candidates kept per frame
JUMP_COST = 0.01            # NCC a path gives up per pixel of shift change (see track)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-6 else 0.0


def down(img, s):
    h, w = img.shape[0] // s * s, img.shape[1] // s * s
    return img[:h, :w].reshape(h // s, s, w // s, s).mean(axis=(1, 3))


class Register:
    """Translation of each frame's plate region relative to the plate."""

    def __init__(self, plate_y, ctx, max_shift):
        self.R = max_shift
        self.pad = -(-(max_shift + 8) // 4) * 4   # divisible by every level's scale
        self.levels = []
        inside = np.pad(np.ones(plate_y.shape), self.pad)   # 0 where the padding is
        for s in (4, 2, 1):
            p = np.pad(plate_y, self.pad, mode="edge")
            if s > 1:
                p, m, v = down(p, s), down(ctx.astype(np.float64), s) > 0.99, down(inside, s) > 0.99
            else:
                m, v = ctx, inside > 0
            self.levels.append((s, p, m, v))

    def score(self, frame, s, p, m, v, dy, dx):
        """NCC of the frame with the plate shifted by (dy, dx), over context
        pixels whose shifted plate position is inside the plate."""
        h, w = m.shape
        oy, ox = self.pad // s - dy, self.pad // s - dx
        mm = m & v[oy:oy + h, ox:ox + w]
        if mm.sum() < 0.3 * m.sum():
            return -1.0
        return ncc(frame[mm], p[oy:oy + h, ox:ox + w][mm])

    def candidates(self, frame_y, k=TOP_K):
        """Up to k+1 (dy, dx, ncc) candidates for one frame: the k best local
        maxima of the coarse search plus zero shift, each refined to full
        resolution. One coarse-to-fine path alone sometimes settled on a
        wrong peak a few pixels off."""
        s, p, m, v = self.levels[0]
        f = down(frame_y, s)[:m.shape[0], :m.shape[1]]
        r = -(-self.R // s)
        sc = {(dy, dx): self.score(f, s, p, m, v, dy, dx)
              for dy in range(-r, r + 1) for dx in range(-r, r + 1)}
        peaks = [c for c in sc if all(sc[c] >= sc.get((c[0] + a, c[1] + b), -2.0)
                                      for a in (-1, 0, 1) for b in (-1, 0, 1))]
        peaks = sorted(peaks, key=sc.get, reverse=True)[:k]
        if (0, 0) not in peaks:
            peaks.append((0, 0))
        out = [self.refine(frame_y, c) for c in peaks]
        uniq = []
        for c in sorted(out, key=lambda c: -c[2]):
            if all(np.hypot(c[0] - u[0], c[1] - u[1]) > 0.5 for u in uniq):
                uniq.append(c)
        return uniq

    def refine(self, frame_y, best):
        for s, p, m, v in self.levels[1:]:
            f = down(frame_y, s) if s > 1 else frame_y
            f = f[:m.shape[0], :m.shape[1]]
            cy, cx = best[0] * 2, best[1] * 2
            cands = [(cy + dy, cx + dx) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)]
            scores = {c: self.score(f, s, p, m, v, *c) for c in cands
                      if abs(c[0]) * s <= self.R and abs(c[1]) * s <= self.R}
            best = max(scores, key=scores.get)
        q = scores[best]
        sub = []
        for axis in (0, 1):
            lo, hi = list(best), list(best)
            lo[axis] -= 1
            hi[axis] += 1
            a, b = scores.get(tuple(lo)), scores.get(tuple(hi))
            den = (a - 2 * q + b) if a is not None and b is not None else 0
            sub.append(best[axis] + (0.5 * (a - b) / den if den < 0 else 0.0))
        return sub[0], sub[1], q


def track(cands, cost=JUMP_COST):
    """Pick one candidate per frame maximising the summed NCC minus `cost`
    per pixel of shift change between consecutive frames (Viterbi). A false
    match that wins for a frame or two by a hair no longer makes the fill
    twitch, while a pan or a real camera jolt, which wins by a lot, is kept."""
    total = np.array([c[2] for c in cands[0]])
    back = []
    for t in range(1, len(cands)):
        prev = np.array([c[:2] for c in cands[t - 1]])
        cur = np.array([c[:2] for c in cands[t]])
        jump = np.hypot(cur[:, None, 0] - prev[None, :, 0], cur[:, None, 1] - prev[None, :, 1])
        step = total[None, :] - cost * jump
        back.append(step.argmax(1))
        total = step.max(1) + np.array([c[2] for c in cands[t]])
    i = int(total.argmax())
    path = [i]
    for b in reversed(back):
        i = int(b[i])
        path.append(i)
    path.reverse()
    return [cands[t][i] for t, i in enumerate(path)]


def main(src, dst, boxes, plate_spec, max_shift="28"):
    w, h, _ = probe(src)
    rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=r_frame_rate", "-of", "csv=p=0", src],
                          capture_output=True, text=True, check=True).stdout.strip()
    Y, U, V = [], [], []
    for f in read_frames(src, w, h):
        y, u, v = planes(f, w, h)
        Y.append(y.copy()); U.append(u.copy()); V.append(v.copy())
    Y, U, V = np.stack(Y), np.stack(U), np.stack(V)
    n = len(Y)

    hole = np.zeros((h, w), bool)
    for b in boxes.split(";"):
        x, y, bw, bh = (int(v) for v in b.split(","))
        hole[y:y + bh, x:x + bw] = True
    hole = ndi.binary_dilation(hole, iterations=MARGIN)
    hole_c = ndi.binary_dilation(hole.reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3)), iterations=1)

    png, at = plate_spec.rsplit("@", 1)
    x0, y0 = (int(v) for v in at.split(","))
    py, pu, pv = rgb_to_yuv(np.asarray(Image.open(png).convert("RGB")).astype(np.float64))
    ph, pw = py.shape
    pool = lambda a: a.reshape(ph // 2, 2, pw // 2, 2).mean(axis=(1, 3))
    pu, pv = pool(pu), pool(pv)
    hole_p = hole[y0:y0 + ph, x0:x0 + pw]
    ctx = ~ndi.binary_dilation(hole_p, iterations=6)
    ctx[:4, :] = ctx[-4:, :] = ctx[:, :4] = ctx[:, -4:] = False
    reg = Register(py, ctx, int(max_shift))

    path = track([reg.candidates(Y[t, y0:y0 + ph, x0:x0 + pw].astype(np.float64)) for t in range(n)])
    shifts = np.array([c[:2] for c in path])
    quality = np.array([c[2] for c in path])
    weight = np.clip((quality - Q_LO) / (Q_HI - Q_LO), 0, 1)
    weight = ndi.uniform_filter1d(ndi.minimum_filter1d(weight, 5, mode="nearest"), 7, mode="nearest")
    if os.environ.get("MPLATE_REG_ONLY"):
        print(json.dumps({"src": src, "status": "reg-only", "q": quality.round(3).tolist(),
                          "shift": shifts.round(2).tolist()}))
        return 0

    solver_y, solver_c = Poisson(hole), Poisson(hole_c)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                            "-s", f"{w}x{h}", "-r", rate, "-i", "-", "-i", src,
                            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
                            "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy",
                            "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    for t in range(n):
        (dy, dx), wt = shifts[t], weight[t]
        planes_out = []
        for seq, plate, solver, sc, (ox, oy) in ((Y, py, solver_y, 1, (x0, y0)),
                                                 (U, pu, solver_c, 2, (x0 // 2, y0 // 2)),
                                                 (V, pv, solver_c, 2, (x0 // 2, y0 // 2))):
            g = np.zeros(seq[t].shape)
            hh, ww = plate.shape
            g[oy:oy + hh, ox:ox + ww] = wt * ndi.shift(plate, (dy / sc, dx / sc), order=1, mode="nearest")
            planes_out.append(np.clip(np.rint(solver.fill(seq[t], g)), 0, 255).astype(np.uint8))
        enc.stdin.write(b"".join(p.tobytes() for p in planes_out))
    enc.stdin.close()
    rc = enc.wait()
    mag = np.hypot(shifts[:, 0], shifts[:, 1])
    if os.environ.get("MPLATE_LOG_DIR"):
        stem = os.path.splitext(os.path.basename(dst))[0]
        with open(os.path.join(os.environ["MPLATE_LOG_DIR"], stem + ".json"), "w", encoding="utf-8") as f:
            json.dump({"q": quality.round(3).tolist(), "w": weight.round(3).tolist(),
                       "shift": shifts.round(2).tolist()}, f)
    print(json.dumps({"src": src, "status": "ok" if rc == 0 else f"encode-failed:{rc}", "frames": n,
                      "guide": "mplate", "q_median": round(float(np.median(quality)), 3),
                      "q_min": round(float(quality.min()), 3),
                      "low_weight_frames": int((weight < 0.5).sum()),
                      "shift_max": round(float(mag.max()), 2),
                      "shift_p90": round(float(np.percentile(mag, 90)), 2)}, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
