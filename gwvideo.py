# -*- coding: utf-8 -*-
"""Remove Gemini/Veo sparkle watermarks from a video by inverting the blend.

The Veo overlay is a static white sparkle alpha-blended onto every frame,

    out = a * white + (1 - a) * frame,

so it inverts exactly once a(x, y) is known. Its shape is GeminiWatermarkTool's
sparkle alpha map and it sits on a fixed grid: size S = 48 px per 1280 px of
width, centred 2*S (current) or 2.5*S (older clips) in from the bottom-right
corner. Image-to-video clips can also carry the source image's own sparkle,
re-rendered into the scene nearby; those are found by correlation on the
temporal mean of the corner and refined.

Opacity is estimated from the step across the sparkle boundary on the
temporal mean: with the right opacity the un-blended mean is continuous
across the edge, and texture only adds zero-mean noise to that median, unlike
total variation, which biases opacity low on static textured backgrounds.
Grid overlays come out at ~0.375 (the current Gemini profile); a sparkle at a
grid spot that is not there comes out ~0 and is skipped. Each sparkle's
on-screen frames are decided per 1-second window.

Frames are processed as raw yuv420p (no RGB round trip), white = (235,128,128)
for tv-range video; the 1-2 px edge band, where encoding rings differently in
every frame, is refilled from both sides. Re-encoded with libx264 CRF 18;
audio is copied.

Usage: python gwvideo.py <in.mp4> <out.mp4>
Prints one JSON line describing what was removed.
"""
import json
import os
import subprocess
import sys

import numpy as np
import scipy.ndimage as ndi

import gwclean
from gwclean import alpha_map, ncc_at
from video_scan import probe

Y_WHITE, C_WHITE = 235.0, 128.0
VEO_SIZE = 48 / 1280                    # sparkle size as a fraction of width
GRID = (2.0, 2.5)                       # centre inset from the corner, in sizes
# An image-to-video source image's own sparkle lands 0.5-1.6 sizes in from
# the corner (its canonical spot after the frame is cropped/scaled to 16:9).
EXTRA_BOX = (0.5, 1.6)
EXTRA_NCC = 0.35                        # candidate search
EXTRA_NCC_BIG, EXTRA_NCC_SMALL = 0.5, 0.4   # acceptance, see extra_ok
EXTRA_NCC_FAR = 0.55                    # big sparkle outside EXTRA_BOX (up to 3 S in)
EXTRA_REACH = 3.0
MIN_OPACITY = 0.15
CLONE_IF_NCC = 0.3                      # residual after inversion -> clone instead


def read_video_overrides(kind="add"):
    """overrides_video.tsv: name<TAB>cx,cy,size adds a sparkle to clone-fill;
    name<TAB>skip:cx,cy drops a detected in-scene sparkle near that point
    (false positives such as scene text)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overrides_video.tsv")
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8-sig"):
            if line.strip() and not line.startswith("#"):
                name, spec = line.rstrip("\r\n").split("\t")[:2]
                is_skip = spec.startswith("skip:")
                if is_skip != (kind == "skip"):
                    continue
                out.setdefault(name, []).append(tuple(float(v) for v in spec.replace("skip:", "").split(",")))
    return out


def read_frames(path, w, h):
    fs = w * h * 3 // 2
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "yuv420p", "-"], stdout=subprocess.PIPE)
    while True:
        buf = p.stdout.read(fs)
        if len(buf) < fs:
            break
        yield np.frombuffer(buf, np.uint8)
    p.wait()


def _crop(img, ox, oy, cx, cy, s):
    """Window of +-1 size around a sparkle; returns (window, its ox, its oy)."""
    x0 = max(0, int(cx - 1.5 * s) - ox)
    y0 = max(0, int(cy - 1.5 * s) - oy)
    x1 = min(img.shape[1], int(cx + 1.5 * s) - ox + 1)
    y1 = min(img.shape[0], int(cy + 1.5 * s) - oy + 1)
    return img[y0:y1, x0:x1], ox + x0, oy + y0


def boundary_opacity(mean_y, ox, oy, cx, cy, s, sigma=0.0, white=Y_WHITE):
    """Opacity that makes the un-blended mean continuous across the edge."""
    mean_y, ox, oy = _crop(mean_y, ox, oy, cx, cy, s)
    unit = alpha_map(mean_y.shape, ox, oy, cx, cy, s, 1.0, sigma)
    star = unit > 0.5
    inner = (star & ~ndi.binary_erosion(star, iterations=3)) & ndi.binary_erosion(star, iterations=1)
    outer = ndi.binary_dilation(star, iterations=4) & ~ndi.binary_dilation(star, iterations=1)
    edge = ndi.binary_dilation(star, iterations=1) & ~ndi.binary_erosion(star, iterations=1)
    gi = ndi.gaussian_filter(inner * 1.0, 2.0)
    go = ndi.gaussian_filter(outer * 1.0, 2.0)
    sel = edge & (gi > 0.05) & (go > 0.05)
    if sel.sum() < 10:
        return 0.0
    best_k, best_d = 0.0, None
    for k in np.arange(0.0, 0.9, 0.005):
        a = k * unit
        r = (mean_y - white * a) / (1 - a)
        d = np.median((ndi.gaussian_filter(r * inner, 2.0) / np.maximum(gi, 1e-6)
                       - ndi.gaussian_filter(r * outer, 2.0) / np.maximum(go, 1e-6))[sel])
        if best_d is None or abs(d) < abs(best_d):
            best_k, best_d = float(k), d
    return best_k


def planes(frame, w, h):
    y = frame[:w * h].reshape(h, w)
    u = frame[w * h:w * h + w * h // 4].reshape(h // 2, w // 2)
    v = frame[w * h + w * h // 4:].reshape(h // 2, w // 2)
    return y, u, v


class Patch:
    """Per-frame inversion of one fitted sparkle, on a small window."""

    def __init__(self, f, w, h, band_sigma=1.2):
        self.w, self.h = w, h
        pad = int(0.6 * f["size"]) + 8
        x0 = max(0, int(f["cx"] - f["size"] / 2) - pad) // 2 * 2
        y0 = max(0, int(f["cy"] - f["size"] / 2) - pad) // 2 * 2
        x1 = min(w, int(f["cx"] + f["size"] / 2) + pad + 2) // 2 * 2
        y1 = min(h, int(f["cy"] + f["size"] / 2) + pad + 2) // 2 * 2
        self.ys, self.xs = slice(y0, y1), slice(x0, x1)
        self.cys, self.cxs = slice(y0 // 2, y1 // 2), slice(x0 // 2, x1 // 2)
        ay = alpha_map((y1 - y0, x1 - x0), x0, y0, f["cx"], f["cy"], f["size"],
                       f["opacity"], f["sigma"])
        self.ay = ay
        # transcoded clips can carry less overlay in chroma than in luma
        self.ac = ay.reshape(ay.shape[0] // 2, 2, ay.shape[1] // 2, 2).mean(axis=(1, 3)) \
            * (f["opacity_c"] / max(f["opacity"], 1e-6))
        self.sig = band_sigma
        # 0.25 rather than 0.5 so the thin tapering tips are inside the band
        star = ay > 0.25 * ay.max()
        self.by = ndi.binary_dilation(star, iterations=2) & ~ndi.binary_erosion(star, iterations=1)
        cstar = self.ac > 0.25 * self.ac.max()
        self.bc = ndi.binary_dilation(cstar, iterations=1) & ~ndi.binary_erosion(cstar, iterations=1)
        # inversion multiplies compression noise by 1/(1-a); where that gain
        # is large (stacked overlays, a ~ 0.7) it shows as dark specks, so
        # blend toward a light blur in proportion to the gain
        self.wy = np.clip((1 / (1 - ay) - 1) / 2.5, 0, 0.8)
        self.wc = np.clip((1 / (1 - self.ac) - 1) / 2.5, 0, 0.8)

    @staticmethod
    def _band_fill(p, band, sig):
        keep = (~band).astype(np.float64)
        num = ndi.gaussian_filter(p * keep, sig)
        den = ndi.gaussian_filter(keep, sig)
        ok = band & (den > 1e-3)
        p[ok] = num[ok] / den[ok]

    def apply(self, frame):
        f = frame.copy()
        yp, up, vp = planes(f, self.w, self.h)
        for plane, sl, a, white, band, sig, wn in (
                (yp, (self.ys, self.xs), self.ay, Y_WHITE, self.by, self.sig, self.wy),
                (up, (self.cys, self.cxs), self.ac, C_WHITE, self.bc, self.sig / 2, self.wc),
                (vp, (self.cys, self.cxs), self.ac, C_WHITE, self.bc, self.sig / 2, self.wc)):
            win = plane[sl].astype(np.float64)
            win = (win - white * a) / (1 - a)
            win += wn * (ndi.gaussian_filter(win, 0.8) - win)
            self._band_fill(win, band, sig)
            plane[sl] = np.clip(np.rint(win), 0, 255).astype(np.uint8)
        return f


class CloneFill:
    """Fill a static re-rendered sparkle by cloning from one fixed offset.

    The offset is chosen once on the temporal mean, so the fill is the
    neighbouring texture of the same frame, moving with the content; each
    frame is Poisson-blended into its own surroundings (system factorised once).
    """

    def __init__(self, f, w, h, mean_y, ox, oy):
        self.w, self.h = w, h
        s = f["size"]
        reach = int(2.5 * s)
        pad = reach + int(s) + 8
        x0 = max(0, int(f["cx"] - pad)) // 2 * 2
        y0 = max(0, int(f["cy"] - pad)) // 2 * 2
        x1 = min(w, int(f["cx"] + pad)) // 2 * 2
        y1 = min(h, int(f["cy"] + pad)) // 2 * 2
        self.ys, self.xs = slice(y0, y1), slice(x0, x1)
        self.cys, self.cxs = slice(y0 // 2, y1 // 2), slice(x0 // 2, x1 // 2)
        unit = alpha_map((y1 - y0, x1 - x0), x0, y0, f["cx"], f["cy"], s, 1.0, f["sigma"])
        hole = ndi.binary_dilation(unit > 0.02, iterations=max(2, int(0.08 * s)))
        hole[0, :] = hole[-1, :] = False
        hole[:, 0] = hole[:, -1] = False
        ring = ndi.binary_dilation(hole, iterations=max(4, int(0.15 * s))) & ~hole
        forbid = ndi.binary_dilation(hole, iterations=4)
        forbid[0, :] = forbid[-1, :] = forbid[:, 0] = forbid[:, -1] = True
        lum = np.zeros((y1 - y0, x1 - x0))
        my0, mx0 = y0 - oy, x0 - ox
        src = mean_y[max(0, my0):my0 + lum.shape[0], max(0, mx0):mx0 + lum.shape[1]]
        lum[max(0, -my0):max(0, -my0) + src.shape[0], max(0, -mx0):max(0, -mx0) + src.shape[1]] = src
        off = gwclean._best_offset(lum, hole, ring, forbid, reach)
        self.off = off or (0, -int(1.5 * s))
        self.ok = off is not None
        hole_c = hole.reshape(hole.shape[0] // 2, 2, hole.shape[1] // 2, 2).any(axis=(1, 3))
        hole_c[0, :] = hole_c[-1, :] = hole_c[:, 0] = hole_c[:, -1] = False
        self.sys_y = self._system(hole)
        self.sys_c = self._system(hole_c)
        self.off_c = (self.off[0] // 2, self.off[1] // 2)

    @staticmethod
    def _system(hole):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        idx = -np.ones(hole.shape, np.int64)
        ys, xs = np.nonzero(hole)
        idx[ys, xs] = np.arange(len(ys))
        n = len(ys)
        rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
        nbrs = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = ys + dy, xs + dx
            inside = idx[ny, nx] >= 0
            rows.append(np.arange(n)[inside])
            cols.append(idx[ny, nx][inside])
            vals.append(-np.ones(inside.sum()))
            nbrs.append((ny, nx, inside))
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))
        return ys, xs, nbrs, spla.factorized(A)

    @staticmethod
    def _solve(win, system, off):
        ys, xs, nbrs, solve = system
        dy, dx = off
        src = np.roll(win, (-dy, -dx), axis=(0, 1))
        b = np.zeros(len(ys))
        for ny, nx, inside in nbrs:
            b += src[ys, xs] - src[ny, nx]
            b[~inside] += win[ny[~inside], nx[~inside]]
        out = win.copy()
        out[ys, xs] = solve(b)
        return out

    def apply(self, frame):
        f = frame.copy()
        yp, up, vp = planes(f, self.w, self.h)
        for plane, sl, system, off in ((yp, (self.ys, self.xs), self.sys_y, self.off),
                                       (up, (self.cys, self.cxs), self.sys_c, self.off_c),
                                       (vp, (self.cys, self.cxs), self.sys_c, self.off_c)):
            win = self._solve(plane[sl].astype(np.float64), system, off)
            plane[sl] = np.clip(np.rint(win), 0, 255).astype(np.uint8)
        return f


def _inverted(win, wx, wy, cx, cy, s, k, sg, white, band_sigma):
    a = alpha_map(win.shape, wx, wy, cx, cy, s, k, sg)
    r = (win - white * a) / (1 - a)
    star = a > 0.5 * a.max()
    band = ndi.binary_dilation(star, iterations=2) & ~ndi.binary_erosion(star, iterations=1)
    Patch._band_fill(r, band, band_sigma)
    return r


def residual_score(mean, mean_u, mean_v, ox, oy, cx, cy, s, k, kc, sg):
    """How sparkle-like the temporal mean still is after inversion + edge fill:
    max |NCC| of luma, of luma edges, and of coloured chroma planes."""
    win, wx, wy = _crop(mean, ox, oy, cx, cy, s)
    r = _inverted(win, wx, wy, cx, cy, s, k, sg, Y_WHITE, 1.2)
    unit = alpha_map(win.shape, wx, wy, cx, cy, s, 1.0, sg)
    grad = lambda img: np.hypot(ndi.sobel(img, 0), ndi.sobel(img, 1))
    scores = [ncc_at(r, unit), ncc_at(grad(r), grad(unit))]
    for m in (mean_u, mean_v):
        wc, cxw, cyw = _crop(m, ox // 2, oy // 2, cx / 2, cy / 2, s / 2)
        if abs(wc.mean() - C_WHITE) > 8:
            rc = _inverted(wc, cxw, cyw, cx / 2, cy / 2, s / 2, kc, sg / 2, C_WHITE, 0.6)
            scores.append(ncc_at(rc, alpha_map(wc.shape, cxw, cyw, cx / 2, cy / 2, s / 2, 1.0)))
    return max(abs(v) for v in scores)


def extra_ok(w, h, cx, cy, s, S, ncc):
    """Accept an in-scene sparkle. Big ones (a source image's watermark) need
    NCC >= 0.5 (real 0.73-0.89, texture <= 0.38 on the 55 test clips). Small
    templates match HUD corners and texture as well as a real third-generation
    sparkle, so a small one is only accepted where those sit: ~0.64 S, 0.70 S
    in from the corner."""
    if s >= 0.55 * S:
        usual = EXTRA_BOX[0] * S <= w - cx <= EXTRA_BOX[1] * S and EXTRA_BOX[0] * S <= h - cy <= EXTRA_BOX[1] * S
        return ncc >= (EXTRA_NCC_BIG if usual else EXTRA_NCC_FAR)
    return ncc >= EXTRA_NCC_SMALL and np.hypot((w - cx) - 0.64 * S, (h - cy) - 0.70 * S) <= 0.25 * S


def refine(mean_y, ox, oy, cx, cy, s):
    """Sub-pixel centre and size that best correlate with the sparkle."""
    mean_y, ox, oy = _crop(mean_y, ox, oy, cx, cy, s)
    best = (-2.0, cx, cy, s)
    for ds in np.arange(-0.08, 0.081, 0.02):
        ss = s * (1 + ds)
        for dx in np.arange(-2, 2.01, 0.5):
            for dy in np.arange(-2, 2.01, 0.5):
                v = ncc_at(mean_y, alpha_map(mean_y.shape, ox, oy, cx + dx, cy + dy, ss, 1.0))
                if v > best[0]:
                    best = (v, cx + dx, cy + dy, ss)
    return best


def edge_sigma(mean_y, ox, oy, cx, cy, s, k):
    """Edge softness of a re-rendered sparkle (grid overlays are sharp)."""
    mean_y, ox, oy = _crop(mean_y, ox, oy, cx, cy, s)

    def tv(sg):
        a = alpha_map(mean_y.shape, ox, oy, cx, cy, s, k, sg)
        r = (mean_y - Y_WHITE * a) / (1 - a)
        return np.abs(np.diff(r, axis=0)).sum() + np.abs(np.diff(r, axis=1)).sum()
    return float(min(np.arange(0, 1.81, 0.15), key=tv))


def main(src, dst):
    w, h, fps = probe(src)
    rate, codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=r_frame_rate,codec_name", "-of", "default=nw=1:nk=1", src],
                                 capture_output=True, text=True, check=True).stdout.split()[:2]
    # Veo delivers H.264; anything else was re-encoded (often rescaled), which
    # smears the overlay so inversion leaves a ghost -> clone-fill instead
    if not rate[0].isdigit():
        rate, codec = codec, rate
    transcoded = codec != "h264"
    # The 48 px template matches Veo's 720p rasterisation; the 1080p clip's
    # 72 px overlay inverted to a visible ghost in luma and chroma alike.
    native = (w, h) == (1280, 720)
    S = VEO_SIZE * w
    side = int(6 * S)
    ox, oy = max(0, (w - side) // 2 * 2), max(0, (h - side) // 2 * 2)
    frames_uv = []
    ys = []
    for f in read_frames(src, w, h):
        y, u, v = planes(f, w, h)
        ys.append(y[oy:, ox:].copy())
        frames_uv.append((u[oy // 2:, ox // 2:].copy(), v[oy // 2:, ox // 2:].copy()))
    ys = np.stack(ys)
    n = len(ys)
    mean = ys.astype(np.float64).mean(axis=0)
    mean_u = np.mean([uv[0] for uv in frames_uv], axis=0, dtype=np.float64)
    mean_v = np.mean([uv[1] for uv in frames_uv], axis=0, dtype=np.float64)
    del frames_uv
    step = max(1, int(round(fps)))

    cands = [("grid", w - g * S, h - g * S, S) for g in GRID]
    # extras include a third-generation sparkle as small as ~0.3 S
    sizes = sorted({max(10, int(round(S * f))) for f in np.arange(0.3, 1.06, 0.05)})
    peaks = gwclean._peaks(np.repeat(mean[..., None], 3, axis=2).astype(np.float32),
                           (0, 0, mean.shape[1], mean.shape[0]), sizes, 1, EXTRA_NCC)
    for x, y, s, sc in sorted(peaks, key=lambda p: -p[3]):   # best-fitting size wins
        cx, cy = ox + x + s / 2, oy + y + s / 2
        if not (EXTRA_BOX[0] * S <= w - cx <= EXTRA_REACH * S and
                EXTRA_BOX[0] * S <= h - cy <= EXTRA_REACH * S):
            continue
        if all(np.hypot(cx - c[1], cy - c[2]) > 0.4 * max(s, c[3]) for c in cands):
            cands.append(("extra", cx, cy, float(s)))

    skips = read_video_overrides("skip").get(os.path.basename(src), [])
    fits = []
    for kind, cx, cy, s in cands:
        v = None
        if kind == "extra":
            v, cx, cy, s = refine(mean, ox, oy, cx, cy, s)
            if not extra_ok(w, h, cx, cy, s, S, v):
                continue
            if any(np.hypot(cx - sx, cy - sy) < 0.5 * S for sx, sy in skips):
                continue
        k = boundary_opacity(mean, ox, oy, cx, cy, s)
        if k < MIN_OPACITY:
            continue
        sg = 0.0 if kind == "grid" else edge_sigma(mean, ox, oy, cx, cy, s, k)
        if sg:
            k = boundary_opacity(mean, ox, oy, cx, cy, s, sg)
        # chroma opacity, measurable only where the background has colour
        kc = k
        cm = [boundary_opacity(m, ox // 2, oy // 2, cx / 2, cy / 2, s / 2, sg / 2, C_WHITE)
              for m in (mean_u, mean_v)
              if abs(_crop(m, ox // 2, oy // 2, cx / 2, cy / 2, s / 2)[0].mean() - C_WHITE) > 8]
        if cm:
            kc = float(np.median(cm))
        # A sparkle re-rendered into the scene (extra), or an overlay that went
        # through a resize/transcode, is no longer an exact blend: inversion
        # leaves a sparkle-shaped residue, so those are clone-filled instead.
        resid = residual_score(mean, mean_u, mean_v, ox, oy, cx, cy, s, k, kc, sg)
        mode = "clone" if kind == "extra" or transcoded or not native or resid >= CLONE_IF_NCC \
            else "invert"
        # GWV_GRID_MODE=invert|clone overrides the choice for grid overlays,
        # for producing both variants of a doubtful clip and comparing them
        if kind == "grid" and os.environ.get("GWV_GRID_MODE") in ("invert", "clone"):
            mode = os.environ["GWV_GRID_MODE"]
        # on-screen per 1-second window: opacity there must be at least half
        present = np.zeros(n, bool)
        for t0 in range(0, n, step):
            m = ys[t0:t0 + step].astype(np.float64).mean(axis=0)
            if boundary_opacity(m, ox, oy, cx, cy, s, sg) >= 0.5 * k:
                present[t0:t0 + step] = True
        if mode == "clone":
            present[:] = True       # in-scene: the cloned fill is harmless where it is faint
        if not present.any():
            continue
        fits.append({"kind": kind, "mode": mode, "ncc": None if v is None else round(v, 2),
                     "cx": round(cx, 2), "cy": round(cy, 2),
                     "size": round(s, 2), "opacity": round(k, 3), "opacity_c": round(kc, 3),
                     "sigma": round(sg, 2), "resid_ncc": round(resid, 2),
                     "frames_on": int(present.sum()), "_present": present})

    # manual additions: sparkles no rule picks up (e.g. one sitting on a
    # high-contrast edge), clone-filled in every frame
    for cx, cy, s in read_video_overrides().get(os.path.basename(src), []):
        fits.append({"kind": "manual", "mode": "clone", "ncc": None, "cx": cx, "cy": cy, "size": s,
                     "opacity": 0.0, "opacity_c": 0.0, "sigma": 0.0, "resid_ncc": None,
                     "frames_on": n, "_present": np.ones(n, bool)})

    if not fits:
        print(json.dumps({"src": src, "status": "no-sparkle", "frames": n}))
        return 2
    patches = [CloneFill(f, w, h, mean, ox, oy) if f["mode"] == "clone" else Patch(f, w, h)
               for f in fits]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                            "-s", f"{w}x{h}", "-r", rate, "-i", "-", "-i", src,
                            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
                            "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy",
                            "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    for t, frame in enumerate(read_frames(src, w, h)):
        for f, patch in zip(fits, patches):
            if t < n and f["_present"][t]:
                frame = patch.apply(frame)
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    rc = enc.wait()
    for f in fits:
        del f["_present"]
    print(json.dumps({"src": src, "status": "ok" if rc == 0 else f"encode-failed:{rc}",
                      "frames": n, "fits": fits}))
    return rc


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
