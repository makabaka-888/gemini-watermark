# -*- coding: utf-8 -*-
"""Repair the smeared rectangles that ffmpeg delogo left in a video.

The pixels inside each delogo box are an interpolation of the box border, so
they carry no information. Each frame's box is rebuilt from:

  1. propagation: optical flow (torchvision RAFT, on half-resolution luma) is
     estimated for every frame pair; inside the boxes it is replaced by a
     harmonic interpolation of the surrounding flow (flow completion). Each box
     pixel is then followed forward and backward through the completed flow
     until it lands on genuine pixels outside the boxes in a nearby frame, and
     takes its value there (both directions are blended by temporal distance).
     This recovers whatever the camera or the moving content shows elsewhere.
  2. texture: pixels never seen outside the boxes get the texture of the same
     frame at one fixed offset (chosen on the temporal median), so the fill
     moves with the content instead of being a frozen patch.
  3. blending: per frame and plane, a Poisson solve keeps the gradients of
     that guide and matches the frame's own pixels around the box.

Works in yuv420p (no RGB round trip); pixels outside the boxes are untouched.
Re-encodes with libx264 CRF 18 and copies the audio.

Usage: python repair_delogo.py <in.mp4> <out.mp4> x,y,w,h[;x,y,w,h...] [guide] [cache.npz]
  guide: clone (default) | harmonic | plate:<png>@x0,y0   (see main())
  Same-frame clone texture failed where the box hid an object (a foot's
  middle became floor), so for static shots the plate guide -- a still of the
  region inpainted by hand -- or the plain harmonic membrane are the options.
Prints one JSON line with how much of the boxes came from propagation.
"""
import json
import subprocess
import sys

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

import gwclean
from gwvideo import planes, read_frames
from video_scan import probe

MAX_STEPS = 48          # how far (frames) to follow a pixel through the flow
MARGIN = 2              # box dilation: delogo also touches the border pixels


class Poisson:
    """Dirichlet Poisson solver on a fixed hole (factorised once)."""

    def __init__(self, hole):
        hole = hole.copy()
        hole[0, :] = hole[-1, :] = hole[:, 0] = hole[:, -1] = False
        self.hole = hole
        idx = -np.ones(hole.shape, np.int64)
        ys, xs = np.nonzero(hole)
        idx[ys, xs] = np.arange(len(ys))
        n = len(ys)
        rows, cols, vals, self.nbrs = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)], []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = ys + dy, xs + dx
            inside = idx[ny, nx] >= 0
            rows.append(np.arange(n)[inside])
            cols.append(idx[ny, nx][inside])
            vals.append(-np.ones(inside.sum()))
            self.nbrs.append((ny, nx, inside))
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))
        self.ys, self.xs, self.solve = ys, xs, spla.factorized(A)

    def fill(self, target, guide=None):
        """Values inside the hole: Laplacian of guide (0 if None), target outside."""
        ys, xs = self.ys, self.xs
        b = np.zeros(len(ys))
        for ny, nx, inside in self.nbrs:
            if guide is not None:
                b += guide[ys, xs] - guide[ny, nx]
            b[~inside] += target[ny[~inside], nx[~inside]]
        out = target.astype(np.float64).copy()
        out[ys, xs] = self.solve(b)
        return out


def raft_flows(lum_half):
    """Forward (t->t+1) and backward (t+1->t) flows, (N-1, 2, H, W) in px."""
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(dev)
    x = torch.from_numpy(lum_half).float().div(255).mul(2).sub(1)       # (N,H,W)
    x = x[:, None].expand(-1, 3, -1, -1)
    fwd, bwd = [], []
    with torch.no_grad():
        for i in range(0, len(x) - 1, 4):
            a = x[i:min(i + 4, len(x) - 1)].to(dev)
            b = x[i + 1:min(i + 5, len(x))].to(dev)
            fwd.append(model(a, b, num_flow_updates=12)[-1].cpu())
            bwd.append(model(b, a, num_flow_updates=12)[-1].cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(fwd).numpy(), torch.cat(bwd).numpy()


def sample(img, pos):
    """Bilinear sample img (H,W[,C]) at float positions pos (M,2) = (x,y)."""
    h, w = img.shape[:2]
    x = np.clip(pos[:, 0], 0, w - 1.001)
    y = np.clip(pos[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    if img.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) +
            (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def propagate(hole, flows_fwd, flows_bwd, scale):
    """For every frame and hole pixel: (frame, x, y) of genuine pixels found by
    following the flow forward and backward, with step counts. scale converts
    plane pixels to flow (half-res) pixels: 2 for full-res luma, 1 for chroma."""
    n = len(flows_fwd) + 1
    h, w = hole.shape
    ys, xs = np.nonzero(hole)
    base = np.stack([xs, ys], 1).astype(np.float64)
    bad = ndi.binary_dilation(hole, iterations=1)
    res = []
    for t in range(n):
        found = {}
        for direction, flows, rng in (("f", flows_fwd, range(t, min(n - 1, t + MAX_STEPS))),
                                      ("b", flows_bwd, range(t - 1, max(-1, t - 1 - MAX_STEPS), -1))):
            pos = base.copy()
            alive = np.ones(len(pos), bool)
            frame = np.full(len(pos), -1)
            src = np.zeros_like(pos)
            steps = np.zeros(len(pos))
            for k, s in enumerate(rng):
                idx = np.nonzero(alive & (frame < 0))[0]
                if len(idx) == 0:
                    break
                fl = flows[s].transpose(1, 2, 0)                       # (H/2, W/2, 2)
                p_half = (pos[idx] + 0.5) / scale - 0.5
                pos[idx] += sample(fl, p_half) * scale
                land = s + 1 if direction == "f" else s
                px, py = pos[idx, 0], pos[idx, 1]
                inb = (px >= 0) & (py >= 0) & (px <= w - 1.001) & (py <= h - 1.001)
                alive[idx[~inb]] = False
                ok = idx[inb]
                genuine = ~bad[np.round(pos[ok, 1]).astype(int), np.round(pos[ok, 0]).astype(int)]
                hit = ok[genuine]
                frame[hit] = land
                src[hit] = pos[hit]
                steps[hit] = k + 1
            found[direction] = (frame, src, steps)
        res.append(found)
    return ys, xs, res


def guided_values(plane_seq, ys, xs, prop, t, off, hole_shape):
    """Guide values for hole pixels of frame t: propagated where found
    (blend of both directions by temporal distance), else same-frame clone."""
    vals = np.zeros(len(ys))
    wsum = np.zeros(len(ys))
    for direction in ("f", "b"):
        frame, src, steps = prop[t][direction]
        for f in np.unique(frame[frame >= 0]):
            sel = frame == f
            wgt = 1.0 / steps[sel]
            vals[sel] += sample(plane_seq[f].astype(np.float64), src[sel]) * wgt
            wsum[sel] += wgt
    got = wsum > 0
    vals[got] /= wsum[got]
    dy, dx = off
    clone = plane_seq[t][np.clip(ys + dy, 0, hole_shape[0] - 1), np.clip(xs + dx, 0, hole_shape[1] - 1)]
    vals[~got] = clone[~got]
    return vals, got


def best_offset_color(med_y, med_u, med_v, hole, ring_w=8, step=3):
    """Offset (dy, dx) whose surroundings best match the hole's ring in colour.
    Matching luma alone picked brown dirt for blue-grey stone."""
    h, w = hole.shape
    up = lambda c: np.repeat(np.repeat(c, 2, 0), 2, 1).astype(np.float64)
    img = np.stack([med_y.astype(np.float64), 2 * up(med_u), 2 * up(med_v)], -1)
    ring = ndi.binary_dilation(hole, iterations=ring_w) & ~hole
    region = hole | ring
    forbid = ndi.binary_dilation(hole, iterations=4)
    ys, xs = np.nonzero(region)
    ry, rx = np.nonzero(ring)
    target = img[ry, rx]
    reach = int(2.0 * max(hole.sum(1).max(), hole.sum(0).max()))
    best, best_off = np.inf, None
    for dy in range(-reach, reach + 1, step):
        for dx in range(-reach, reach + 1, step):
            if ys.min() + dy < 0 or xs.min() + dx < 0 or ys.max() + dy >= h or xs.max() + dx >= w:
                continue
            if forbid[ys + dy, xs + dx].any():
                continue
            c = float(((img[ry + dy, rx + dx] - target) ** 2).mean())
            if c < best:
                best, best_off = c, (dy, dx)
    if best_off is None:
        return (0, -int(hole.sum(1).max()) - 4)
    by, bx = best_off
    for dy in range(by - step + 1, by + step):
        for dx in range(bx - step + 1, bx + step):
            if ys.min() + dy < 0 or xs.min() + dx < 0 or ys.max() + dy >= h or xs.max() + dx >= w:
                continue
            if forbid[ys + dy, xs + dx].any():
                continue
            c = float(((img[ry + dy, rx + dx] - target) ** 2).mean())
            if c < best:
                best, best_off = c, (dy, dx)
    return best_off


def save_prop(path, parts):
    """parts: [(ys, xs, res), ...] for luma and chroma; only found entries kept."""
    out = {}
    for k, (ys, xs, res) in enumerate(parts):
        out[f"ys{k}"], out[f"xs{k}"] = ys, xs
        rows = []
        for t, found in enumerate(res):
            for di, direction in enumerate(("f", "b")):
                frame, src_, steps = found[direction]
                idx = np.nonzero(frame >= 0)[0]
                rows.append(np.stack([np.full(len(idx), t), np.full(len(idx), di), idx, frame[idx],
                                      src_[idx, 0], src_[idx, 1], steps[idx]], 1))
        out[f"rows{k}"] = np.concatenate(rows) if rows else np.zeros((0, 7))
        out[f"n{k}"] = np.array([len(res)])
    np.savez_compressed(path, **out)


def load_prop(path):
    z = np.load(path)
    parts = []
    for k in range(2):
        ys, xs, rows, n = z[f"ys{k}"], z[f"xs{k}"], z[f"rows{k}"], int(z[f"n{k}"][0])
        res = []
        for t in range(n):
            found = {}
            for di, direction in enumerate(("f", "b")):
                frame = np.full(len(ys), -1)
                src_ = np.zeros((len(ys), 2))
                steps = np.zeros(len(ys))
                r = rows[(rows[:, 0] == t) & (rows[:, 1] == di)]
                if len(r):
                    i = r[:, 2].astype(int)
                    frame[i] = r[:, 3].astype(int)
                    src_[i] = r[:, 4:6]
                    steps[i] = r[:, 6]
                found[direction] = (frame, src_, steps)
            res.append(found)
        parts.append((ys, xs, res))
    return parts


def plate_guide(spec, h, w):
    """'<png>@x0,y0' -> full-frame (Y, U, V) guide arrays (NaN outside the plate)."""
    from plate_fix import rgb_to_yuv
    png, at = spec.rsplit("@", 1)
    x0, y0 = (int(v) for v in at.split(","))
    rgb = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(png).convert("RGB")).astype(np.float64)
    py, pu, pv = rgb_to_yuv(rgb)
    ph, pw = py.shape
    gy = np.full((h, w), np.nan)
    gy[y0:y0 + ph, x0:x0 + pw] = py
    pool = lambda a: a.reshape(ph // 2, 2, pw // 2, 2).mean(axis=(1, 3))
    gu, gv = np.full((h // 2, w // 2), np.nan), np.full((h // 2, w // 2), np.nan)
    gu[y0 // 2:(y0 + ph) // 2, x0 // 2:(x0 + pw) // 2] = pool(pu)
    gv[y0 // 2:(y0 + ph) // 2, x0 // 2:(x0 + pw) // 2] = pool(pv)
    return gy, gu, gv


def main(src, dst, boxes, guide="clone", cache=None):
    """guide: clone (same-frame texture at a fixed offset), harmonic (smooth
    membrane, no texture) or plate:<png>@x0,y0 (a hand-inpainted still of the
    region, used as the gradient guide in every frame). Propagated pixels are
    used in all modes. cache: .npz to store/reuse the propagation result."""
    w, h, fps = probe(src)
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

    if cache and __import__("os").path.exists(cache):
        (ys_y, xs_y, prop_y), (ys_c, xs_c, prop_c) = load_prop(cache)
    else:
        # flow on half-res luma, completed inside the (half-res) hole
        lum_half = Y.reshape(n, h // 2, 2, w // 2, 2).mean(axis=(2, 4))
        fwd, bwd = raft_flows(lum_half)
        pf = Poisson(ndi.binary_dilation(hole_c, iterations=3))
        for fl in (fwd, bwd):
            for i in range(len(fl)):
                for c in range(2):
                    fl[i, c] = pf.fill(fl[i, c])
        ys_y, xs_y, prop_y = propagate(hole, fwd, bwd, 2.0)
        ys_c, xs_c, prop_c = propagate(hole_c, fwd, bwd, 1.0)
        del fwd, bwd
        if cache:
            save_prop(cache, [(ys_y, xs_y, prop_y), (ys_c, xs_c, prop_c)])

    off = (0, 0)
    if guide == "clone":
        # one fixed clone offset for never-seen pixels, chosen on the temporal median
        off = best_offset_color(np.median(Y[::4], axis=0), np.median(U[::4], axis=0),
                                np.median(V[::4], axis=0), hole)
    off_c = (off[0] // 2, off[1] // 2)
    plate = plate_guide(guide[6:], h, w) if guide.startswith("plate:") else None
    py, pc = Poisson(hole), Poisson(hole_c)

    def guide_plane(seq, ys, xs, prop, t, offset, shape, plate_plane, solver):
        """Guide image for frame t: propagated values where found; elsewhere
        same-frame clone texture, the plate, or (harmonic) a smooth membrane."""
        vals, got = guided_values(seq, ys, xs, prop, t, offset, shape)
        if plate_plane is not None:
            pv = plate_plane[ys, xs]
            use = ~got & ~np.isnan(pv)
            vals[use] = pv[use]
        g = seq[t].astype(np.float64).copy()
        g[ys, xs] = vals
        if guide == "harmonic" and (~got).any():
            if (~got).all():
                g = solver.fill(g)                     # membrane over the whole hole
            else:
                unk = np.zeros(g.shape, bool)
                unk[ys[~got], xs[~got]] = True
                g = Poisson(unk).fill(g)
        return g, got

    found_frac = []
    out_y, out_u, out_v = Y.copy(), U.copy(), V.copy()
    for t in range(n):
        g, got = guide_plane(Y, ys_y, xs_y, prop_y, t, off, hole.shape,
                             None if plate is None else plate[0], py)
        out_y[t] = np.clip(np.rint(py.fill(Y[t], g)), 0, 255).astype(np.uint8)
        found_frac.append(float(got.mean()))
        for k, (src_seq, dst_seq) in enumerate(((U, out_u), (V, out_v))):
            gc, _ = guide_plane(src_seq, ys_c, xs_c, prop_c, t, off_c, hole_c.shape,
                                None if plate is None else plate[1 + k], pc)
            dst_seq[t] = np.clip(np.rint(pc.fill(src_seq[t], gc)), 0, 255).astype(np.uint8)

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                            "-s", f"{w}x{h}", "-r", rate, "-i", "-", "-i", src,
                            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
                            "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy",
                            "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    for t in range(n):
        enc.stdin.write(out_y[t].tobytes() + out_u[t].tobytes() + out_v[t].tobytes())
    enc.stdin.close()
    rc = enc.wait()
    print(json.dumps({"src": src, "status": "ok" if rc == 0 else f"encode-failed:{rc}", "frames": n,
                      "guide": guide, "clone_offset": list(off),
                      "propagated_mean": round(float(np.mean(found_frac)), 3),
                      "propagated_min": round(float(np.min(found_frac)), 3)}, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
