# -*- coding: utf-8 -*-
"""Find and erase Gemini sparkle watermarks that GeminiWatermarkTool cannot.

GeminiWatermarkTool inverts the alpha blend of the *current* overlay at its
canonical spot, which is lossless. Images that were edited in Gemini several
times also carry older sparkles that the model re-rendered into the picture
(not an exact alpha blend), sometimes at shifted positions or scales, and a
wrong inversion leaves a dark star-shaped ghost. This module locates every
bright sparkle and dark ghost in the bottom-right area by multi-scale
normalised cross-correlation against the tool's own alpha map, then fills
each one by cloning the best-matching nearby texture and Poisson-blending it
into the surroundings.

CLI (images):
  python gwclean.py <src_dir> <pass1_dir> <pass1_log> <final_dir> <report.tsv> [name ...]
    src_dir    original images
    pass1_dir  GeminiWatermarkTool output (used as the base when present)
    pass1_log  its `-v` log, for where each reverse-alpha removal happened
    final_dir  where cleaned images are written
    name ...   optional subset of file names
  env GWC_MODE=clone skips the fitted-inversion step (clone-fill only); the
  two modes win on different images, so both are worth producing and comparing.
"""
import os
import sys

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = np.load(os.path.join(HERE, "templates", "alpha96_v2.npy"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

REGION_X, REGION_Y = 0.60, 0.60    # verification sheets show this corner
# Sparkle size scales with the long side: 96 px at 2816. Real sparkles,
# including doubled ones from repeated edits, sit within ~0.9 sizes of the
# canonical centre (1.1667 sizes in from the corner); hits farther out were
# UI icons and texture in a sweep of all 198 test images.
SIZE_LO, SIZE_HI, SCALE_STEP = 0.6, 1.6, 1.07
MAX_DIST = 0.9
THR_BRIGHT, THR_DARK = 0.60, 0.50


def _template(size):
    t = torch.from_numpy(ALPHA)[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", antialias=True,
                         align_corners=False)[0, 0]


def luminance(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def ncc_maps(gray, sizes):
    """Yield (size, ncc_map) with ncc_map indexed by template top-left."""
    img = torch.from_numpy(gray).to(DEV)[None, None]
    for s in sizes:
        t = _template(s).to(DEV)
        t = t - t.mean()
        tn = torch.sqrt((t * t).sum())
        num = F.conv2d(img, t[None, None])
        ones = torch.ones(1, 1, s, s, device=DEV)
        s1 = F.conv2d(img, ones)
        s2 = F.conv2d(img * img, ones)
        var = (s2 - s1 * s1 / (s * s)).clamp_min(1e-6)
        ncc = num / (torch.sqrt(var) * tn)
        # flat patches give meaningless correlation
        ncc[var / (s * s) < 4.0] = 0
        yield s, ncc[0, 0]


def expected_size(w, h):
    return 96.0 * max(w, h) / 2816.0


def _peaks(rgb, box, sizes, sign, thr):
    """Local NCC maxima of the given polarity inside box=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    gray = luminance(rgb[y0:y1, x0:x1]).astype(np.float32)
    out = []
    for size, ncc in ncc_maps(gray, sizes):
        if ncc.shape[0] < 1 or ncc.shape[1] < 1:
            continue
        k = max(3, size // 2) | 1
        m = ncc * sign
        peak = (m == F.max_pool2d(m[None, None], k, 1, k // 2)[0, 0]) & (m >= thr)
        for yy, xx in torch.nonzero(peak).tolist():
            out.append((x0 + xx, y0 + yy, size, sign * float(m[yy, xx])))
    return out


def detect(rgb, ghost_hint=None):
    """Sparkles near the canonical spot, plus a dark ghost at ghost_hint.

    ghost_hint is (x, y, size) of an earlier reverse-alpha removal; a wrong
    removal leaves a dark sparkle there. Returns every accepted candidate
    [(x, y, size, score)] (overlapping hits are kept so their masks union).
    """
    h, w = rgb.shape[:2]
    e = expected_size(w, h)
    cx0, cy0 = w - 1.1667 * e, h - 1.1667 * e
    sizes, s = [], SIZE_LO * e
    while s <= SIZE_HI * e:
        sizes.append(max(8, int(round(s))))
        s *= SCALE_STEP
    sizes = sorted(set(sizes))
    reach = (MAX_DIST + SIZE_HI / 2) * e + 2
    box = (max(0, int(cx0 - reach)), max(0, int(cy0 - reach)), w, h)
    found = []
    for x, y, size, sc in _peaks(rgb, box, sizes, 1, THR_BRIGHT):
        if np.hypot(x + size / 2 - cx0, y + size / 2 - cy0) <= MAX_DIST * e:
            found.append((x, y, size, sc))
    if ghost_hint:
        gx, gy, gs = ghost_hint
        r = int(0.6 * gs) + 2
        gbox = (max(0, gx - r), max(0, gy - r), min(w, gx + gs + r), min(h, gy + gs + r))
        gsizes = sorted({max(8, int(round(gs * f))) for f in (0.85, 0.92, 1.0, 1.08, 1.17)})
        for x, y, size, sc in _peaks(rgb, gbox, gsizes, -1, THR_DARK):
            if np.hypot(x + size / 2 - gx - gs / 2, y + size / 2 - gy - gs / 2) <= 0.35 * gs:
                found.append((x, y, size, sc))
    return found


def summarize(dets):
    """Collapse overlapping candidates for reporting (strongest first)."""
    kept = []
    for x, y, size, sc in sorted(dets, key=lambda d: -abs(d[3])):
        cx, cy = x + size / 2, y + size / 2
        if any(np.hypot(cx - kx - ks / 2, cy - ky - ks / 2) < 0.3 * max(size, ks)
               for kx, ky, ks, _ in kept):
            continue
        kept.append((x, y, size, sc))
    return kept


A_UNIT = ALPHA / ALPHA.max()


def alpha_map(shape, ox, oy, cx, cy, s, k, sigma=0.0):
    """Sparkle alpha (peak k) on a grid whose (0,0) pixel sits at image (ox, oy).

    cx, cy, s are the sparkle centre and size in image pixels (sub-pixel ok);
    sigma softens the edge. k < 0 models a dark ghost: inverting with
    a' = -a/(1-a) re-applies a blend that an earlier removal took out wrongly.
    """
    py, px = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    u = (px + ox + 0.5 - (cx - s / 2)) * (96.0 / s) - 0.5
    v = (py + oy + 0.5 - (cy - s / 2)) * (96.0 / s) - 0.5
    a = ndi.map_coordinates(A_UNIT, [v, u], order=1, cval=0.0)
    if sigma > 0:
        a = ndi.gaussian_filter(a, sigma)
    return k * a


def ncc_at(win, a):
    m = np.abs(a) > 0.02 * np.abs(a).max() if np.abs(a).max() > 0 else np.zeros(a.shape, bool)
    if m.sum() < 10:
        return 0.0
    x, t = win[m] - win[m].mean(), a[m] - a[m].mean()
    return float((x * t).sum() / (np.sqrt((x * x).sum() * (t * t).sum()) + 1e-9))


def _window(gray, cx, cy, s, pad_frac=0.6):
    pad = int(pad_frac * s) + 4
    x0, y0 = max(0, int(cx - s / 2) - pad), max(0, int(cy - s / 2) - pad)
    x1, y1 = min(gray.shape[1], int(cx + s / 2) + pad + 1), min(gray.shape[0], int(cy + s / 2) + pad + 1)
    return x0, y0, x1, y1


def locked_ncc(gray, cand, jitter=2):
    """Best |NCC| with the sparkle at a known spot (+-jitter px, +-5% size)."""
    x, y, s = cand
    best = 0.0
    for f in (0.95, 1.0, 1.05):
        ss = s * f
        for dx in range(-jitter, jitter + 1):
            for dy in range(-jitter, jitter + 1):
                cx, cy = x + s / 2 + dx, y + s / 2 + dy
                x0, y0, x1, y1 = _window(gray, cx, cy, ss, 0.3)
                v = ncc_at(gray[y0:y1, x0:x1], alpha_map((y1 - y0, x1 - x0), x0, y0, cx, cy, ss, 1.0))
                if abs(v) > abs(best):
                    best = v
    return best


def boundary_k(win, wx, wy, cx, cy, s, sigma=0.0, white=255.0,
               ks=np.arange(-0.6, 0.9, 0.005)):
    """Signed opacity that makes the un-blended window continuous across the
    sparkle edge (median inside-minus-outside step ~ 0). Texture only adds
    zero-mean noise to that median; total variation instead rewards the
    contrast change that inversion causes and drifts to a wrong opacity."""
    unit = alpha_map(win.shape, wx, wy, cx, cy, s, 1.0, sigma)
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
    for k in ks:
        a = k * unit
        r = (win - white * a) / (1 - a)
        d = np.median((ndi.gaussian_filter(r * inner, 2.0) / np.maximum(gi, 1e-6)
                       - ndi.gaussian_filter(r * outer, 2.0) / np.maximum(go, 1e-6))[sel])
        if best_d is None or abs(d) < abs(best_d):
            best_k, best_d = float(k), d
    return best_k


def fit_residual(gray, cand, sign=1.0):
    """Fit centre/size (by correlation), then signed opacity (boundary step)
    and edge softness (smallest leftover correlation) of a sparkle at a known
    spot. Returns ([cx, cy, s, k, sigma], leftover NCC)."""
    x, y, s = cand
    cx0, cy0 = x + s / 2, y + s / 2
    x0, y0, x1, y1 = _window(gray, cx0, cy0, s * 1.2)
    win = gray[y0:y1, x0:x1]
    best = (-2.0, cx0, cy0, float(s))
    for ds in np.arange(-0.08, 0.081, 0.02):
        for dx in np.arange(-2, 2.01, 0.5):
            for dy in np.arange(-2, 2.01, 0.5):
                v = sign * ncc_at(win, alpha_map(win.shape, x0, y0, cx0 + dx, cy0 + dy, s * (1 + ds), 1.0))
                if v > best[0]:
                    best = (v, cx0 + dx, cy0 + dy, s * (1 + ds))
    _, cx, cy, ss = best
    unit = alpha_map(win.shape, x0, y0, cx, cy, ss, 1.0)
    result = None
    for sg in (0.0, 0.6, 1.2):
        k = boundary_k(win, x0, y0, cx, cy, ss, sg)
        a = alpha_map(win.shape, x0, y0, cx, cy, ss, k, sg)
        left = ncc_at((win - 255.0 * a) / (1 - a), unit)
        if result is None or abs(left) < abs(result[1]):
            result = ([cx, cy, ss, k, sg], left)
    return result


def apply_inversion(rgb, p, band_sigma=1.2):
    """Invert the fitted blend on all channels, then refill the 1-2 px edge
    band (where re-rendering and compression differ from the template)."""
    cx, cy, s, k, sg = p
    x0, y0, x1, y1 = _window(rgb[..., 0], cx, cy, s)
    a = alpha_map((y1 - y0, x1 - x0), x0, y0, cx, cy, s, k, sg)
    win = rgb[y0:y1, x0:x1].astype(np.float64)
    win = (win - 255.0 * a[..., None]) / (1 - a[..., None])
    # a narrow band: widening it with sigma ate into text under the sparkle
    star = np.abs(a) > 0.5 * np.abs(a).max()
    band = ndi.binary_dilation(star, iterations=2) & ~ndi.binary_erosion(star, iterations=1)
    keep = (~band).astype(np.float64)
    den = ndi.gaussian_filter(keep, band_sigma)
    ok = band & (den > 1e-3)
    for c in range(win.shape[2]):
        num = ndi.gaussian_filter(win[..., c] * keep, band_sigma)
        win[..., c][ok] = num[ok] / den[ok]
    out = rgb.astype(np.float64).copy()
    out[y0:y1, x0:x1] = win
    return np.clip(out, 0, 255)


def star_mask(shape, dets):
    """Dilated union of the detected sparkle shapes."""
    mask = np.zeros(shape[:2], bool)
    for x, y, size, _ in dets:
        a = _template(size).numpy() > 0.02 * ALPHA.max()
        pad = max(3, int(round(0.12 * size)))
        a = ndi.binary_dilation(np.pad(a, pad), iterations=pad)
        ys, xs = slice(y - pad, y - pad + a.shape[0]), slice(x - pad, x - pad + a.shape[1])
        sub = mask[max(ys.start, 0):ys.stop, max(xs.start, 0):xs.stop]
        sub |= a[max(0, -ys.start):max(0, -ys.start) + sub.shape[0],
                 max(0, -xs.start):max(0, -xs.start) + sub.shape[1]]
    return mask


def _best_offset(img, hole, ring, forbid, reach):
    """Offset (dy, dx) whose ring best matches the target ring."""
    h, w = hole.shape
    ys, xs = np.nonzero(hole | ring)
    ry, rx = np.nonzero(ring)
    target = img[ry, rx]
    best, best_off = np.inf, None

    def cost(dy, dx):
        if ys.min() + dy < 0 or xs.min() + dx < 0 or ys.max() + dy >= h or xs.max() + dx >= w:
            return np.inf
        if forbid[ys + dy, xs + dx].any():
            return np.inf
        return float(((img[ry + dy, rx + dx] - target) ** 2).mean())

    step = max(2, reach // 24)
    for dy in range(-reach, reach + 1, step):
        for dx in range(-reach, reach + 1, step):
            c = cost(dy, dx)
            if c < best:
                best, best_off = c, (dy, dx)
    if best_off is None:
        return None
    by, bx = best_off
    for dy in range(by - step, by + step + 1):
        for dx in range(bx - step, bx + step + 1):
            c = cost(dy, dx)
            if c < best:
                best, best_off = c, (dy, dx)
    return best_off


def _poisson(target, source, hole):
    """Seamless clone: Laplacian of source inside hole, target on boundary."""
    idx = -np.ones(hole.shape, np.int64)
    ys, xs = np.nonzero(hole)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)
    rows, cols, vals = [], [], []
    b = np.zeros((n, target.shape[2]), np.float64)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        b += source[ys, xs] - source[ny, nx]
        inside = idx[ny, nx] >= 0
        rows.append(np.arange(n)[inside])
        cols.append(idx[ny, nx][inside])
        vals.append(-np.ones(inside.sum()))
        b[~inside] += target[ny[~inside], nx[~inside]]
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(np.full(n, 4.0))
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    solve = spla.factorized(A)
    out = target.copy()
    for c in range(target.shape[2]):
        out[ys, xs, c] = solve(b[:, c])
    return out


def fill(rgb, mask, no_source=None):
    """Fill every connected masked blob from nearby texture.

    no_source: optional bool mask of pixels that must not be cloned from
    (e.g. a HUD element next to the sparkle).
    """
    out = rgb.astype(np.float64)
    labels, n = ndi.label(mask)
    forbid_all = ndi.binary_dilation(mask, iterations=4)
    if no_source is not None:
        forbid_all |= no_source
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        reach = int(2.5 * max(bh, bw))
        y0, y1 = max(0, sl[0].start - reach - 8), min(mask.shape[0], sl[0].stop + reach + 8)
        x0, x1 = max(0, sl[1].start - reach - 8), min(mask.shape[1], sl[1].stop + reach + 8)
        # 1px edge padding so holes touching the image border still have
        # neighbours for the Poisson stencil
        win = np.pad(out[y0:y1, x0:x1], ((1, 1), (1, 1), (0, 0)), mode="edge")
        hole = np.pad(labels[y0:y1, x0:x1] == i, 1)
        forbid = np.pad(forbid_all[y0:y1, x0:x1], 1, constant_values=True)
        ringw = max(4, int(0.15 * max(bh, bw)))
        ring = ndi.binary_dilation(hole, iterations=ringw) & ~hole
        off = _best_offset(luminance(win), hole, ring, forbid, reach)
        if off is None:
            continue
        dy, dx = off
        src = np.roll(win, (-dy, -dx), axis=(0, 1))
        out[y0:y1, x0:x1] = _poisson(win, src, hole)[1:-1, 1:-1]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def relaxed_canonical(rgb):
    """Best sparkle within 0.5 sizes of the canonical spot, NCC >= 0.35;
    for images the tool skipped and the strict search found nothing."""
    h, w = rgb.shape[:2]
    e = expected_size(w, h)
    cx0, cy0 = w - 1.1667 * e, h - 1.1667 * e
    sizes = sorted({max(8, int(round(e * f))) for f in np.arange(0.8, 1.26, 0.05)})
    r = int(0.5 * e + 0.7 * e) + 2
    box = (max(0, int(cx0 - r)), max(0, int(cy0 - r)), min(w, int(cx0 + r)), min(h, int(cy0 + r)))
    peaks = [p for p in _peaks(rgb, box, sizes, 1, 0.35)
             if np.hypot(p[0] + p[2] / 2 - cx0, p[1] + p[2] / 2 - cy0) <= 0.5 * e]
    return max(peaks, key=lambda p: p[3]) if peaks else None


def _same(a, b):
    return np.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[2] / 2 - b[1] - b[2] / 2) \
        < 0.3 * max(a[2], b[2])


def clean_array(rgb, removal=None, rects=(), rounds=2, invert_first=True):
    """Remove every sparkle near the canonical spot.

    1. Candidates: the reverse-alpha removal spot of GeminiWatermarkTool, the
       strict search hits (bright sparkles; a dark ghost at that spot), and
       for untouched images a relaxed search at the canonical spot.
    2. Where a candidate still correlates with the sparkle, fit and invert a
       blend with signed opacity: this restores the content under a faint
       left-over sparkle (and undoes an over-removal) instead of painting
       over it.
    3. Whatever still shows, plus any manual rectangles, is clone-filled.
    Returns (image, log entries).
    """
    out = rgb.astype(np.float64)
    log = []
    cands = [tuple(removal)] if removal else []
    dets = summarize(detect(out.astype(np.float32), removal))
    for x, y, s, sc in dets:
        if not any(_same((x, y, s), c) for c in cands):
            cands.append((x, y, s))
    if removal is None and not dets:
        best = relaxed_canonical(out.astype(np.float32))
        if best:
            cands.append(best[:3])
    for c in (cands if invert_first else []):
        g = luminance(out)
        v = locked_ncc(g, c)
        if not (v >= 0.2 or v <= -0.3):
            continue
        p, _ = fit_residual(g, c, 1.0 if v > 0 else -1.0)
        if abs(p[3]) < 0.03:
            continue
        trial = apply_inversion(out, p)
        v2 = locked_ncc(luminance(trial), c)
        if abs(v2) < min(abs(v), 0.25):
            out = trial
            log.append(f"inv{c}k{p[3]:+.2f}s{p[4]}:{v:+.2f}>{v2:+.2f}")
    for r in range(rounds):
        g = luminance(out)
        left = [c + (1.0,) for c in cands if abs(locked_ncc(g, c)) >= 0.3]
        more = summarize(detect(out.astype(np.float32))) if r == 0 else []
        mask = star_mask(out.shape, left + more)
        if r == 0:
            for x, y, ww, hh in rects:
                mask[max(0, y):y + hh, max(0, x):x + ww] = True
        if not mask.any():
            break
        log.append(f"clone{[c[:3] for c in left + more]}{'+rects' if r == 0 and rects else ''}")
        out = fill(out, mask).astype(np.float64)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), log


def save_like(arr, src_path, dst_path):
    im = Image.fromarray(arr)
    if os.path.splitext(dst_path)[1].lower() in (".jpg", ".jpeg"):
        im.save(dst_path, quality=95, subsampling=0)
    else:
        im.save(dst_path)


def read_overrides(path):
    """name -> [(x, y, w, h)] rectangles to clone-fill (manual fixes)."""
    rects = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                name, box = line.rstrip("\n").split("\t")[:2]
                rects.setdefault(name, []).append(tuple(int(v) for v in box.split(",")))
    return rects


def main(src_dir, pass1_dir, pass1_log, final_dir, report, *only):
    from montage_images import parse_log
    removals = {n: i["pos"] for n, i in parse_log(pass1_log).items() if i["saved"]}
    overrides = read_overrides(os.path.join(HERE, "overrides.tsv"))
    os.makedirs(final_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(src_dir)
                   if os.path.splitext(n)[1].lower() in (".png", ".jpg", ".jpeg", ".webp")
                   and (not only or n in only))
    with open(report, "w", encoding="utf-8") as rep:
        rep.write("name\tbase\tactions\n")
        for k, n in enumerate(names):
            base_path = os.path.join(pass1_dir, n)
            base = "pass1" if os.path.exists(base_path) else "orig"
            if base == "orig":
                base_path = os.path.join(src_dir, n)
            rgb = np.asarray(Image.open(base_path).convert("RGB"))
            out, log = clean_array(rgb, removals.get(n) if base == "pass1" else None,
                                   overrides.get(n, ()),
                                   invert_first=os.environ.get("GWC_MODE", "invert") != "clone")
            save_like(out, base_path, os.path.join(final_dir, n))
            rep.write(f"{n}\t{base}\t{' '.join(log)}\n")
            rep.flush()
            print(f"[{k + 1}/{len(names)}] {base} {len(log)} actions", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
