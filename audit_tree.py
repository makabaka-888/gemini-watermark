# -*- coding: utf-8 -*-
"""Find which images and videos under a folder tree still carry a Gemini/Veo mark.

Usage: python audit_tree.py <out.jsonl> <root> [<root> ...]

Images: gwclean.detect (sparkle near the canonical bottom-right spot).
Videos: video_audit.audit (Veo grid sparkle opacity, in-scene sparkles, static
"Veo" text pixels). Read-only; one JSON line per file with a "marked" verdict.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

import gwclean
import video_audit

IMAGES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEOS = {".mp4", ".mov", ".webm"}
GRID_MIN_OPACITY = 0.15     # gwvideo.MIN_OPACITY: below this nothing is removed
TEXT_MIN_PX = 40            # bright static pixels in the text box ("Veo" mark)


def image_verdict(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB")).astype(np.float32)
    dets = gwclean.summarize(gwclean.detect(rgb))
    return {"kind": "image", "marked": bool(dets),
            "dets": [[int(x), int(y), int(s), round(float(sc), 2)] for x, y, s, sc in dets[:3]]}


def video_verdict(path):
    a = video_audit.audit(path)
    grid = max(g["opacity"] for g in a["grid"])
    marked = grid >= GRID_MIN_OPACITY or bool(a["extras"]) or a["text_px"] >= TEXT_MIN_PX
    return {"kind": "video", "marked": marked, "grid_opacity": grid, "extras": len(a["extras"]),
            "text_px": a["text_px"], "frames": a["frames"], "w": a["w"], "h": a["h"], "fps": a["fps"]}


def verdict(p):
    try:
        r = image_verdict(p) if os.path.splitext(p)[1].lower() in IMAGES else video_verdict(p)
    except Exception as e:  # report and go on
        r = {"kind": "error", "marked": None, "error": str(e)[:200]}
    r["path"] = p
    return r


def main(out_path, *roots):
    """Appends to out_path and skips files already in it, so a stopped scan resumes."""
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["path"] for l in open(out_path, encoding="utf-8") if l.strip()}
    kinds = {"images": IMAGES, "videos": VIDEOS}.get(os.environ.get("AUDIT_KIND", ""), IMAGES | VIDEOS)
    todo = [os.path.join(d, n) for root in roots for d, _, files in os.walk(root) for n in sorted(files)
            if os.path.splitext(n)[1].lower() in kinds and os.path.join(d, n) not in done]
    # memory, not CPU, is the limit: 12 workers on a 16 GB machine got one killed
    workers = int(os.environ.get("AUDIT_WORKERS", "4"))
    with open(out_path, "a", encoding="utf-8") as fh, ProcessPoolExecutor(workers) as pool:
        for r in pool.map(verdict, todo, chunksize=2):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()


if __name__ == "__main__":
    main(*sys.argv[1:])
