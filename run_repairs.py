# -*- coding: utf-8 -*-
"""Batch-run repair_delogo.py for every video in a delogo box list.

Usage: python run_repairs.py <boxes.tsv> <video_dir> <out_root> <cache_dir> <mode> [<mode> ...]
       mode: harmonic | clone | plate:<plate_dir> | mplate:<plate_dir>
             (plate_dir holds <id>.json with {"video": name, "x0":.., "y0":..} and
             <id>_fixed.png; videos without one are skipped. mplate runs
             mplate_fix.py, the others repair_delogo.py)
Outputs go to <out_root>/<mode name>/<video>. Two videos run at a time; the
first repair_delogo.py run for a video computes and caches the propagation.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def boxes_list(path):
    rows = []
    for line in open(path, encoding="utf-8-sig"):
        if line.strip() and not line.startswith("#"):
            name, boxes = line.rstrip("\r\n").split("\t")[:2]
            rows.append((name, boxes))
    return rows


def one(name, boxes, video_dir, out_root, cache_dir, modes):
    stem = os.path.splitext(name)[0]
    cache = os.path.join(cache_dir, stem + ".npz")
    log = []
    for mode in modes:
        kind = mode.split(":")[0]
        if kind in ("plate", "mplate"):
            pdir = mode[len(kind) + 1:]
            m = next((json.load(open(os.path.join(pdir, j), encoding="utf-8")) | {"_id": j[:-5]}
                      for j in sorted(os.listdir(pdir)) if j.endswith(".json")
                      and json.load(open(os.path.join(pdir, j), encoding="utf-8")).get("video") == name), None)
            png = os.path.join(pdir, m["_id"] + "_fixed.png") if m else ""
            if not os.path.exists(png):
                log.append({"mode": kind, "status": "no-plate"})
                continue
            spec, sub = f"{png}@{m['x0']},{m['y0']}", kind
        else:
            spec, sub = mode, mode
        out_dir = os.path.join(out_root, sub)
        os.makedirs(out_dir, exist_ok=True)
        if kind == "mplate":
            cmd = ["mplate_fix.py", os.path.join(video_dir, name), os.path.join(out_dir, name), boxes, spec]
        else:
            guide = f"plate:{spec}" if kind == "plate" else spec
            cmd = ["repair_delogo.py", os.path.join(video_dir, name), os.path.join(out_dir, name), boxes,
                   guide, cache]
        r = subprocess.run([sys.executable, os.path.join(HERE, cmd[0])] + cmd[1:],
                           capture_output=True, text=True, encoding="utf-8")
        lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
        log.append(json.loads(lines[-1]) if lines else {"mode": sub, "status": f"crash:{r.returncode}",
                                                         "err": r.stderr[-300:]})
    return name, log


def main(boxes_tsv, video_dir, out_root, cache_dir, *modes):
    """Env RR_ONLY=4,7 limits the run to those row numbers of boxes.tsv."""
    os.makedirs(cache_dir, exist_ok=True)
    only = {int(v) for v in os.environ.get("RR_ONLY", "").split(",") if v.strip()}
    rows = [(k, n, b) for k, (n, b) in enumerate(boxes_list(boxes_tsv)) if not only or k in only]
    with ThreadPoolExecutor(2) as pool:
        futs = [(k, pool.submit(one, n, b, video_dir, out_root, cache_dir, modes)) for k, n, b in rows]
        for k, f in futs:
            name, log = f.result()
            print(json.dumps({"k": k, "name": name, "runs": [
                {key: l[key] for key in ("guide", "mode", "status", "propagated_mean", "q_median", "q_min",
                                         "low_weight_frames", "shift_max", "shift_p90", "err") if key in l}
                for l in log]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
