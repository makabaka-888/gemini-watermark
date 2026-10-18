# -*- coding: utf-8 -*-
"""Run gwvideo.py over every .mp4 in a folder, two at a time.

Usage: python run_videos.py <video_dir> <out_dir> <results.jsonl> [name ...]
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def one(src, dst):
    r = subprocess.run([sys.executable, os.path.join(HERE, "gwvideo.py"), src, dst],
                       capture_output=True, text=True, encoding="utf-8")
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if lines:
        return lines[-1]
    return json.dumps({"src": src, "status": f"crash:{r.returncode}", "err": r.stderr[-400:]})


def main(video_dir, out_dir, results, *only):
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(video_dir)
                   if n.lower().endswith(".mp4") and (not only or n in only))
    with ThreadPoolExecutor(2) as pool, open(results, "w", encoding="utf-8") as fh:
        futs = [pool.submit(one, os.path.join(video_dir, n), os.path.join(out_dir, n)) for n in names]
        for k, f in enumerate(futs):
            line = f.result()
            fh.write(line + "\n")
            fh.flush()
            print(f"[{k + 1}/{len(names)}] {json.loads(line)['status']}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
