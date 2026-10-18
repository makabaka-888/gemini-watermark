# -*- coding: utf-8 -*-
"""Recover the Gemini sparkle alpha maps that GeminiWatermarkTool embeds.

The tool removes a watermark by inverting  out = a*255 + (1-a)*bg.  Feeding it
a flat image of value G and forcing removal at a known region gives
out = (G - 255a) / (1 - a), so  a = (G - out) / (255 - out).

Writes templates/alpha{48,96}_{v1,v2}.npy (float32, 0..~0.52).
"""
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "bin", "GeminiWatermarkTool.exe")
G = 250


def recover(size, legacy):
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "flat.png")
        out = os.path.join(tmp, "out.png")
        Image.fromarray(np.full((400, 400, 3), G, np.uint8)).save(src)
        args = [EXE, "--no-banner", "-q", "-i", src, "-o", out, "--force",
                "--region", f"100,100,{size},{size}", "--denoise", "off"]
        args.append("--legacy" if legacy else "--no-legacy")
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        o = np.asarray(Image.open(out).convert("L"), np.float32)[100:100 + size, 100:100 + size]
    return np.clip((G - o) / np.maximum(255 - o, 1e-3), 0, 1).astype(np.float32)


def main():
    os.makedirs(os.path.join(HERE, "templates"), exist_ok=True)
    for size in (48, 96):
        for tag, legacy in (("v1", True), ("v2", False)):
            a = recover(size, legacy)
            np.save(os.path.join(HERE, "templates", f"alpha{size}_{tag}.npy"), a)
            print(f"alpha{size}_{tag}: max={a.max():.3f} mean={a.mean():.4f} "
                  f"nonzero={(a > 0.02).mean():.2%}")


if __name__ == "__main__":
    main()
