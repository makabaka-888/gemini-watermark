# -*- coding: utf-8 -*-
"""Compare GeminiWatermarkTool cleanup modes on residual (baked-in) stars.

Usage: python denoise_variants.py <in_dir> <work_dir> <sheet.png> name:x:y:s ...
Each sample row: input crop, then one crop per cleanup variant.
"""
import os
import subprocess
import sys

from PIL import Image, ImageDraw

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "GeminiWatermarkTool.exe")
VARIANTS = {
    "off": ["--denoise", "off"],
    "telea": ["--denoise", "telea", "--radius", "10", "--strength", "300"],
    "ns": ["--denoise", "ns", "--radius", "10", "--strength", "300"],
    "soft": ["--denoise", "soft", "--radius", "10", "--strength", "300"],
    "ai": ["--denoise", "ai", "--sigma", "120", "--strength", "300"],
}
T = 180


def main(in_dir, work, sheet_path, *samples):
    os.makedirs(work, exist_ok=True)
    rows = []
    for spec in samples:
        name, x, y, s = spec.rsplit(":", 3)
        x, y, s = int(x), int(y), int(s)
        src = os.path.join(in_dir, name)
        box = (x - s, y - s, x + 2 * s, y + 2 * s)
        crops = [Image.open(src).convert("RGB").crop(box).resize((T, T))]
        for tag, extra in VARIANTS.items():
            out = os.path.join(work, f"{tag}_{name}")
            subprocess.run([EXE, "--no-banner", "-q", "-i", src, "-o", out, "--force",
                            "--region", f"{x},{y},{s},{s}", *extra], check=True)
            crops.append(Image.open(out).convert("RGB").crop(box).resize((T, T)))
        rows.append(crops)
    labels = ["input"] + list(VARIANTS)
    sheet = Image.new("RGB", (len(labels) * (T + 4), len(rows) * (T + 4) + 16), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    for c, lab in enumerate(labels):
        d.text((c * (T + 4) + 4, 2), lab, fill=(255, 255, 0))
    for r, crops in enumerate(rows):
        for c, im in enumerate(crops):
            sheet.paste(im, (c * (T + 4), 16 + r * (T + 4)))
    sheet.save(sheet_path)
    print(sheet_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
