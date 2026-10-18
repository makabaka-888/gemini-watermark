# gemini-watermark

Remove the visible Gemini / Veo sparkle watermark from images and videos,
including the cases that a plain reverse-alpha tool leaves behind: older
sparkles that Gemini re-rendered into the picture, sparkles that moved or
changed size, the "Veo" text mark, and smeared rectangles left by an earlier
`ffmpeg delogo` pass.

It builds on [GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool)
by AllenK (Kwyshell), which does the exact, lossless removal of the current
overlay. This repository handles what is left after that.

> **Scope.** This only removes the *visible* overlay. It does not touch, detect
> or defeat Google's invisible SynthID watermark, and it does not change file
> metadata. Use it on content you have the right to edit, and follow the terms
> of the service that produced it and of any platform you publish to. Where AI
> disclosure is required, removing the visible mark does not remove that duty.

## Contents

- [How the watermark looks](#how-the-watermark-looks)
- [Requirements](#requirements)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Pipeline reference](#pipeline-reference)
- [Verification tools](#verification-tools)
- [Known limits](#known-limits)
- [License](#license)

## How the watermark looks

- **Current overlay**: a white sparkle alpha-blended at a fixed spot. Images:
  96 px on a 2816-px long side, centred 1.1667 sizes in from the bottom-right
  corner. Veo video: 48 px per 1280 px of width, centred 2 S (current) or
  2.5 S (older clips) in from the corner; peak opacity about 0.375.
- **Older sparkles**: images edited several times in Gemini carry earlier
  sparkles that the model re-rendered into the picture (not exact blends),
  sometimes shifted, sometimes merged with HUD digits or turned into an
  emblem. Image-to-video clips carry the source image's sparkle (and
  occasionally a third, smaller one at about 0.64 S, 0.70 S from the corner).
- Positions differ per file, so everything is located per file.

## Requirements

- Python 3.10+ (developed on 3.12)
- `numpy`, `scipy`, `Pillow`, `torch` (CPU works; CUDA is used when present)
- `torchvision`, only for `repair_delogo.py` (RAFT optical flow)
- [`ffmpeg` and `ffprobe`](https://ffmpeg.org/) on `PATH`, for anything video
- `GeminiWatermarkTool.exe` (Windows build) in `bin/`, for the first image
  pass and for `extract_alpha.py`

The video tools and `gwclean.py` are plain Python and run on any OS. Only the
GeminiWatermarkTool pass needs its Windows binary (or a build of it for your
platform).

## Setup

```bash
git clone https://github.com/makabaka-888/gemini-watermark.git
cd gemini-watermark
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download GeminiWatermarkTool v0.3.2 from its
[releases page](https://github.com/allenk/GeminiWatermarkTool/releases),
check the checksum published there, and put `GeminiWatermarkTool.exe` in
`bin/` (the folder is git-ignored).

`templates/` already holds the alpha maps the tools match against. To
regenerate them from your copy of the tool, run `python extract_alpha.py`.

## Quick start

Images:

```bash
mkdir -p logs sheets
bin/GeminiWatermarkTool.exe -v -i in/ -o pass1/ > logs/pass1.log
python gwclean.py in/ pass1/ logs/pass1.log out/ logs/report.tsv
python verify_sheet.py logs/report.tsv in/ out/ sheets/images   # look at every sheet
```

Videos:

```bash
python run_videos.py in/ out/ logs/videos.jsonl
python video_compare.py logs/videos.jsonl sheets/ videos   # before/after sheets
```

Every script prints its full usage in its module docstring
(`python -c "import gwclean; help(gwclean)"`, or open the file).

## Pipeline reference

### Images

1. `bin/GeminiWatermarkTool.exe -v -i <dir> -o <out>`: exact reverse-alpha
   removal of the current overlay. Keep its `-v` log; step 2 reads it.
2. `gwclean.py`: finds what is left near the canonical spot (multi-scale
   NCC against the tool's own alpha map in `templates/`), fits and inverts
   faint leftovers (signed opacity from the boundary step), and clone-fills
   the rest (best-offset clone plus Poisson blend). `GWC_MODE=clone` skips
   the inversion; the two modes win on different images, so produce both and
   compare.
3. Manual rectangles go in `overrides.tsv` in the repository root (format in
   [`examples/overrides.tsv`](examples/overrides.tsv)).

### Videos

- `run_videos.py <dir> <out> <results.jsonl>` runs `gwvideo.py` per clip:
  YUV-domain reverse alpha of the grid overlay (opacity from the boundary
  step on the temporal mean, edge band refilled per frame, noise gain
  damped), clone-fill for re-rendered sparkles, rescaled clips and
  `overrides_video.tsv` entries. Re-encodes with libx264 CRF 18 and copies
  the audio.
- `overrides_video.tsv` (format in
  [`examples/overrides_video.tsv`](examples/overrides_video.tsv)) adds a
  sparkle to fill, or with `name<TAB>skip:cx,cy` drops a false in-scene
  detection (scene text such as a kanji or "PHOTO MODE" that matched the
  template).

### Other static overlays

- `gwstatic.py <in> <out> x0,y0,x1,y1 [thr] [frame_fixes.tsv]` removes any
  static overlay learned from the clip itself (for example the "Veo" text
  mark): per-pixel opacity by least squares over all frames, luma
  un-blended, chroma interpolated. The optional `frame_fixes.tsv`
  clone-fills a moving sparkle in listed frames
  ([`examples/frame_fixes.tsv`](examples/frame_fixes.tsv)).

### Sparkles over HUD text or other static detail

- `plate_fix.py extract <video> x0,y0,w,h <plate.png>` saves the temporal
  median of a static region. Inpaint it by hand to `<plate_fixed.png>`, then
  `plate_fix.py apply ...` adds that edit to every frame. This keeps HUD text
  that a per-frame fill would smear. Only for regions where the scene does
  not move.

### Old `ffmpeg delogo` boxes

Blurry, streaky rectangles from an earlier delogo pass. List the boxes per
video in a TSV ([`examples/delogo_boxes.tsv`](examples/delogo_boxes.tsv)).

- `mplate_fix.py <in> <out> <boxes> <plate_fixed.png>@x0,y0 [max_shift]` is
  the recommended one. A hand-inpainted plate of the region (the median still
  from `plate_fix.py extract`, box painted over) is registered to every frame
  (translation, zero-mean NCC on the context around the box) and used as the
  Poisson guide, weighted by how well the context matched. The fill follows
  camera shake and pans, and fades to a smooth membrane where the scene does
  not match the plate (a cut, a limb crossing). `MPLATE_LOG_DIR` dumps the
  per-frame match and shift; `MPLATE_REG_ONLY=1` skips the encode.
- `repair_delogo.py` is the earlier approach: RAFT flow propagation plus a
  clone / harmonic / static-plate guide. Propagation pulls wrong content into
  the box during fast motion, and a static plate looks like a sticker when the
  camera moves. Kept for reference.
- `run_repairs.py <boxes.tsv> <video_dir> <out_root> <cache_dir> <mode>...`
  batches either (`mplate:<plate_dir>`, `plate:<plate_dir>`, `harmonic`,
  `clone`; env `RR_ONLY=4,7` limits the rows).

### Hard single images

`experiments/try_fix.py` (locate / invert / clone / rect / blob / zoom) is a
toolkit for fixing one image by hand. Template inversion misfits flat
(re-rendered) stars; those are better fixed with a flat-alpha inversion or a
guided clone along the lines that cross the star.

`experiments/` also holds the one-off probes used to calibrate thresholds.

## Verification tools

Look at every result. Detectors miss, and fills can go wrong.

- Images: `verify_sheet.py`, `zoom_sheet.py`, `pair_zoom.py`, `zoom_pair.py`,
  `compare_dirs.py`, `montage_images.py`.
- Videos: `video_compare.py` (temporal mean before/after plus frames),
  `compare_videos.py` (two versions side by side), `video_scan.py` (survey
  the sparkle across a folder).
- `video_audit.py <out.jsonl> <dir>...` measures what overlay is left in each
  video (opacity per grid spot, in-scene sparkles, bright text pixels)
  without changing anything. `audit_tree.py` does the same check over a whole
  folder tree of images and videos.
- `review_sheets.py <boxes.tsv> <old_dir> "label=dir|label=dir" <out_dir>`
  for delogo repairs: 12 frames of the old video and each repaired version,
  the frames around the largest jump inside the box, and motion numbers (a
  fill that stays still while its surroundings move is a sticker).
- `assemble.py choices.tsv <dest>` copies the chosen version of each file
  into a delivery folder.

## Known limits

- Overlays that went through a resize or transcode, and sparkles the model
  drew into text or a HUD, cannot be inverted exactly. The fill leaves a faint
  trace and cannot recreate text that was under the sparkle.
- Do not run it again on images that were already cleaned (by any tool). The
  detector fires on the faint dotted outline the old removal left, and a
  second pass stamps a new dark or grey star. In one test, 94 of 100
  "changed" re-processed images were this kind of damage.
- The invisible SynthID mark is untouched (see [Scope](#gemini-watermark)).

## License

Apache License 2.0, see [LICENSE](LICENSE). The alpha maps in `templates/`
are derived from GeminiWatermarkTool, which is MIT-licensed; see
[NOTICE](NOTICE).

Not affiliated with or endorsed by Google. "Gemini" and "Veo" are trademarks
of Google LLC.
