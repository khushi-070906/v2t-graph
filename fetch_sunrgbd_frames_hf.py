"""
Pulls N RGB frames from the wyrx/SUNRGBD_seg dataset on Hugging Face using
the `datasets` library directly (streaming), instead of the lightweight
Datasets Server REST API used by fetch_sunrgbd_frames.py.

Use this if the REST API version 422s on you (it did in testing here,
even though wyrx/SUNRGBD_seg's own web viewer renders images fine --
the REST API and the library apparently disagree on this dataset's
subset/config resolution). The `datasets` library resolves that
internally instead of requiring you to guess the exact REST parameter
names, at the cost of a pandas/pyarrow dependency (same tradeoff as
fetch_nyu_frames.py vs fetch_nyu_frames_api.py).

wyrx/SUNRGBD_seg has "image", "depth", and "label" fields per row (per
its dataset card) -- we only save "image" (RGB). detect_depth.py
estimates its own depth via Depth-Anything-V2 rather than using
ground-truth depth, so "depth"/"label" aren't needed for this pipeline.

Usage:
    pip install datasets pillow
    python fetch_sunrgbd_frames_hf.py --n 50 --out frames_sunrgbd

Always smoke-test with --n 1 first and open the saved image before
trusting a full run.
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of frames to save")
    parser.add_argument("--out", type=str, default="frames_sunrgbd", help="output directory")
    parser.add_argument(
        "--dataset", type=str, default="wyrx/SUNRGBD_seg",
        help="HF dataset repo id. Default verified to render real RGB/depth/label images "
             "via its own Data Studio viewer.",
    )
    parser.add_argument("--config", type=str, default="default", help="dataset config/subset name")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.dataset} (config={args.config}, split={args.split}, streaming)...")
    try:
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    except Exception as e:
        print(f"Failed to load with config={args.config!r}: {e}")
        print("Try without a config (some datasets only have one, unnamed config):")
        print(f'  python -c "from datasets import load_dataset; ds=load_dataset(\'{args.dataset}\', split=\'{args.split}\', streaming=True); print(next(iter(ds)).keys())"')
        return

    saved = 0
    for i, example in enumerate(ds):
        if saved >= args.n:
            break
        img = example.get("image") or example.get("rgb")
        if img is None:
            print(f"  skip idx {i}: no 'image'/'rgb' field found, keys={list(example.keys())}")
            continue
        out_path = os.path.join(args.out, f"sunrgbd_{saved:04d}.jpg")
        img.convert("RGB").save(out_path, quality=95)
        saved += 1
        if saved % 10 == 0:
            print(f"  saved {saved}/{args.n}")

    print(f"Done. Saved {saved} frames to {args.out}/")
    if saved == 0:
        print("No frames saved -- check the dataset's actual field names by inspecting one example:")
        print(f'  python -c "from datasets import load_dataset; ds=load_dataset(\'{args.dataset}\', \'{args.config}\', split=\'{args.split}\', streaming=True); print(next(iter(ds)).keys())"')
    else:
        print(f"Smoke-test tip: open {args.out}/sunrgbd_0000.jpg and confirm it's an actual "
              f"indoor RGB scene before trusting a full run.")


if __name__ == "__main__":
    main()
