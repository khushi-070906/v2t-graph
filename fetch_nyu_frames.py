"""
Pulls N frames from the NYU Depth V2 labeled dataset (mirrored on Hugging
Face) and saves just the RGB images to a folder, ready for
eval/batch_eval.py.

We only need RGB here -- detect_depth.py estimates its own depth via
Depth-Anything-V2 rather than consuming NYU's ground-truth depth maps, so
the ground-truth depth channel in this dataset isn't used by this
pipeline (it would only matter if you later want to *validate* the depth
estimates against ground truth, which is a separate, optional step).

Usage:
    pip install datasets pillow
    python fetch_nyu_frames.py --n 50 --out frames
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of frames to save")
    parser.add_argument("--out", type=str, default="frames", help="output directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sayakpaul/nyu_depth_v2",
        help="HF dataset repo id (mirrors the official NYU Depth V2 labeled subset)",
    )
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.dataset} ({args.split} split, streaming)...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    saved = 0
    for i, example in enumerate(ds):
        if saved >= args.n:
            break
        img = example.get("image") or example.get("rgb")
        if img is None:
            print(f"  skip idx {i}: no 'image'/'rgb' field found, keys={list(example.keys())}")
            continue
        out_path = os.path.join(args.out, f"nyu_{saved:04d}.jpg")
        img.convert("RGB").save(out_path, quality=95)
        saved += 1
        if saved % 10 == 0:
            print(f"  saved {saved}/{args.n}")

    print(f"Done. Saved {saved} frames to {args.out}/")
    if saved == 0:
        print("No frames saved -- check the dataset's actual field names by inspecting one example:")
        print('  python -c "from datasets import load_dataset; ds=load_dataset(\'%s\', split=\'%s\', streaming=True); print(next(iter(ds)).keys())"' % (args.dataset, args.split))


if __name__ == "__main__":
    main()
