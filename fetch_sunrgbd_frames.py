"""
Pulls N RGB frames from a SUN RGB-D dataset mirror on Hugging Face, using
the lightweight Datasets Server REST API (same approach as
fetch_nyu_frames_api.py -- only `requests` + `Pillow` needed, no `datasets`
library / pandas/pyarrow dependency).

IMPORTANT -- verify the mirror before a full run:
No single HF mirror of SUN RGB-D has been verified working end-to-end yet
(the same problem README's known-issues log describes for the original NYU
mirror, which 500'd on image rendering and had to be swapped for
jagennath-hari/nyuv2). Candidates worth checking, in order:
    - wyrx/SUNRGBD_seg  -- VERIFIED: the HF dataset page's Data Studio
      viewer renders real RGB/depth/label images for "default" config,
      "train" (5.29k rows) and "test" (5.05k rows) splits. Use this as
      --dataset wyrx/SUNRGBD_seg --config default --split train.
      (kasurashan/RGBD-Instance-Segmentation was tried first and its
      Datasets Server job crashed -- "missing heartbeats" 501 -- don't
      use it.)
    - if wyrx/SUNRGBD_seg ever breaks too, search
      https://huggingface.co/datasets?search=sunrgbd for alternatives
Before running with --n set high, always smoke-test with --n 1 first (see
bottom of this docstring) and eyeball the saved image.

We only save RGB -- detect_depth.py estimates its own depth via
Depth-Anything-V2 rather than using dataset ground-truth depth, so the
depth channel isn't needed here (same rationale as fetch_nyu_frames.py).

Usage:
    pip install requests pillow
    python fetch_sunrgbd_frames.py --n 1 --out frames_sunrgbd --dataset wyrx/SUNRGBD_seg --config default --split train
    # once the smoke test looks right:
    python fetch_sunrgbd_frames.py --n 50 --out frames_sunrgbd --dataset wyrx/SUNRGBD_seg --config default --split train

Note: the datasets-server API paginates in blocks (server caps length,
usually <=100 rows/request), so this script does its own pagination.
"""

from __future__ import annotations

import argparse
import io
import os
import time

import requests
from PIL import Image

API_URL = "https://datasets-server.huggingface.co/rows"


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    params = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of frames to save")
    parser.add_argument("--out", type=str, default="frames_sunrgbd", help="output directory")
    parser.add_argument(
        "--dataset", type=str, default="wyrx/SUNRGBD_seg",
        help="HF dataset repo id for a SUN RGB-D mirror. Default is wyrx/SUNRGBD_seg, verified "
             "working via its Data Studio viewer (see module docstring). Always smoke-test with "
             "--n 1 first regardless.",
    )
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--page-size", type=int, default=50, help="rows per API request (server caps this, usually <=100)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    saved = 0
    offset = 0
    while saved < args.n:
        length = min(args.page_size, args.n - saved)
        print(f"Fetching rows {offset}..{offset+length} ...")
        try:
            data = fetch_rows(args.dataset, args.config, args.split, offset, length)
        except requests.HTTPError as e:
            print(f"API error at offset {offset}: {e}")
            print(f"Response body: {e.response.text[:500]}")
            print("If this 500s or times out, the mirror likely doesn't have a working "
                  "Datasets Server viewer -- try a different --dataset (see module docstring).")
            break

        rows = data.get("rows", [])
        if not rows:
            print("No more rows returned -- stopping (dataset may have fewer rows than requested).")
            break

        for row in rows:
            row_data = row.get("row", {})
            # SUN RGB-D mirrors vary in field naming -- try the common ones.
            img_field = (
                row_data.get("image")
                or row_data.get("rgb")
                or row_data.get("rgb_image")
                or row_data.get("color")
            )
            if img_field is None:
                print(f"  skip row_idx={row.get('row_idx')}: no image field found. keys={list(row_data.keys())}")
                continue

            # datasets-server image fields are typically {"src": url, "height":..., "width":...}
            img_url = img_field.get("src") if isinstance(img_field, dict) else img_field
            if not img_url:
                print(f"  skip row_idx={row.get('row_idx')}: image field has no 'src' url: {img_field}")
                continue

            try:
                img_bytes = requests.get(img_url, timeout=30).content
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                out_path = os.path.join(args.out, f"sunrgbd_{saved:04d}.jpg")
                img.save(out_path, quality=95)
                saved += 1
            except Exception as e:
                print(f"  failed to save row_idx={row.get('row_idx')}: {e}")

            if saved >= args.n:
                break

        offset += length
        time.sleep(0.2)  # be polite to the API

    print(f"Done. Saved {saved} frames to {args.out}/")
    if saved == 0:
        print("Nothing saved. Debug with:")
        print(f'  python -c "import requests; print(requests.get(\'{API_URL}\', params={{\'dataset\':\'{args.dataset}\',\'config\':\'{args.config}\',\'split\':\'{args.split}\',\'offset\':0,\'length\':1}}).json())"')
    elif saved > 0:
        print(f"Smoke-test tip: open {args.out}/sunrgbd_0000.jpg and confirm it's actually an "
              f"indoor RGB scene (not a depth map / thumbnail / broken image) before trusting a full run.")


if __name__ == "__main__":
    main()