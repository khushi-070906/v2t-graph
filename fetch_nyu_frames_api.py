"""
Pulls N frames from the NYU Depth V2 labeled dataset (mirrored on Hugging
Face) using the lightweight Datasets Server REST API instead of the full
`datasets` library. This deliberately avoids `datasets`' pandas/pyarrow
dependency -- only `requests` + `Pillow` are needed, both of which were
already working in this environment. Use this if `datasets` itself fails
to import (e.g. a blocked pandas DLL, as happened here) -- if `datasets`
works fine for you, the streaming version (fetch_nyu_frames.py) is simpler.

We only save RGB -- detect_depth.py estimates its own depth via
Depth-Anything-V2 rather than using NYU's ground-truth depth, so the
ground-truth depth channel isn't needed for this pipeline (it would only
matter if you later want to validate depth estimates against ground
truth -- a separate, optional step).

Usage:
    pip install requests pillow
    python fetch_nyu_frames_api.py --n 50 --out frames

Note: the datasets-server API paginates in blocks (default max length is
100 rows/request), so this script does its own pagination internally.
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
    parser.add_argument("--out", type=str, default="frames", help="output directory")
    parser.add_argument("--dataset", type=str, default="jagennath-hari/nyuv2",
                         help="HF dataset repo id. Default is a pre-converted Parquet mirror of the "
                              "official NYU Depth V2 labeled set with a working dataset viewer -- "
                              "sayakpaul/nyu_depth_v2 (the previous default) currently 500s on the "
                              "Datasets Server API for image rendering.")
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
            break

        rows = data.get("rows", [])
        if not rows:
            print("No more rows returned -- stopping (dataset may have fewer rows than requested).")
            break

        for row in rows:
            row_data = row.get("row", {})
            img_field = row_data.get("rgb") or row_data.get("image")
            if img_field is None:
                print(f"  skip row_idx={row.get('row_idx')}: no 'image'/'rgb' field. keys={list(row_data.keys())}")
                continue

            # datasets-server image fields are typically {"src": url, "height":..., "width":...}
            img_url = img_field.get("src") if isinstance(img_field, dict) else img_field
            if not img_url:
                print(f"  skip row_idx={row.get('row_idx')}: image field has no 'src' url: {img_field}")
                continue

            try:
                img_bytes = requests.get(img_url, timeout=30).content
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                out_path = os.path.join(args.out, f"nyu_{saved:04d}.jpg")
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


if __name__ == "__main__":
    main()