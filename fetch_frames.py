"""
Pulls N RGB frames from a Hugging Face dataset mirror (NYU Depth V2,
SUN RGB-D, or any other image dataset with an "image"/"rgb"-style
field), saved as JPEGs ready for eval/batch_eval.py or
calibrate_depth_from_images.py.

Replaces fetch_nyu_frames.py, fetch_nyu_frames_api.py,
fetch_sunrgbd_frames.py, and fetch_sunrgbd_frames_hf.py, which were
~80% identical pagination/download code differing only in the dataset
default and which backend (the `datasets` library vs. the lightweight
Datasets Server REST API) they used. That split existed because, in
practice, one backend sometimes fails where the other works for a given
dataset mirror (see the "Known mirror issues" note below) -- this script
keeps both backends but as a single --source flag instead of four files,
so a fix to the shared download/retry logic only has to happen once.

Both backends only ever save the RGB image -- detect_depth.py estimates
its own depth via Depth-Anything-V2 rather than consuming a dataset's
ground-truth depth/label channels, so those fields are never needed here
(they'd only matter if you later want to *validate* depth estimates
against ground truth, which is a separate, optional step).

Usage:
    pip install requests pillow
    # REST API backend (no `datasets`/pandas/pyarrow dependency):
    python fetch_frames.py --dataset jagennath-hari/nyuv2 --source rest --n 50 --out frames --prefix nyu

Always smoke-test a new --dataset with --n 1 first and open the saved
image before trusting a full run -- see "Known mirror issues" below.
This matters more than it sounds like: it is exactly the check that
would have caught the wyrx/SUNRGBD_seg problem documented below before
an entire batch-eval run and a paper draft were built on top of it.

Known mirror issues (carried over from the scripts this replaces):
    - sayakpaul/nyu_depth_v2 500s on the Datasets Server API for image
      rendering; jagennath-hari/nyuv2 is a working Parquet mirror of the
      same NYU Depth V2 labeled set.
    - kasurashan/RGBD-Instance-Segmentation's Datasets Server job crashes
      (501, "missing heartbeats").
    - DO NOT USE wyrx/SUNRGBD_seg. It loads without error and its own
      Data Studio viewer renders images fine, which is why it was
      initially treated as a working SUN RGB-D mirror -- but visual
      inspection of the fetched frames (via visualize_pruning.py, after
      an entire batch_eval run had already been built on top of it)
      showed it is actually furniture-retailer product photography
      (price tags, "Sale" stickers, staged showroom shots), not real SUN
      RGB-D indoor scenes. A batch_eval / paper result built on this
      mirror is invalid regardless of how clean the pipeline metrics
      look -- see the main project README's known issue #6. If you need
      a SUN RGB-D sample, search
      https://huggingface.co/datasets?search=sunrgbd for an alternative
      and smoke-test it visually (not just "does it load") before
      trusting a full run.
    - If you hit a different dataset with neither backend working,
      search https://huggingface.co/datasets?search=<name> for another
      mirror rather than assuming the dataset is unusable.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time


IMAGE_FIELD_CANDIDATES = ("image", "rgb", "rgb_image", "color")

# Datasets confirmed (via visual inspection, not just "does it load") to
# NOT be what they claim to be. The docstring's "Known mirror issues"
# section already documented wyrx/SUNRGBD_seg as furniture-retailer
# product photography rather than real SUN RGB-D indoor scenes -- but
# that warning lived in a comment only, and this exact mirror got fetched
# and built into a full batch_eval run anyway (see README known issue
# #6). This dict is the same information made unbypassable: --dataset is
# checked against it before any network call, and matching it is a hard
# error, not a warning, since a printed warning is exactly what already
# didn't work here.
BLOCKED_DATASETS = {
    "wyrx/SUNRGBD_seg": (
        "confirmed via visual inspection to be furniture-retailer product "
        "photography (price tags, \"Sale\" stickers, staged showroom shots), "
        "not real SUN RGB-D indoor scenes. Any pipeline result built on this "
        "mirror is invalid. See this file's module docstring and the main "
        "project README's known issue #6. For a real SUN RGB-D sample, "
        "search https://huggingface.co/datasets?search=sunrgbd for an "
        "alternative and visually smoke-test a handful of saved frames "
        "yourself before trusting a full run -- do not rely on the "
        "dataset's own Data Studio viewer, which also renders this one "
        "fine despite it being mislabeled."
    ),
}


def _check_not_blocked(dataset: str) -> None:
    reason = BLOCKED_DATASETS.get(dataset)
    if reason is not None:
        print(f"ERROR: --dataset {dataset!r} is blocked. {reason}", file=sys.stderr)
        sys.exit(1)


def _fetch_streaming(args) -> int:
    from datasets import load_dataset

    print(f"Loading {args.dataset} (config={args.config}, split={args.split}, streaming)...")
    load_kwargs = {"split": args.split, "streaming": True}
    if args.config:
        load_kwargs["name"] = args.config
    try:
        ds = load_dataset(args.dataset, **load_kwargs)
    except Exception as e:
        print(f"Failed to load with config={args.config!r}: {e}")
        print("Try without a config (some datasets only have one, unnamed config) by "
              "passing --config '' , or inspect the dataset's fields directly:")
        print(f'  python -c "from datasets import load_dataset; ds=load_dataset(\'{args.dataset}\', split=\'{args.split}\', streaming=True); print(next(iter(ds)).keys())"')
        return 0

    saved = 0
    for i, example in enumerate(ds):
        if saved >= args.n:
            break
        img = None
        for field in IMAGE_FIELD_CANDIDATES:
            img = example.get(field)
            if img is not None:
                break
        if img is None:
            print(f"  skip idx {i}: no image field found (tried {IMAGE_FIELD_CANDIDATES}), keys={list(example.keys())}")
            continue
        out_path = os.path.join(args.out, f"{args.prefix}_{saved:04d}.jpg")
        img.convert("RGB").save(out_path, quality=95)
        saved += 1
        if saved % 10 == 0:
            print(f"  saved {saved}/{args.n}")

    if saved == 0:
        print("No frames saved -- check the dataset's actual field names:")
        print(f'  python -c "from datasets import load_dataset; ds=load_dataset(\'{args.dataset}\', split=\'{args.split}\', streaming=True); print(next(iter(ds)).keys())"')
    return saved


def _fetch_rest(args) -> int:
    import requests
    from PIL import Image

    api_url = "https://datasets-server.huggingface.co/rows"

    def fetch_rows(offset: int, length: int) -> dict:
        params = {
            "dataset": args.dataset, "config": args.config or "default",
            "split": args.split, "offset": offset, "length": length,
        }
        resp = requests.get(api_url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    saved = 0
    offset = 0
    while saved < args.n:
        length = min(args.page_size, args.n - saved)
        print(f"Fetching rows {offset}..{offset + length} ...")
        try:
            data = fetch_rows(offset, length)
        except requests.HTTPError as e:
            print(f"API error at offset {offset}: {e}")
            print(f"Response body: {e.response.text[:500]}")
            print("If this 500s/422s, this mirror's REST path may not work -- try --source streaming instead.")
            break

        rows = data.get("rows", [])
        if not rows:
            print("No more rows returned -- stopping (dataset may have fewer rows than requested).")
            break

        for row in rows:
            row_data = row.get("row", {})
            img_field = None
            for field in IMAGE_FIELD_CANDIDATES:
                img_field = row_data.get(field)
                if img_field is not None:
                    break
            if img_field is None:
                print(f"  skip row_idx={row.get('row_idx')}: no image field found (tried {IMAGE_FIELD_CANDIDATES}). keys={list(row_data.keys())}")
                continue

            img_url = img_field.get("src") if isinstance(img_field, dict) else img_field
            if not img_url:
                print(f"  skip row_idx={row.get('row_idx')}: image field has no 'src' url: {img_field}")
                continue

            try:
                img_bytes = requests.get(img_url, timeout=30).content
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                out_path = os.path.join(args.out, f"{args.prefix}_{saved:04d}.jpg")
                img.save(out_path, quality=95)
                saved += 1
            except Exception as e:
                print(f"  failed to save row_idx={row.get('row_idx')}: {e}")

            if saved >= args.n:
                break

        offset += length
        time.sleep(0.2)  # be polite to the API

    if saved == 0:
        print("Nothing saved. Debug with:")
        print(f'  python -c "import requests; print(requests.get(\'{api_url}\', params={{\'dataset\':\'{args.dataset}\',\'config\':\'{args.config or "default"}\',\'split\':\'{args.split}\',\'offset\':0,\'length\':1}}).json())"')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="HF dataset repo id, e.g. jagennath-hari/nyuv2 (verified-good NYU Depth V2 mirror). See this file's docstring for a SUN RGB-D mirror to avoid.")
    parser.add_argument(
        "--source", choices=["streaming", "rest"], default="rest",
        help="'rest' uses the lightweight Datasets Server REST API (requests + pillow only, "
             "no pandas/pyarrow). 'streaming' uses the `datasets` library directly -- use this "
             "if --source rest 422s/500s on your dataset even though the dataset's own web "
             "viewer renders fine (the two paths have disagreed on subset/config resolution "
             "before). Default: rest.",
    )
    parser.add_argument("--n", type=int, default=50, help="number of frames to save")
    parser.add_argument("--out", type=str, default="frames", help="output directory")
    parser.add_argument("--prefix", type=str, default=None, help="output filename prefix (frames saved as <prefix>_0000.jpg, ...). Defaults to the last path segment of --dataset.")
    parser.add_argument("--config", type=str, default="default", help="dataset config/subset name (pass '' for datasets with no named config, streaming source only)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--page-size", type=int, default=50, help="rows per API request, rest source only (server caps this, usually <=100)")
    args = parser.parse_args()

    _check_not_blocked(args.dataset)

    if args.prefix is None:
        args.prefix = args.dataset.rstrip("/").split("/")[-1].lower().replace("-", "_")

    os.makedirs(args.out, exist_ok=True)

    if args.source == "streaming":
        saved = _fetch_streaming(args)
    else:
        saved = _fetch_rest(args)

    print(f"\nDone. Saved {saved} frames to {args.out}/")
    if saved > 0:
        print(f"Smoke-test tip: open {args.out}/{args.prefix}_0000.jpg and confirm it's an actual "
              f"indoor RGB scene (not a depth map / thumbnail / broken image, and not staged "
              f"product/showroom photography -- look for price tags or sale stickers) before "
              f"trusting a full run. Loading without error or a clean-looking web viewer is NOT "
              f"sufficient confirmation -- that was exactly the case for wyrx/SUNRGBD_seg.")


if __name__ == "__main__":
    main()