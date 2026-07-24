"""
Batch evaluation — run the full pipeline over a directory of frames and
aggregate compute_compression() output into the numbers a results table
actually needs (mean +- std, not one photo at a time).

This is the "next step" after single-photo verification in pipeline.py /
README known-issues #4-5: those confirmed the pipeline is *correct* on one
real photo. This script is what turns that into a *paper number* by running
it over a dataset (NYU Depth V2, SUN RGB-D, or any folder of frames).

Usage:
    python eval/batch_eval.py path/to/frames_dir --heading 0.0 --output-json results.json
    python eval/batch_eval.py path/to/frames_dir --detector-weights yolov8s-worldv2.pt --conf 0.15

Notes:
- A frame that fails (unreadable, zero detections, model error) is recorded
  in `failures` and excluded from the aggregate stats, not silently dropped
  — the summary always reports how many frames succeeded vs. how many were
  attempted, so a bad run can't quietly look like a clean one.
- Per-frame results are kept in full (not just the aggregate) so you can
  re-slice by scene type, detector, etc. later without re-running.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import run_pipeline  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FrameResult:
    filename: str
    raw_nodes: int
    raw_edges: int
    pruned_nodes: int
    pruned_edges: int
    node_compression: float
    edge_compression: float
    critical_nodes_total: int
    critical_nodes_retained: int
    critical_retention: float
    raw_labels: list[str]
    pruned_labels: list[str]


def find_frames(frames_dir: str) -> list[str]:
    files = sorted(
        f for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    return [os.path.join(frames_dir, f) for f in files]


def run_batch(
    frames_dir: str,
    heading_rad: float = 0.0,
    device: str = "cuda",
    detector_weights: str = "yolov10n.pt",
    conf_threshold: float = 0.35,
) -> dict:
    frame_paths = find_frames(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"no image files ({sorted(IMAGE_EXTS)}) found in {frames_dir}")

    results: list[FrameResult] = []
    failures: list[dict] = []

    for path in frame_paths:
        fname = os.path.basename(path)
        try:
            out = run_pipeline(
                path,
                heading_rad=heading_rad,
                device=device,
                output="matrix",  # audio JSON not needed for aggregate stats; skip for speed
                detector_weights=detector_weights,
                conf_threshold=conf_threshold,
            )
            comp = out["compression"]
            results.append(
                FrameResult(
                    filename=fname,
                    raw_nodes=comp.raw_nodes,
                    raw_edges=comp.raw_edges,
                    pruned_nodes=comp.pruned_nodes,
                    pruned_edges=comp.pruned_edges,
                    node_compression=comp.node_compression,
                    edge_compression=comp.edge_compression,
                    critical_nodes_total=comp.critical_nodes_total,
                    critical_nodes_retained=comp.critical_nodes_retained,
                    critical_retention=comp.critical_retention,
                    raw_labels=out["raw_labels"],
                    pruned_labels=out["pruned_labels"],
                )
            )
            print(f"[ok]   {fname}: {comp.summary()}")
        except Exception as e:  # noqa: BLE001 - one bad frame must not kill the batch
            failures.append({"filename": fname, "error": f"{type(e).__name__}: {e}"})
            print(f"[FAIL] {fname}: {type(e).__name__}: {e}")

    return {
        "frames_attempted": len(frame_paths),
        "frames_succeeded": len(results),
        "frames_failed": len(failures),
        "failures": failures,
        "per_frame": [asdict(r) for r in results],
        "aggregate": _aggregate(results),
    }


def _aggregate(results: list[FrameResult]) -> dict:
    if not results:
        return {}

    def mean_std(vals: list[float]) -> dict:
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 4), "std": round(std, 4)}

    # Critical retention is only meaningful over frames that actually had a
    # critical-class object (door/stairs/obstacle) in the raw scene —
    # frames with zero critical objects report retention=1.0 vacuously
    # (see compression_ratio.py) and would inflate the aggregate if included.
    retention_frames = [r for r in results if r.critical_nodes_total > 0]

    return {
        "n_frames": len(results),
        "node_compression": mean_std([r.node_compression for r in results]),
        "edge_compression": mean_std([r.edge_compression for r in results]),
        "critical_retention": (
            mean_std([r.critical_retention for r in retention_frames])
            if retention_frames
            else None
        ),
        "n_frames_with_critical_objects": len(retention_frames),
        "mean_raw_nodes": round(statistics.mean([r.raw_nodes for r in results]), 2),
        "mean_pruned_nodes": round(statistics.mean([r.pruned_nodes for r in results]), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir", type=str, help="directory of frames (NYU Depth V2, SUN RGB-D, or any images)")
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-weights", default="yolov10n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--output-json", default=None, help="path to write full results JSON")
    args = parser.parse_args()

    t0 = time.time()
    batch = run_batch(
        args.frames_dir,
        heading_rad=args.heading,
        device=args.device,
        detector_weights=args.detector_weights,
        conf_threshold=args.conf,
    )
    elapsed = time.time() - t0

    print()
    print(f"Attempted: {batch['frames_attempted']}  Succeeded: {batch['frames_succeeded']}  Failed: {batch['frames_failed']}")
    if batch["frames_failed"]:
        print("Failures:")
        for f in batch["failures"]:
            print(f"  - {f['filename']}: {f['error']}")

    agg = batch["aggregate"]
    if agg:
        nc, ec = agg["node_compression"], agg["edge_compression"]
        print()
        print(f"Node compression:     {nc['mean']:.1%} +- {nc['std']:.1%}  (n={agg['n_frames']})")
        print(f"Edge compression:     {ec['mean']:.1%} +- {ec['std']:.1%}  (n={agg['n_frames']})")
        if agg["critical_retention"]:
            cr = agg["critical_retention"]
            print(f"Critical retention:   {cr['mean']:.1%} +- {cr['std']:.1%}  (n={agg['n_frames_with_critical_objects']} frames w/ critical objects)")
        else:
            print("Critical retention:   n/a (no frames contained door/stairs/obstacle)")
        print(f"Mean nodes:           {agg['mean_raw_nodes']} raw -> {agg['mean_pruned_nodes']} pruned")

    print(f"\nElapsed: {elapsed:.1f}s")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(batch, f, indent=2)
        print(f"Full results written to {args.output_json}")
