"""
End-to-end pipeline: frame -> detections -> graph -> pruned graph -> outputs.

Usage:
    python pipeline.py path/to/frame.jpg --heading 0.0 --output both
    python pipeline.py path/to/frame.jpg --detector-weights yolov8s-worldv2.pt --conf 0.15
"""

from __future__ import annotations

import argparse
import os
import sys
import cv2

from detect_depth import DetectorDepthEstimator
from graph_builder import build_graph, DEFAULT_CLASS_VOCAB
from pruning import prune_graph, PruningConfig
from encoders import encode_haptic_matrix, encode_spatial_audio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))
from compression_ratio import compute_compression  # noqa: E402


def run_pipeline(
    image_path: str,
    heading_rad: float = 0.0,
    device: str = "cuda",
    output: str = "both",
    detector_weights: str = "yolov10n.pt",
    conf_threshold: float = 0.35,
):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)
    h, w = frame.shape[:2]

    # Phase 1
    detector = DetectorDepthEstimator(
        device=device, detector_weights=detector_weights, conf_threshold=conf_threshold
    )
    detections = detector.run(frame)
    labels = [d.label for d in detections]

    # Phase 2
    graph = build_graph(detections, frame_size=(w, h))
    raw_edge_count = graph.edge_index.shape[1]

    # Phase 3 — the core contribution
    pruned = prune_graph(graph, detections_labels=labels, heading_rad=heading_rad, config=PruningConfig())
    pruned_labels = [labels[i] for i in pruned.kept_node_indices]

    # Real (not synthetic) numbers for eval/compression_ratio.py's table —
    # this is the same compute_compression() the eval script uses, run
    # against this actual photo's detections instead of the hardcoded
    # toy example in that file's __main__ block.
    compression = compute_compression(
        raw_labels=labels,
        pruned_labels=pruned_labels,
        raw_edge_count=raw_edge_count,
        pruned_edge_count=pruned.edge_index.shape[1],
    )

    # Phase 4
    results = {}
    if output in ("matrix", "both"):
        results["matrix"] = encode_haptic_matrix(pruned)
    if output in ("audio", "both"):
        # Pass the FULL original labels list — encode_spatial_audio uses
        # pruned.kept_node_indices (set by prune_graph) to map pruned nodes
        # back to their correct original labels.
        results["audio_json"] = encode_spatial_audio(pruned, labels, w, h)

    results["raw_labels"] = labels
    results["pruned_labels"] = pruned_labels
    results["compression"] = compression
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--heading", type=float, default=0.0, help="user heading in radians, 0 = facing right in image plane")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", choices=["matrix", "audio", "both"], default="both")
    parser.add_argument(
        "--detector-weights",
        default="yolov10n.pt",
        help='e.g. "yolov8s-worldv2.pt" for open-vocabulary detection (see DetectorDepthEstimator docstring)',
    )
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    out = run_pipeline(
        args.image_path,
        heading_rad=args.heading,
        device=args.device,
        output=args.output,
        detector_weights=args.detector_weights,
        conf_threshold=args.conf,
    )

    print("Raw detections:   ", out["raw_labels"])
    print("Kept after prune: ", out["pruned_labels"])
    print(out["compression"].summary())

    if "matrix" in out:
        print("Haptic matrix shape:", out["matrix"].shape)
    if "audio_json" in out:
        print(out["audio_json"])