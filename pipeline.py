"""
End-to-end pipeline: frame -> detections -> graph -> pruned graph -> outputs.

Usage:
    python pipeline.py path/to/frame.jpg --heading 0.0 --output both
"""

from __future__ import annotations

import argparse
import cv2

from detect_depth import DetectorDepthEstimator
from graph_builder import build_graph, DEFAULT_CLASS_VOCAB
from pruning import prune_graph, PruningConfig
from encoders import encode_haptic_matrix, encode_spatial_audio


def run_pipeline(
    image_path: str,
    heading_rad: float = 0.0,
    device: str = "cuda",
    output: str = "both",
):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)
    h, w = frame.shape[:2]

    # Phase 1
    detector = DetectorDepthEstimator(device=device)
    detections = detector.run(frame)
    labels = [d.label for d in detections]

    # Phase 2
    graph = build_graph(detections, frame_size=(w, h))

    # Phase 3 — the core contribution
    pruned = prune_graph(graph, detections_labels=labels, heading_rad=heading_rad, config=PruningConfig())

    # Phase 4
    results = {}
    if output in ("matrix", "both"):
        results["matrix"] = encode_haptic_matrix(pruned)
    if output in ("audio", "both"):
        # Pass the FULL original labels list — encode_spatial_audio uses
        # pruned.kept_node_indices (set by prune_graph) to map pruned nodes
        # back to their correct original labels.
        results["audio_json"] = encode_spatial_audio(pruned, labels, w, h)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--heading", type=float, default=0.0, help="user heading in radians, 0 = facing right in image plane")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", choices=["matrix", "audio", "both"], default="both")
    args = parser.parse_args()

    out = run_pipeline(args.image_path, heading_rad=args.heading, device=args.device, output=args.output)

    if "matrix" in out:
        print("Haptic matrix shape:", out["matrix"].shape)
    if "audio_json" in out:
        print(out["audio_json"])
