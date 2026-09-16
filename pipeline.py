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

from detect_depth import DetectorDepthEstimator, load_calibration
from graph_builder import build_graph
from pruning import prune_graph, PruningConfig
from encoders import encode_haptic_matrix, encode_spatial_audio
from navigation_planner import generate_instructions, instructions_to_speech_text

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))
from compression_ratio import compute_compression  # noqa: E402


def build_detector(
    device: str = "cuda",
    detector_weights: str = "yolov10n.pt",
    conf_threshold: float = 0.35,
    calibration: dict | None = None,
) -> DetectorDepthEstimator:
    """
    Constructs the Phase 1 detector once so it can be reused across frames.

    Loading YOLO + Depth-Anything costs seconds and hundreds of MB. Batch
    callers (eval/batch_eval.py) must build the detector ONCE and pass it to
    run_pipeline(detector=...) — previously run_pipeline constructed its own
    on every call, so a 51-frame NYU batch paid the full model-load cost 51
    times over.
    """
    return DetectorDepthEstimator(
        device=device,
        detector_weights=detector_weights,
        conf_threshold=conf_threshold,
        calibration=calibration,
    )


def run_pipeline(
    image_path: str,
    heading_rad: float = 0.0,
    device: str = "cuda",
    output: str = "both",
    detector_weights: str = "yolov10n.pt",
    conf_threshold: float = 0.35,
    detector: DetectorDepthEstimator | None = None,
    config: PruningConfig | None = None,
    calibration: dict | None = None,
):
    """
    detector: an already-constructed DetectorDepthEstimator to reuse. When
        None, one is built per call (fine for a single image, very wasteful
        in a loop — see build_detector).
    config: PruningConfig to use; defaults to PruningConfig(). Pass an
        ablated config to re-run the same frames with a term disabled.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)
    h, w = frame.shape[:2]

    if config is None:
        config = PruningConfig()

    # Phase 1
    if detector is None:
        detector = build_detector(
            device=device,
            detector_weights=detector_weights,
            conf_threshold=conf_threshold,
            calibration=calibration,
        )
    detections = detector.run(frame)
    labels = [d.label for d in detections]

    # Phase 2
    graph = build_graph(detections, frame_size=(w, h))
    raw_edge_count = graph.edge_index.shape[1]

    # Phase 3 — the core contribution
    pruned = prune_graph(graph, detections_labels=labels, heading_rad=heading_rad, config=config)
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

    # Phase 6/7 — navigation instructions + speech text, derived from the
    # same spatial-audio JSON (computed above if output includes "audio";
    # computed fresh here otherwise, since instructions don't depend on
    # which output modes the caller asked for).
    audio_json = results.get("audio_json") or encode_spatial_audio(pruned, labels, w, h)
    instructions = generate_instructions(audio_json, max_instructions=3)
    results["instructions"] = [instr.text for instr in instructions]
    results["speech_text"] = instructions_to_speech_text(instructions)

    results["raw_labels"] = labels
    results["pruned_labels"] = pruned_labels
    results["compression"] = compression
    results["ablation"] = config.ablation_name()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument(
        "--heading", type=float, default=0.0,
        help="user heading in radians relative to the camera's optical axis; "
             "0 = looking straight ahead (matches graph_builder's ego convention, "
             "where an object at the horizontal center of the frame has bearing 0)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", choices=["matrix", "audio", "both"], default="both")
    parser.add_argument(
        "--detector-weights",
        default="yolov10n.pt",
        help='e.g. "yolov8s-worldv2.pt" for open-vocabulary detection (see DetectorDepthEstimator docstring)',
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument(
        "--calibration", default=None,
        help="path to calibration.json from calibrate_depth.py; applies the fitted depth correction",
    )
    parser.add_argument("--tau", type=float, default=PruningConfig.tau,
                        help="distance decay constant in pruning.edge_weight")
    parser.add_argument("--gamma", type=float, default=PruningConfig.heading_gamma,
                        help="heading cone sharpness exponent")
    parser.add_argument("--prune-threshold", type=float, default=PruningConfig.prune_threshold)
    parser.add_argument("--max-nodes", type=int, default=PruningConfig.max_nodes)
    parser.add_argument(
        "--ablate", nargs="*", default=[], choices=["affordance", "distance", "heading"],
        help="disable one or more terms of the pruning formula, for term-wise ablation",
    )
    parser.add_argument(
        "--speak", action="store_true",
        help="Speak the navigation instructions aloud via pyttsx3 (offline TTS). "
             "Off by default -- run_pipeline()/batch_eval.py callers should never "
             "trigger audio playback implicitly.",
    )
    args = parser.parse_args()

    config = PruningConfig(
        tau=args.tau,
        heading_gamma=args.gamma,
        prune_threshold=args.prune_threshold,
        max_nodes=args.max_nodes,
        use_affordance="affordance" not in args.ablate,
        use_distance="distance" not in args.ablate,
        use_heading="heading" not in args.ablate,
    )

    out = run_pipeline(
        args.image_path,
        heading_rad=args.heading,
        device=args.device,
        output=args.output,
        detector_weights=args.detector_weights,
        conf_threshold=args.conf,
        config=config,
        calibration=load_calibration(args.calibration),
    )

    print("Pruning terms:    ", out["ablation"])
    print("Raw detections:   ", out["raw_labels"])
    print("Kept after prune: ", out["pruned_labels"])
    print(out["compression"].summary())

    if "matrix" in out:
        print("Haptic matrix shape:", out["matrix"].shape)
    if "audio_json" in out:
        print(out["audio_json"])

    print()
    print("Navigation instructions:")
    for line in out["instructions"]:
        print(" ", line)
    print("Speech text:", out["speech_text"])

    if args.speak:
        from speech_output import speak_instructions  # lazy import — pyttsx3 optional
        speak_instructions(out["speech_text"])