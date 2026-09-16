"""
Visualizes what the pipeline actually kept vs. pruned on a single real
frame, by drawing bounding boxes directly on the image.

Exists specifically to spot-check the open question flagged in the
paper's Limitations section and README known-issue #6: SUN RGB-D showed
74.0% node compression vs. NYU's 52.6%, and 14/21 SUN RGB-D frames pruned
down to only ['bed'] -- including scenes where person/table/sofa/wall
were also detected but dropped. Aggregate compression numbers alone
can't distinguish "the heading/distance math is correctly deciding bed
is more relevant" from "something about this dataset mirror or the
scoring is systematically biased toward bed." Looking at the actual
boxes/depths/scores on the actual photo can.

Color coding:
    GREEN solid box   = kept (survived pruning), label + depth + score shown
    RED dashed box    = pruned (score below threshold, not force-kept)
    ORANGE dashed box = pruned but was a forced-keep-eligible class
                        (door/stairs/obstacle) that nonetheless didn't
                        survive -- shouldn't normally happen given
                        pruning.py's force-keep logic, so this color is
                        also an implicit correctness check: if you ever
                        see orange, something is wrong upstream.

Usage:
    python visualize_pruning.py path/to/frame.jpg --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu --heading 0.0
    python visualize_pruning.py frames_sunrgbd/00000.jpg --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu --out annotated_00000.jpg

Run this against a handful of the SUN RGB-D frames that pruned to
['bed'] only (see results_sunrgbd.json's per_frame list for exactly
which filenames) and look at where the dropped objects' boxes actually
are relative to the bed's -- that's the fastest way to tell whether this
is a real heading/distance effect or not.
"""

from __future__ import annotations

import argparse

import cv2

from detect_depth import DetectorDepthEstimator
from graph_builder import build_graph
from pruning import prune_graph, PruningConfig, load_pruning_config, AFFORDANCE_PRIORITY, FORCE_KEEP_MIN_PRIORITY

KEPT_COLOR = (0, 200, 0)       # green, BGR
PRUNED_COLOR = (0, 0, 220)     # red, BGR
FORCE_KEEP_MISS_COLOR = (0, 140, 255)  # orange, BGR -- see module docstring


def visualize(
    image_path: str,
    out_path: str,
    detector: DetectorDepthEstimator,
    heading_rad: float = 0.0,
    calibration: dict | None = None,
    verbose: bool = True,
    pruning_config: PruningConfig = PruningConfig(),
    affordance_priority: dict[str, float] = AFFORDANCE_PRIORITY,
) -> list[str]:
    """
    Runs detection+graph+pruning on one image and writes an annotated
    copy. Takes an already-constructed DetectorDepthEstimator so batch
    mode (see main()) can reuse one loaded model across many images
    instead of reloading YOLO-World + Depth-Anything per file, which
    would make a 21-frame batch take as long as running batch_eval.py
    itself just to produce pictures.

    Returns the list of KEPT labels for this frame, so batch mode can
    print a summary of which frames match the flagged bed-only pattern
    without the caller needing to re-parse this function's printed output.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)
    h, w = frame.shape[:2]

    detections = detector.run(frame)

    if calibration is not None:
        from calibrate_depth import apply_calibration_to_detections
        detections = apply_calibration_to_detections(detections, calibration)

    labels = [d.label for d in detections]
    graph = build_graph(detections, frame_size=(w, h))
    pruned = prune_graph(
        graph, detections_labels=labels, heading_rad=heading_rad,
        config=pruning_config, affordance_priority=affordance_priority,
    )
    kept_original_indices = set(pruned.kept_node_indices)
    kept_labels = [labels[i] for i in pruned.kept_node_indices]

    if verbose:
        print(f"Raw detections ({len(detections)}):")
        for i, d in enumerate(detections):
            kept = i in kept_original_indices
            status = "KEPT" if kept else "pruned"
            print(f"  [{status:6s}] {d.label:12s} depth={d.depth_m:.2f}m  bbox={tuple(round(v) for v in d.bbox_xyxy)}")

    annotated = frame.copy()
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
        kept = i in kept_original_indices
        is_force_keep_class = affordance_priority.get(d.label, 0) >= FORCE_KEEP_MIN_PRIORITY

        if kept:
            color = KEPT_COLOR
            thickness = 3
            line_type = cv2.LINE_AA
        elif is_force_keep_class:
            # A door/stairs/obstacle that did NOT survive -- shouldn't
            # happen given pruning.py's force-keep logic. Flagged
            # distinctly so it's visually obvious if it ever occurs.
            color = FORCE_KEEP_MISS_COLOR
            thickness = 2
            line_type = cv2.LINE_AA
        else:
            color = PRUNED_COLOR
            thickness = 1
            line_type = cv2.LINE_AA

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, line_type)
        label_text = f"{d.label} {d.depth_m:.1f}m"
        cv2.putText(
            annotated, label_text, (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )

    legend_y = 20
    for text, color in [
        ("KEPT", KEPT_COLOR), ("pruned", PRUNED_COLOR), ("force-keep MISS (bug?)", FORCE_KEEP_MISS_COLOR)
    ]:
        cv2.putText(annotated, text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        legend_y += 20

    cv2.imwrite(out_path, annotated)
    if verbose:
        print(f"\nSaved annotated image to {out_path}")
        print(f"Kept {len(kept_original_indices)}/{len(detections)} nodes.")
    return kept_labels


if __name__ == "__main__":
    import os

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", help="a single image file, OR a directory to process every image in it (batch mode)")
    parser.add_argument("--out", default=None, help="output path (single-file mode) or output directory (batch mode); defaults sensibly if omitted")
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detector-weights", default="yolov10n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--calibration", default=None, help="path to calibration.json from calibrate_depth.py")
    parser.add_argument(
        "--pruning-config", default=None,
        help="path to a JSON file (see pruning_config.example.json) overriding "
             "tau/heading_gamma/prune_threshold/max_nodes/affordance_priority",
    )
    parser.add_argument(
        "--only-single-label", action="store_true",
        help="batch mode only: only print/flag frames that pruned down to exactly one distinct "
             "kept label (the pattern flagged for SUN RGB-D in README known-issue #6) instead of "
             "printing a line for every frame",
    )
    args = parser.parse_args()

    calibration = None
    if args.calibration:
        from calibrate_depth import load_calibration
        calibration = load_calibration(args.calibration)

    pruning_config, affordance_priority = load_pruning_config(args.pruning_config)

    print("Loading detector + depth models (once, reused for every image)...")
    detector = DetectorDepthEstimator(
        device=args.device, detector_weights=args.detector_weights, conf_threshold=args.conf
    )

    if os.path.isdir(args.image_path):
        out_dir = args.out or (args.image_path.rstrip("/\\") + "_annotated")
        os.makedirs(out_dir, exist_ok=True)
        image_files = sorted(
            f for f in os.listdir(args.image_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not image_files:
            raise FileNotFoundError(f"no image files found in {args.image_path}")

        print(f"Processing {len(image_files)} images from {args.image_path} -> {out_dir}\n")
        flagged = []  # frames matching the single-distinct-label pattern
        for fname in image_files:
            in_path = os.path.join(args.image_path, fname)
            out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + "_annotated.jpg")
            try:
                kept_labels = visualize(
                    in_path, out_path, detector,
                    heading_rad=args.heading, calibration=calibration, verbose=False,
                    pruning_config=pruning_config, affordance_priority=affordance_priority,
                )
            except Exception as e:  # noqa: BLE001 - one bad frame shouldn't kill the batch
                print(f"[FAIL] {fname}: {type(e).__name__}: {e}")
                continue

            distinct_labels = set(kept_labels)
            is_single_label = len(distinct_labels) == 1 and len(kept_labels) >= 1
            if is_single_label:
                flagged.append((fname, kept_labels))

            if not args.only_single_label or is_single_label:
                marker = " <-- single kept label" if is_single_label else ""
                print(f"{fname}: kept {kept_labels}{marker}")

        print(f"\n{len(flagged)}/{len(image_files)} frames pruned to a single distinct kept label:")
        for fname, kept_labels in flagged:
            print(f"  {fname}: {kept_labels}")
        print(f"\nAnnotated images written to {out_dir}/ -- open the ones listed above first.")

    else:
        out_path = args.out or (
            args.image_path.rsplit(".", 1)[0] + "_annotated." + args.image_path.rsplit(".", 1)[1]
        )
        visualize(
            args.image_path, out_path, detector,
            heading_rad=args.heading, calibration=calibration, verbose=True,
            pruning_config=pruning_config, affordance_priority=affordance_priority,
        )