"""
Depth calibration — measures how far off Depth-Anything-V2's metric depth
estimate is at KNOWN real distances, and fits a correction.

Why this exists: a live run reported "person at 12 o'clock, 1.5 meters"
for someone sitting directly in front of a laptop webcam (true distance
more like 0.4-0.6m). Depth-Anything-V2-Metric-Indoor was trained on
room-scale scenes (NYU Depth V2-style, ~1-10m); close-range accuracy at
webcam distances is untested territory for this checkpoint, and laptop
webcams often have wider FOV than the model's implicit assumption, which
distorts near-field scale specifically. This script measures whether the
error is:
    - a fixed OFFSET (e.g. always +0.9m too far) -- easy to correct with
      one constant, and
    - a SCALE error that grows with distance (e.g. 20% too far at 0.5m,
      10% too far at 2m) -- points to the FOV mismatch being the real
      cause, and matters for how trustworthy far-range readings are too,
      not just close-range ones.

Fits corrected_depth = (raw_depth - offset) / scale via least squares over
however many (true_distance, predicted_depth) pairs you capture, and saves
the result to calibration.json for detect_depth.py / pipeline.py to
optionally apply (see apply_correction() below and the --calibration flag
this enables downstream).

Usage:
    python calibrate_depth.py --detector-weights yolov8s-worldv2.pt --device cpu

    Stand (or sit) at a few different KNOWN distances from the camera
    (tape measure, or floor markers at 0.5m/1m/1.5m/2m/3m -- cover the
    range you actually care about, including close range since that's
    where the reported error was). At each distance, enter the true
    distance in meters when prompted and press Enter to capture. Do at
    least 3 distances (5+ recommended) for a reliable fit.

Output: calibration.json, e.g.
    {"offset": 0.87, "scale": 1.02, "n_points": 5, "residual_std_m": 0.11,
     "points": [{"true_m": 0.5, "predicted_m": 1.41}, ...]}
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from detect_depth import DetectorDepthEstimator


def capture_point(detector: DetectorDepthEstimator, cap: cv2.VideoCapture, true_distance_m: float) -> float | None:
    """
    Grabs one frame, runs detection, and returns the predicted depth of
    whichever "person" detection is closest to frame center (i.e. the
    person the operator is presumably standing as, not someone else
    incidentally in frame). Returns None if no person was detected --
    caller should ask the operator to retry.
    """
    ok, frame = cap.read()
    if not ok:
        print("  Frame grab failed.")
        return None

    h, w = frame.shape[:2]
    center = (w / 2, h / 2)

    detections = detector.run(frame)
    people = [d for d in detections if d.label == "person"]
    if not people:
        print("  No 'person' detected in this frame -- make sure you're clearly in view, retry.")
        return None

    def dist_to_center(d):
        cx, cy = d.centroid_px
        return ((cx - center[0]) ** 2 + (cy - center[1]) ** 2) ** 0.5

    closest = min(people, key=dist_to_center)
    print(f"  Detected person at depth={closest.depth_m:.2f}m (true={true_distance_m}m)")
    return closest.depth_m


def fit_correction(points: list[tuple[float, float]]) -> dict:
    """
    Fits predicted = scale * true + offset via least squares, then
    reports the INVERSE (how to correct a raw prediction back to true
    distance): corrected = (predicted - offset) / scale.

    Also reports residual_std_m -- how much scatter is left after the fit.
    A small residual relative to the offset/scale correction means the
    linear model explains the error well (systematic, fixable). A large
    residual means there's noise/nonlinearity a simple linear correction
    won't fully fix -- worth flagging as a real limitation rather than
    quietly correcting for it.
    """
    true_vals = np.array([p[0] for p in points])
    pred_vals = np.array([p[1] for p in points])

    # predicted = scale * true + offset  (fit in this direction since
    # predicted is the noisy measurement, true is the known/controlled value)
    A = np.vstack([true_vals, np.ones_like(true_vals)]).T
    scale, offset = np.linalg.lstsq(A, pred_vals, rcond=None)[0]

    residuals = pred_vals - (scale * true_vals + offset)
    residual_std = float(np.std(residuals))

    return {
        "offset": float(offset),
        "scale": float(scale),
        "n_points": len(points),
        "residual_std_m": round(residual_std, 3),
        "points": [{"true_m": t, "predicted_m": p} for t, p in points],
    }


def apply_correction(raw_depth_m: float, calibration: dict) -> float:
    """
    Applies a saved calibration (from calibration.json) to a raw depth
    reading. detect_depth.py / pipeline.py can import and call this on
    Detection.depth_m if a --calibration file is provided (not wired in
    by default -- this script only measures and saves the correction;
    applying it everywhere is a deliberate separate step so you can
    inspect the numbers first).
    """
    scale = calibration["scale"]
    offset = calibration["offset"]
    if abs(scale) < 1e-6:
        return raw_depth_m
    return (raw_depth_m - offset) / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detector-weights", default="yolov10n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--out", default="calibration.json")
    args = parser.parse_args()

    print("Loading detector + depth models...")
    detector = DetectorDepthEstimator(
        device=args.device, detector_weights=args.detector_weights, conf_threshold=args.conf
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    points: list[tuple[float, float]] = []
    print("\nStand at a known distance, enter it in meters, press Enter to capture.")
    print("Cover a few distances including close range (e.g. 0.5, 1, 1.5, 2, 3).")
    print("Type 'done' when finished (need at least 3 points).\n")

    try:
        while True:
            raw = input(f"[{len(points)} points so far] True distance in meters (or 'done'): ").strip()
            if raw.lower() == "done":
                if len(points) < 3:
                    print(f"Only {len(points)} points -- need at least 3 for a fit. Keep going.")
                    continue
                break
            try:
                true_dist = float(raw)
            except ValueError:
                print("  Enter a number (e.g. 1.5) or 'done'.")
                continue

            input("  Get in position, then press Enter to capture...")
            predicted = capture_point(detector, cap, true_dist)
            if predicted is not None:
                points.append((true_dist, predicted))
    finally:
        cap.release()

    calibration = fit_correction(points)
    with open(args.out, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nFit: predicted_m = {calibration['scale']:.3f} * true_m + {calibration['offset']:.3f}")
    print(f"Residual std: {calibration['residual_std_m']}m")
    if calibration["residual_std_m"] > 0.2:
        print(
            "  NOTE: residual std > 0.2m -- there's meaningful scatter a simple linear "
            "correction won't fully fix. Treat close-range readings as approximate even "
            "after correction."
        )
    if abs(calibration["scale"] - 1.0) > 0.15:
        print(
            f"  NOTE: scale={calibration['scale']:.2f} is far from 1.0 -- error grows/shrinks "
            "with distance, consistent with a FOV mismatch rather than just a fixed offset."
        )
    print(f"Saved to {args.out}")
    print(
        "\nTo correct a raw reading: corrected_m = (raw_m - offset) / scale "
        f"= (raw_m - {calibration['offset']:.3f}) / {calibration['scale']:.3f}"
    )


if __name__ == "__main__":
    main()
