"""
Depth calibration from a manifest of photos, instead of a live webcam
session (see calibrate_depth.py for that version).

Why this exists: the live capture loop requires typing a true distance
and immediately capturing a frame, in sequence, standing in position --
easy to mislabel under time pressure (this project's wall calibration
was mislabeled 1.5m when the true distance was 2.5m, for exactly this
reason). Taking photos first, measuring/confirming each one's true
distance separately, and only THEN running them through calibration
removes that time pressure -- you can double-check the manifest before
anything gets fit.

This does NOT invent depth data. Every predicted_m value below comes
from actually running the real detector+depth model on a real photo you
provide; only the true_m labels are something you supply, and you
supply them for photos you already took (i.e. you know the real
distance because you set up and measured the shot), not for anything
synthetic.

Usage:
    1. Take a few photos of the target class at different KNOWN
       distances (tape measure / floor markers, same as the live
       version) -- e.g. wall_05m.jpg, wall_10m.jpg, wall_15m.jpg.
    2. Write a manifest JSON:
       [
         {"path": "wall_05m.jpg", "true_m": 0.5, "class": "wall"},
         {"path": "wall_10m.jpg", "true_m": 1.0, "class": "wall"},
         {"path": "wall_15m.jpg", "true_m": 1.5, "class": "wall"}
       ]
       (a class can appear multiple times across separate manifest
       entries/files -- this script fits and merges per class exactly
       like calibrate_depth.py's --target-class does)
    3. Run:
       python calibrate_depth_from_images.py manifest.json --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu --out calibration.json

The script prints each photo's detected depth before fitting anything,
same as the live version's real-time check -- read it and fix the
manifest (wrong path, wrong true_m, wrong class) before trusting the
fit, exactly like you'd redo a bad live capture.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2

from detect_depth import DetectorDepthEstimator
from calibrate_depth import fit_correction, load_calibration, closest_detection_to_center


def find_detection_depth(
    detector: DetectorDepthEstimator, image_path: str, target_class: str
) -> float | None:
    """
    Runs detection on one image file and returns the depth of whichever
    detection of target_class is closest to frame center. Mirrors
    calibrate_depth.py's capture_point() by sharing its
    closest_detection_to_center() matching helper -- only the frame
    source differs (a saved image via cv2.imread here, vs. a live
    webcam grab there).
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  Could not read image: {image_path}")
        return None

    h, w = frame.shape[:2]
    detections = detector.run(frame)

    closest = closest_detection_to_center(
        detections, target_class, w, h,
        not_found_msg=(
            f"  No '{target_class}' detected in {image_path} -- "
            f"detected classes were: {sorted(set(d.label for d in detections)) or 'none'}"
        ),
        multiple_found_msg=f"  WARNING: multiple '{target_class}' detections in {image_path} -- using the one closest to center.",
    )
    return closest.depth_m if closest is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="path to a manifest JSON: list of {path, true_m, class}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detector-weights", default="yolov10n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--out", default="calibration.json")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    if not isinstance(manifest, list) or not manifest:
        raise ValueError(
            "manifest must be a non-empty JSON list of "
            '{"path": ..., "true_m": ..., "class": ...} objects'
        )

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))

    print("Loading detector + depth models...")
    detector = DetectorDepthEstimator(
        device=args.device, detector_weights=args.detector_weights, conf_threshold=args.conf
    )

    # points_by_class[class_name] = list of (true_m, predicted_m)
    points_by_class: dict[str, list[tuple[float, float]]] = {}

    print(f"\nProcessing {len(manifest)} manifest entries...\n")
    for entry in manifest:
        path = entry["path"]
        if not os.path.isabs(path):
            path = os.path.join(manifest_dir, path)
        true_m = entry["true_m"]
        target_class = entry["class"]

        predicted_m = find_detection_depth(detector, path, target_class)
        if predicted_m is None:
            print(f"[SKIP] {entry['path']}: no usable detection\n")
            continue

        print(f"[ok] {entry['path']}: class={target_class}  true={true_m}m  predicted={predicted_m:.2f}m")
        points_by_class.setdefault(target_class, []).append((true_m, predicted_m))

    if not points_by_class:
        raise RuntimeError("No usable detections across the whole manifest -- nothing to fit.")

    # Merge with any existing calibration.json data, same behavior as
    # calibrate_depth.py's --target-class resume logic, so running this
    # against a manifest for one class doesn't wipe out other classes
    # already calibrated (e.g. via the live script).
    all_calibration: dict = {}
    if os.path.exists(args.out):
        try:
            all_calibration = load_calibration(args.out)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"\nCould not load existing {args.out} ({e}) -- starting fresh.")

    print()
    for class_name, points in points_by_class.items():
        existing = all_calibration.get(class_name)
        if existing:
            prior_points = [(p["true_m"], p["predicted_m"]) for p in existing.get("points", [])]
            print(f"'{class_name}': merging {len(points)} new point(s) with {len(prior_points)} existing")
            points = prior_points + points

        if len(points) < 3:
            print(f"'{class_name}': only {len(points)} point(s) total -- need at least 3, "
                  f"not fitting yet. Add more manifest entries for this class.")
            continue

        cal = fit_correction(points)
        all_calibration[class_name] = cal
        print(f"'{class_name}' fit: predicted_m = {cal['scale']:.3f} * true_m + {cal['offset']:.3f}  "
              f"residual_std={cal['residual_std_m']}m")
        if cal["residual_std_m"] > 0.2:
            print(f"  WARNING: residual std > 0.2m for '{class_name}' -- points are inconsistent, "
                  f"check the list below before trusting this:")
            for p in cal["points"]:
                print(f"    true={p['true_m']}m  predicted={p['predicted_m']:.2f}m")
        if cal["scale"] < 0.3:
            print(f"  WARNING: scale={cal['scale']:.2f} for '{class_name}' is implausible for a "
                  f"working depth model -- do not use this calibration.")

    with open(args.out, "w") as f:
        json.dump(all_calibration, f, indent=2)
    print(f"\nSaved to {args.out}. Calibrated classes: {list(all_calibration.keys())}")


if __name__ == "__main__":
    main()
