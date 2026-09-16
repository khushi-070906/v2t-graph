"""
Phase 1 — Frontend pipeline.

Takes a single RGB frame (numpy array, HxWx3, BGR as read by cv2) and returns
a list of structured detections: class, bbox, centroid, and a per-object
depth estimate pulled from a monocular depth map.

This module is intentionally "dumb" — no graph logic, no prioritization.
It only answers: what objects are in the frame, where are they, and how far
away are they. Everything downstream (graph_builder.py) consumes this
output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import cv2
from PIL import Image


def load_calibration(path: str | None) -> dict | None:
    """Loads a calibration.json produced by calibrate_depth.py, or None."""
    if not path:
        return None
    import json

    with open(path) as f:
        calib = json.load(f)
    if "scale" not in calib or "offset" not in calib:
        raise ValueError(f"{path} is not a calibration file (needs 'scale' and 'offset')")
    return calib


def apply_calibration(raw_depth_m: float, calibration: dict | None) -> float:
    """
    Applies the affine correction fitted by calibrate_depth.py:
        corrected = (raw - offset) / scale

    Lives here (not in calibrate_depth.py) so the runtime path can apply a
    calibration without importing the interactive capture script. Clamped at
    0 because a corrected depth below zero is physically meaningless and
    would otherwise sail through pruning as the closest possible object.
    """
    if not calibration:
        return raw_depth_m
    scale = calibration.get("scale", 1.0)
    offset = calibration.get("offset", 0.0)
    if abs(scale) < 1e-6:
        return raw_depth_m
    return max(0.0, (raw_depth_m - offset) / scale)


@dataclass
class Detection:
    obj_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    centroid_px: tuple[float, float]
    depth_m: float  # estimated distance in meters (relative scale unless calibrated)
    extra: dict = field(default_factory=dict)


class DetectorDepthEstimator:
    """
    Wraps a YOLO detector and a monocular depth model behind one call.

    Swap `detector_weights` / `depth_model_name` for whatever checkpoints
    you have locally — the rest of the pipeline only depends on the
    Detection dataclass shape, not on these specific models.

    detector_weights: pass an open-vocabulary checkpoint (e.g.
        "yolov8s-worldv2.pt") to detect classes a stock COCO-trained YOLO
        never can — COCO has no "door", "stairs", "wall", "cabinet", or
        "obstacle" class, which are exactly the highest-priority classes
        in pruning.py's AFFORDANCE_PRIORITY (door/stairs are force-kept
        by pruning.py regardless of score, but a COCO detector can never
        emit those labels for that force-keep to act on in the first
        place). A "-world" checkpoint name triggers open_vocab_classes
        below via YOLO.set_classes(); a standard checkpoint ignores it.

    open_vocab_classes: text prompts for a "-world" checkpoint. Defaults
        to graph_builder.DEFAULT_CLASS_VOCAB (minus "unknown", which is
        pruning.py's fallback label, not a real detectable class) so the
        detector's vocabulary and the rest of the pipeline's vocabulary
        can't silently drift apart the way "sofa" (this project's vocab)
        vs. "couch" (stock COCO's actual label) did.
    """

    def __init__(
        self,
        detector_weights: str = "yolov10n.pt",
        depth_model_name: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        device: str = "cuda",
        conf_threshold: float = 0.35,
        open_vocab_classes: list[str] | None = None,
        calibration: dict | None = None,
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        # Fitted depth correction from calibrate_depth.py. Previously that
        # script measured the error (calibration.json records a +1.37m offset
        # at webcam range) and then nothing consumed it, so every downstream
        # distance the user heard spoken was the uncorrected one.
        self.calibration = calibration

        # Lazy imports so this file can be inspected/tested without the
        # heavy deps installed.
        from ultralytics import YOLO
        from transformers import pipeline as hf_pipeline

        self.detector = YOLO(detector_weights)

        if "-world" in detector_weights:
            from graph_builder import DEFAULT_CLASS_VOCAB

            classes = open_vocab_classes or [c for c in DEFAULT_CLASS_VOCAB if c != "unknown"]
            self.detector.set_classes(classes)

        self.depth_estimator = hf_pipeline(
            task="depth-estimation", model=depth_model_name, device=device
        )

    def run(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- detection ---
        results = self.detector.predict(frame_rgb, conf=self.conf_threshold, verbose=False)[0]

        # --- monocular depth map for the whole frame ---
        depth_out = self.depth_estimator(Image.fromarray(frame_rgb))
        # NOTE: depth_out["depth"] is a PIL Image rescaled to 0-255 for
        # DISPLAY only — it is not real depth. The actual per-pixel depth
        # values (meters, for a metric checkpoint) are in
        # depth_out["predicted_depth"], a torch tensor shaped [1, H', W'].
        depth_map = depth_out["predicted_depth"].squeeze().cpu().numpy()  # H'xW', meters
        depth_map = cv2.resize(depth_map, (w, h))

        detections: list[Detection] = []
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            label = self.detector.names[cls_id]
            conf = float(box.conf[0].item())

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Sample depth from a small patch around the centroid rather than
            # a single pixel — more robust to noisy depth maps at edges.
            patch = self._depth_patch(depth_map, (x1, y1, x2, y2))
            if patch.size == 0:
                continue
            finite = patch[np.isfinite(patch)]
            if finite.size == 0:
                # Depth model returned nan/inf over the whole object — better
                # to drop the detection than to feed a nan into pruning,
                # where comparisons against prune_threshold are all False and
                # the object silently disappears with no explanation.
                continue
            depth_val = apply_calibration(float(np.median(finite)), self.calibration)

            detections.append(
                Detection(
                    obj_id=i,
                    label=label,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    centroid_px=(cx, cy),
                    depth_m=depth_val,
                )
            )

        return detections

    @staticmethod
    def _depth_patch(depth_map: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
        """
        Samples depth from the central region of the object's own bounding
        box, sized as a fraction of the box rather than a fixed 5px radius.

        A fixed radius is wrong in both directions: on a distant 12px-wide
        door it reaches past the object into background wall depth, and on a
        near-field sofa filling half the frame it samples a 10x10 speck whose
        median is dominated by depth-map noise. Scaling with the box keeps
        the sample inside the object at every scale, and clamping to the box
        guarantees background pixels are never mixed in.
        """
        h, w = depth_map.shape
        bx1, by1, bx2, by2 = bbox_xyxy
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        # Central ~30% of the box in each dimension, at least 1px, at most 24px.
        rx = int(min(24, max(1, 0.15 * (bx2 - bx1))))
        ry = int(min(24, max(1, 0.15 * (by2 - by1))))
        x0, x1 = max(0, int(cx - rx)), min(w, int(cx + rx) + 1)
        y0, y1 = max(0, int(cy - ry)), min(h, int(cy + ry) + 1)
        if x0 >= x1 or y0 >= y1:
            return np.empty((0,), dtype=depth_map.dtype)
        return depth_map[y0:y1, x0:x1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run detection + depth on a single image.")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--detector-weights",
        default="yolov10n.pt",
        help='e.g. "yolov8s-worldv2.pt" for open-vocabulary detection (see DetectorDepthEstimator docstring)',
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument(
        "--calibration",
        default=None,
        help="path to calibration.json from calibrate_depth.py; applies the fitted depth correction",
    )
    args = parser.parse_args()

    frame = cv2.imread(args.image_path)
    if frame is None:
        raise FileNotFoundError(args.image_path)

    pipeline = DetectorDepthEstimator(
        device=args.device,
        detector_weights=args.detector_weights,
        conf_threshold=args.conf,
        calibration=load_calibration(args.calibration),
    )
    dets = pipeline.run(frame)
    for d in dets:
        print(f"{d.label:15s} conf={d.confidence:.2f} depth={d.depth_m:.2f} centroid={d.centroid_px}")