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
    """

    def __init__(
        self,
        detector_weights: str = "yolov10n.pt",
        depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cuda",
        conf_threshold: float = 0.35,
    ):
        self.conf_threshold = conf_threshold
        self.device = device

        # Lazy imports so this file can be inspected/tested without the
        # heavy deps installed.
        from ultralytics import YOLO
        from transformers import pipeline as hf_pipeline

        self.detector = YOLO(detector_weights)
        self.depth_estimator = hf_pipeline(
            task="depth-estimation", model=depth_model_name, device=device
        )

    def run(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- detection ---
        results = self.detector.predict(frame_rgb, conf=self.conf_threshold, verbose=False)[0]

        # --- monocular depth map for the whole frame ---
        depth_out = self.depth_estimator(frame_rgb)  # HF pipeline accepts PIL/np/array-like
        depth_map = np.array(depth_out["depth"])  # HxW, relative depth units
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
            patch = self._depth_patch(depth_map, cx, cy, radius=5)
            depth_val = float(np.median(patch))

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
    def _depth_patch(depth_map: np.ndarray, cx: float, cy: float, radius: int) -> np.ndarray:
        h, w = depth_map.shape
        x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius))
        y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius))
        return depth_map[y0:y1, x0:x1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run detection + depth on a single image.")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frame = cv2.imread(args.image_path)
    if frame is None:
        raise FileNotFoundError(args.image_path)

    pipeline = DetectorDepthEstimator(device=args.device)
    dets = pipeline.run(frame)
    for d in dets:
        print(f"{d.label:15s} conf={d.confidence:.2f} depth={d.depth_m:.2f} centroid={d.centroid_px}")
