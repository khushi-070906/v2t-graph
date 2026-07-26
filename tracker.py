"""
Phase 5.5 — Frame-to-frame object tracking.

Sits between per-frame detection (detect_depth.py) and the graph/pruning
pipeline (graph_builder.py / pruning.py), which stay exactly as they are —
this module only adds PERSISTENCE across frames on top of them:
    - track IDs so the same physical object is recognized frame-to-frame
    - real velocity (depth_m/sec, not just a single-frame snapshot) so
      "person approaching" is an actual measurement, not a guess
    - a short confirmation window before a new detection is trusted, and a
      short grace window before a track is dropped -- this is what kills
      the frame-to-frame flicker the stateless live_camera_loop.py has on
      borderline detections

Deliberately NOT a learned tracker (no ReID embeddings, no Kalman filter,
no ByteTrack/DeepSORT dependency). Matching is greedy nearest-centroid
within the same class label, gated by a max-distance threshold. This is
the right complexity level for indoor, mostly-static, single-camera
assistive navigation -- objects don't move fast or unpredictably enough
between ~10-30fps frames to need motion-model prediction, and pulling in
a heavier tracker would be new project surface area, not a fix for what's
actually broken (see live_camera_loop.py's docstring on why it stayed
stateless -- this module is what closes that specific gap without
touching the graph/pruning core).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from detect_depth import Detection


@dataclass
class Track:
    track_id: int
    label: str
    centroid_px: tuple[float, float]
    depth_m: float
    last_seen: float                  # time.time() of last matched detection
    first_seen: float
    hits: int = 1                     # consecutive frames matched (confirmation)
    misses: int = 0                   # consecutive frames NOT matched (grace period)
    depth_velocity_mps: float = 0.0   # smoothed d(depth)/dt, negative = approaching
    confirmed: bool = False           # True once hits >= confirm_frames


class ObjectTracker:
    """
    Call update(detections) once per frame with the raw Detection list from
    DetectorDepthEstimator.run(). Returns the SAME list of Detections
    (unmodified) plus a parallel dict {obj_id: Track} for whichever
    detections matched an existing or newly-confirmed track -- callers that
    don't care about tracking can ignore the second return value entirely
    and use the pipeline exactly as before.
    """

    def __init__(
        self,
        max_match_distance_norm: float = 0.15,
        confirm_frames: int = 3,
        max_missed_frames: int = 5,
        velocity_smoothing: float = 0.6,
    ):
        """
        max_match_distance_norm: max centroid distance (normalized 0-1
            frame-diagonal units) for a detection to match an existing
            track of the same label. Tune down if fast-moving objects in a
            cluttered scene start swapping track IDs; tune up if a
            slow-panning camera loses tracks it shouldn't.
        confirm_frames: a track must be matched this many consecutive
            frames before `confirmed=True` -- this is what suppresses a
            one-frame false-positive detection from ever being spoken.
        max_missed_frames: a track survives this many consecutive
            unmatched frames (e.g. brief occlusion, one bad detection)
            before being dropped -- this is what suppresses flicker from a
            detection that drops out for a frame or two but is still
            really there.
        velocity_smoothing: exponential moving average weight (0-1, higher
            = more responsive to the latest frame, lower = smoother/more
            lag) for depth_velocity_mps.
        """
        self.max_match_distance_norm = max_match_distance_norm
        self.confirm_frames = confirm_frames
        self.max_missed_frames = max_missed_frames
        self.velocity_smoothing = velocity_smoothing
        self._tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(
        self, detections: list[Detection], frame_w: int, frame_h: int
    ) -> dict[int, Track]:
        """
        detections: this frame's raw Detection list (obj_id is per-frame,
            NOT stable across frames -- do not rely on it for identity).
        Returns {detection_index_in_this_frame's_list: Track} for every
        detection that matched a track confirmed this frame (i.e. only
        confirmed, still-alive tracks are returned -- a brand new,
        not-yet-confirmed detection won't appear here yet).
        """
        now = time.time()
        diag = math.hypot(frame_w, frame_h)
        unmatched_track_ids = set(self._tracks.keys())
        detection_to_track: dict[int, Track] = {}

        for i, det in enumerate(detections):
            ncx = det.centroid_px[0] / frame_w
            ncy = det.centroid_px[1] / frame_h

            best_track_id = None
            best_dist = self.max_match_distance_norm
            for tid in unmatched_track_ids:
                track = self._tracks[tid]
                if track.label != det.label:
                    continue
                tcx = track.centroid_px[0] / frame_w
                tcy = track.centroid_px[1] / frame_h
                dist = math.hypot(ncx - tcx, ncy - tcy)
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = tid

            if best_track_id is not None:
                track = self._tracks[best_track_id]
                dt = max(1e-3, now - track.last_seen)
                raw_velocity = (det.depth_m - track.depth_m) / dt
                track.depth_velocity_mps = (
                    self.velocity_smoothing * raw_velocity
                    + (1 - self.velocity_smoothing) * track.depth_velocity_mps
                )
                track.centroid_px = det.centroid_px
                track.depth_m = det.depth_m
                track.last_seen = now
                track.hits += 1
                track.misses = 0
                if track.hits >= self.confirm_frames:
                    track.confirmed = True
                unmatched_track_ids.discard(best_track_id)
            else:
                new_track = Track(
                    track_id=self._next_id,
                    label=det.label,
                    centroid_px=det.centroid_px,
                    depth_m=det.depth_m,
                    last_seen=now,
                    first_seen=now,
                )
                self._tracks[self._next_id] = new_track
                track = new_track
                self._next_id += 1

            if track.confirmed:
                detection_to_track[i] = track

        # age out tracks that weren't matched this frame
        for tid in unmatched_track_ids:
            track = self._tracks[tid]
            track.misses += 1
            if track.misses > self.max_missed_frames:
                del self._tracks[tid]

        return detection_to_track

    def is_approaching(self, track: Track, threshold_mps: float = 0.15) -> bool:
        """True if a track's depth is decreasing faster than threshold_mps
        (i.e. genuinely getting closer, not just measurement noise)."""
        return track.depth_velocity_mps < -threshold_mps
