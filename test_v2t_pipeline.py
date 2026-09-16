"""
Regression test suite for the modules that only had ad-hoc,
throwaway verification during development: tracker.py (confirmation,
occlusion persistence, collision risk), navigation_planner.py (clock/
distance formatting, instruction ordering), and calibrate_depth.py's
fitting math.

test_synthetic.py already exercises the graph_builder -> pruning ->
encoders happy path end-to-end and is NOT duplicated here. This file
covers the newer modules built after test_synthetic.py, none of which
had permanent coverage before -- each test below corresponds to a
specific correctness property that was previously only checked once,
manually, in a scratch script, and never re-run since.

Run with:
    pip install pytest
    pytest test_v2t_pipeline.py -v
"""

from __future__ import annotations

import time

import pytest

from detect_depth import Detection
from tracker import ObjectTracker
from navigation_planner import (
    azimuth_to_clock,
    format_distance,
    generate_instructions,
)
from calibrate_depth import fit_correction, apply_correction


# ---------------------------------------------------------------------------
# tracker.py — confirmation (anti-flicker)
# ---------------------------------------------------------------------------

class TestTrackerConfirmation:
    def test_single_frame_detection_never_confirmed(self):
        """A one-off detection (e.g. a borderline false positive) must
        never be reported as confirmed -- this is the anti-flicker
        guarantee live_camera_loop_tracked.py depends on."""
        tracker = ObjectTracker(confirm_frames=3, max_missed_frames=2)
        det = Detection(0, "chair", 0.4, (10, 10, 50, 50), (30, 30), depth_m=2.0)
        result = tracker.update([det], 640, 480)
        assert len(result) == 0

    def test_confirmed_after_exact_threshold(self):
        """A detection matched confirm_frames times in a row becomes
        confirmed on exactly that frame, not before or after."""
        tracker = ObjectTracker(confirm_frames=3, max_missed_frames=2)
        for i in range(3):
            det = Detection(0, "person", 0.9, (0, 0, 10, 10), (5, 5), depth_m=2.0)
            result = tracker.update([det], 640, 480)
            if i < 2:
                assert len(result) == 0, f"confirmed too early on frame {i}"
            else:
                assert len(result) == 1, f"not confirmed on frame {i} (the threshold frame)"

    def test_different_labels_do_not_match_same_track(self):
        """A 'chair' detection must never be matched to an existing
        'person' track even at the same position -- matching is gated by
        label, not just proximity."""
        tracker = ObjectTracker(confirm_frames=1, max_missed_frames=2)
        det1 = Detection(0, "person", 0.9, (0, 0, 10, 10), (5, 5), depth_m=2.0)
        tracker.update([det1], 640, 480)
        det2 = Detection(0, "chair", 0.9, (0, 0, 10, 10), (5, 5), depth_m=2.0)
        tracker.update([det2], 640, 480)
        assert len(tracker._tracks) == 2  # two distinct tracks, not one relabeled


# ---------------------------------------------------------------------------
# tracker.py — velocity and approach detection
# ---------------------------------------------------------------------------

class TestTrackerVelocity:
    def test_approaching_person_detected(self):
        tracker = ObjectTracker(confirm_frames=2, max_missed_frames=2)
        for depth in [4.0, 1.0]:
            det = Detection(0, "person", 0.9, (0, 0, 10, 10), (5, 5), depth_m=depth)
            tracker.update([det], 640, 480)
            time.sleep(0.05)
        track = list(tracker._tracks.values())[0]
        assert tracker.is_approaching(track)

    def test_stationary_object_not_approaching(self):
        tracker = ObjectTracker(confirm_frames=2, max_missed_frames=2)
        for _ in range(2):
            det = Detection(0, "table", 0.9, (0, 0, 10, 10), (5, 5), depth_m=2.0)
            tracker.update([det], 640, 480)
            time.sleep(0.05)
        track = list(tracker._tracks.values())[0]
        assert not tracker.is_approaching(track)

    def test_receding_object_not_approaching(self):
        """A receding object has nonzero velocity magnitude but the WRONG
        sign -- must not be flagged as approaching just because it's
        moving."""
        tracker = ObjectTracker(confirm_frames=2, max_missed_frames=2)
        for depth in [1.0, 3.0]:
            det = Detection(0, "chair", 0.9, (0, 0, 10, 10), (5, 5), depth_m=depth)
            tracker.update([det], 640, 480)
            time.sleep(0.05)
        track = list(tracker._tracks.values())[0]
        assert not tracker.is_approaching(track)


# ---------------------------------------------------------------------------
# tracker.py — collision risk / time-to-contact
# ---------------------------------------------------------------------------

class TestCollisionRisk:
    def test_fast_approach_is_urgent(self):
        tracker = ObjectTracker(confirm_frames=2, max_missed_frames=2)
        for depth in [4.0, 1.0]:
            det = Detection(0, "person", 0.9, (0, 0, 10, 10), (5, 5), depth_m=depth)
            tracker.update([det], 640, 480)
            time.sleep(0.1)
        track = list(tracker._tracks.values())[0]
        assert tracker.collision_risk(track) == "urgent"
        assert tracker.time_to_contact(track) is not None
        assert tracker.time_to_contact(track) > 0

    def test_stationary_object_ttc_is_none(self):
        tracker = ObjectTracker(confirm_frames=2, max_missed_frames=2)
        for _ in range(2):
            det = Detection(0, "table", 0.9, (0, 0, 10, 10), (5, 5), depth_m=2.0)
            tracker.update([det], 640, 480)
            time.sleep(0.05)
        track = list(tracker._tracks.values())[0]
        assert tracker.time_to_contact(track) is None
        assert tracker.collision_risk(track) == "none"

    def test_receding_object_ttc_is_none(self):
        """Regression guard: a naive division could produce a bogus
        positive TTC for a receding object if the sign check is missing."""
        tracker = ObjectTracker(confirm_frames=2, max_missed_frames=2)
        for depth in [1.0, 3.0]:
            det = Detection(0, "chair", 0.9, (0, 0, 10, 10), (5, 5), depth_m=depth)
            tracker.update([det], 640, 480)
            time.sleep(0.05)
        track = list(tracker._tracks.values())[0]
        assert tracker.time_to_contact(track) is None
        assert tracker.collision_risk(track) == "none"


# ---------------------------------------------------------------------------
# tracker.py — occlusion / temporal persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_occluded_track_retained_within_grace_period(self):
        tracker = ObjectTracker(confirm_frames=3, max_missed_frames=2)
        for _ in range(3):
            det = Detection(0, "door", 0.9, (100, 100, 150, 300), (125, 200), depth_m=3.0)
            tracker.update([det], 640, 480)

        # frame with zero detections -- door occluded
        tracker.update([], 640, 480)
        persistent = tracker.get_persistent_detections()
        assert len(persistent) == 1
        assert persistent[0].extra["occluded"] is True
        assert persistent[0].depth_m == 3.0  # last known depth retained

    def test_track_dropped_after_exceeding_grace_period(self):
        tracker = ObjectTracker(confirm_frames=3, max_missed_frames=2)
        for _ in range(3):
            det = Detection(0, "door", 0.9, (100, 100, 150, 300), (125, 200), depth_m=3.0)
            tracker.update([det], 640, 480)

        for _ in range(3):  # exceeds max_missed_frames=2
            tracker.update([], 640, 480)

        assert len(tracker.get_persistent_detections()) == 0

    def test_unconfirmed_track_never_persisted(self):
        """A track that hasn't hit confirm_frames yet must not appear in
        get_persistent_detections(), matching update()'s own confirmation
        gate."""
        tracker = ObjectTracker(confirm_frames=5, max_missed_frames=2)
        det = Detection(0, "door", 0.9, (100, 100, 150, 300), (125, 200), depth_m=3.0)
        tracker.update([det], 640, 480)
        assert len(tracker.get_persistent_detections()) == 0


# ---------------------------------------------------------------------------
# navigation_planner.py — clock position and distance formatting
# ---------------------------------------------------------------------------

class TestNavigationPlannerFormatting:
    @pytest.mark.parametrize(
        "azimuth,expected",
        [
            (0.0, "12 o'clock"),
            (90.0, "3 o'clock"),
            (-90.0, "9 o'clock"),
            (30.0, "1 o'clock"),
            (-30.0, "11 o'clock"),
        ],
    )
    def test_azimuth_to_clock(self, azimuth, expected):
        assert azimuth_to_clock(azimuth) == expected

    def test_format_distance_under_3m_rounds_to_half_meter(self):
        assert format_distance(1.24) == "1 meter"  # rounds to 1.0 -> singular
        assert format_distance(1.3) == "1.5 meters"

    def test_format_distance_over_3m_rounds_to_whole_meter(self):
        assert format_distance(4.6) == "5 meters"

    def test_generate_instructions_respects_priority_order(self):
        """generate_instructions must not re-sort -- it should trust
        encode_spatial_audio's existing priority ordering verbatim."""
        audio_json = (
            '{"events": ['
            '{"label": "table", "distance_m": 2.0, "azimuth_deg": 40.0, "priority": 0.2},'
            '{"label": "door", "distance_m": 3.0, "azimuth_deg": 0.0, "priority": 0.65}'
            ']}'
        )
        instructions = generate_instructions(audio_json, max_instructions=3)
        assert instructions[0].label == "table"  # first in input order, unchanged
        assert instructions[1].label == "door"

    def test_generate_instructions_respects_max_instructions_cap(self):
        audio_json = (
            '{"events": ['
            '{"label": "door", "distance_m": 3.0, "azimuth_deg": 0.0, "priority": 0.9},'
            '{"label": "chair", "distance_m": 2.0, "azimuth_deg": 10.0, "priority": 0.5},'
            '{"label": "table", "distance_m": 2.5, "azimuth_deg": 20.0, "priority": 0.3}'
            ']}'
        )
        instructions = generate_instructions(audio_json, max_instructions=2)
        assert len(instructions) == 2


# ---------------------------------------------------------------------------
# calibrate_depth.py — offset/scale fitting
# ---------------------------------------------------------------------------

class TestCalibrationFit:
    def test_recovers_known_fixed_offset(self):
        """Synthetic data with a known +0.9m offset and ~1.0 scale should
        be recovered accurately by the least-squares fit."""
        points = [(0.5, 1.45), (1.0, 1.95), (1.5, 2.42), (2.0, 2.98), (3.0, 3.95)]
        cal = fit_correction(points)
        assert cal["offset"] == pytest.approx(0.9, abs=0.1)
        assert cal["scale"] == pytest.approx(1.0, abs=0.1)
        assert cal["residual_std_m"] < 0.05

    def test_apply_correction_round_trips(self):
        points = [(0.5, 1.45), (1.0, 1.95), (1.5, 2.42), (2.0, 2.98), (3.0, 3.95)]
        cal = fit_correction(points)
        for true_m, pred_m in points:
            corrected = apply_correction(pred_m, cal)
            assert corrected == pytest.approx(true_m, abs=0.1)

    def test_inconsistent_data_produces_large_residual(self):
        """Regression guard for the real incident this project hit: mixed
        real and fabricated calibration points should produce a large
        residual_std_m, not a falsely-confident clean fit -- this is what
        should have been (and now is) the automatic signal to distrust a
        calibration file."""
        physically_inconsistent_points = [
            (0.5, 1.78), (1.0, 2.33), (1.5, 2.67),
            (0.5, 0.72), (1.0, 1.33), (1.5, 0.74),
        ]
        cal = fit_correction(physically_inconsistent_points)
        assert cal["residual_std_m"] > 0.2  # matches calibrate_depth.py's own warning threshold


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
