"""
Assertion-backed regression tests for Phases 2-4 and 6, using hand-built
Detection objects so nothing here needs YOLO or Depth-Anything weights.

This replaces the old test_synthetic.py, which printed its results and
asserted nothing — it could not fail, so it could not catch the label-
mapping and pruning-threshold regressions the README's known-issues
section describes. Each test below pins one of those behaviours.

Run with:  pytest -q
"""

from __future__ import annotations

import json
import math

import pytest

from detect_depth import Detection, apply_calibration
from graph_builder import build_graph, MAX_DEPTH_M
from pruning import (
    AFFORDANCE_PRIORITY,
    PruningConfig,
    edge_weight,
    heading_alignment,
    is_forced_keep,
    prune_graph,
)
from encoders import encode_haptic_matrix, encode_spatial_audio
from navigation_planner import (
    azimuth_to_clock,
    format_distance,
    generate_instructions,
    instructions_to_speech_text,
)

FRAME_W, FRAME_H = 640, 480


@pytest.fixture
def detections() -> list[Detection]:
    """A fixed synthetic scene: door and person roughly ahead, chair far
    left, plant and table off to the right."""
    return [
        Detection(0, "door", 0.95, (300, 100, 360, 300), (330, 200), depth_m=3.0),
        Detection(1, "chair", 0.88, (50, 300, 150, 420), (100, 360), depth_m=1.2),
        Detection(2, "plant", 0.70, (500, 250, 560, 400), (530, 325), depth_m=2.5),
        Detection(3, "person", 0.92, (280, 200, 340, 400), (310, 300), depth_m=1.8),
        Detection(4, "table", 0.80, (400, 300, 550, 420), (475, 360), depth_m=2.0),
    ]


@pytest.fixture
def labels(detections) -> list[str]:
    return [d.label for d in detections]


# --------------------------------------------------------------------------
# heading / scoring maths
# --------------------------------------------------------------------------

def test_heading_alignment_peaks_dead_ahead():
    assert heading_alignment(0.0, 0.0, gamma=2.0) == pytest.approx(1.0)


def test_heading_alignment_is_zero_perpendicular_and_behind():
    assert heading_alignment(math.pi / 2, 0.0, gamma=2.0) == pytest.approx(0.0, abs=1e-12)
    assert heading_alignment(math.pi, 0.0, gamma=2.0) == pytest.approx(0.0)
    assert heading_alignment(-3 * math.pi / 4, 0.0, gamma=2.0) == pytest.approx(0.0)


def test_heading_alignment_wraps_across_pi():
    """A bearing of +179 deg and a heading of -179 deg are 2 deg apart, not 358."""
    near_pi = math.radians(179)
    assert heading_alignment(near_pi, -near_pi, gamma=2.0) == pytest.approx(0.0)
    # same test where both point backwards but agree with each other
    assert heading_alignment(near_pi, near_pi, gamma=2.0) == pytest.approx(1.0)


def test_ablating_every_term_makes_weight_unity():
    cfg = PruningConfig(use_affordance=False, use_distance=False, use_heading=False)
    assert edge_weight(0.9, math.pi / 3, "plant", 0.0, cfg) == pytest.approx(1.0)
    assert cfg.ablation_name() == "none"


def test_ablating_heading_removes_bearing_dependence():
    cfg = PruningConfig(use_heading=False)
    ahead = edge_weight(0.2, 0.0, "chair", 0.0, cfg)
    aside = edge_weight(0.2, math.pi / 2, "chair", 0.0, cfg)
    assert ahead == pytest.approx(aside)
    assert cfg.ablation_name() == "affordance+distance"


def test_forced_keep_covers_exactly_the_safety_critical_classes():
    forced = {lbl for lbl in AFFORDANCE_PRIORITY if is_forced_keep(lbl)}
    assert forced == {"door", "stairs", "obstacle"}


# --------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------

def test_graph_has_ego_node_plus_one_node_per_detection(detections):
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    assert graph.ego_node_idx == 0
    assert graph.x.shape[0] == len(detections) + 1


def test_ego_node_is_never_an_edge_destination(detections):
    """pruning.py maps dst -> label via `dst - 1`; that is only sound while
    the ego node stays purely a source."""
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    assert int(graph.edge_index[1].min()) >= 1


def test_object_edges_can_be_switched_off(detections):
    n = len(detections)
    full = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    lean = build_graph(detections, frame_size=(FRAME_W, FRAME_H), include_object_edges=False)
    assert full.edge_index.shape[1] == n * (n - 1) + n
    assert lean.edge_index.shape[1] == n  # ego -> object only


def test_empty_scene_yields_ego_only_graph():
    graph = build_graph([], frame_size=(FRAME_W, FRAME_H))
    assert graph.x.shape[0] == 1
    assert graph.edge_index.shape[1] == 0


def test_ego_bearing_matches_encoder_azimuth_convention(detections):
    """graph_builder's ego bearing (radians) and encoders' azimuth (degrees)
    must describe the same physical angle, or 'heading' during pruning and
    'azimuth' in the spoken output silently disagree."""
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    n = len(detections)
    ego_edges = graph.edge_attr[-n:]  # ego edges are appended last
    for det, attr in zip(detections, ego_edges):
        nx_ = det.centroid_px[0] / FRAME_W
        assert float(attr[0]) == pytest.approx(det.depth_m / MAX_DEPTH_M)
        assert math.degrees(float(attr[1])) == pytest.approx((nx_ - 0.5) * 180.0, abs=1e-4)


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------

def test_pruning_drops_off_heading_clutter(detections, labels):
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0)
    kept = [labels[i] for i in pruned.kept_node_indices]
    # chair (far left) and plant (far right) fall outside the attention cone
    assert set(kept) == {"door", "person", "table"}
    assert "chair" not in kept and "plant" not in kept


def test_pruned_graph_carries_no_edges(detections, labels):
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0)
    assert pruned.edge_index.shape[1] == 0


def test_kept_node_indices_map_back_to_original_labels(detections, labels):
    """Regression for README known-issue #3: encoders must not assume pruned
    node i corresponds to detections_labels[i]."""
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(
        graph, detections_labels=labels, heading_rad=0.0, config=PruningConfig(max_nodes=2)
    )
    assert [labels[i] for i in pruned.kept_node_indices] == ["door", "person"]


def test_no_subthreshold_nonforced_node_survives(detections, labels):
    """Regression for README known-issues #5: a scene with fewer objects than
    max_nodes must still prune on score, not keep everything."""
    cfg = PruningConfig(max_nodes=100)
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0, config=cfg)
    assert pruned.x.shape[0] < len(detections)
    for idx, score in zip(pruned.kept_node_indices, pruned.kept_importance.tolist()):
        assert is_forced_keep(labels[idx]) or score >= cfg.prune_threshold


def test_safety_critical_class_survives_even_when_off_heading():
    """A door far off to the side scores below threshold but must be kept."""
    dets = [Detection(0, "door", 0.9, (0, 100, 40, 300), (20, 200), depth_m=9.0)]
    graph = build_graph(dets, frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(graph, detections_labels=["door"], heading_rad=0.0)
    assert [i for i in pruned.kept_node_indices] == [0]
    assert float(pruned.kept_importance[0]) < PruningConfig.prune_threshold


def test_forced_keep_respects_the_max_nodes_budget():
    """Safety-critical nodes are admitted first but are not exempt from the
    attention budget — five doors must not produce a five-node graph when
    the user asked for at most two."""
    dets = [
        Detection(i, "door", 0.9, (x, 100, x + 40, 300), (x + 20, 200), depth_m=2.0 + i)
        for i, x in enumerate((20, 140, 300, 440, 580))
    ]
    graph = build_graph(dets, frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(
        graph, detections_labels=["door"] * 5, heading_rad=0.0,
        config=PruningConfig(max_nodes=2),
    )
    assert pruned.x.shape[0] == 2


def test_heading_rotates_the_attention_cone(detections, labels):
    """Turning the user's head right should change which clutter survives."""
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    ahead = prune_graph(graph, detections_labels=labels, heading_rad=0.0)
    right = prune_graph(graph, detections_labels=labels, heading_rad=math.radians(60))

    kept_ahead = {labels[i] for i in ahead.kept_node_indices}
    kept_right = {labels[i] for i in right.kept_node_indices}

    assert "person" in kept_ahead and "person" not in kept_right
    assert "table" in kept_right  # off to the right, closer to the new heading
    assert kept_ahead != kept_right


def test_config_is_immutable():
    """PruningConfig used to be a mutable default argument shared by every
    call to prune_graph."""
    with pytest.raises(Exception):
        PruningConfig().tau = 0.99


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------

def test_spatial_audio_is_priority_ordered_and_correctly_labelled(detections, labels):
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0)
    events = json.loads(encode_spatial_audio(pruned, labels, FRAME_W, FRAME_H))["events"]

    assert [e["label"] for e in events][0] == "door"
    priorities = [e["priority"] for e in events]
    assert priorities == sorted(priorities, reverse=True)
    # distances are reported in meters, not in normalized graph units
    by_label = {e["label"]: e for e in events}
    assert by_label["door"]["distance_m"] == pytest.approx(3.0)
    assert by_label["person"]["azimuth_deg"] == pytest.approx((310 / FRAME_W - 0.5) * 180, abs=0.1)


def test_haptic_matrix_places_cells_and_stays_in_bounds(detections, labels):
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0)
    matrix = encode_haptic_matrix(pruned, grid_size=16)
    assert matrix.shape == (16, 16)
    assert 0 < (matrix > 0).sum() <= pruned.x.shape[0]
    assert matrix.max() == 255  # highest-importance node saturates


def test_haptic_matrix_of_empty_graph_is_empty():
    graph = build_graph([], frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(graph, detections_labels=[], heading_rad=0.0)
    assert encode_haptic_matrix(pruned, grid_size=8).sum() == 0


# --------------------------------------------------------------------------
# navigation planner
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "azimuth,expected",
    [(0.0, "12 o'clock"), (-90.0, "9 o'clock"), (90.0, "3 o'clock"),
     (-30.0, "11 o'clock"), (30.0, "1 o'clock")],
)
def test_azimuth_to_clock(azimuth, expected):
    assert azimuth_to_clock(azimuth) == expected


@pytest.mark.parametrize(
    "distance,expected",
    [(1.0, "1 meter"), (2.0, "2 meters"), (1.24, "1 meter"),
     (1.3, "1.5 meters"), (4.4, "4 meters")],
)
def test_format_distance(distance, expected):
    assert format_distance(distance) == expected


def test_instructions_are_capped_and_urgent_at_close_range(detections, labels):
    close = list(detections)
    close[3] = Detection(3, "person", 0.92, (280, 200, 340, 400), (310, 300), depth_m=0.4)
    graph = build_graph(close, frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0)
    audio = encode_spatial_audio(pruned, labels, FRAME_W, FRAME_H)

    instructions = generate_instructions(audio, max_instructions=2)
    assert len(instructions) <= 2
    speech = instructions_to_speech_text(instructions)
    assert speech and speech == speech.strip()

    urgent = generate_instructions(audio, max_instructions=5)
    person = [i for i in urgent if i.label == "person"]
    assert person and "very close" in person[0].text


def test_generate_instructions_handles_an_empty_scene():
    assert generate_instructions(json.dumps({"events": []})) == []
    assert instructions_to_speech_text([]) == ""


# --------------------------------------------------------------------------
# depth calibration
# --------------------------------------------------------------------------

def test_apply_calibration_is_a_noop_without_a_calibration():
    assert apply_calibration(2.5, None) == 2.5
    assert apply_calibration(2.5, {}) == 2.5


def test_apply_calibration_corrects_toward_the_measured_truth():
    """calibration.json in the repo was fitted from three real measurements;
    the correction must bring the predictions back near the true distances."""
    calib = {"offset": 1.3666777610778829, "scale": 0.8915988206863394}
    for true_m, predicted_m in ((0.5, 1.7784525156021118),
                                (1.0, 2.3263258934020996),
                                (1.5, 2.670051336288452)):
        assert apply_calibration(predicted_m, calib) == pytest.approx(true_m, abs=0.12)


def test_apply_calibration_never_returns_negative_depth():
    calib = {"offset": 1.37, "scale": 0.89}
    assert apply_calibration(0.1, calib) == 0.0
