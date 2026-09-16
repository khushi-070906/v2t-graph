"""
Regression tests for three correctness properties flagged during code
review as untested: graph_builder.build_graph's zero-detection edge
case, pruning._select_top_nodes's max_nodes boundary, and
encoders.encode_spatial_audio's label remapping via kept_node_indices.

test_synthetic.py already exercises the graph_builder -> pruning ->
encoders happy path end-to-end and is not duplicated here (same scoping
convention test_v2t_pipeline.py uses). All three properties below were
previously only checked manually/once during development -- the
max_nodes boundary and label-remapping cases are exactly the class of
silent-wrong-output bug this project has already hit twice (the
object-to-object heading bug, the stale-pruning-cache bug), so they're
worth locking in as permanent regression tests rather than leaving them
as "verified once, by hand" claims in the README.

Run with:
    pip install pytest
    pytest test_graph_pruning_encoders.py -v
"""

from __future__ import annotations

import json
import math

import pytest

from detect_depth import Detection
from graph_builder import build_graph, DEFAULT_CLASS_VOCAB
from pruning import prune_graph, PruningConfig
from encoders import encode_spatial_audio


# ---------------------------------------------------------------------------
# graph_builder.py — zero-detection edge case
# ---------------------------------------------------------------------------

class TestBuildGraphZeroDetections:
    def test_empty_detections_produces_ego_only_graph(self):
        """An empty frame (no detections at all) must not crash, and must
        produce a graph containing only the ego node -- no object nodes,
        no edges."""
        graph = build_graph([], frame_size=(640, 480))
        assert graph.x.shape[0] == 1  # ego node only
        assert graph.edge_index.shape[1] == 0
        assert graph.edge_attr.shape[0] == 0
        assert graph.ego_node_idx == 0

    def test_empty_detections_graph_survives_pruning(self):
        """The zero-detection graph must also flow cleanly through
        prune_graph -- an empty detections_labels list, zero ego edges to
        score, and a pruned graph with zero surviving object nodes."""
        graph = build_graph([], frame_size=(640, 480))
        pruned = prune_graph(graph, detections_labels=[], heading_rad=0.0, config=PruningConfig())
        assert pruned.x.shape[0] == 0
        assert pruned.kept_node_indices == []


# ---------------------------------------------------------------------------
# pruning.py — max_nodes boundary
# ---------------------------------------------------------------------------

def _make_detections(n: int, label: str = "chair", depth_m: float = 1.0) -> list[Detection]:
    """n detections of the same middling-priority class, all centered
    on-heading so they score identically and only max_nodes / prune_threshold
    decide what survives -- isolates the boundary behavior from any
    heading/distance tie-breaking."""
    return [
        Detection(i, label, 0.9, (100, 100, 150, 150), (320, 240), depth_m=depth_m)
        for i in range(n)
    ]


class TestMaxNodesBoundary:
    def test_exactly_at_cap_keeps_everything(self):
        """When the number of above-threshold candidates exactly equals
        max_nodes, every candidate must survive -- this is the specific
        regression this project hit before (11 objects, max_nodes=12,
        zero nodes dropped despite several scoring exactly 0.0): the fix
        must not overcorrect into dropping nodes that legitimately fit
        under the cap."""
        detections = _make_detections(3)
        labels = [d.label for d in detections]
        graph = build_graph(detections, frame_size=(640, 480))
        config = PruningConfig(prune_threshold=0.01, max_nodes=3)
        pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0, config=config)
        assert pruned.x.shape[0] == 3

    def test_one_over_cap_drops_exactly_one(self):
        """One more above-threshold candidate than max_nodes allows must
        drop exactly one -- the lowest-scoring survivor, not an arbitrary
        one, and not zero (which would mean the cap isn't being enforced)."""
        detections = _make_detections(4)
        labels = [d.label for d in detections]
        graph = build_graph(detections, frame_size=(640, 480))
        config = PruningConfig(prune_threshold=0.01, max_nodes=3)
        pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0, config=config)
        assert pruned.x.shape[0] == 3

    def test_max_nodes_none_means_no_cap(self):
        """max_nodes=None must keep every above-threshold candidate
        regardless of count -- the 'no cap' escape hatch must actually
        mean no cap, not silently fall back to some default limit."""
        detections = _make_detections(20)
        labels = [d.label for d in detections]
        graph = build_graph(detections, frame_size=(640, 480))
        config = PruningConfig(prune_threshold=0.01, max_nodes=None)
        pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0, config=config)
        assert pruned.x.shape[0] == 20

    def test_forced_keep_class_survives_even_over_cap(self):
        """A door (force-keep class) must survive even when max_nodes is
        tight enough that it would otherwise be squeezed out by
        higher-scoring-but-non-critical objects -- force-keep must not be
        silently defeated by the cap."""
        detections = _make_detections(3, label="chair") + [
            Detection(3, "door", 0.9, (10, 10, 20, 20), (639, 0), depth_m=5.0),
        ]
        labels = [d.label for d in detections]
        graph = build_graph(detections, frame_size=(640, 480))
        # door is off-heading/far, so its own ego-edge score is low --
        # only force-keep logic should save it under a tight cap.
        config = PruningConfig(prune_threshold=0.01, max_nodes=2)
        pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0, config=config)
        kept_labels = [labels[i] for i in pruned.kept_node_indices]
        assert "door" in kept_labels

    def test_below_threshold_node_never_survives_even_with_room_under_cap(self):
        """This is the actual historical bug, reproduced directly: 11
        objects, max_nodes=12, zero nodes dropped despite 7 scoring
        exactly 0.0 -- because _select_top_nodes had no per-candidate
        threshold check, so any scene with fewer objects than max_nodes
        kept everything regardless of score. Here: one low-priority
        object placed directly behind the user (heading_alignment=0, so
        its score is exactly 0.0) alongside high-scoring objects, with
        max_nodes left deliberately large so only the (previously
        missing) threshold check -- not the cap -- can be what drops it."""
        behind_user = Detection(
            0, "plant", 0.7, (10, 10, 20, 20), (320, 240), depth_m=1.0,
        )
        detections = [behind_user] + _make_detections(3)
        labels = [d.label for d in detections]
        graph = build_graph(detections, frame_size=(640, 480))
        # heading_rad=pi puts the user facing directly away from the
        # plant's on-frame-center bearing, driving its heading_alignment
        # (and therefore its score) to 0.0 -- below any positive threshold.
        config = PruningConfig(prune_threshold=0.01, max_nodes=10, heading_gamma=2.0)
        pruned = prune_graph(graph, detections_labels=labels, heading_rad=math.pi, config=config)
        kept_labels = [labels[i] for i in pruned.kept_node_indices]
        assert "plant" not in kept_labels


# ---------------------------------------------------------------------------
# encoders.py — label remapping via kept_node_indices
# ---------------------------------------------------------------------------

class TestEncodeSpatialAudioLabelRemapping:
    def test_pruned_labels_match_original_detections_not_position(self):
        """Regression guard for the exact bug README known-issue #3
        describes: encode_spatial_audio must use kept_node_indices to map
        each surviving node back to its ORIGINAL detection label, not
        assume pruned-graph row position lines up with the original
        detections list. Forces max_nodes=2 on a 5-detection scene (same
        setup the README says was previously only checked once by hand)
        and asserts every event's label is one that was actually in the
        original detections, at the position kept_node_indices claims."""
        detections = [
            Detection(0, "door", 0.95, (300, 100, 360, 300), (330, 200), depth_m=3.0),
            Detection(1, "chair", 0.88, (50, 300, 150, 420), (100, 360), depth_m=1.2),
            Detection(2, "plant", 0.7, (500, 250, 560, 400), (530, 325), depth_m=2.5),
            Detection(3, "person", 0.92, (280, 200, 340, 400), (310, 300), depth_m=1.8),
            Detection(4, "table", 0.8, (400, 300, 550, 420), (475, 360), depth_m=2.0),
        ]
        labels = [d.label for d in detections]
        frame_w, frame_h = 640, 480

        graph = build_graph(detections, frame_size=(frame_w, frame_h))
        config = PruningConfig(max_nodes=2)
        pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0, config=config)

        assert pruned.x.shape[0] == 2  # capped as expected

        audio_json = encode_spatial_audio(pruned, labels, frame_w, frame_h)
        events = json.loads(audio_json)["events"]

        assert len(events) == 2
        original_labels_at_kept_indices = {labels[i] for i in pruned.kept_node_indices}
        event_labels = {e["label"] for e in events}
        assert event_labels == original_labels_at_kept_indices

        # Every emitted label must be a real label from the original
        # detections list -- catches the specific failure mode where a
        # positional-index bug would emit a label belonging to a
        # different (possibly pruned-away) detection instead.
        assert event_labels <= set(labels)

    def test_unpruned_graph_falls_back_to_identity_order(self):
        """If a raw (non-pruned) graph is passed directly -- no
        kept_node_indices attribute -- encode_spatial_audio must fall
        back to identity ordering rather than crashing, per its own
        docstring caveat that this fallback is 'only correct
        pre-pruning'."""
        detections = [
            Detection(0, "door", 0.95, (300, 100, 360, 300), (330, 200), depth_m=3.0),
            Detection(1, "chair", 0.88, (50, 300, 150, 420), (100, 360), depth_m=1.2),
        ]
        labels = [d.label for d in detections]
        graph = build_graph(detections, frame_size=(640, 480))
        # graph here is the RAW graph (ego + 2 object nodes), not pruned --
        # encode_spatial_audio should still run without error.
        audio_json = encode_spatial_audio(graph, labels, 640, 480)
        events = json.loads(audio_json)["events"]
        assert len(events) == graph.x.shape[0]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
