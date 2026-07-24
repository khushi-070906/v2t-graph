"""
Phase 3 — Adaptive graph pruning (the paper's actual contribution).

Everything upstream (detect_depth.py, graph_builder.py) is standard
detection + depth + geometric graph construction. This module is where
the claimed novelty has to live, so the formula is written out explicitly
and documented — you want to be able to point at one function and say
"this is the method" rather than "an attention mechanism, generally."

Edge weight w(i -> j) is a function of three signals:
    1. distance:            closer objects matter more (inverse falloff)
    2. heading alignment:   objects near the user's current heading vector
                             matter more than objects off to the side
    3. affordance priority: some object classes matter more regardless of
                             position (e.g. a "door" vs a "plant")

w(i->j) = affordance_weight(j) * exp(-distance / tau) * heading_alignment(i, j, heading)

IMPORTANT — heading is scored from the ego node, not between objects:
graph_builder.py attaches an explicit ego/user node (graph.ego_node_idx,
normally 0) with ego -> object edges carrying the object's true bearing
from the user. This module only scores those ego edges for node
importance/pruning: an earlier version scored object-to-object edges
instead, which measures "is object B to the right of object A in the
photo" — a meaningless quantity for "should the user pay attention to
this," and one that silently zeroed every object's importance whenever
the (arbitrary) object-to-object edge angle exceeded 90 degrees from
`heading_rad`, regardless of how close or important the object actually
was. Object-to-object edges (also present in the input graph, for
possible future GNN structural use) are ignored here.

The final pruned graph carries no edges at all — its only job is to
select and score object NODES. Per-node importance (the ego-edge weight
that got each surviving node kept) is attached directly as
`kept_importance`, which encoders.py reads instead of reconstructing a
score from edge topology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from graph_builder import DEFAULT_CLASS_VOCAB


# Per-class affordance priority. Tune per deployment / user study.
# Higher = more important to surface regardless of position.
AFFORDANCE_PRIORITY = {
    "door": 1.0,
    "stairs": 1.0,
    "obstacle": 0.9,
    "person": 0.8,
    "chair": 0.5,
    "table": 0.5,
    "sofa": 0.4,
    "bed": 0.4,
    "cabinet": 0.3,
    "wall": 0.3,
    "plant": 0.2,
    "tv": 0.15,
    "unknown": 0.3,
}


@dataclass
class PruningConfig:
    tau: float = 0.35          # distance decay constant (graph is in normalized coords)
    heading_gamma: float = 2.0  # how sharply off-heading objects get down-weighted
    prune_threshold: float = 0.15
    max_nodes: int | None = 12  # hard cap on nodes kept after pruning; None = no cap


def heading_alignment(angle_rad: float, heading_rad: float, gamma: float) -> float:
    """
    1.0 when the edge angle matches the user's heading exactly, decaying
    with angular deviation. cos-based falloff, raised to gamma to sharpen
    or soften the cone of attention.
    """
    delta = angle_rad - heading_rad
    delta = (delta + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
    raw = max(0.0, math.cos(delta))  # 0 for anything behind the user
    return raw ** gamma


def edge_weight(
    distance: float,
    angle_rad: float,
    dst_label: str,
    heading_rad: float,
    config: PruningConfig,
) -> float:
    affordance = AFFORDANCE_PRIORITY.get(dst_label, AFFORDANCE_PRIORITY["unknown"])
    distance_term = math.exp(-distance / config.tau)
    heading_term = heading_alignment(angle_rad, heading_rad, config.heading_gamma)
    return affordance * distance_term * heading_term


def prune_graph(
    graph: Data,
    detections_labels: list[str],
    heading_rad: float,
    config: PruningConfig = PruningConfig(),
) -> Data:
    """
    Scores every ego -> object edge with edge_weight() above, drops nodes
    whose score falls below `prune_threshold` (unless the node's class is
    high-affordance, e.g. door/stairs, in which case it's force-kept
    regardless of score), then caps the survivors at `max_nodes`.
    """
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    ego_idx = getattr(graph, "ego_node_idx", None)

    weights = []
    for e in range(edge_index.shape[1]):
        dst = edge_index[1, e].item()
        distance, angle = edge_attr[e].tolist()
        # dst is always an object node (>=1) when an ego node is present,
        # since graph_builder.py never makes the ego node an edge
        # destination — map back to the 0-based detections_labels index.
        label_idx = dst - 1 if ego_idx is not None else dst
        label = detections_labels[label_idx]
        w = edge_weight(distance, angle, label, heading_rad, config)
        weights.append(w)

    weights_t = torch.tensor(weights, dtype=torch.float32) if weights else torch.zeros((0,))

    if ego_idx is not None and edge_index.numel() > 0:
        is_ego_edge = edge_index[0] == ego_idx
        keep_mask = (weights_t >= config.prune_threshold) & is_ego_edge
    else:
        keep_mask = weights_t >= config.prune_threshold

    pruned_edge_index = edge_index[:, keep_mask]
    pruned_weights = weights_t[keep_mask]

    node_importance = _node_importance(
        pruned_edge_index, pruned_weights, num_nodes=graph.x.shape[0]
    )

    keep_nodes = _select_top_nodes(
        node_importance, detections_labels, config.max_nodes, config.prune_threshold, ego_idx=ego_idx
    )

    new_graph = _subgraph(graph, keep_nodes, node_importance)

    # kept_node_indices maps each surviving row back to its ORIGINAL,
    # 0-based detections_labels index, exactly as encoders.py expects.
    new_graph.kept_node_indices = (
        [i - 1 for i in keep_nodes] if ego_idx is not None else list(keep_nodes)
    )

    _assert_pruning_invariant(keep_nodes, node_importance, detections_labels, config, ego_idx)

    return new_graph


def _assert_pruning_invariant(
    keep_nodes: list[int],
    node_importance: torch.Tensor,
    labels: list[str],
    config: PruningConfig,
    ego_idx: int | None,
) -> None:
    """
    Fails loudly if a non-forced node with importance below prune_threshold
    ever survives into the pruned graph. This exact silent failure (11
    objects, max_nodes=12, zero nodes dropped despite 7 of them scoring
    exactly 0.0) previously shipped a broken 0%-node-compression graph
    without raising — see this module's git history / README known-issues
    for the real-photo repro that caught it. This assertion turns any
    recurrence (e.g. a stale/shadowed copy of _select_top_nodes missing its
    threshold check) into an immediate, obvious crash instead of quietly
    bad output.
    """
    offset = 1 if ego_idx is not None else 0
    for i in keep_nodes:
        label = labels[i - offset]
        is_forced = AFFORDANCE_PRIORITY.get(label, 0) >= 0.9
        score = node_importance[i].item()
        if not is_forced and score < config.prune_threshold:
            raise AssertionError(
                f"pruning invariant violated: node {i} (label={label!r}) kept with "
                f"importance={score:.4f} < prune_threshold={config.prune_threshold} "
                f"and is not a forced-keep affordance class. This means a node that "
                f"should have been pruned survived — check for a stale/shadowed "
                f"pruning.py (see module docstring)."
            )


def _node_importance(edge_index: torch.Tensor, weights: torch.Tensor, num_nodes: int) -> torch.Tensor:
    importance = torch.zeros(num_nodes)
    if edge_index.numel() == 0:
        return importance
    dst = edge_index[1]
    importance.scatter_add_(0, dst, weights)
    return importance


def _select_top_nodes(
    importance: torch.Tensor,
    labels: list[str],
    max_nodes: int | None,
    prune_threshold: float,
    ego_idx: int | None = None,
) -> list[int]:
    # Always keep high-affordance nodes (doors, stairs) even if their ego
    # edge fell below threshold, since these are safety-critical regardless
    # of measured importance.
    offset = 1 if ego_idx is not None else 0
    forced_keep = {
        i + offset for i, lbl in enumerate(labels) if AFFORDANCE_PRIORITY.get(lbl, 0) >= 0.9
    }

    candidate_nodes = range(offset, offset + len(labels))
    ranked = sorted(candidate_nodes, key=lambda i: importance[i].item(), reverse=True)
    keep = set(forced_keep)
    for i in ranked:
        if i in keep:
            continue
        # This is the check that was missing: a candidate's own score has
        # to clear prune_threshold, independent of how much room is left
        # under max_nodes. Without it, any scene with fewer objects than
        # max_nodes keeps every node regardless of score -- which is
        # exactly what was happening (see pruning.py's module history /
        # the real-photo repro that caught this: 11 objects, max_nodes=12,
        # zero nodes dropped despite 7 of them scoring exactly 0.0).
        if importance[i].item() < prune_threshold:
            continue
        if max_nodes is not None and len(keep) >= max_nodes:
            break
        keep.add(i)
    return sorted(keep)


def _subgraph(graph: Data, keep_nodes: list[int], node_importance: torch.Tensor) -> Data:
    """
    Builds the final pruned graph from the surviving object nodes. No edges
    are carried over — see this module's docstring for why edges aren't a
    meaningful pruned-graph output right now. Per-node importance (the
    score that got each node kept) is attached as `kept_importance`,
    aligned row-for-row with `x`.
    """
    x = graph.x[keep_nodes]
    if node_importance.numel() > 0:
        importance = node_importance[keep_nodes]
    else:
        importance = torch.zeros(len(keep_nodes))

    pruned = Data(
        x=x,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 2)),
    )
    pruned.kept_importance = importance
    return pruned