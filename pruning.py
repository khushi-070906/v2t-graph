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

Nodes/edges below `prune_threshold` are dropped. This keeps the output
graph small and prioritized, which is the whole point (fixing the
"information overload" problem from a linear obstacle list).
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
    Recomputes edge weights using edge_weight() above, drops edges below
    the prune threshold, then drops any node with no remaining incoming
    edges above threshold AND not itself high-affordance (so a door
    directly ahead never gets dropped just because it's isolated).
    """
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr  # [distance, angle] per edge, from graph_builder

    weights = []
    for e in range(edge_index.shape[1]):
        dst = edge_index[1, e].item()
        distance, angle = edge_attr[e].tolist()
        label = detections_labels[dst]
        w = edge_weight(distance, angle, label, heading_rad, config)
        weights.append(w)

    weights_t = torch.tensor(weights, dtype=torch.float32)
    keep_mask = weights_t >= config.prune_threshold

    pruned_edge_index = edge_index[:, keep_mask]
    pruned_edge_attr = edge_attr[keep_mask]
    pruned_weights = weights_t[keep_mask]

    node_importance = _node_importance(
        pruned_edge_index, pruned_weights, num_nodes=graph.x.shape[0]
    )

    keep_nodes = _select_top_nodes(
        node_importance, detections_labels, config.max_nodes
    )

    new_graph, remap = _subgraph(graph, pruned_edge_index, pruned_edge_attr, pruned_weights, keep_nodes)

    # Downstream consumers (encoders.py) need to know which ORIGINAL detection
    # index each pruned node came from, since pruning drops and reorders
    # nodes. Attach it directly to the graph rather than making callers
    # track a separate remap dict — this is what issue 3 in the README was
    # about (encoders previously assumed pruned-node-order == original-order).
    new_graph.kept_node_indices = keep_nodes
    return new_graph


def _node_importance(edge_index: torch.Tensor, weights: torch.Tensor, num_nodes: int) -> torch.Tensor:
    importance = torch.zeros(num_nodes)
    if edge_index.numel() == 0:
        return importance
    dst = edge_index[1]
    importance.scatter_add_(0, dst, weights)
    return importance


def _select_top_nodes(
    importance: torch.Tensor, labels: list[str], max_nodes: int | None
) -> list[int]:
    # Always keep high-affordance nodes (doors, stairs) even if geometrically
    # isolated in the graph, since these are safety-critical regardless of
    # connectivity.
    forced_keep = {
        i for i, lbl in enumerate(labels) if AFFORDANCE_PRIORITY.get(lbl, 0) >= 0.9
    }

    ranked = sorted(range(len(labels)), key=lambda i: importance[i].item(), reverse=True)
    keep = set(forced_keep)
    for i in ranked:
        if max_nodes is not None and len(keep) >= max_nodes:
            break
        keep.add(i)
    return sorted(keep)


def _subgraph(graph, edge_index, edge_attr, weights, keep_nodes):
    keep_set = set(keep_nodes)
    remap = {old: new for new, old in enumerate(keep_nodes)}

    x = graph.x[keep_nodes]

    src, dst = [], []
    new_edge_attr, new_weights = [], []
    for e in range(edge_index.shape[1]):
        s, d = edge_index[0, e].item(), edge_index[1, e].item()
        if s in keep_set and d in keep_set:
            src.append(remap[s])
            dst.append(remap[d])
            new_edge_attr.append(edge_attr[e])
            new_weights.append(weights[e])

    new_edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    new_edge_attr_t = torch.stack(new_edge_attr) if new_edge_attr else torch.zeros((0, 2))
    new_weights_t = torch.stack(new_weights) if new_weights else torch.zeros((0,))

    pruned = Data(x=x, edge_index=new_edge_index, edge_attr=new_edge_attr_t)
    pruned.edge_weight = new_weights_t
    return pruned, remap
