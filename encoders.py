"""
Phase 4 — Output encoders.

Both encoders consume the SAME pruned graph (from pruning.py). This is
what makes the "hardware-agnostic" claim real: swap the encoder, not
the pipeline. If these two encoders ever need different upstream data,
the hardware-agnostic claim breaks — keep them both strictly downstream
of the pruned Data object and nothing else.
"""

from __future__ import annotations

import json
import math
import numpy as np
from torch_geometric.data import Data


def encode_haptic_matrix(pruned_graph: Data, grid_size: int = 64) -> np.ndarray:
    """
    Projects node positions (normalized x, y stored in node features at
    indices [-2, -1], see graph_builder.py) onto a grid_size x grid_size
    integer matrix. Cell value = affordance-weighted intensity, 0 = empty.

    This matrix maps directly onto a refreshable haptic pin array API.
    """
    matrix = np.zeros((grid_size, grid_size), dtype=np.int32)
    if pruned_graph.x.shape[0] == 0:
        return matrix

    xs = pruned_graph.x[:, -2].numpy()
    ys = pruned_graph.x[:, -1].numpy()

    # node_importance derived from incoming edge weights, if present
    if hasattr(pruned_graph, "edge_weight") and pruned_graph.edge_weight.numel() > 0:
        importance = np.zeros(pruned_graph.x.shape[0])
        dst = pruned_graph.edge_index[1].numpy()
        for d, w in zip(dst, pruned_graph.edge_weight.numpy()):
            importance[d] += w
        if importance.max() > 0:
            importance = importance / importance.max()
    else:
        importance = np.ones(pruned_graph.x.shape[0])

    for x, y, imp in zip(xs, ys, importance):
        gx = min(grid_size - 1, max(0, int(x * grid_size)))
        gy = min(grid_size - 1, max(0, int(y * grid_size)))
        intensity = int(round(imp * 255))
        matrix[gy, gx] = max(matrix[gy, gx], intensity)

    return matrix


def encode_spatial_audio(
    pruned_graph: Data,
    detections_labels: list[str],
    frame_w: int,
    frame_h: int,
) -> str:
    """
    Emits a JSON schema of {label, distance, azimuth, priority} objects,
    suitable for driving a spatial audio synthesizer (e.g. HRTF panning
    by azimuth, volume/pitch by distance and priority).

    detections_labels must be the FULL, ORIGINAL label list (same one
    passed into build_graph / prune_graph) — not pre-filtered. This
    function uses pruned_graph.kept_node_indices (set by pruning.py) to
    map each pruned node back to its original label, since pruning drops
    and reorders nodes and pruned-node position i does NOT correspond to
    detections_labels[i] in general.
    """
    events = []
    n = pruned_graph.x.shape[0]

    kept_indices = getattr(pruned_graph, "kept_node_indices", None)
    if kept_indices is None:
        # Graph wasn't produced by prune_graph (e.g. raw graph passed directly) —
        # fall back to assuming identity order, but this is only correct
        # pre-pruning.
        kept_indices = list(range(n))

    for i in range(n):
        nx_ = pruned_graph.x[i, -2].item()
        ny_ = pruned_graph.x[i, -1].item()
        depth = pruned_graph.x[i, -3].item()

        # azimuth: map normalized x in [0,1] to [-90, 90] degrees (left..right)
        azimuth_deg = (nx_ - 0.5) * 180.0

        priority = 0.0
        if hasattr(pruned_graph, "edge_weight") and pruned_graph.edge_weight.numel() > 0:
            dst = pruned_graph.edge_index[1]
            mask = dst == i
            if mask.any():
                priority = pruned_graph.edge_weight[mask].max().item()

        orig_idx = kept_indices[i]
        label = (
            detections_labels[orig_idx]
            if orig_idx < len(detections_labels)
            else "unknown"
        )

        events.append(
            {
                "label": label,
                "distance_m": round(depth, 2),
                "azimuth_deg": round(azimuth_deg, 1),
                "priority": round(priority, 3),
            }
        )

    events.sort(key=lambda e: e["priority"], reverse=True)
    return json.dumps({"events": events}, indent=2)
