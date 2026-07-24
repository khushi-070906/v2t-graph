"""
Phase 2 — Spatial topology graph construction.

Converts a list of Detection objects (from detect_depth.py) into a
PyTorch Geometric graph:
    - nodes: objects, features = [class embedding, depth, normalized centroid]
    - edges: every object pair, edge_attr = [relative distance, relative angle]

This is the *baseline* graph — pure geometry, no prioritization yet.
Phase 3 (pruning.py) consumes this graph and reweights/prunes edges.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch_geometric.data import Data

from detect_depth import Detection


# Fixed vocabulary for a quick embedding lookup. In practice, swap this for
# a learned embedding table sized to your detector's class list.
DEFAULT_CLASS_VOCAB = [
    "person", "chair", "table", "door", "wall", "stairs", "sofa",
    "bed", "obstacle", "plant", "tv", "cabinet", "unknown",
]


def class_to_index(label: str, vocab: list[str] = DEFAULT_CLASS_VOCAB) -> int:
    return vocab.index(label) if label in vocab else vocab.index("unknown")


def build_graph(
    detections: list[Detection],
    frame_size: tuple[int, int],
    class_vocab: list[str] = DEFAULT_CLASS_VOCAB,
) -> Data:
    """
    frame_size: (width, height) in pixels, used to normalize centroids to [0, 1]
    """
    w, h = frame_size
    n = len(detections)

    if n == 0:
        return Data(
            x=torch.zeros((0, len(class_vocab) + 3)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2)),
        )

    # --- node features: one-hot class + depth + normalized (x, y) ---
    node_feats = []
    for d in detections:
        one_hot = np.zeros(len(class_vocab), dtype=np.float32)
        one_hot[class_to_index(d.label, class_vocab)] = 1.0
        nx_, ny_ = d.centroid_px[0] / w, d.centroid_px[1] / h
        feat = np.concatenate([one_hot, [d.depth_m, nx_, ny_]])
        node_feats.append(feat)
    x = torch.tensor(np.stack(node_feats), dtype=torch.float32)

    # --- edges: fully connected (n * (n-1) directed pairs) ---
    src, dst, edge_attrs = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            edge_attrs.append(_relative_edge_attr(detections[i], detections[j], w, h))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.stack(edge_attrs), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _relative_edge_attr(
    a: Detection, b: Detection, frame_w: int, frame_h: int, max_depth: float = 5.0
) -> np.ndarray:
    """
    Returns [relative_distance, relative_angle_rad] from object a -> object b,
    combining pixel-plane offset with the depth difference to approximate a
    3D relative position. This is a geometric proxy, not a calibrated 3D
    reconstruction — good enough for graph topology, not for metric SLAM.

    max_depth normalizes the depth difference onto roughly the same [0,1]
    scale as the normalized pixel-plane offsets (dx, dy). Without this, raw
    meters and normalized pixel units get combined directly into one
    Euclidean distance, which over-inflates distance and causes
    pruning.py's exp(-distance / tau) term to collapse to ~0 for nearly
    every edge regardless of tau. Set max_depth to roughly your expected
    room scale in meters (5m is a reasonable indoor default) and tune per
    deployment.
    """
    ax, ay = a.centroid_px
    bx, by = b.centroid_px

    dx = (bx - ax) / frame_w
    dy = (by - ay) / frame_h
    ddepth = (b.depth_m - a.depth_m) / max_depth

    distance = math.sqrt(dx**2 + dy**2 + ddepth**2)
    angle = math.atan2(dy, dx)  # image-plane bearing; swap for true heading-relative angle downstream

    return np.array([distance, angle], dtype=np.float32)
