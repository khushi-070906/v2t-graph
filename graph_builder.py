"""
Phase 2 — Spatial topology graph construction.

Converts a list of Detection objects (from detect_depth.py) into a
PyTorch Geometric graph with an explicit EGO NODE representing the user:

    - node 0            = ego (the user), fixed at the bottom-center of the
                           frame — the natural position of a chest/head-
                           mounted forward-facing camera's own body/ground
                           contact point.
    - nodes 1..n         = detected objects, features = [class embedding,
                           depth, normalized centroid]
    - ego -> object edges = [distance, heading-relative bearing] from the
                           USER to each object. This is what pruning.py's
                           heading_alignment() actually needs to score
                           "is this object in the direction I'm facing."
    - object <-> object edges = [relative distance, image-plane bearing]
                           between detected objects, kept for potential
                           future graph-structural (message-passing) use.
                           NOT currently consumed by pruning.py's importance
                           scoring or by encoders.py — those rely solely on
                           the ego edges now (see pruning.py's module
                           docstring for why the old object-to-object-only
                           design produced meaningless heading scores).

This is the *baseline* graph — pure geometry, no prioritization yet.
Phase 3 (pruning.py) consumes this graph and scores/prunes nodes.
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

# Normalized (x, y) position of the ego/user node within the frame.
# Bottom-center approximates a forward-facing chest/head-mounted camera:
# the user's own body/the ground they're standing on sits at the bottom
# of the image, and "straight ahead" is the horizontal center.
EGO_NORM_POS = (0.5, 1.0)


def class_to_index(label: str, vocab: list[str] = DEFAULT_CLASS_VOCAB) -> int:
    return vocab.index(label) if label in vocab else vocab.index("unknown")


def _ego_node_feature(class_vocab: list[str]) -> np.ndarray:
    one_hot = np.zeros(len(class_vocab), dtype=np.float32)
    return np.concatenate([one_hot, [0.0, EGO_NORM_POS[0], EGO_NORM_POS[1]]])


def build_graph(
    detections: list[Detection],
    frame_size: tuple[int, int],
    class_vocab: list[str] = DEFAULT_CLASS_VOCAB,
    max_depth: float = 20.0,
) -> Data:
    """
    frame_size: (width, height) in pixels, used to normalize centroids to [0, 1]
    max_depth: expected room/scene scale in meters, used to normalize depth
        differences onto the same ~[0,1] scale as normalized pixel offsets.
        Should match the depth model's own effective range (20m for the
        Depth-Anything-V2-Metric-Indoor checkpoint used in detect_depth.py).

    Node 0 is always the ego/user node — see EGO_NORM_POS and this module's
    docstring. graph.ego_node_idx is set to 0 so downstream code (pruning.py)
    doesn't have to hardcode it.
    """
    w, h = frame_size
    n = len(detections)

    if n == 0:
        x = torch.tensor(np.stack([_ego_node_feature(class_vocab)]), dtype=torch.float32)
        graph = Data(
            x=x,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2)),
        )
        graph.ego_node_idx = 0
        return graph

    # --- node features: ego first, then one-hot class + depth + normalized (x, y) ---
    node_feats = [_ego_node_feature(class_vocab)]
    for d in detections:
        one_hot = np.zeros(len(class_vocab), dtype=np.float32)
        one_hot[class_to_index(d.label, class_vocab)] = 1.0
        nx_, ny_ = d.centroid_px[0] / w, d.centroid_px[1] / h
        feat = np.concatenate([one_hot, [d.depth_m, nx_, ny_]])
        node_feats.append(feat)
    x = torch.tensor(np.stack(node_feats), dtype=torch.float32)

    # Object node i (0-indexed in `detections`) lives at graph index i + 1,
    # since index 0 is reserved for the ego node.
    src, dst, edge_attrs = [], [], []

    # --- object <-> object edges (structural only, not used for pruning) ---
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src.append(i + 1)
            dst.append(j + 1)
            edge_attrs.append(_relative_edge_attr(detections[i], detections[j], w, h, max_depth))

    # --- ego -> object edges (what pruning.py actually scores) ---
    for j in range(n):
        src.append(0)
        dst.append(j + 1)
        edge_attrs.append(_ego_edge_attr(detections[j], w, h, max_depth))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.stack(edge_attrs), dtype=torch.float32)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.ego_node_idx = 0
    return graph


def _relative_edge_attr(
    a: Detection, b: Detection, frame_w: int, frame_h: int, max_depth: float = 20.0
) -> np.ndarray:
    """
    Returns [relative_distance, relative_angle_rad] from object a -> object b,
    combining pixel-plane offset with the depth difference to approximate a
    3D relative position. This is a geometric proxy, not a calibrated 3D
    reconstruction — good enough for graph topology, not for metric SLAM.

    NOTE: this measures object-to-object position, not object-relative-to-
    user. It is retained for structural/future GNN use but pruning.py's
    heading scoring no longer reads it — see _ego_edge_attr below and
    pruning.py's module docstring.

    max_depth normalizes the depth difference onto roughly the same [0,1]
    scale as the normalized pixel-plane offsets (dx, dy). Without this, raw
    meters and normalized pixel units get combined directly into one
    Euclidean distance, which over-inflates distance and causes
    pruning.py's exp(-distance / tau) term to collapse to ~0 for nearly
    every edge regardless of tau. Set max_depth to roughly your expected
    room scale in meters — 20m matches the training range of the
    Depth-Anything-V2-Metric-Indoor checkpoint used in detect_depth.py;
    tune per deployment if you swap in a different depth model.
    """
    ax, ay = a.centroid_px
    bx, by = b.centroid_px

    dx = (bx - ax) / frame_w
    dy = (by - ay) / frame_h
    ddepth = (b.depth_m - a.depth_m) / max_depth

    distance = math.sqrt(dx**2 + dy**2 + ddepth**2)
    angle = math.atan2(dy, dx)  # image-plane bearing between two objects

    return np.array([distance, angle], dtype=np.float32)


def _ego_edge_attr(
    obj: Detection, frame_w: int, frame_h: int, max_depth: float = 20.0
) -> np.ndarray:
    """
    Returns [distance, heading_bearing_rad] from the USER (ego) to a
    detected object — this is what pruning.py's heading_alignment() should
    actually be scored against, not the object-to-object angle from
    _relative_edge_attr.

    distance: the object's own metric depth IS the ego->object distance
        (the camera/user is the origin objects were measured from in the
        first place), so no pixel-offset geometry is needed here — just
        normalize by max_depth onto the ~[0,1] scale pruning.py expects.

    angle: a pure left/right bearing, 0 = straight ahead (object at
        horizontal frame center), increasing toward +-pi/2 at the frame
        edges. This intentionally ignores vertical (dy) position — which
        way the user needs to turn to face an object depends on its
        horizontal position, not its height in the frame. This also
        matches the azimuth_deg convention already used by
        encoders.encode_spatial_audio, so "heading" during pruning and
        "azimuth" in the final output are the same physical quantity.
    """
    nx_ = obj.centroid_px[0] / frame_w
    distance = obj.depth_m / max_depth
    angle = (nx_ - 0.5) * math.pi
    return np.array([distance, angle], dtype=np.float32)