"""
Sanity test for Phases 2-4 using hand-built Detection objects, so it runs
without downloading YOLO/Depth-Anything weights. Run this first to confirm
the graph/pruning/encoding logic is wired correctly before plugging in
real model weights in detect_depth.py.
"""

import math
from detect_depth import Detection
from graph_builder import build_graph
from pruning import prune_graph, PruningConfig
from encoders import encode_haptic_matrix, encode_spatial_audio

FRAME_W, FRAME_H = 640, 480

detections = [
    Detection(0, "door", 0.95, (300, 100, 360, 300), (330, 200), depth_m=3.0),
    Detection(1, "chair", 0.88, (50, 300, 150, 420), (100, 360), depth_m=1.2),
    Detection(2, "plant", 0.7, (500, 250, 560, 400), (530, 325), depth_m=2.5),
    Detection(3, "person", 0.92, (280, 200, 340, 400), (310, 300), depth_m=1.8),
    Detection(4, "table", 0.8, (400, 300, 550, 420), (475, 360), depth_m=2.0),
]
labels = [d.label for d in detections]

graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
print(f"Raw graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")

# heading = 0 rad -> facing straight ahead (image-plane "right"); the door
# and person are roughly centered/ahead in this synthetic layout
pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0, config=PruningConfig())
print(f"Pruned graph: {pruned.x.shape[0]} nodes, {pruned.edge_index.shape[1]} edges")

matrix = encode_haptic_matrix(pruned, grid_size=16)
print("Haptic matrix non-zero cells:", (matrix > 0).sum())

audio_json = encode_spatial_audio(pruned, labels, FRAME_W, FRAME_H)
print(audio_json)
