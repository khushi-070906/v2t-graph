# V-to-T Graph — pipeline scaffold

Monocular RGB -> navigation-aware semantic graph -> hardware-agnostic
tactile/audio output, for assistive scene understanding.

## Structure

```
src/
  detect_depth.py     Phase 1 — YOLOv10 + Depth Anything V2 -> Detection list
  graph_builder.py     Phase 2 — Detection list -> PyG Data (baseline graph)
  pruning.py            Phase 3 — CORE CONTRIBUTION: heading/affordance-aware
                          adaptive edge pruning
  encoders.py            Phase 4 — pruned graph -> haptic matrix / spatial audio JSON
  pipeline.py             ties Phases 1-4 together, CLI entry point
  test_synthetic.py        exercises Phases 2-4 with hand-built detections,
                            no model weights needed — run this first
  eval/
    compression_ratio.py  node/edge compression + critical-object retention metric
    simulate_walk.py        3-way baseline comparison (linear / distance-only / v2t)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Verify graph/pruning/encoding logic without downloading any model weights
python src/test_synthetic.py

# 2. Once you have detector + depth weights available, run the full pipeline
python src/pipeline.py path/to/frame.jpg --heading 0.0 --output both

# 3. Run the (currently dependency-free) evaluation scripts
python src/eval/simulate_walk.py
python src/eval/compression_ratio.py
```

## Known issues

1. ~~Unit mismatch in `graph_builder._relative_edge_attr`~~ — **FIXED.**
   Depth difference is now normalized by `max_depth` (default 5m) before
   combining with normalized pixel-plane offsets. Verified via
   `test_synthetic.py`: pruning now keeps 2/20 edges instead of 0/20, and
   correctly ranks `person` (close, on-heading) above `door` (far).
2. ~~`eval/simulate_walk.py`'s toy room doesn't yet differentiate
   `distance_only` from `v2t_pruned`~~ — **FIXED.** Root cause was two-fold:
   (a) the original diagonal start->goal heading capped angular deviation
   at 45°, so "off-heading" clutter was never truly off-heading —
   fixed by decoupling heading from goal direction (agent now walks a
   straight hallway, heading independent of obstacle placement); (b) raw
   collision counts don't differentiate ranking quality since a forgiving
   reactive dodge lets any strategy avoid collisions once close enough —
   replaced with an **attention_precision@K** metric (fraction of limited
   attention slots spent on a still-ahead critical hazard vs. wasted on
   clutter or a stale already-passed hazard). Verified: `v2t_pruned`
   attention precision (26.3%) now clearly beats `distance_only` (10.6%)
   in the adversarial scene. See `simulate_walk.py`'s module docstring for
   the full debugging story — worth citing in the paper's evaluation
   section, since the direction-aware ground truth (an object stops
   counting as a "hazard" once the agent has passed it) is itself a
   defensible methodological point.
3. ~~`encoders.py` label indexing assumed pruned-graph node order matches
   the original `labels` list positionally~~ — **FIXED.** `prune_graph` now
   attaches `kept_node_indices` (original detection index per surviving
   node) directly to the returned graph, and `encode_spatial_audio` uses it
   to map labels correctly. Verified with a forced `max_nodes=2` test: with
   5 detections pruned down to 2, the surviving nodes were correctly
   labeled `door` and `person` (not mislabeled by position).
4. `detect_depth.py` and `pipeline.py` are untested against real model
   weights in this environment (no network access to download YOLOv10 /
   Depth-Anything checkpoints here) — logic follows the `ultralytics` and
   `transformers` APIs but run it locally to confirm before relying on it.
   Still open.

## Next implementation steps

- Swap in real YOLOv10 + Depth Anything V2 weights, run `pipeline.py` on a
  few NYU Depth V2 frames, sanity-check detections visually (issue 4 —
  still open)
- Only after that: run `eval/compression_ratio.py` over the full dataset
  for the actual paper numbers
- Consider extending `simulate_walk.py`'s adversarial scenario to multiple
  blockers / more clutter to get a distribution of attention_precision
  values rather than one fixed scene per seed, if you want error bars for
  the paper's results table
