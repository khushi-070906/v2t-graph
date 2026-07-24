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
4. ~~`detect_depth.py` and `pipeline.py` are untested against real model
   weights~~ — **FIXED / VERIFIED.** Ran end-to-end on a real photo with
   real `yolov8s-worldv2.pt` + Depth-Anything-V2 weights: 11 raw detections
   (`sofa, bed, chair, wall, cabinet, chair, tv, table, sofa, cabinet,
   door`) correctly pruned to 4 (`chair, cabinet, chair, door`), 63.6% node
   compression, `door` correctly force-kept as the one critical-affordance
   node. Confirms the `ultralytics`/`transformers` API usage in
   `detect_depth.py` is correct against real weights, not just plausible.
5. **Silent pruning regression from a stale/shadowed `pruning.py`** —
   **FIXED.** A run against the real photo in issue 4 above initially
   produced `node compression: 0.0%` — all 11 objects survived pruning,
   including 7 nodes with `priority: 0.0` exactly, which should be
   impossible: `_select_top_nodes`'s `if importance[i] < prune_threshold:
   continue` check (added for the exact same class of bug — see the old
   issue-4-era history in this section) should have dropped every one of
   them. Root cause: the environment was executing a stale/shadowed copy
   of `pruning.py` predating that fix, not the file actually being edited.
   Re-running with the correct file in place fixed it — same real depth
   values (`door` at 7.28m, `chair` at 5.42m, unchanged), correctly pruned
   to 4/11 nodes. `pruning.py` now also carries a loud
   `_assert_pruning_invariant` check (raises `AssertionError` naming the
   offending node) so this class of regression fails immediately instead
   of silently shipping a broken graph. Verified the assertion is silent
   on the correct code path and fires correctly when the old buggy
   `_select_top_nodes` is reintroduced.
   **Takeaway:** if a "fixed" pruning/scoring bug appears to resurface,
   check `python3 -c "import pruning; print(pruning.__file__)"` and for
   stale `__pycache__`/shadowing copies before assuming the fix regressed.

## Next implementation steps

- Real YOLOv8-World + Depth Anything V2 weights are now confirmed working
  end-to-end on a real photo, with correct pruning (issues 4 and 5 —
  closed). Next: scale from single-photo spot checks to a real batch —
  run `pipeline.py` over a folder of NYU Depth V2 / SUN RGB-D frames and
  aggregate `compute_compression` output into the actual results-table
  numbers, rather than one photo at a time.
- Run `eval/simulate_walk.py` for the `attention_precision@K` comparison
  number now that pruning correctness is verified on real weights, not
  just synthetic detections.
- Consider extending `simulate_walk.py`'s adversarial scenario to multiple
  blockers / more clutter to get a distribution of attention_precision
  values rather than one fixed scene per seed, if you want error bars for
  the paper's results table



python pipeline.py photo.jpeg --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu

python eval\batch_eval.py frames --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu --output-json results.json


DATASET - https://drive.google.com/drive/folders/1GDm53IaI4Hqfdp5uamKJNJcVuZH3237G?usp=sharing
