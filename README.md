# V-to-T Graph — pipeline scaffold

Monocular RGB -> navigation-aware semantic graph -> hardware-agnostic
tactile/audio output -> spoken navigation instructions, for assistive
scene understanding.

## Structure

```
src/
  detect_depth.py       Phase 1 — YOLOv10/YOLOv8-World + Depth Anything V2
                          -> Detection list
  graph_builder.py       Phase 2 — Detection list -> PyG Data (baseline
                          graph, explicit ego/user node)
  pruning.py               Phase 3 — CORE CONTRIBUTION: heading/affordance-
                            aware adaptive edge pruning
  encoders.py                Phase 4 — pruned graph -> haptic matrix /
                              spatial audio JSON
  navigation_planner.py        Phase 6/7 — spatial audio JSON -> rule-based
                                navigation instructions + speech text
                                (deterministic, not LLM-generated — see
                                module docstring for why)
  speech_output.py                Phase 7 — offline TTS (pyttsx3) for the
                                   speech text above; optional, lazy-
                                   imported, off by default
  pipeline.py                       ties Phases 1-4 + 6/7 together, CLI
                                     entry point; --speak flag for TTS
  test_synthetic.py                  exercises Phases 2-4 + 6 with hand-
                                      built detections, no model weights
                                      needed — run this first
  eval/
    compression_ratio.py  node/edge compression + critical-object
                           retention metric
    batch_eval.py          runs the full pipeline over a directory of
                            frames, aggregates compute_compression output
                            into mean +- std for a results table
    simulate_walk.py        3-way baseline comparison (linear / distance-
                             only / v2t), including a multi-blocker
                             adversarial ablation for attention_precision@K
                             error bars
fetch_nyu_frames.py / fetch_nyu_frames_api.py    NYU Depth V2 frame fetchers
fetch_sunrgbd_frames.py / fetch_sunrgbd_frames_hf.py    SUN RGB-D frame
                                                          fetchers (see
                                                          Known issues #6)
generate_results_figure.py    builds the two-panel results figure (ablation
                               curve + real-photo compression bars) from
                               results.json / results_sunrgbd.json
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Verify graph/pruning/encoding/navigation logic without downloading
#    any model weights
python src/test_synthetic.py

# 2. Once you have detector + depth weights available, run the full
#    pipeline on one frame (add --speak for offline TTS)
python src/pipeline.py path/to/frame.jpg --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu --output both --speak

# 3. Run the full pipeline over a folder of frames and aggregate results
python src/eval/batch_eval.py frames_dir --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu --output-json results.json

# 4. Run the (dependency-free) simulated-walk ablation
python src/eval/simulate_walk.py

# 5. Build the paper results figure from batch_eval output
python generate_results_figure.py --results-json results.json --out results_figure.png
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
5. ~~Silent pruning regression from a stale/shadowed `pruning.py`~~ —
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
6. **SUN RGB-D compression number needs a visual sanity check before
   citing.** Batch run on 21 real SUN RGB-D frames (`wyrx/SUNRGBD_seg`)
   succeeded 21/21 with node compression 74.0% vs. NYU's 52.6% — but
   14/21 SUN RGB-D frames pruned down to only `['bed']`, including scenes
   where `person`, `table`, `sofa`, `wall` were also detected and dropped.
   Not yet confirmed whether this is a real heading/distance effect or a
   scene-mix artifact (this mirror looks bedroom-subset-heavy). **Not
   fixed yet** — spot-check a few of these frames visually before using
   the NYU-vs-SUN-RGB-D comparison in the paper.
7. **Critical-object retention is statistically thin across both real
   datasets.** NYU: 7/51 frames had a critical object (door/stairs/
   obstacle). SUN RGB-D: 1/21. n=8 total. Both report 100% retention, but
   that's not yet a defensible general claim at this sample size — **not
   fixed**, needs a deliberately critical-object-heavy frame sample if
   this number is going in the results section.
8. Getting a Hugging Face SUN RGB-D mirror working was its own detour:
   `kasurashan/RGBD-Instance-Segmentation`'s Datasets Server job crashed
   (501, "missing heartbeats"); the `datasets` library itself then hit a
   blocked-pandas-DLL error on the target Windows machine (Application
   Control policy); the REST API version 422'd against `wyrx/SUNRGBD_seg`
   despite its own web viewer rendering fine. Resolved by downloading
   frames manually from the browser instead of scripting the fetch — see
   `fetch_sunrgbd_frames.py` / `fetch_sunrgbd_frames_hf.py` docstrings for
   the full debugging trail if this needs revisiting for a larger SUN
   RGB-D sample.

## Next implementation steps

<<<<<<< HEAD
- **Done:** real-photo batch eval on two datasets — NYU Depth V2 (51
  frames: 52.6% +- 29.8% node compression) and SUN RGB-D (21 frames: 74.0%
  +- 13.9%, see known issue #6 caveat). `batch_eval.py` now also captures
  full tracebacks on frame failures instead of swallowing them.
- **Done:** `simulate_walk.py`'s adversarial ablation extended to multiple
  blockers (`n_blockers` param) instead of one fixed scene per seed.
  `v2t_pruned` holds a stable 2.2-2.5x attention_precision@K margin over
  `distance_only` across n_blockers=1-4 (26.3% -> 89.5% vs. 10.6% -> 38.4%).
- **Done:** Phase 6/7 — `navigation_planner.py` (deterministic spatial
  JSON -> spoken instruction sentences, e.g. "Door at 12 o'clock, 3 meters
  away.") and `speech_output.py` (offline pyttsx3 TTS, opt-in via
  `pipeline.py --speak`). Full stack now runs frame -> detection -> depth
  -> graph -> pruning -> spatial audio -> spoken instruction, verified on
  both synthetic detections and real photos.
- **Not started — live camera loop.** Everything above takes a static
  image path. No webcam/video capture, no per-frame real-time timing
  budget measured yet.
- **Not started — temporal/tracking layer.** "Person approaching" is
  currently a single-frame distance snapshot, not a real velocity
  estimate; no object permanence across frames.
- **Not started — Jetson Orin Nano deployment.** Everything has only been
  run on a CPU dev machine; no on-device latency numbers yet.
- **Patent track (parallel to the above, not blocking it):** provisional
  filing can happen now, before the rest of the system is built — India
  requires enough disclosure to reduce the invention to practice for a
  skilled reader, not a finished product, and a provisional gives 12
  months to file the complete specification (during which the camera
  loop / tracking / deployment work above can proceed and be folded in).
  Next step is GTBIT/GGSIPU's IP/tech transfer cell, to confirm
  ownership/inventorship (likely co-inventor with Dr. Amandeep Kaur)
  before drafting. India has no public-disclosure grace period, so this
  needs to happen before any paper submission or conference presentation.
=======
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
>>>>>>> 4760b106664a236b66a04ded6b4e3583312f72a4
