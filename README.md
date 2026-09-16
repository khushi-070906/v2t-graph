# V-to-T Graph

Monocular RGB → navigation-aware semantic graph → hardware-agnostic
tactile/audio output → spoken navigation instructions, for assistive scene
understanding.

The claimed contribution is Phase 3 (`pruning.py`): a heading- and
affordance-aware scoring function that decides which objects in a scene are
worth a blind or low-vision user's limited attention. Everything upstream is
standard detection + depth; everything downstream is deterministic encoding.

---

## Layout

All modules live at the repository root (flat, no `src/` package) — run
everything from the repo root.

| Path | Phase | What it does |
| --- | --- | --- |
| `detect_depth.py` | 1 | YOLOv8-World / YOLOv10 + Depth Anything V2 → `Detection` list. Applies a fitted depth calibration if one is supplied. |
| `graph_builder.py` | 2 | `Detection` list → PyG `Data` with an explicit ego/user node. |
| `pruning.py` | 3 | **Core contribution.** Heading/affordance/distance-aware node scoring and pruning. |
| `encoders.py` | 4 | Pruned graph → haptic matrix / spatial-audio JSON. |
| `tracker.py` | 5.5 | Frame-to-frame track IDs, confirmation windows, real closing velocity. |
| `navigation_planner.py` | 6/7 | Spatial-audio JSON → deterministic spoken instructions (rule-based by design — see module docstring). |
| `speech_output.py` | 7 | Offline TTS (pyttsx3). Optional, lazily imported, off by default. |
| `pipeline.py` | — | Ties Phases 1–4 and 6/7 together. CLI entry point for a single frame. |
| `live_camera_loop_tracked.py` | — | Webcam loop with the tracker in the path. |
| `calibrate_depth.py` | — | Measures Depth-Anything's metric error at known distances, fits `corrected = (raw − offset) / scale`, writes `calibration.json`. |
| `demo_synthetic.py` | — | Prints a Phase 2–6 walkthrough on hand-built detections. No model weights needed. |
| `tests/` | — | `pytest` suite covering Phases 2–4 and 6. No model weights needed. |
| `eval/compression_ratio.py` | — | Node compression + critical-object retention metrics. |
| `eval/batch_eval.py` | — | Runs the pipeline over a directory of frames, aggregates to mean ± std. |
| `eval/simulate_walk.py` | — | 3-way baseline comparison (linear / distance-only / v2t) with `attention_precision@K`. |
| `figures/generate_results_figure.py` | — | Builds the two-panel results figure from `results.json`. |
| `fetch_nyu_frames*.py`, `fetch_sunrgbd_frames*.py` | — | Dataset frame fetchers. |

## Quickstart

```bash
pip install -r requirements-dev.txt   # or requirements.txt for runtime only

# 1. Verify graph/pruning/encoding/navigation logic — no model weights needed
pytest                     # assertion-backed suite
python demo_synthetic.py   # human-readable walkthrough of the same scene

# 2. One real frame, end to end (add --speak for offline TTS)
python pipeline.py photo.jpeg \
    --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu \
    --calibration calibration.json --output both

# 3. A folder of frames, aggregated into results-table numbers
python eval/batch_eval.py frames \
    --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu \
    --calibration calibration.json --output-json results.json

# 4. Term-wise ablation on the same frames (run once per term)
for term in affordance distance heading; do
  python eval/batch_eval.py frames \
      --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu \
      --ablate $term --output-json results_no_$term.json
done

# 5. Dependency-free simulated-walk ablation
python eval/simulate_walk.py

# 6. Paper results figure
python figures/generate_results_figure.py --results-json results.json --out results_figure.png
```

Dataset used for the batch runs:
<https://drive.google.com/drive/folders/1GDm53IaI4Hqfdp5uamKJNJcVuZH3237G>

### Conventions worth knowing before reading the code

- **Heading.** `--heading` is in radians relative to the camera's optical
  axis. `0` means looking straight ahead; an object at the horizontal centre
  of the frame has bearing `0`. This is the same physical quantity as
  `encoders.encode_spatial_audio`'s `azimuth_deg`, in radians.
- **Depth normalization.** `graph_builder.MAX_DEPTH_M` (20 m) is the single
  scale that turns metric depth into the `~[0,1]` units `pruning.tau` is
  calibrated against. Swapping the depth model without updating it silently
  rescales every distance term in the pruning formula.
- **Forced keep.** Any class with affordance ≥ `FORCED_KEEP_MIN_AFFORDANCE`
  (door, stairs, obstacle) survives pruning regardless of score — but is
  still subject to `max_nodes`.

## Closed issues

1. **Unit mismatch in `graph_builder._relative_edge_attr`.** Depth difference
   is normalized by `max_depth` before being combined with normalized
   pixel-plane offsets.
2. **`eval/simulate_walk.py` couldn't tell `distance_only` from `v2t_pruned`.**
   Two causes: the diagonal start→goal heading capped angular deviation at
   45°, so "off-heading" clutter was never off-heading; and raw collision
   counts don't measure ranking quality once a forgiving reactive dodge is in
   play. Fixed by decoupling heading from goal direction and replacing
   collisions with **attention_precision@K** — the fraction of limited
   attention slots spent on a still-ahead critical hazard rather than on
   clutter or an already-passed hazard. `v2t_pruned` 26.3% vs `distance_only`
   10.6% in the adversarial scene. The direction-aware ground truth (an object
   stops counting as a hazard once passed) is itself a defensible
   methodological point worth stating in the paper.
3. **`encoders.py` assumed pruned node order matched the original label list.**
   `prune_graph` now attaches `kept_node_indices`; pinned by
   `tests/test_pipeline_logic.py::test_kept_node_indices_map_back_to_original_labels`.
4. **`detect_depth.py` / `pipeline.py` untested against real weights.**
   Verified end to end with `yolov8s-worldv2.pt` + Depth-Anything-V2: 11 raw
   detections pruned to 4, 63.6% node compression, `door` correctly
   force-kept.
5. **Silent pruning regression from a stale/shadowed `pruning.py`.** A
   real-photo run reported 0.0% node compression with seven nodes at
   `priority: 0.0` surviving. Root cause was a stale copy on the import path,
   not the fix regressing. `pruning._assert_pruning_invariant` now raises
   immediately on recurrence. **If a fixed pruning bug appears to resurface,
   check `python -c "import pruning; print(pruning.__file__)"` and for stale
   `__pycache__` before assuming the fix regressed.**
6. **Hugging Face SUN RGB-D fetch.** `kasurashan/RGBD-Instance-Segmentation`'s
   Datasets Server job crashed (501); the `datasets` library hit a
   blocked-pandas-DLL error under Windows Application Control; the REST API
   422'd against `wyrx/SUNRGBD_seg`. Resolved by downloading frames manually.
   See the fetcher docstrings if a larger sample is needed.
7. **Depth calibration was measured but never applied.** `calibrate_depth.py`
   fitted a +1.37 m offset at webcam range and wrote `calibration.json`, but
   nothing consumed it, so every spoken distance was the uncorrected one.
   `detect_depth.apply_calibration` is now wired into the detector and exposed
   as `--calibration` on `detect_depth.py`, `pipeline.py` and `batch_eval.py`.
8. **Batch eval reloaded both models on every frame.** `run_pipeline` built its
   own `DetectorDepthEstimator` per call, so a 51-frame run paid the YOLO +
   Depth-Anything load cost 51 times over. `batch_eval` now builds the detector
   once via `pipeline.build_detector` and passes it in.
9. **`test_synthetic.py` asserted nothing.** It printed results and always
   passed, so it could not have caught issues 3 or 5. Replaced by
   `tests/test_pipeline_logic.py` (assertion-backed, still weight-free);
   the printing version survives as `demo_synthetic.py`.

## Open issues

1. **The heading term dominates, and at `heading=0` the method reduces to a
   centre-crop.** With the defaults `tau=0.35`, `MAX_DEPTH_M=20`, `gamma=2`,
   `prune_threshold=0.15`, the distance term only falls from 1.0 to ~0.7
   across an entire indoor room, while `cos²` drives the score to zero by
   ~45° off-axis. For a `bed` (affordance 0.4) at 2.5 m, the score clears
   threshold only while the object sits inside roughly the middle 47% of the
   frame. Every evaluation so far runs at `heading=0`, so what is being
   measured is close to "keep centred, high-affordance objects." This is the
   likeliest explanation for open issue 2 below, and it is the first thing a
   reviewer will probe. Run the term-wise ablation (`--ablate heading` /
   `distance` / `affordance`) and a `tau`/`gamma` sweep before claiming the
   three-term formula is doing three separate things.
2. **SUN RGB-D compression number needs a visual sanity check before being
   cited.** 21/21 frames succeeded at 74.0% node compression vs NYU's 52.6%,
   but 14/21 pruned down to only `['bed']`, including scenes where `person`,
   `table`, `sofa` and `wall` were detected and dropped. Not yet confirmed
   whether this is a real heading/distance effect, a scene-mix artifact (this
   mirror looks bedroom-heavy), or open issue 1.
3. **Critical-object retention is statistically thin.** NYU: 7/51 frames had a
   critical object. SUN RGB-D: 1/21. n = 8. Both report 100% retention, which
   is not a defensible general claim at that sample size. Needs a deliberately
   critical-object-heavy frame sample.
4. **Compression alone is not a defensible headline metric.** A pruner that
   keeps nothing scores 100%. Report compression jointly with retention at a
   fixed attention budget, or against a per-frame annotation of which objects
   *should* have been surfaced. Do not report **edge** compression at all: the
   pruned graph has zero edges by construction, so it is 100% on every frame
   and measures the output format, not the method.
5. **`figures/generate_results_figure.py` hardcodes the simulate_walk
   numbers**, copied out of stdout. Have `simulate_walk.py` write JSON and read
   that instead, or the figure will silently drift from the code.
6. **Not started — real-time budget.** No per-frame latency measured, and no
   Jetson Orin Nano numbers. A paper claiming real-time assistive use needs
   both.
7. **Not started — temporal reasoning inside the graph.** `tracker.py` measures
   closing velocity, but `pruning.py` never consumes it: velocity only
   annotates the final sentence. Feeding time-to-contact into the scoring
   function is the obvious next contribution.
8. **No licence file.** Given the patent track below, choose this deliberately
   rather than by default — publishing a repository is a disclosure decision,
   not just a packaging one.

## Next steps

- **Ablation and sensitivity first.** Term-wise ablation on the real frames
  (now a flag), plus a `tau` / `gamma` / `prune_threshold` sweep. The reported
  52.6% ± 29.8% is a function of four hand-set constants, and that ± is larger
  than most of the effects being claimed.
- **Live camera loop timing.** `live_camera_loop_tracked.py` exists but has no
  measured per-frame budget.
- **Jetson Orin Nano deployment.** CPU dev machine only so far.
- **Patent track (parallel, not blocking).** A provisional filing can happen
  before the rest of the system is built: India requires enough disclosure to
  reduce the invention to practice for a skilled reader, not a finished
  product, and a provisional gives 12 months for the complete specification.
  Next step is the institution's IP / tech-transfer cell, to settle ownership
  and inventorship before drafting. India has no public-disclosure grace
  period, so this must precede any paper submission, preprint, conference
  presentation — or making this repository public.
