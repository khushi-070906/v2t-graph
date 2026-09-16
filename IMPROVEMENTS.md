# What changed in this pass

Grouped by severity. Nothing here changes the pruning formula itself — the
scoring function is unchanged, so previously reported numbers remain
reproducible (except where a bug was making them wrong, noted below).

## Correctness

**`README.md` contained an unresolved git merge conflict.** The
`<<<<<<< HEAD` / `=======` / `>>>>>>> 4760b10` markers were committed verbatim,
so the "Next implementation steps" section shipped two contradictory versions
of itself. Rewritten from both halves.

**Depth calibration was measured and then discarded.** `calibrate_depth.py`
fitted `offset = 1.37 m`, `scale = 0.89` from three real measurements and wrote
`calibration.json`, and its own `apply_correction()` docstring said the
correction was "not wired in by default." Nothing imported it. Every distance
the system spoke aloud was the uncorrected one — a person at 0.5 m was
announced at 1.8 m. Fixed:

- `detect_depth.apply_calibration()` / `load_calibration()` now own the
  formula (so the runtime path doesn't import the interactive capture script),
  clamped at 0 m so a corrected depth can't go negative and become the closest
  object in the scene.
- `DetectorDepthEstimator(calibration=...)` applies it per detection.
- `--calibration calibration.json` added to `detect_depth.py`, `pipeline.py`
  and `eval/batch_eval.py`.
- `calibrate_depth.apply_correction()` kept as a thin alias, so there is one
  implementation rather than two that can drift.

**Forced-keep nodes ignored the `max_nodes` budget.** `_select_top_nodes`
seeded `keep` with every door/stairs/obstacle *before* the cap was consulted,
so a corridor with six doors returned six nodes under `max_nodes=2`. The whole
point of the cap is the user's attention budget. Forced nodes are now ranked
among themselves by importance and truncated to the cap.

**Depth was sampled from a fixed 5 px radius regardless of object size.** On a
distant 12 px-wide door that patch reaches past the object into background
wall; on a near sofa filling the frame it samples a 10×10 speck dominated by
depth-map noise. Now sampled from the central ~30% of the object's own bounding
box, clamped to the box, so background pixels can never be mixed in.

**A `nan`/`inf` depth silently deleted an object.** `np.median` of a patch
containing `nan` returns `nan`; `nan >= prune_threshold` is `False`, so the
object vanished from the graph with no error. Non-finite values are now
filtered, and a detection with no finite depth is dropped explicitly.

**`PruningConfig()` was a mutable default argument** on `prune_graph`,
evaluated once at import and shared by every call. Now `config: … | None = None`
with per-call construction, and the dataclass is `frozen=True`.

## Performance

**`eval/batch_eval.py` reloaded YOLO + Depth-Anything on every frame.**
`run_pipeline` constructed its own `DetectorDepthEstimator` per call, so a
51-frame NYU run paid both checkpoint loads 51 times. Added
`pipeline.build_detector()`; `run_pipeline(detector=...)` reuses it, and
`run_batch` builds it once. This is the single largest win in the repo and it
changes no numbers, only wall-clock time.

**`prune_graph` scored all `n²` object↔object edges and then masked them out.**
They were computed, label-looked-up, weighted, and discarded. Now only the
`n` ego edges are scored.

**`build_graph(include_object_edges=False)`** skips constructing those edges at
all; `live_camera_loop_tracked.py` uses it, since nothing in the live path
reads them.

## Testing

**`test_synthetic.py` asserted nothing.** It printed and always passed — it
could not have caught the label-mapping bug (old known-issue 3) or the
threshold bug (old known-issue 5) that it was supposedly guarding. Now:

- `tests/test_pipeline_logic.py` — 29 assertion-backed test functions (more cases after parametrization), still needing no
  model weights. Includes explicit regression tests for both of those bugs,
  the heading wrap-around at ±π, the forced-keep budget, the ego/azimuth
  convention agreement between `graph_builder` and `encoders`, empty-scene
  handling, and the calibration fit against the three real measurements.
- `demo_synthetic.py` — the old printing script, renamed so `pytest` doesn't
  collect a file that can't fail.
- `pyproject.toml` with `pytest` and `ruff` config.

## Evaluation

**Term-wise ablation is now a flag.** `PruningConfig.use_affordance /
use_distance / use_heading`, surfaced as `--ablate heading distance affordance`
on `pipeline.py` and `eval/batch_eval.py`. A disabled term contributes `1.0`
rather than being removed, so `prune_threshold` stays comparable across runs.
`batch_eval`'s output JSON now records which terms were active along with
`tau`, `gamma`, `prune_threshold` and `max_nodes`, so a results file is
self-describing.

**Hyperparameters are exposed as flags** (`--tau`, `--gamma`,
`--prune-threshold`, `--max-nodes`) so a sensitivity sweep doesn't need source
edits.

**Edge compression demoted.** `prune_graph` returns a graph with zero edges by
construction, so edge compression is 100% on every frame with any edge at all.
It measures the output format, not the method. Still written to JSON; no longer
printed as a headline number.

## Hygiene

- `README.md` rewritten: real (flat) file layout instead of a `src/` tree that
  never existed, so every quickstart command now runs as written; issues split
  into closed and open.
- `result graph/` → `figures/` (a directory name with a space in it breaks the
  shell commands in the docs); `result_figures.py` → `generate_results_figure.py`
  to match the name the README and its own usage string already used.
- `requirements.txt` was missing `pillow`, which `detect_depth.py` imports
  directly. Split runtime from `requirements-dev.txt` (`pytest`, `matplotlib`,
  `requests`, `datasets`) so a Jetson image needn't carry matplotlib.
- `--heading`'s help text said "0 = facing right in image plane"; the ego
  convention is 0 = straight ahead. Corrected, and the convention documented
  once in the README.
- `FORCED_KEEP_MIN_AFFORDANCE` constant plus `is_forced_keep()` replace the
  `0.9` literal that was duplicated across `_select_top_nodes` and
  `_assert_pruning_invariant`, where the two copies could drift.
- `graph_builder.MAX_DEPTH_M` replaces the `20.0` literal repeated across three
  signatures; `pruning.tau` is calibrated against that scale, so it needed one
  owner.
- Dead code removed: unused `math` / `field` / `DEFAULT_CLASS_VOCAB` imports,
  the unused `diag` in `ObjectTracker.update`. `pyflakes` is now clean.
- `encode_spatial_audio`'s `frame_w` / `frame_h` were never used (positions
  arrive pre-normalized). Made optional rather than removed, so existing call
  sites keep working.

## Not changed, deliberately

- The pruning formula, the affordance table, and all four default constants.
- `eval/simulate_walk.py` and the fetchers.
- The `sys.path.insert` import shims in `pipeline.py` / `eval/batch_eval.py`.
  Making `eval/` a real package means shadowing the builtin name `eval`;
  renaming it to `evaluation/` is the right fix but touches every documented
  command, so it's a decision rather than a cleanup.
- No licence file added — see README open issue 8. Publishing this repository
  is a disclosure event on the patent track, so the choice is yours to make
  deliberately.

## Verification status

`pyflakes` is clean and every module compiles. The `pytest` suite was written
against hand-computed expected values but **has not been executed here** —
`torch` and `torch-geometric` aren't installed in this environment. Run
`pytest` once locally before trusting it; any failure is far more likely to be
a wrong expectation in the test than a new bug in the pipeline, since no
scoring logic changed.
