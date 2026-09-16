# V-to-T Graph — Results Summary

## 1. Real-photo batch evaluation (NYU Depth V2, n=51 frames)

Real `yolov8s-worldv2.pt` + Depth-Anything-V2-Metric-Indoor-Small-hf weights, full pipeline (`detect_depth.py` -> `graph_builder.py` -> `pruning.py`). 51/51 frames succeeded, 0 failures.

| Metric | Mean | Std |
|---|---|---|
| Node compression | 52.6% | 29.8% |
| Edge compression | 100.0% | 0.0% |
| Critical object retention | 100.0% | 0.0% |
| Mean raw nodes / frame | 4.59 | — |
| Mean pruned nodes / frame | 2.10 | — |

Critical retention is computed over 7/51 frames that contained a door/stairs/obstacle — the other 44 vacuously score 1.0 (no critical object present) and are excluded, per `batch_eval.py`'s `_aggregate`.

**Caveat for the paper:** n=7 for critical retention is a small subgroup — a 100% result here should be reported as "no critical-object misses observed in the frames tested," not oversold as a general guarantee. A dataset run with more door/stairs frames would strengthen this number.

## 2. Attention precision ablation (simulated walking, adversarial hallway)

`simulate_walk.py`, K=2 attention slots, 200 trials/point, critical blockers spread along the hallway (not one fixed layout — see `adversarial_room`'s `n_blockers` param).

| Blockers in scene | linear_baseline | distance_only | v2t_pruned (ours) | v2t margin over distance_only |
|---|---|---|---|---|
| 1 | 0.0% | 10.6% ± 0.3% | 26.3% ± 0.0% | 2.5x |
| 2 | 0.0% | 28.3% ± 1.6% | 62.7% ± 2.0% | 2.2x |
| 3 | 0.0% | 34.8% ± 2.1% | 81.6% ± 0.0% | 2.3x |
| 4 | 0.0% | 38.4% ± 1.7% | 89.5% ± 1.7% | 2.3x |

Collision rate stays ~0 for both `distance_only` and `v2t_pruned` at every blocker count (the reactive dodge is forgiving enough that it doesn't differentiate strategies) — `attention_precision@K` is the metric that actually separates the three approaches, consistent with README's stated rationale for that metric.

The heading-aware margin over distance-only ranking is stable (2.2–2.5x) across scene difficulty, not an artifact of one adversarial layout.

## Figure

See `results_figure.png` — left panel: attention_precision@K vs. number of blockers with error bars; right panel: real-photo compression/retention bar chart.