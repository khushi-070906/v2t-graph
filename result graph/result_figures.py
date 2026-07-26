"""
Generates results_figure.png (two-panel results plot) from:
  1. simulate_walk.py's multi-blocker adversarial ablation output
     (hardcoded below -- copy fresh numbers out of simulate_walk.py's
     stdout if you re-run the ablation with different params)
  2. results.json -- the aggregate block produced by eval/batch_eval.py's
     real NYU Depth V2 batch run

Usage:
    python generate_results_figure.py --results-json results.json --out results_figure.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Panel 1 data: from simulate_walk.py's adversarial ablation, n_blockers=1..4 ---
# (attention_precision@K, K=2, 200 trials/point -- see run_comparison() in simulate_walk.py)
N_BLOCKERS = [1, 2, 3, 4]
DISTANCE_ONLY_MEAN = [10.6, 28.3, 34.8, 38.4]
DISTANCE_ONLY_STD = [0.3, 1.6, 2.1, 1.7]
V2T_MEAN = [26.3, 62.7, 81.6, 89.5]
V2T_STD = [0.0, 2.0, 0.0, 1.7]
LINEAR_MEAN = [0.0, 0.0, 0.0, 0.0]


def build_figure(results_json_path: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel 1: attention_precision@K vs scene difficulty ---
    ax = axes[0]
    ax.errorbar(
        N_BLOCKERS, DISTANCE_ONLY_MEAN, yerr=DISTANCE_ONLY_STD,
        marker="o", capsize=4, label="distance_only",
    )
    ax.errorbar(
        N_BLOCKERS, V2T_MEAN, yerr=V2T_STD,
        marker="s", capsize=4, label="v2t_pruned (ours)",
    )
    ax.plot(
        N_BLOCKERS, LINEAR_MEAN,
        marker="x", linestyle="--", color="gray", label="linear_baseline",
    )
    ax.set_xlabel("Number of critical blockers in adversarial hallway")
    ax.set_ylabel("attention_precision@K (%)")
    ax.set_title(
        "Attention precision vs. scene difficulty\n"
        "(adversarial room, K=2, n=200 trials/point)"
    )
    ax.set_xticks(N_BLOCKERS)
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Panel 2: real-photo batch_eval aggregate (from results.json) ---
    with open(results_json_path) as f:
        d = json.load(f)
    agg = d["aggregate"]

    ax2 = axes[1]
    labels = ["Node\ncompression", "Edge\ncompression", "Critical\nretention"]
    means = [
        agg["node_compression"]["mean"] * 100,
        agg["edge_compression"]["mean"] * 100,
        agg["critical_retention"]["mean"] * 100,
    ]
    stds = [
        agg["node_compression"]["std"] * 100,
        agg["edge_compression"]["std"] * 100,
        agg["critical_retention"]["std"] * 100,
    ]
    bars = ax2.bar(labels, means, yerr=stds, capsize=6, color=["#4C72B0", "#55A868", "#C44E52"])
    ax2.set_ylabel("%")
    ax2.set_ylim(0, 110)
    ax2.set_title(f"Real-photo batch eval\n(NYU Depth V2, n={agg['n_frames']} frames)")
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + 3, f"{m:.1f}%", ha="center", fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", default="results.json", help="path to batch_eval.py output")
    parser.add_argument("--out", default="results_figure.png")
    args = parser.parse_args()
    build_figure(args.results_json, args.out)