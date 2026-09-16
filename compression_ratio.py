"""
eval/compression_ratio.py — node/edge compression + critical-object
retention metric.

Reconstructed to match the interface pipeline.py already expects
(compute_compression(raw_labels, pruned_labels, raw_edge_count,
pruned_edge_count) -> an object with .summary()) and the output field
names already baked into results_sunrgbd.json / results.json:
    node_compression, edge_compression,
    critical_nodes_total, critical_nodes_retained, critical_retention

"Critical" objects are the force-keep affordance classes from
pruning.py (door / stairs / obstacle — see pruning.FORCE_KEEP_MIN_PRIORITY
and AFFORDANCE_PRIORITY). Retention is 1.0 (vacuously) when a frame had
zero critical objects to begin with, matching every "critical_nodes_total":
0 -> "critical_retention": 1.0 row already in results_sunrgbd.json.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Kept as a local constant (not imported from pruning.py) so this module
# stays usable standalone (e.g. batch_eval.py aggregating over many
# pipeline runs) without a hard dependency on pruning.py's internals.
# Mirrors pruning.AFFORDANCE_PRIORITY's force-keep classes
# (priority >= pruning.FORCE_KEEP_MIN_PRIORITY = 0.9).
CRITICAL_CLASSES = {"door", "stairs", "obstacle"}


@dataclass
class CompressionResult:
    raw_node_count: int
    pruned_node_count: int
    raw_edge_count: int
    pruned_edge_count: int
    node_compression: float
    edge_compression: float
    critical_nodes_total: int
    critical_nodes_retained: int
    critical_retention: float
    raw_labels: list[str] = field(default_factory=list)
    pruned_labels: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Node compression: {self.node_compression * 100:.1f}% "
            f"({self.raw_node_count} -> {self.pruned_node_count})\n"
            f"Edge compression: {self.edge_compression * 100:.1f}% "
            f"({self.raw_edge_count} -> {self.pruned_edge_count})\n"
            f"Critical-object retention: {self.critical_retention * 100:.1f}% "
            f"({self.critical_nodes_retained}/{self.critical_nodes_total})"
        )

    def to_dict(self) -> dict:
        return {
            "raw_nodes": self.raw_node_count,
            "raw_edges": self.raw_edge_count,
            "pruned_nodes": self.pruned_node_count,
            "pruned_edges": self.pruned_edge_count,
            "node_compression": self.node_compression,
            "edge_compression": self.edge_compression,
            "critical_nodes_total": self.critical_nodes_total,
            "critical_nodes_retained": self.critical_nodes_retained,
            "critical_retention": self.critical_retention,
            "raw_labels": self.raw_labels,
            "pruned_labels": self.pruned_labels,
        }


def compute_compression(
    raw_labels: list[str],
    pruned_labels: list[str],
    raw_edge_count: int,
    pruned_edge_count: int,
    critical_classes: set[str] = CRITICAL_CLASSES,
) -> CompressionResult:
    """
    node_compression / edge_compression: fraction REMOVED (1.0 = fully
    pruned away, 0.0 = nothing dropped) -- matches results_sunrgbd.json,
    e.g. raw_nodes=4, pruned_nodes=1 -> node_compression=0.75.

    critical_retention: fraction of critical-class objects (door/stairs/
    obstacle) present in raw_labels that survived into pruned_labels.
    Counted with multiplicity via Counter, so e.g. two "door" detections
    in raw_labels with only one surviving in pruned_labels correctly
    reports 1/2, not 1/1. Vacuously 1.0 when there were no critical
    objects in the raw frame at all (nothing to lose).
    """
    raw_node_count = len(raw_labels)
    pruned_node_count = len(pruned_labels)

    node_compression = (
        1.0 - (pruned_node_count / raw_node_count) if raw_node_count > 0 else 0.0
    )
    edge_compression = (
        1.0 - (pruned_edge_count / raw_edge_count) if raw_edge_count > 0 else 0.0
    )

    raw_counts = Counter(raw_labels)
    pruned_counts = Counter(pruned_labels)

    critical_nodes_total = sum(
        count for label, count in raw_counts.items() if label in critical_classes
    )
    critical_nodes_retained = sum(
        min(raw_counts[label], pruned_counts.get(label, 0))
        for label in raw_counts
        if label in critical_classes
    )
    critical_retention = (
        critical_nodes_retained / critical_nodes_total
        if critical_nodes_total > 0
        else 1.0
    )

    return CompressionResult(
        raw_node_count=raw_node_count,
        pruned_node_count=pruned_node_count,
        raw_edge_count=raw_edge_count,
        pruned_edge_count=pruned_edge_count,
        node_compression=node_compression,
        edge_compression=edge_compression,
        critical_nodes_total=critical_nodes_total,
        critical_nodes_retained=critical_nodes_retained,
        critical_retention=critical_retention,
        raw_labels=list(raw_labels),
        pruned_labels=list(pruned_labels),
    )


if __name__ == "__main__":
    # Toy example, matches this module's original standalone smoke test.
    raw = ["door", "chair", "plant", "person", "table"]
    pruned = ["door", "person"]
    result = compute_compression(
        raw_labels=raw, pruned_labels=pruned,
        raw_edge_count=20, pruned_edge_count=2,
    )
    print(result.summary())
