"""
Structural compression ratio: how much of the raw scene graph gets stripped
away by pruning, while checking whether safety-critical nodes (doors,
stairs, obstacles) survive.

Run this over a dataset (e.g. SUN RGB-D) of pre-annotated frames to produce
the number reviewers will want in the results table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompressionResult:
    raw_nodes: int
    raw_edges: int
    pruned_nodes: int
    pruned_edges: int
    critical_nodes_total: int
    critical_nodes_retained: int

    @property
    def node_compression(self) -> float:
        return 1.0 - (self.pruned_nodes / self.raw_nodes) if self.raw_nodes else 0.0

    @property
    def edge_compression(self) -> float:
        return 1.0 - (self.pruned_edges / self.raw_edges) if self.raw_edges else 0.0

    @property
    def critical_retention(self) -> float:
        return (
            self.critical_nodes_retained / self.critical_nodes_total
            if self.critical_nodes_total
            else 1.0
        )

    def summary(self) -> str:
        return (
            f"node compression: {self.node_compression:.1%} | "
            f"edge compression: {self.edge_compression:.1%} | "
            f"critical node retention: {self.critical_retention:.1%} "
            f"({self.critical_nodes_retained}/{self.critical_nodes_total})"
        )


def compute_compression(
    raw_labels: list[str],
    pruned_labels: list[str],
    raw_edge_count: int,
    pruned_edge_count: int,
    critical_classes: set[str] = frozenset({"door", "stairs", "obstacle"}),
) -> CompressionResult:
    critical_total = sum(1 for l in raw_labels if l in critical_classes)
    critical_retained = sum(1 for l in pruned_labels if l in critical_classes)

    return CompressionResult(
        raw_nodes=len(raw_labels),
        raw_edges=raw_edge_count,
        pruned_nodes=len(pruned_labels),
        pruned_edges=pruned_edge_count,
        critical_nodes_total=critical_total,
        critical_nodes_retained=critical_retained,
    )


if __name__ == "__main__":
    # Example usage on a single synthetic frame's worth of labels
    raw = ["chair", "table", "door", "plant", "person", "tv", "cabinet"]
    pruned = ["door", "person", "chair"]
    result = compute_compression(raw, pruned, raw_edge_count=42, pruned_edge_count=6)
    print(result.summary())
