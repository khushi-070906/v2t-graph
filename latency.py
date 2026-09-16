"""
Per-stage latency tracking for the live camera loops.

Exists specifically to close the README gap: "no per-frame real-time
timing budget measured yet." Before any claim about real-time operation
or Jetson deployment feasibility, you need actual per-stage timing data
from a real run, not an assumption. This module only measures and
reports -- it makes no decisions and doesn't change pipeline behavior.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from contextlib import contextmanager


class LatencyTracker:
    """
    Usage per frame:
        tracker = LatencyTracker(report_every=30, csv_path="latency.csv")
        with tracker.stage("detect_depth"):
            ...
        with tracker.stage("track"):
            ...
        with tracker.stage("graph_prune"):
            ...
        with tracker.stage("encode_instructions"):
            ...
        tracker.end_frame()   # call once per frame, after all stages

    Prints a rolling mean/max per stage (in ms) plus overall FPS every
    `report_every` frames. If csv_path is given, appends one row per
    frame with every stage's duration -- raw data for a latency
    breakdown plot/table later, not just the printed rolling summary.
    """

    def __init__(self, report_every: int = 30, csv_path: str | None = None):
        self.report_every = report_every
        self.csv_path = csv_path
        self._stage_durations: dict[str, list[float]] = defaultdict(list)
        self._frame_start: float | None = None
        self._frame_count = 0
        self._current_frame_stages: dict[str, float] = {}
        self._csv_file = None
        self._csv_writer = None
        self._csv_header_written = False

    @contextmanager
    def stage(self, name: str):
        if self._frame_start is None:
            self._frame_start = time.perf_counter()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._stage_durations[name].append(dt)
            self._current_frame_stages[name] = dt

    def end_frame(self) -> None:
        """Call once per frame after all stage() blocks for that frame."""
        if self._frame_start is None:
            return  # no stages were timed this frame, nothing to record
        frame_total = time.perf_counter() - self._frame_start
        self._stage_durations["total"].append(frame_total)
        self._current_frame_stages["total"] = frame_total
        self._frame_count += 1

        if self.csv_path is not None:
            self._write_csv_row()

        if self._frame_count % self.report_every == 0:
            self._print_summary()

        self._frame_start = None
        self._current_frame_stages = {}

    def _write_csv_row(self) -> None:
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "a", newline="")
            self._csv_writer = csv.writer(self._csv_file)
        if not self._csv_header_written:
            # write header only if file is empty (fresh file, not appending to existing)
            if self._csv_file.tell() == 0:
                self._csv_writer.writerow(["frame", "stage", "duration_ms"])
            self._csv_header_written = True
        for stage_name, dt in self._current_frame_stages.items():
            self._csv_writer.writerow([self._frame_count, stage_name, round(dt * 1000, 3)])
        self._csv_file.flush()

    def _print_summary(self) -> None:
        window = self.report_every
        print(f"\n--- Latency summary (last {window} frames) ---")
        for stage_name, durations in self._stage_durations.items():
            recent = durations[-window:]
            mean_ms = 1000 * sum(recent) / len(recent)
            max_ms = 1000 * max(recent)
            print(f"  {stage_name:20s} mean={mean_ms:7.1f}ms  max={max_ms:7.1f}ms")
        total_recent = self._stage_durations.get("total", [])[-window:]
        if total_recent:
            fps = 1.0 / (sum(total_recent) / len(total_recent))
            print(f"  {'-> effective FPS':20s} {fps:.2f}")
        print()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
