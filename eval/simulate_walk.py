"""
Simulated user-walking evaluation: compares three "ranking" strategies on
a synthetic hallway with obstacles, without needing any real hardware or a
real blind participant.

Strategies compared (this is the three-way comparison your paper needs —
two-way vs. only "no graph" is not enough, reviewers will ask why GNN
pruning beats simple heuristics):
    1. linear_baseline   — obstacles are announced in raw list order,
                            no priority at all
    2. distance_only      — obstacles ranked by distance only, no heading/affordance
    3. v2t_pruned         — full heading + affordance + distance pruning (this project)

Primary metric: attention_precision@K — as the agent advances step by step,
each strategy proposes its top-K ranked obstacles as the ones "worth telling
the user about" (K = limited attention slots, simulating the information-
overload constraint this project exists to solve). attention_precision@K is
the fraction of those K slots spent on a LIVE critical hazard — i.e. an
object that is both a real navigational hazard (door/stairs/obstacle) AND
still ahead of the agent (not already passed). This is the direct, literal
measurement of the paper's core claim; higher is better.

Ground truth is intentionally direction-aware (x > agent.x), not just a
static class label. This was discovered to matter during development:
distance_only correctly identifies the on-path blocker once close to it,
but then KEEPS flagging it as "nearest" indefinitely after the agent has
already passed it, since raw Euclidean distance can't distinguish ahead
from behind. A static ground-truth label wrongly credited that stale
flagging as correct, making distance_only look artificially competitive.
With direction-aware ground truth, v2t_pruned's heading term (which
correctly drops to ~0 for anything behind the agent) outperforms
distance_only as expected: ~26% vs ~11% attention precision in the
adversarial scene.

Secondary metric: collisions — included as a sanity check, but in this toy
simulation the agent's reactive dodge is forgiving enough that collisions
stay near zero for all non-linear strategies regardless of ranking quality.
attention_precision is the metric that actually differentiates the three
strategies; state this explicitly in the paper rather than leading with
collision rate.

Two room modes:
    - random_room: general scattered-obstacle scene, broad but not very
      discriminative between strategies (most rankings converge when
      obstacles are uniformly scattered — this is included for completeness,
      not as the main result)
    - adversarial_room: specifically constructed to separate distance-only
      ranking from heading-aware ranking — THIS is the meaningful ablation
      for the paper

v2t_pruned scoring: this now calls the REAL graph_builder.py/pruning.py
pipeline instead of reimplementing edge_weight() locally (an earlier
version duplicated the formula here so this file could avoid a torch/PyG
dependency, which meant the eval numbers cited in the paper weren't
actually produced by the described method — see _priority_v2t below for
the top-down-world -> camera-frame projection this requires).
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass

# eval/ is a subdirectory of the project root, where graph_builder.py,
# pruning.py, and detect_depth.py actually live.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from detect_depth import Detection  # noqa: E402
from graph_builder import build_graph  # noqa: E402
from pruning import prune_graph, PruningConfig  # noqa: E402


@dataclass
class Obstacle:
    x: int
    y: int
    label: str
    critical: bool = False  # ground-truth: is this a real navigational hazard?


@dataclass
class SimResult:
    collisions: int
    steps: int
    avg_attention_precision: float  # mean fraction of top-K slots spent on critical objects
    reached_goal: bool


CRITICAL_LABELS = {"door", "stairs", "obstacle"}  # matches pruning.py's forced-keep threshold (>=0.9)


def random_room(
    width: int, height: int, n_obstacles: int, seed: int
) -> tuple[list[Obstacle], tuple[int, int], tuple[int, int], tuple[float, float]]:
    rng = random.Random(seed)
    labels = ["chair", "table", "person", "plant", "door", "obstacle"]
    mid_y = height // 2
    obstacles = [
        Obstacle(
            x=rng.randint(0, width - 1),
            y=(y := rng.randint(0, height - 1)),
            label=(lbl := rng.choice(labels)),
            critical=(lbl in CRITICAL_LABELS and y == mid_y),
        )
        for _ in range(n_obstacles)
    ]
    start = (0, mid_y)
    goal = (width - 1, mid_y)
    heading = (1.0, 0.0)
    return obstacles, start, goal, heading


def adversarial_room(
    width: int, height: int, n_clutter: int, seed: int
) -> tuple[list[Obstacle], tuple[int, int], tuple[int, int], tuple[float, float]]:
    """
    Stress-tests the specific failure mode distance-only ranking has and
    heading-aware ranking doesn't.

    Setup: a straight hallway, agent starts at the left-center and heads
    due right (heading = (1, 0)) toward the goal at the right-center.

    - Clutter objects (plant/chair/table) are placed CLOSE to the agent in
      Euclidean distance but well OFF to the side (large |dy| relative to
      dx) — i.e. near, but not in the direction the agent is walking.
    - ONE real blocker ("obstacle", critical=True) is placed FURTHER away
      in Euclidean distance but DIRECTLY ON the heading line (dy=0) — i.e.
      far, but squarely in the walking path.

    distance_only ranks by raw distance, so early in the walk (when the
    agent is still near the clutter) it burns its limited attention slots
    on the close-but-irrelevant clutter instead of the farther, genuinely
    critical blocker. v2t_pruned's heading term should down-weight the
    off-path clutter (large angular deviation from heading) despite it
    being closer, and correctly prioritize the on-path blocker despite it
    being farther — this is exactly what attention_precision measures.
    """
    rng = random.Random(seed)
    mid_y = height // 2
    start = (0, mid_y)
    goal = (width - 1, mid_y)
    heading = (1.0, 0.0)

    obstacles = []
    clutter_labels = ["plant", "chair", "table"]
    for _ in range(n_clutter):
        dx = rng.randint(1, 3)               # close in x
        dy_sign = rng.choice([-1, 1])
        dy = dy_sign * rng.randint(3, 4)      # far off to the side in y
        cy = max(0, min(height - 1, mid_y + dy))
        obstacles.append(Obstacle(x=dx, y=cy, label=rng.choice(clutter_labels), critical=False))

    # Real blocker: further ahead (large dx), but exactly on the heading line (dy=0)
    blocker_x = min(width - 1, width // 2)
    obstacles.append(Obstacle(x=blocker_x, y=mid_y, label="obstacle", critical=True))

    return obstacles, start, goal, heading


def _priority_linear(obstacles: list[Obstacle], agent_pos: tuple[int, int]) -> list[Obstacle]:
    return obstacles  # raw order, as given — this is the "chaotic list" baseline


def _priority_distance_only(obstacles: list[Obstacle], agent_pos: tuple[int, int]) -> list[Obstacle]:
    ax, ay = agent_pos
    return sorted(obstacles, key=lambda o: (o.x - ax) ** 2 + (o.y - ay) ** 2)


def _bearing_and_distance(
    agent_pos: tuple[int, int], heading: tuple[float, float], obj_pos: tuple[float, float]
) -> tuple[float, float]:
    """
    (bearing_rad, distance) of obj_pos relative to agent_pos and heading.
    bearing_rad is signed, 0 = straight ahead, +-pi = directly behind —
    the same "angle relative to where the camera is pointed" convention
    graph_builder._ego_edge_attr assumes when it derives angle from a
    normalized frame x-position.
    """
    ax, ay = agent_pos
    hx, hy = heading
    hnorm = math.hypot(hx, hy) or 1.0
    hx, hy = hx / hnorm, hy / hnorm

    dx, dy = obj_pos[0] - ax, obj_pos[1] - ay
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return 0.0, 0.0

    heading_angle = math.atan2(hy, hx)
    obj_angle = math.atan2(dy, dx)
    bearing = obj_angle - heading_angle
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
    return bearing, dist


def _priority_v2t(
    obstacles: list[Obstacle],
    agent_pos: tuple[int, int],
    heading: tuple[float, float],
    max_depth: float,
) -> list[Obstacle]:
    """
    Ranks obstacles by calling the REAL graph_builder.build_graph() /
    pruning.prune_graph() pipeline, instead of a local reimplementation
    of edge_weight() (see this module's docstring for why that mattered).

    Projection: the sim's top-down (agent, heading-vector, x/y obstacles)
    world becomes the pipeline's forward-facing-camera model by treating
    each obstacle's position relative to the agent as a (depth, bearing)
    pair, as if a chest-mounted camera were pointed exactly along
    `heading` — bearing 0 = straight ahead, matching graph_builder's
    ego->object edge convention (see _ego_edge_attr). Obstacles outside
    the camera's +-90 degree horizontal field of view get their bearing
    CLAMPED to +-pi/2 rather than dropped from the detection list:
    pruning.py's heading_alignment() already scores anything past +-90
    degrees to exactly 0 (cos-based falloff, floored at 0 "behind the
    user"), so clamping and excluding produce identical scores — clamping
    is just simpler than filtering detections up front.

    max_depth is passed straight through to build_graph()'s own
    normalization (see graph_builder.py) — it should roughly match the
    room's scale in the same units as agent/obstacle coordinates, the
    same role it plays for real metric depth in meters.
    """
    detections: list[Detection] = []
    labels: list[str] = []
    for i, o in enumerate(obstacles):
        bearing, dist = _bearing_and_distance(agent_pos, heading, (o.x, o.y))
        clipped_bearing = max(-math.pi / 2, min(math.pi / 2, bearing))
        # Inverse of graph_builder._ego_edge_attr's `angle = (nx - 0.5) * pi`.
        nx = 0.5 + clipped_bearing / math.pi
        detections.append(
            Detection(
                obj_id=i,
                label=o.label,
                confidence=1.0,
                bbox_xyxy=(0.0, 0.0, 0.0, 0.0),  # unused by build_graph/prune_graph
                centroid_px=(nx, 0.5),  # frame_size=(1,1) below, so this IS the normalized position; ny is unused for ego-edge scoring
                depth_m=max(dist, 1e-3),
            )
        )
        labels.append(o.label)

    graph = build_graph(detections, frame_size=(1.0, 1.0), max_depth=max_depth)

    # prune_threshold=-1 and max_nodes=None: nothing should be dropped
    # here — every obstacle needs a score so the sim can pick its own
    # top-K via attention_slots. Pruning to a fixed node budget is a
    # separate concern (compression_ratio.py), not this ranking.
    pruned = prune_graph(
        graph,
        detections_labels=labels,
        heading_rad=0.0,  # bearings above are already heading-relative
        config=PruningConfig(prune_threshold=-1.0, max_nodes=None),
    )

    # Nothing was dropped above, so kept_node_indices == range(n) and
    # kept_importance is aligned row-for-row with it (see pruning.py's
    # _subgraph) — i.e. importance[k] is obstacles[k]'s real score.
    importance = pruned.kept_importance.tolist()
    ranked_idx = sorted(range(len(obstacles)), key=lambda i: importance[i], reverse=True)
    return [obstacles[i] for i in ranked_idx]


def simulate(
    strategy: str,
    width: int,
    height: int,
    n_obstacles: int,
    seed: int,
    room_mode: str = "random",
    attention_slots: int = 2,
) -> SimResult:
    if room_mode == "random":
        obstacles, start, goal, heading = random_room(width, height, n_obstacles, seed)
    elif room_mode == "adversarial":
        obstacles, start, goal, heading = adversarial_room(width, height, n_obstacles, seed)
    else:
        raise ValueError(room_mode)

    n_critical_total = sum(1 for o in obstacles if o.critical)

    pos = list(start)
    collisions = 0
    steps = 0
    precisions = []
    max_steps = width * height * 2
    obstacle_set = {(o.x, o.y): o for o in obstacles}

    while pos[0] < goal[0]:
        steps += 1
        if steps > max_steps:
            avg_p = sum(precisions) / len(precisions) if precisions else 0.0
            return SimResult(collisions=collisions, steps=steps, avg_attention_precision=avg_p, reached_goal=False)

        if strategy == "linear_baseline":
            ranked = _priority_linear(obstacles, tuple(pos))
        elif strategy == "distance_only":
            ranked = _priority_distance_only(obstacles, tuple(pos))
        elif strategy == "v2t_pruned":
            ranked = _priority_v2t(obstacles, tuple(pos), heading, max_depth=float(max(width, height)))
        else:
            raise ValueError(strategy)

        known = ranked[:attention_slots]

        # attention_precision@K for this step: fraction of the K flagged
        # objects that are LIVE critical hazards — i.e. genuinely on-path
        # AND still ahead of the agent (x > pos[0]). An object the agent
        # has already passed is no longer worth flagging even if it's a
        # critical class; a strategy that keeps flagging it (distance_only
        # does exactly this, since raw distance doesn't distinguish ahead
        # from behind) is wasting an attention slot, not using it well.
        if n_critical_total > 0:
            n_live_critical_flagged = sum(1 for o in known if o.critical and o.x > pos[0])
            precisions.append(n_live_critical_flagged / attention_slots)

        # Collision check: did a critical object sitting directly in the
        # next cell go unflagged?
        straight_ahead = (pos[0] + 1, pos[1])
        known_coords = {(o.x, o.y) for o in known}
        if straight_ahead in obstacle_set and straight_ahead not in known_coords:
            collisions += 1
            next_pos = [pos[0] + 1, pos[1] + (1 if pos[1] < height - 1 else -1)]
        elif straight_ahead in known_coords:
            # flagged in time -> swerve around it
            next_pos = [pos[0] + 1, pos[1] + (1 if pos[1] < height - 1 else -1)]
        else:
            next_pos = list(straight_ahead)

        pos = next_pos

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    return SimResult(collisions=collisions, steps=steps, avg_attention_precision=avg_p, reached_goal=True)


def run_comparison(
    n_trials: int = 200,
    width: int = 20,
    height: int = 20,
    n_obstacles: int = 15,
    room_mode: str = "random",
    attention_slots: int = 2,
):
    strategies = ["linear_baseline", "distance_only", "v2t_pruned"]
    results = {s: [] for s in strategies}

    for seed in range(n_trials):
        for s in strategies:
            results[s].append(
                simulate(s, width, height, n_obstacles, seed, room_mode=room_mode, attention_slots=attention_slots)
            )

    print(f"-- room_mode={room_mode}, attention_slots={attention_slots} --")
    print(f"{'strategy':16s} {'avg_collisions':>15s} {'avg_steps':>10s} {'attn_precision':>15s} {'goal_rate':>10s}")
    for s in strategies:
        rs = results[s]
        avg_collisions = sum(r.collisions for r in rs) / len(rs)
        avg_steps = sum(r.steps for r in rs) / len(rs)
        avg_precision = sum(r.avg_attention_precision for r in rs) / len(rs)
        goal_rate = sum(r.reached_goal for r in rs) / len(rs)
        print(f"{s:16s} {avg_collisions:15.2f} {avg_steps:10.1f} {avg_precision:15.1%} {goal_rate:10.1%}")


if __name__ == "__main__":
    # General random-room stats (broad, less discriminative between strategies)
    run_comparison(room_mode="random", n_obstacles=15, attention_slots=3)
    print()
    # The actual ablation: adversarial scenes built to separate
    # distance-only ranking from heading-aware ranking. n_obstacles here
    # controls clutter count, not total obstacles (see adversarial_room).
    run_comparison(room_mode="adversarial", n_obstacles=4, attention_slots=2)