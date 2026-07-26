"""Terminal collision analysis for planned trajectories."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .collision import (
    CollisionMetrics,
    DEFAULT_CLEARANCE_MARGIN,
    evaluate_trajectory_collision,
)
from .scene import PlanningScene
from .trajectory_utils import representative_waypoint_map

# Re-export margin used in summary text (single source: collision.DEFAULT_CLEARANCE_MARGIN)
_CLEARANCE_MARGIN = DEFAULT_CLEARANCE_MARGIN


def print_collision_sample(metrics: CollisionMetrics, index: int, count: int) -> None:
    line = (
        f"  [{index:3d}/{count - 1}] "
        f"env: signed_d={metrics.signed_distance_m:+.4f} m  "
        f"pen={metrics.penetration_m * 1000:.2f} mm  "
        f"clr={metrics.clearance_m * 1000:.2f} mm  "
        f"valid={metrics.valid}  "
        f"penalty={metrics.penalty:.4f}"
    )
    if metrics.self_penetration_m > 1e-6:
        line += (
            f"  |  self_pen={metrics.self_penetration_m * 1000:.2f} mm "
            f"(ignored for valid/penalty)"
        )
    print(line, flush=True)


def print_collision_profile_summary(metrics: Sequence[CollisionMetrics]) -> None:
    n = len(metrics)
    if n == 0:
        return
    n_invalid = sum(1 for m in metrics if not m.valid)
    n_penalty = sum(1 for m in metrics if m.penalty > 0.0)
    signed = [m.signed_distance_m for m in metrics]
    penalties = [m.penalty for m in metrics]
    print("\n--- Collision summary (planner inputs) ---")
    print(f"Waypoints: {n}  |  invalid (pen > 0.1 mm): {n_invalid}  |  penalty > 0: {n_penalty}")
    print(
        f"Signed distance d (m): min={min(signed):+.4f}  max={max(signed):+.4f}  "
        f"(+ separated, − penetrating; clearance margin={_CLEARANCE_MARGIN} m)"
    )
    print(f"collision_penalty: max={max(penalties):.4f}  mean={sum(penalties) / n:.4f}")
    worst = min(range(n), key=lambda i: signed[i])
    m = metrics[worst]
    print(
        f"Worst sample [{worst}]: d={m.signed_distance_m:+.4f} m, "
        f"pen={m.penetration_m * 1000:.2f} mm, clr={m.clearance_m * 1000:.2f} mm, "
        f"valid={m.valid}, penalty={m.penalty:.4f}"
    )
    print("---\n")


def run_collision_analysis(
    scene: PlanningScene,
    trajectory: np.ndarray,
    stride: int,
    exact_clearance: bool,
) -> list[CollisionMetrics]:
    n = len(trajectory)
    representatives, _ = representative_waypoint_map(trajectory)
    n_unique = len(representatives)
    mode = "exact per-link clearance" if exact_clearance else "fast whole-body clearance"
    print(
        f"\nCollision analysis ({n} waypoints, {n_unique} unique poses, "
        f"stride={stride} on unique, {mode})"
    )
    print(
        "Columns: env signed_d, env penetration/clearance (mm), valid & penalty "
        "(environment only; self_pen shown separately if present)\n"
    )

    def on_sample(i: int, _n: int, m: CollisionMetrics) -> None:
        print_collision_sample(m, i, n)

    profile = evaluate_trajectory_collision(
        scene,
        trajectory,
        stride=stride,
        exact_clearance=exact_clearance,
        progress=True,
        on_sample=on_sample,
    )
    print_collision_profile_summary(profile)
    return profile
