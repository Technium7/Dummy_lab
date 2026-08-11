"""Sequential convex programming (SCP) for collision-aware trajectories.

Phase 1 — nominal discrete iLQR (no collision).
Phase 2 — linearize env constraints ``c(q) <= 0``, solve convex iLQR subproblem, repeat.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .collision import (
    batch_constraint_linearization,
    batch_constraint_values,
    batch_scp_linearized_penalties,
    is_state_valid,
    signed_obstacle_distance,
)
from .config import (
    DEFAULT_CLEARANCE_MARGIN,
    DEFAULT_ENDPOINT_RELAX_FRACTION,
    DEFAULT_ENDPOINT_SKIP_FRACTION,
    DEFAULT_HORIZON,
    DEFAULT_MAX_CONTROL_STEP_RAD,
    DEFAULT_N_ILQR_ITERATIONS,
    DEFAULT_Q_WEIGHT,
    DEFAULT_QF_WEIGHT,
    DEFAULT_R_WEIGHT,
    DEFAULT_REG,
    DEFAULT_RESAMPLE_TO,
    DEFAULT_SCP_CONSTRAINT_WEIGHT,
    DEFAULT_SCP_ILQR_PER_OUTER,
    DEFAULT_SCP_OUTER_ITERATIONS,
    DEFAULT_SCP_TRUST_REGION_RAD,
    DEFAULT_SCP_VIOLATION_TOL,
    DEFAULT_SCP_WEIGHT_SCALE,
    DEFAULT_PLANNER_COLLISION_STRIDE,
)
from .ilqr_core import (
    clip_joint_states,
    default_line_search_alphas,
    get_ilqr_solver,
    ilqr_iteration,
    resolve_joint_guess,
)
from .scene import PlanningScene
from .trajectory_utils import (
    pin_trajectory_endpoints,
    resample_trajectory,
    spread_terminal_approach,
)


def _clearance_margin_schedule(
    n: int,
    base_margin: float,
    goal_margin: float,
    endpoint_relax_fraction: float,
) -> np.ndarray:
    n = max(1, int(n))
    margins = np.full(n, float(base_margin), dtype=np.float64)
    frac = float(np.clip(endpoint_relax_fraction, 0.0, 1.0))
    tail = max(1, int(round(n * frac))) if frac > 0.0 else 0
    if tail <= 0:
        return margins
    start = max(0, n - tail)
    for i in range(start, n):
        t = (i - start) / max(1, n - 1 - start)
        margins[i] = (1.0 - t) * base_margin + t * goal_margin
    return margins


def _scp_weight_schedule(
    n: int,
    base_weight: float,
    endpoint_relax_fraction: float,
    endpoint_weight_frac: float = 0.15,
) -> np.ndarray:
    n = max(1, int(n))
    weights = np.full(n, float(base_weight), dtype=np.float64)
    frac = float(np.clip(endpoint_relax_fraction, 0.0, 1.0))
    w_end = base_weight * float(np.clip(endpoint_weight_frac, 0.0, 1.0))
    tail = max(1, int(round(n * frac))) if frac > 0.0 else 0
    if tail <= 0:
        return weights
    start = max(0, n - tail)
    for i in range(start, n):
        t = (i - start) / max(1, n - 1 - start)
        weights[i] = (1.0 - t) * base_weight + t * w_end
    return weights


def _apply_trust_region(
    traj: np.ndarray,
    ref: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    delta_rad: float,
    active: np.ndarray | None = None,
    *,
    pin_terminal_joint: bool = True,
) -> np.ndarray:
    out = np.array(traj, dtype=np.float64, copy=True)
    delta = float(delta_rad)
    if active is not None:
        active = np.asarray(active, dtype=bool)
        for i in range(len(out)):
            if active[i]:
                out[i] = ref[i] + np.clip(traj[i] - ref[i], -delta, delta)
    else:
        out = ref + np.clip(traj - ref, -delta, delta)
    out[0] = start
    if pin_terminal_joint:
        out[-1] = goal
    return out


def _finalize_trajectory(
    traj: np.ndarray,
    start: np.ndarray,
    joint_goal: np.ndarray,
    resample_to: int,
    max_control_step: float,
) -> np.ndarray:
    dense = resample_trajectory(traj, resample_to)
    dense = spread_terminal_approach(dense, joint_goal, max_step_rad=max_control_step)
    return pin_trajectory_endpoints(dense, start, joint_goal, pin_goal=True)


def _run_nominal_ilqr(
    scene: PlanningScene,
    start: np.ndarray,
    stage_goal: np.ndarray,
    horizon: int,
    n_ilqr_iterations: int,
    low: np.ndarray,
    high: np.ndarray,
    alphas: list[float],
    max_control_step: float,
    q_weight: float,
    r_weight: float,
    qf_weight: float,
    reg: float,
    verbose: bool,
    *,
    joint_terminal_goal: np.ndarray | None = None,
    joint_guess_mode: str | None = None,
) -> np.ndarray:
    solver = get_ilqr_solver()
    traj = resolve_joint_guess(start, stage_goal, horizon, joint_guess_mode)
    traj[0] = start
    jterm = joint_terminal_goal if joint_terminal_goal is not None else stage_goal
    traj[-1] = jterm
    traj = clip_joint_states(traj, low, high)
    u_nom = traj[1:] - traj[:-1]
    zero_pen = np.zeros(horizon, dtype=np.float64)
    zero_grad = np.zeros((horizon, start.shape[0]), dtype=np.float64)

    if verbose:
        mode_s = joint_guess_mode or "linear"
        print(
            f"SCP phase 1: nominal iLQR ({n_ilqr_iterations} iters, no collision, "
            f"joint goal, guess={mode_s})",
            flush=True,
        )
    for _ in range(max(1, int(n_ilqr_iterations))):
        traj, u_nom, accepted = ilqr_iteration(
            solver,
            scene,
            traj,
            u_nom,
            start,
            stage_goal,
            low,
            high,
            alphas,
            max_control_step,
            q_weight,
            r_weight,
            qf_weight,
            reg,
            zero_pen,
            zero_grad,
            terminal_goal=jterm,
        )
        if not accepted:
            break
    return traj


def plan_nominal_ilqr_trajectory(
    scene: PlanningScene,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    horizon: int = DEFAULT_HORIZON,
    n_ilqr_iterations: int = DEFAULT_N_ILQR_ITERATIONS,
    q_weight: float = DEFAULT_Q_WEIGHT,
    qf_weight: float = DEFAULT_QF_WEIGHT,
    r_weight: float = DEFAULT_R_WEIGHT,
    reg: float = DEFAULT_REG,
    max_control_step: float = DEFAULT_MAX_CONTROL_STEP_RAD,
    line_search_alphas: Optional[Sequence[float]] = None,
    resample_to: Optional[int] = None,
    verbose: bool = True,
    joint_guess_mode: str | None = None,
) -> np.ndarray:
    """Collision-free discrete iLQR."""
    start = np.asarray(start_q, dtype=np.float64)
    stage_goal = np.asarray(goal_q, dtype=np.float64)
    horizon = max(4, int(horizon))
    out_samples = int(resample_to) if resample_to is not None else horizon
    low = np.asarray(scene.lower_limits, dtype=np.float64)
    high = np.asarray(scene.upper_limits, dtype=np.float64)
    alphas = default_line_search_alphas(line_search_alphas)

    traj = _run_nominal_ilqr(
        scene,
        start,
        stage_goal,
        horizon,
        n_ilqr_iterations,
        low,
        high,
        alphas,
        max_control_step,
        q_weight,
        r_weight,
        qf_weight,
        reg,
        verbose,
        joint_terminal_goal=stage_goal,
        joint_guess_mode=joint_guess_mode,
    )
    return _finalize_trajectory(traj, start, stage_goal, out_samples, max_control_step)


def plan_scp_from_nominal(
    scene: PlanningScene,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    nominal_traj: np.ndarray,
    horizon: int = DEFAULT_HORIZON,
    clearance_margin: float = DEFAULT_CLEARANCE_MARGIN,
    q_weight: float = DEFAULT_Q_WEIGHT,
    qf_weight: float = DEFAULT_QF_WEIGHT,
    r_weight: float = DEFAULT_R_WEIGHT,
    reg: float = DEFAULT_REG,
    max_control_step: float = DEFAULT_MAX_CONTROL_STEP_RAD,
    line_search_alphas: Optional[Sequence[float]] = None,
    resample_to: int = DEFAULT_RESAMPLE_TO,
    scp_outer_iterations: int = DEFAULT_SCP_OUTER_ITERATIONS,
    scp_ilqr_per_outer: int = DEFAULT_SCP_ILQR_PER_OUTER,
    scp_constraint_weight: float = DEFAULT_SCP_CONSTRAINT_WEIGHT,
    scp_weight_scale: float = DEFAULT_SCP_WEIGHT_SCALE,
    scp_violation_tol: float = DEFAULT_SCP_VIOLATION_TOL,
    scp_trust_region_rad: float = DEFAULT_SCP_TRUST_REGION_RAD,
    collision_stride: int = DEFAULT_PLANNER_COLLISION_STRIDE,
    exact_clearance: bool = False,
    endpoint_relax_fraction: float = DEFAULT_ENDPOINT_RELAX_FRACTION,
    endpoint_skip_fraction: float = DEFAULT_ENDPOINT_SKIP_FRACTION,
    verbose: bool = True,
) -> np.ndarray:
    """SCP collision refinement from a collision-free nominal trajectory."""
    start = np.asarray(start_q, dtype=np.float64)
    stage_goal = np.asarray(goal_q, dtype=np.float64)
    horizon = max(4, int(horizon))
    low = np.asarray(scene.lower_limits, dtype=np.float64)
    high = np.asarray(scene.upper_limits, dtype=np.float64)
    alphas = default_line_search_alphas(line_search_alphas)
    solver = get_ilqr_solver()
    stride = max(1, int(collision_stride))
    skip_tail = max(1, int(round(horizon * float(endpoint_skip_fraction))))

    d_ref = signed_obstacle_distance(scene, stage_goal, exact_clearance=exact_clearance)
    goal_margin = max(0.0, min(float(clearance_margin), d_ref - 0.001))
    margin_profile = _clearance_margin_schedule(
        horizon, clearance_margin, goal_margin, endpoint_relax_fraction
    )

    coarse = resample_trajectory(np.asarray(nominal_traj, dtype=np.float64), horizon)
    traj = clip_joint_states(
        pin_trajectory_endpoints(coarse, start, stage_goal, pin_goal=True),
        low,
        high,
    )
    u_nom = traj[1:] - traj[:-1]

    violations = batch_constraint_values(
        scene, traj, margin_profile, exact_clearance=exact_clearance, stride=stride
    )
    check = violations[: max(1, horizon - skip_tail)]
    nominal_max_v = float(np.max(check))

    if verbose:
        mode = "exact per-link" if exact_clearance else "fast arm-link"
        print(
            f"SCP phase 2: sequential convex programming "
            f"({scp_outer_iterations} linearize + solve, stride={stride}, {mode}, joint goal)",
            flush=True,
        )
        print(
            f"  margin {clearance_margin * 1000:.1f} mm -> {goal_margin * 1000:.1f} mm at goal; "
            f"skip SCP on last {skip_tail} waypoints",
            flush=True,
        )
        print(
            f"  nominal max violation={nominal_max_v * 1000:.2f} mm",
            flush=True,
        )

    # Already clear of env (+ margin): keep nominal iLQR path, skip SCP solve.
    if nominal_max_v <= float(scp_violation_tol):
        if verbose:
            print(
                f"  skip SCP (nominal already feasible, "
                f"tol {scp_violation_tol * 1000:.2f} mm)",
                flush=True,
            )
        return _finalize_trajectory(traj, start, stage_goal, resample_to, max_control_step)

    scp_weight = float(scp_constraint_weight)

    for outer in range(max(1, int(scp_outer_iterations))):
        ref = np.array(traj, dtype=np.float64, copy=True)
        weight_profile = _scp_weight_schedule(horizon, scp_weight, endpoint_relax_fraction)
        c_ref, grad_c = batch_constraint_linearization(
            scene,
            ref,
            margin_profile,
            stride=stride,
            active_only=True,
            exact_clearance=exact_clearance,
            skip_tail=skip_tail,
        )

        if verbose:
            n_active = int(np.sum(c_ref > 0.0))
            print(
                f"  SCP iter {outer + 1}: weight={scp_weight:.0f}, "
                f"active linearized constraints={n_active}/{horizon - skip_tail}",
                flush=True,
            )

        for _inner in range(max(1, int(scp_ilqr_per_outer))):
            col_pen, col_grad = batch_scp_linearized_penalties(
                traj, ref, c_ref, grad_c, weight_profile
            )
            traj, u_nom, accepted = ilqr_iteration(
                solver,
                scene,
                traj,
                u_nom,
                start,
                stage_goal,
                low,
                high,
                alphas,
                max_control_step,
                q_weight,
                r_weight,
                qf_weight,
                reg,
                col_pen,
                col_grad,
                terminal_goal=stage_goal,
            )
            if not accepted:
                break

        traj = _apply_trust_region(
            traj,
            ref,
            start,
            stage_goal,
            float(scp_trust_region_rad),
            active=(c_ref > 1e-6),
            pin_terminal_joint=True,
        )
        traj = clip_joint_states(traj, low, high)
        u_nom = traj[1:] - traj[:-1]

        violations = batch_constraint_values(
            scene, traj, margin_profile, exact_clearance=exact_clearance, stride=stride
        )
        check = violations[: max(1, horizon - skip_tail)]
        max_v = float(np.max(check))
        mean_v = float(np.mean(np.maximum(check, 0.0)))

        if verbose:
            print(
                f"  true constraint: max+={max_v * 1000:.2f} mm, mean+={mean_v * 1000:.2f} mm",
                flush=True,
            )

        if max_v <= scp_violation_tol:
            if verbose:
                print(
                    f"  SCP converged (max violation <= {scp_violation_tol * 1000:.2f} mm)",
                    flush=True,
                )
            break

        scp_weight *= float(scp_weight_scale)

    return _finalize_trajectory(traj, start, stage_goal, resample_to, max_control_step)


_ILQR_KW = (
    "q_weight",
    "qf_weight",
    "r_weight",
    "reg",
    "max_control_step",
    "line_search_alphas",
    "joint_guess_mode",
)
# SCP refinement reuses iLQR weights but not the initial-guess mode (nominal already built).
_SCP_KW = (
    "q_weight",
    "qf_weight",
    "r_weight",
    "reg",
    "max_control_step",
    "line_search_alphas",
    "scp_constraint_weight",
    "scp_weight_scale",
    "scp_violation_tol",
    "scp_trust_region_rad",
    "exact_clearance",
    "endpoint_relax_fraction",
    "endpoint_skip_fraction",
)


def plan_scp_trajectory(
    scene: PlanningScene,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    horizon: int = DEFAULT_HORIZON,
    n_ilqr_iterations: int = DEFAULT_N_ILQR_ITERATIONS,
    resample_to: int = DEFAULT_RESAMPLE_TO,
    clearance_margin: float = DEFAULT_CLEARANCE_MARGIN,
    scp_outer_iterations: int = DEFAULT_SCP_OUTER_ITERATIONS,
    scp_ilqr_per_outer: int = DEFAULT_SCP_ILQR_PER_OUTER,
    scp_constraint_weight: float = DEFAULT_SCP_CONSTRAINT_WEIGHT,
    scp_weight_scale: float = DEFAULT_SCP_WEIGHT_SCALE,
    scp_violation_tol: float = DEFAULT_SCP_VIOLATION_TOL,
    scp_trust_region_rad: float = DEFAULT_SCP_TRUST_REGION_RAD,
    collision_stride: int = DEFAULT_PLANNER_COLLISION_STRIDE,
    exact_clearance: bool = False,
    endpoint_relax_fraction: float = DEFAULT_ENDPOINT_RELAX_FRACTION,
    endpoint_skip_fraction: float = DEFAULT_ENDPOINT_SKIP_FRACTION,
    verbose: bool = True,
    **kwargs,
) -> Optional[np.ndarray]:
    """Nominal iLQR, then SCP collision refinement."""
    if not is_state_valid(scene, start_q) or not is_state_valid(scene, goal_q):
        return None

    ilqr_kw = {k: kwargs[k] for k in _ILQR_KW if k in kwargs}
    scp_kw = {k: kwargs[k] for k in _SCP_KW if k in kwargs}

    nominal = plan_nominal_ilqr_trajectory(
        scene,
        start_q,
        goal_q,
        horizon=horizon,
        n_ilqr_iterations=n_ilqr_iterations,
        resample_to=horizon,
        verbose=verbose,
        **ilqr_kw,
    )

    return plan_scp_from_nominal(
        scene,
        start_q,
        goal_q,
        nominal,
        horizon=horizon,
        clearance_margin=clearance_margin,
        resample_to=resample_to,
        scp_outer_iterations=scp_outer_iterations,
        scp_ilqr_per_outer=scp_ilqr_per_outer,
        collision_stride=collision_stride,
        exact_clearance=exact_clearance,
        endpoint_relax_fraction=endpoint_relax_fraction,
        endpoint_skip_fraction=endpoint_skip_fraction,
        verbose=verbose,
        **scp_kw,
    )
