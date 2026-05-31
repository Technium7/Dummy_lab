"""Discrete-time iLQR entry point; SCP when collision planning is enabled."""

from __future__ import annotations

from typing import Optional, Sequence

from .config import (
    DEFAULT_CLEARANCE_MARGIN,
    DEFAULT_ENDPOINT_RELAX_FRACTION,
    DEFAULT_ENDPOINT_SKIP_FRACTION,
    DEFAULT_HORIZON,
    DEFAULT_N_ILQR_ITERATIONS,
    DEFAULT_PLANNER_COLLISION_STRIDE,
    DEFAULT_RESAMPLE_TO,
    DEFAULT_SCP_CONSTRAINT_WEIGHT,
    DEFAULT_SCP_ILQR_PER_OUTER,
    DEFAULT_SCP_OUTER_ITERATIONS,
    DEFAULT_SCP_TRUST_REGION_RAD,
    DEFAULT_SCP_VIOLATION_TOL,
    DEFAULT_SCP_WEIGHT_SCALE,
)
from .scene import PlanningScene


def plan_lqr_trajectory(
    scene: PlanningScene,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    horizon: int = DEFAULT_HORIZON,
    n_ilqr_iterations: int = DEFAULT_N_ILQR_ITERATIONS,
    q_weight: float = 8.0,
    qf_weight: float = 120.0,
    r_weight: float = 2.5,
    clearance_margin: float = DEFAULT_CLEARANCE_MARGIN,
    reg: float = 1e-2,
    max_control_step: float = 0.12,
    line_search_alphas: Optional[Sequence[float]] = None,
    resample_to: int = DEFAULT_RESAMPLE_TO,
    enable_collision_penalty: bool = False,
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
) -> Optional[np.ndarray]:
    """
    Optimize a joint trajectory with discrete iLQR (JAX Riccati).

    With ``enable_collision_penalty``, runs nominal iLQR then SCP collision refinement.
    Terminal cost always tracks ``goal_q`` joint angles.
    """
    if enable_collision_penalty:
        from .scp_planner import plan_scp_trajectory

        return plan_scp_trajectory(
            scene,
            start_q,
            goal_q,
            horizon=horizon,
            n_ilqr_iterations=n_ilqr_iterations,
            resample_to=resample_to,
            clearance_margin=clearance_margin,
            q_weight=q_weight,
            qf_weight=qf_weight,
            r_weight=r_weight,
            reg=reg,
            max_control_step=max_control_step,
            line_search_alphas=line_search_alphas,
            scp_outer_iterations=scp_outer_iterations,
            scp_ilqr_per_outer=scp_ilqr_per_outer,
            scp_constraint_weight=scp_constraint_weight,
            scp_weight_scale=scp_weight_scale,
            scp_violation_tol=scp_violation_tol,
            scp_trust_region_rad=scp_trust_region_rad,
            collision_stride=collision_stride,
            exact_clearance=exact_clearance,
            endpoint_relax_fraction=endpoint_relax_fraction,
            endpoint_skip_fraction=endpoint_skip_fraction,
            verbose=verbose,
        )

    from .scp_planner import plan_nominal_ilqr_trajectory

    return plan_nominal_ilqr_trajectory(
        scene,
        start_q,
        goal_q,
        horizon=horizon,
        n_ilqr_iterations=n_ilqr_iterations,
        q_weight=q_weight,
        qf_weight=qf_weight,
        r_weight=r_weight,
        reg=reg,
        max_control_step=max_control_step,
        line_search_alphas=line_search_alphas,
        resample_to=resample_to,
        verbose=verbose,
    )
