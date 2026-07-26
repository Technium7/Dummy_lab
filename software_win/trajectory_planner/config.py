"""Shared planner defaults (iLQR + SCP)."""

from __future__ import annotations

from dataclasses import dataclass

from .collision import DEFAULT_CLEARANCE_MARGIN, DEFAULT_PLANNER_EXACT_CLEARANCE

# Discrete iLQR
DEFAULT_HORIZON = 80
DEFAULT_RESAMPLE_TO = 150
DEFAULT_N_ILQR_ITERATIONS = 35
DEFAULT_Q_WEIGHT = 8.0
DEFAULT_QF_WEIGHT = 120.0
DEFAULT_R_WEIGHT = 2.5
DEFAULT_REG = 1e-2
DEFAULT_MAX_CONTROL_STEP_RAD = 0.12
DEFAULT_LINE_SEARCH_ALPHAS = (1.0, 0.5, 0.25, 0.1, 0.05)

# SCP collision refinement
DEFAULT_SCP_OUTER_ITERATIONS = 4
DEFAULT_SCP_ILQR_PER_OUTER = 3
DEFAULT_SCP_CONSTRAINT_WEIGHT = 60.0
DEFAULT_SCP_WEIGHT_SCALE = 1.5
DEFAULT_SCP_VIOLATION_TOL = 1e-3
DEFAULT_SCP_TRUST_REGION_RAD = 0.1
DEFAULT_PLANNER_COLLISION_STRIDE = 12
DEFAULT_ENDPOINT_RELAX_FRACTION = 0.2
DEFAULT_ENDPOINT_SKIP_FRACTION = 0.1


@dataclass(frozen=True)
class IlqrConfig:
    horizon: int = DEFAULT_HORIZON
    n_iterations: int = DEFAULT_N_ILQR_ITERATIONS
    resample_to: int = DEFAULT_RESAMPLE_TO
    q_weight: float = DEFAULT_Q_WEIGHT
    qf_weight: float = DEFAULT_QF_WEIGHT
    r_weight: float = DEFAULT_R_WEIGHT
    reg: float = DEFAULT_REG
    max_control_step: float = DEFAULT_MAX_CONTROL_STEP_RAD


@dataclass(frozen=True)
class ScpConfig:
    clearance_margin: float = DEFAULT_CLEARANCE_MARGIN
    outer_iterations: int = DEFAULT_SCP_OUTER_ITERATIONS
    ilqr_per_outer: int = DEFAULT_SCP_ILQR_PER_OUTER
    constraint_weight: float = DEFAULT_SCP_CONSTRAINT_WEIGHT
    weight_scale: float = DEFAULT_SCP_WEIGHT_SCALE
    violation_tol: float = DEFAULT_SCP_VIOLATION_TOL
    trust_region_rad: float = DEFAULT_SCP_TRUST_REGION_RAD
    collision_stride: int = DEFAULT_PLANNER_COLLISION_STRIDE
    exact_clearance: bool = DEFAULT_PLANNER_EXACT_CLEARANCE
    endpoint_relax_fraction: float = DEFAULT_ENDPOINT_RELAX_FRACTION
    endpoint_skip_fraction: float = DEFAULT_ENDPOINT_SKIP_FRACTION
