"""Shared discrete iLQR helpers (JAX solver + one iteration step)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .config import DEFAULT_LINE_SEARCH_ALPHAS
from .lqr_jax import IlqrJaxSolver

_JAX_SOLVER: Optional[IlqrJaxSolver] = None


def get_ilqr_solver() -> IlqrJaxSolver:
    global _JAX_SOLVER
    if _JAX_SOLVER is None:
        _JAX_SOLVER = IlqrJaxSolver()
    return _JAX_SOLVER


def clip_joint_states(traj: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    out = traj.copy()
    np.clip(out, low, high, out=out)
    return out


def linear_joint_guess(start_q: np.ndarray, goal_q: np.ndarray, horizon: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, horizon)
    return start_q + t[:, None] * (goal_q - start_q)[None, :]


def ilqr_iteration(
    solver: IlqrJaxSolver,
    scene,
    traj: np.ndarray,
    u_nom: np.ndarray,
    start: np.ndarray,
    stage_goal: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    alphas: Sequence[float],
    max_control_step: float,
    q_weight: float,
    r_weight: float,
    qf_weight: float,
    reg: float,
    col_pen: np.ndarray,
    col_grad: np.ndarray,
    *,
    terminal_goal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """One iLQR iteration; returns updated ``(traj, u_nom)`` and whether a step was accepted."""
    del scene  # FK not used in joint-space terminal cost
    eff_grad = col_grad
    tg = terminal_goal if terminal_goal is not None else stage_goal

    total = solver.total_cost(
        traj,
        u_nom,
        stage_goal,
        col_pen,
        q_weight,
        r_weight,
        qf_weight,
        terminal_goal=tg,
    )
    k_ff, k_fb = solver.backward(
        traj,
        u_nom,
        stage_goal,
        eff_grad,
        q_weight,
        r_weight,
        qf_weight,
        reg,
        terminal_goal=tg,
    )
    candidates = solver.forward_candidates(
        traj, u_nom, k_ff, k_fb, start, stage_goal, alphas, low, high, max_control_step
    )
    for candidate in candidates:
        new_traj = np.array(candidate, dtype=np.float64, copy=True)
        new_traj[0] = start
        new_u = new_traj[1:] - new_traj[:-1]
        trial = solver.total_cost(
            new_traj,
            new_u,
            stage_goal,
            col_pen,
            q_weight,
            r_weight,
            qf_weight,
            terminal_goal=tg,
        )
        if trial <= total * 1.02:
            return new_traj, new_u, True
    return traj, u_nom, False


def default_line_search_alphas(
    alphas: Optional[Sequence[float]] = None,
) -> list[float]:
    return list(alphas or DEFAULT_LINE_SEARCH_ALPHAS)
