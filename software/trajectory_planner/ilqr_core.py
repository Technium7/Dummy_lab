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


def phased_base_joint_guess(
    start_q: np.ndarray,
    goal_q: np.ndarray,
    horizon: int,
    *,
    base_idx: int = 0,
    phase_frac: float = 0.35,
    mode: str = "linear",
) -> np.ndarray:
    """Initial joint path with base (J1) timed early, late, or linearly.

    Prefer ``plan_joint_trajectory(..., joint_guess_mode=early|late)`` which
    sequences two full plans (J1 alone first / last). This guess is the
    single-plan fallback when that midpoint is invalid.

    ``early``: J1 finishes in the first ``phase_frac`` of the horizon, then other joints.
    ``late``: other joints first, J1 in the last ``phase_frac``.
    ``linear``: same as ``linear_joint_guess``.
    """
    horizon = max(2, int(horizon))
    start = np.asarray(start_q, dtype=np.float64).reshape(-1)
    goal = np.asarray(goal_q, dtype=np.float64).reshape(-1)
    n_j = int(start.shape[0])
    mode_norm = str(mode).strip().lower()
    if mode_norm in ("", "linear", "none"):
        return linear_joint_guess(start, goal, horizon)

    frac = float(np.clip(phase_frac, 0.05, 0.9))
    n_phase = max(1, min(horizon - 1, int(round(horizon * frac))))
    bi = int(np.clip(base_idx, 0, n_j - 1))
    out = np.zeros((horizon, n_j), dtype=np.float64)
    out[0] = start
    out[-1] = goal

    if mode_norm == "early":
        # [0, n_phase]: move base only; [n_phase, end]: move other joints, base held at goal.
        for k in range(1, horizon - 1):
            q = start.copy()
            if k <= n_phase:
                a = k / float(n_phase)
                q[bi] = start[bi] + a * (goal[bi] - start[bi])
            else:
                q[bi] = goal[bi]
                a = (k - n_phase) / float(max(1, horizon - 1 - n_phase))
                for j in range(n_j):
                    if j == bi:
                        continue
                    q[j] = start[j] + a * (goal[j] - start[j])
            out[k] = q
    elif mode_norm == "late":
        # [0, split]: other joints; [split, end]: base.
        split = max(1, horizon - 1 - n_phase)
        for k in range(1, horizon - 1):
            q = start.copy()
            if k <= split:
                a = k / float(split)
                for j in range(n_j):
                    if j == bi:
                        continue
                    q[j] = start[j] + a * (goal[j] - start[j])
                q[bi] = start[bi]
            else:
                for j in range(n_j):
                    if j != bi:
                        q[j] = goal[j]
                a = (k - split) / float(max(1, horizon - 1 - split))
                q[bi] = start[bi] + a * (goal[bi] - start[bi])
            out[k] = q
    else:
        raise ValueError(f"joint guess mode must be early|late|linear, got {mode!r}")
    return out


def resolve_joint_guess(
    start_q: np.ndarray,
    goal_q: np.ndarray,
    horizon: int,
    joint_guess_mode: str | None = None,
) -> np.ndarray:
    mode = "linear" if joint_guess_mode is None else str(joint_guess_mode)
    return phased_base_joint_guess(start_q, goal_q, horizon, mode=mode)


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
