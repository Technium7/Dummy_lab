"""JIT-compiled discrete iLQR (JAX). PyBullet collision terms are supplied from NumPy."""

from __future__ import annotations

from functools import partial
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Match NumPy float64 used by the planner (avoids float64 truncation warnings).
jax.config.update("jax_enable_x64", True)

# Dynamics: q_{k+1} = q_k + u_k  =>  f_x = I, f_u = I


def _writable_numpy(x) -> np.ndarray:
    """JAX->NumPy buffers are often read-only; copy so callers can assign in-place."""
    return np.array(x, dtype=np.float64, copy=True)


@partial(jax.jit, static_argnums=())
def _ilqr_backward_pass(
    traj: jnp.ndarray,
    u_nom: jnp.ndarray,
    goal: jnp.ndarray,
    terminal_goal: jnp.ndarray,
    col_grad: jnp.ndarray,
    q_weight: float,
    r_weight: float,
    qf_weight: float,
    reg: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Riccati backward pass; returns feedforward k (H-1, dof) and feedback K (H-1, dof, dof)."""
    dof = traj.shape[1]
    eye = jnp.eye(dof)

    err_n = traj[-1] - terminal_goal
    vx = 2.0 * qf_weight * err_n + col_grad[-1]
    vxx = 2.0 * qf_weight * eye

    def body(carry: Tuple[jnp.ndarray, jnp.ndarray], inp: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        vx_c, vxx_c = carry
        x_k, u_k, cg_k = inp
        err = x_k - goal
        lx = 2.0 * q_weight * err + cg_k
        lu = 2.0 * r_weight * u_k
        qx = lx + vx_c
        qu = lu + vx_c
        qxx = 2.0 * q_weight * eye + vxx_c
        quu = 2.0 * r_weight * eye + vxx_c
        qux = vxx_c

        quu_reg = quu + reg * eye
        inv_quu = jnp.linalg.inv(quu_reg)
        k_i = -inv_quu @ qu
        k_i_fb = -inv_quu @ qux

        vx_new = qx + k_i_fb.T @ quu_reg @ k_i + k_i_fb.T @ qu + qux.T @ k_i
        vxx_new = qxx + k_i_fb.T @ quu_reg @ k_i_fb + k_i_fb.T @ qux + qux.T @ k_i_fb
        vxx_new = 0.5 * (vxx_new + vxx_new.T)
        return (vx_new, vxx_new), (k_i, k_i_fb)

    # Stages k = H-2 .. 0
    xs = traj[:-1][::-1]
    us = u_nom[::-1]
    cgs = col_grad[:-1][::-1]

    (_, _), (ks_rev, kfb_rev) = jax.lax.scan(body, (vx, vxx), (xs, us, cgs))
    k_ff = ks_rev[::-1]
    k_fb = kfb_rev[::-1]
    return k_ff, k_fb


@partial(jax.jit, static_argnums=())
def _ilqr_forward_pass(
    traj: jnp.ndarray,
    u_nom: jnp.ndarray,
    k_ff: jnp.ndarray,
    k_fb: jnp.ndarray,
    start: jnp.ndarray,
    goal: jnp.ndarray,
    alpha: float,
    low: jnp.ndarray,
    high: jnp.ndarray,
    max_control_step: float,
) -> jnp.ndarray:
    """Apply affine control law for one line-search scale ``alpha``."""

    def step(x: jnp.ndarray, inp: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        kf, k_mat, x_nom, u_n = inp
        dx = x - x_nom
        u = u_n + alpha * kf + k_mat @ dx
        norm = jnp.linalg.norm(u)
        scale = jnp.minimum(1.0, max_control_step / (norm + 1e-12))
        u = u * scale
        x_new = jnp.clip(x + u, low, high)
        return x_new, x_new

    _, xs = jax.lax.scan(step, start, (k_ff, k_fb, traj[:-1], u_nom))
    out = jnp.vstack([start[None, :], xs])
    out = out.at[0].set(start)
    return out


@partial(jax.jit, static_argnums=())
def _quadratic_tracking_cost(
    traj: jnp.ndarray,
    u: jnp.ndarray,
    goal: jnp.ndarray,
    terminal_goal: jnp.ndarray,
    col_penalties: jnp.ndarray,
    q_weight: float,
    r_weight: float,
    qf_weight: float,
) -> jnp.ndarray:
    err_stages = traj[:-1] - goal
    stage = q_weight * jnp.sum(err_stages * err_stages)
    stage = stage + r_weight * jnp.sum(u * u) + jnp.sum(col_penalties[:-1])
    err_t = traj[-1] - terminal_goal
    terminal = qf_weight * jnp.sum(err_t * err_t) + col_penalties[-1]
    return stage + terminal


@partial(jax.jit, static_argnums=())
def _forward_all_alphas(
    traj: jnp.ndarray,
    u_nom: jnp.ndarray,
    k_ff: jnp.ndarray,
    k_fb: jnp.ndarray,
    start: jnp.ndarray,
    goal: jnp.ndarray,
    alphas: jnp.ndarray,
    low: jnp.ndarray,
    high: jnp.ndarray,
    max_control_step: float,
) -> jnp.ndarray:
    """Roll out forward pass for each line-search alpha; shape (n_alpha, H, dof)."""
    return jax.vmap(
        _ilqr_forward_pass,
        in_axes=(None, None, None, None, None, None, 0, None, None, None),
    )(traj, u_nom, k_ff, k_fb, start, goal, alphas, low, high, max_control_step)


class IlqrJaxSolver:
    """Wraps JIT iLQR passes; collision penalties/gradients stay in NumPy + PyBullet."""

    def __init__(self) -> None:
        # Warm up compilation with a tiny problem (first real solve still recompiles on shape).
        dof = 6
        h = 4
        traj = jnp.zeros((h, dof))
        u = jnp.zeros((h - 1, dof))
        g = jnp.zeros(dof)
        cg = jnp.zeros((h, dof))
        cp = jnp.zeros(h)
        al = jnp.array([1.0])
        lo = jnp.full(dof, -3.14)
        hi = jnp.full(dof, 3.14)
        _ilqr_backward_pass(traj, u, g, g, cg, 1.0, 1.0, 1.0, 1e-2)
        _ilqr_forward_pass(traj, u, jnp.zeros((h - 1, dof)), jnp.zeros((h - 1, dof, dof)), g, g, 1.0, lo, hi, 0.1)
        _forward_all_alphas(traj, u, jnp.zeros((h - 1, dof)), jnp.zeros((h - 1, dof, dof)), g, g, al, lo, hi, 0.1)
        _quadratic_tracking_cost(traj, u, g, g, cp, 1.0, 1.0, 1.0)

    def backward(
        self,
        traj: np.ndarray,
        u_nom: np.ndarray,
        goal: np.ndarray,
        col_grad: np.ndarray,
        q_weight: float,
        r_weight: float,
        qf_weight: float,
        reg: float,
        terminal_goal: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        tg = np.asarray(terminal_goal if terminal_goal is not None else goal, dtype=np.float64)
        k_ff, k_fb = _ilqr_backward_pass(
            jnp.asarray(traj),
            jnp.asarray(u_nom),
            jnp.asarray(goal),
            jnp.asarray(tg),
            jnp.asarray(col_grad),
            q_weight,
            r_weight,
            qf_weight,
            reg,
        )
        return _writable_numpy(k_ff), _writable_numpy(k_fb)

    def forward_candidates(
        self,
        traj: np.ndarray,
        u_nom: np.ndarray,
        k_ff: np.ndarray,
        k_fb: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray,
        alphas: Sequence[float],
        low: np.ndarray,
        high: np.ndarray,
        max_control_step: float,
    ) -> np.ndarray:
        candidates = _forward_all_alphas(
            jnp.asarray(traj),
            jnp.asarray(u_nom),
            jnp.asarray(k_ff),
            jnp.asarray(k_fb),
            jnp.asarray(start),
            jnp.asarray(goal),
            jnp.asarray(list(alphas), dtype=jnp.float64),
            jnp.asarray(low),
            jnp.asarray(high),
            max_control_step,
        )
        return _writable_numpy(candidates)

    def total_cost(
        self,
        traj: np.ndarray,
        u: np.ndarray,
        goal: np.ndarray,
        col_penalties: np.ndarray,
        q_weight: float,
        r_weight: float,
        qf_weight: float,
        terminal_goal: np.ndarray | None = None,
        terminal_extra_cost: float = 0.0,
    ) -> float:
        tg = np.asarray(terminal_goal if terminal_goal is not None else goal, dtype=np.float64)
        return float(
            _quadratic_tracking_cost(
                jnp.asarray(traj),
                jnp.asarray(u),
                jnp.asarray(goal),
                jnp.asarray(tg),
                jnp.asarray(col_penalties),
                q_weight,
                r_weight,
                qf_weight,
            )
            + terminal_extra_cost
        )
