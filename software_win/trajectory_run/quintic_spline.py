"""Piecewise quintic joint splines with velocity/acceleration retiming."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def quintic_coeffs(
    q0: float,
    q1: float,
    v0: float,
    v1: float,
    a0: float,
    a1: float,
    T: float,
) -> np.ndarray:
    """Coefficients ``a0..a5`` for ``q(t) = sum(a_k t^k)``, ``t in [0, T]``."""
    if T <= 1e-9:
        return np.array([q0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    dq = q1 - q0
    return np.array(
        [
            q0,
            v0,
            0.5 * a0,
            (20.0 * dq - (8.0 * v1 + 12.0 * v0) * T - (3.0 * a0 - a1) * T2) / (2.0 * T3),
            (30.0 * (q0 - q1) + (14.0 * v1 + 16.0 * v0) * T + (3.0 * a0 - 2.0 * a1) * T2) / (2.0 * T4),
            (12.0 * dq - (6.0 * v1 + 6.0 * v0) * T - (a0 - a1) * T2) / (2.0 * T5),
        ],
        dtype=np.float64,
    )


def eval_quintic(coeffs: np.ndarray, t: float) -> tuple[float, float, float]:
    """Position, velocity, acceleration at local time ``t``."""
    a0, a1, a2, a3, a4, a5 = coeffs
    q = a0 + t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    v = a1 + t * (2.0 * a2 + t * (3.0 * a3 + t * (4.0 * a4 + t * 5.0 * a5)))
    acc = 2.0 * a2 + t * (6.0 * a3 + t * (12.0 * a4 + t * 20.0 * a5))
    return float(q), float(v), float(acc)


@dataclass
class QuinticJointSpline:
    """Piecewise quintic through joint-space knots with per-segment durations."""

    knots: np.ndarray  # (N, dof)
    segment_durations: np.ndarray  # (N-1,)
    segment_coeffs: np.ndarray  # (N-1, dof, 6)
    knot_times: np.ndarray  # (N,) cumulative

    @property
    def duration(self) -> float:
        return float(self.knot_times[-1]) if len(self.knot_times) else 0.0

    @property
    def n_dof(self) -> int:
        return int(self.knots.shape[1])

    def _locate_segment(self, t: float) -> tuple[int, float]:
        t = float(np.clip(t, 0.0, self.duration))
        idx = int(np.searchsorted(self.knot_times, t, side="right") - 1)
        idx = max(0, min(idx, len(self.segment_durations) - 1))
        local = t - self.knot_times[idx]
        local = float(np.clip(local, 0.0, self.segment_durations[idx]))
        return idx, local

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx, local = self._locate_segment(t)
        dof = self.n_dof
        q = np.zeros(dof, dtype=np.float64)
        v = np.zeros(dof, dtype=np.float64)
        a = np.zeros(dof, dtype=np.float64)
        for j in range(dof):
            qj, vj, aj = eval_quintic(self.segment_coeffs[idx, j], local)
            q[j] = qj
            v[j] = vj
            a[j] = aj
        return q, v, a

    def resample(
        self,
        dt: float,
        *,
        max_points: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(times, positions, velocities, accelerations)`` sampled uniformly."""
        dt = max(1e-3, float(dt))
        if self.duration <= 0.0:
            q = self.knots[:1].copy()
            return np.array([0.0]), q, np.zeros_like(q), np.zeros_like(q)
        n = min(max_points, max(2, int(self.duration / dt) + 1))
        times = np.linspace(0.0, self.duration, n)
        n = len(times)
        dof = self.n_dof
        pos = np.zeros((n, dof), dtype=np.float64)
        vel = np.zeros((n, dof), dtype=np.float64)
        acc = np.zeros((n, dof), dtype=np.float64)
        for i, t in enumerate(times):
            pos[i], vel[i], acc[i] = self.sample(float(t))
        pos[-1] = self.knots[-1]
        vel[-1] = 0.0
        acc[-1] = 0.0
        return times, pos, vel, acc

    def scaled(self, time_scale: float) -> QuinticJointSpline:
        """Stretch time by ``time_scale`` (>1 slows down, reduces vel/acc)."""
        s = max(1e-6, float(time_scale))
        return QuinticJointSpline(
            knots=self.knots.copy(),
            segment_durations=self.segment_durations * s,
            segment_coeffs=self.segment_coeffs.copy(),
            knot_times=self.knot_times * s,
        )


def _initial_knot_times(
    waypoints: np.ndarray,
    max_vel_deg_s: float,
    max_acc_deg_s2: float,
) -> np.ndarray:
    """Cumulative knot times from per-segment vel/accel limits."""
    wp = np.asarray(waypoints, dtype=np.float64)
    n = len(wp)
    if n <= 1:
        return np.array([0.0], dtype=np.float64)
    vmax = max(1.0, float(max_vel_deg_s))
    amax = max(1.0, float(max_acc_deg_s2))
    times = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta = float(np.max(np.abs(wp[i] - wp[i - 1])))
        t_vel = delta / vmax
        t_acc = 2.0 * math.sqrt(delta / amax) if delta > 1e-9 else 0.0
        times[i] = times[i - 1] + max(t_vel, t_acc, 0.015)
    return times


def _estimate_knot_velocities(knots: np.ndarray, knot_times: np.ndarray) -> np.ndarray:
    n, dof = knots.shape
    vel = np.zeros((n, dof), dtype=np.float64)
    if n <= 1:
        return vel
    vel[0] = 0.0
    vel[-1] = 0.0
    for i in range(1, n - 1):
        dt = knot_times[i + 1] - knot_times[i - 1]
        if dt > 1e-9:
            vel[i] = (knots[i + 1] - knots[i - 1]) / dt
    return vel


def _estimate_knot_accelerations(knots: np.ndarray, knot_times: np.ndarray) -> np.ndarray:
    n, dof = knots.shape
    acc = np.zeros((n, dof), dtype=np.float64)
    if n <= 2:
        return acc
    acc[0] = 0.0
    acc[-1] = 0.0
    for i in range(1, n - 1):
        dt = knot_times[i + 1] - knot_times[i - 1]
        if dt > 1e-9:
            acc[i] = 2.0 * (
                (knots[i + 1] - knots[i]) / (knot_times[i + 1] - knot_times[i])
                - (knots[i] - knots[i - 1]) / (knot_times[i] - knot_times[i - 1])
            ) / dt
    return acc


def build_quintic_spline(
    waypoints: np.ndarray,
    *,
    knot_times: np.ndarray | None = None,
    max_vel_deg_s: float = 25.0,
    max_acc_deg_s2: float = 12.0,
    zero_endpoint_derivatives: bool = True,
) -> QuinticJointSpline:
    """Fit a piecewise quintic spline through ``waypoints`` (deg), shape ``(N, dof)``."""
    knots = np.asarray(waypoints, dtype=np.float64)
    if knots.ndim == 1:
        knots = knots.reshape(1, -1)
    n, dof = knots.shape
    if n < 2:
        return QuinticJointSpline(
            knots=knots,
            segment_durations=np.array([], dtype=np.float64),
            segment_coeffs=np.zeros((0, dof, 6), dtype=np.float64),
            knot_times=np.array([0.0], dtype=np.float64),
        )

    if knot_times is None:
        times = _initial_knot_times(knots, max_vel_deg_s, max_acc_deg_s2)
    else:
        times = np.asarray(knot_times, dtype=np.float64)
        if len(times) != n:
            raise ValueError("knot_times length must match number of waypoints")

    if zero_endpoint_derivatives:
        vels = _clamped_knot_velocities(knots, times, max_vel_deg_s)
        accs = np.zeros((n, dof), dtype=np.float64)
    else:
        vels = _estimate_knot_velocities(knots, times)
        accs = _estimate_knot_accelerations(knots, times)
        vels[0] = 0.0
        vels[-1] = 0.0
        accs[0] = 0.0
        accs[-1] = 0.0

    durations = np.diff(times)
    coeffs = np.zeros((n - 1, dof, 6), dtype=np.float64)
    for i in range(n - 1):
        T = float(durations[i])
        for j in range(dof):
            coeffs[i, j] = quintic_coeffs(
                knots[i, j],
                knots[i + 1, j],
                vels[i, j],
                vels[i + 1, j],
                accs[i, j],
                accs[i + 1, j],
                T,
            )

    return QuinticJointSpline(
        knots=knots,
        segment_durations=durations,
        segment_coeffs=coeffs,
        knot_times=times,
    )


def retime_spline_to_limits(
    spline: QuinticJointSpline,
    *,
    max_vel_deg_s: float,
    max_acc_deg_s2: float,
    sample_dt: float = 0.02,
    max_iterations: int = 30,
) -> tuple[QuinticJointSpline, float, float, float]:
    """
    Uniformly scale spline time until sampled |vel| and |acc| are within limits.

    Returns ``(scaled_spline, time_scale, max_vel, max_acc)``.
    """
    max_vel = max(1.0, float(max_vel_deg_s))
    max_acc = max(1.0, float(max_acc_deg_s2))
    dt = max(1e-3, float(sample_dt))

    scale = 1.0
    peak_v = float("inf")
    peak_a = float("inf")

    for _ in range(max_iterations):
        scaled = spline.scaled(scale)
        if scaled.duration > 120.0:
            break
        _, _, vel, acc = scaled.resample(min(dt, 0.05), max_points=400)
        peak_v = float(np.max(np.abs(vel))) if vel.size else 0.0
        peak_a = float(np.max(np.abs(acc))) if acc.size else 0.0
        if peak_v <= max_vel and peak_a <= max_acc:
            return _compress_spline_to_target(
                spline,
                scale,
                max_vel_deg_s=max_vel,
                max_acc_deg_s2=max_acc,
                sample_dt=dt,
            )

        vel_ratio = peak_v / max_vel if peak_v > max_vel else 1.0
        acc_ratio = math.sqrt(peak_a / max_acc) if peak_a > max_acc else 1.0
        scale *= max(vel_ratio, acc_ratio, 1.05)
        scale = min(scale, 20.0)

    return spline.scaled(scale), scale, peak_v, peak_a


def _compress_spline_to_target(
    spline: QuinticJointSpline,
    scale: float,
    *,
    max_vel_deg_s: float,
    max_acc_deg_s2: float,
    sample_dt: float,
    target_vel_frac: float = 0.92,
) -> tuple[QuinticJointSpline, float, float, float]:
    """Shorten duration to use more of the allowed speed (without crushing below min duration)."""
    max_vel = max(1.0, float(max_vel_deg_s))
    max_acc = max(1.0, float(max_acc_deg_s2))
    target_v = target_vel_frac * max_vel
    dt = min(0.02, float(sample_dt))

    scaled = spline.scaled(scale)
    _, _, vel, acc = scaled.resample(dt, max_points=400)
    peak_v = float(np.max(np.abs(vel))) if vel.size else 0.0
    peak_a = float(np.max(np.abs(acc))) if acc.size else 0.0

    if peak_v < 1e-3 or peak_v >= target_v:
        return scaled, scale, peak_v, peak_a

    # v ~ 1/scale  =>  scale_new = scale * (peak_v / target_v)
    new_scale = scale * (peak_v / target_v)
    new_scale = max(new_scale, scale * 0.05)
    scaled = spline.scaled(new_scale)
    _, _, vel, acc = scaled.resample(dt, max_points=400)
    peak_v = float(np.max(np.abs(vel))) if vel.size else 0.0
    peak_a = float(np.max(np.abs(acc))) if acc.size else 0.0

    if peak_a > max_acc * 1.02:
        acc_scale = math.sqrt(peak_a / max_acc)
        new_scale = min(scale, new_scale * acc_scale)
        new_scale = max(new_scale, scale * 0.05)
        scaled = spline.scaled(new_scale)
        _, _, vel, acc = scaled.resample(dt, max_points=400)
        peak_v = float(np.max(np.abs(vel))) if vel.size else 0.0
        peak_a = float(np.max(np.abs(acc))) if acc.size else 0.0

    return scaled, new_scale, peak_v, peak_a


def collapse_near_duplicate_knots(
    waypoints: np.ndarray,
    *,
    min_step_deg: float = 0.05,
) -> np.ndarray:
    """Drop consecutive knots whose max joint step is below ``min_step_deg`` (keep ends)."""
    wp = np.asarray(waypoints, dtype=np.float64)
    if len(wp) < 3:
        return wp
    keep = [0]
    thr = max(0.0, float(min_step_deg))
    for i in range(1, len(wp) - 1):
        if float(np.max(np.abs(wp[i] - wp[keep[-1]]))) >= thr:
            keep.append(i)
    keep.append(len(wp) - 1)
    # Ensure last kept point is distinct from previous when possible.
    if len(keep) >= 2 and float(np.max(np.abs(wp[keep[-1]] - wp[keep[-2]]))) < 1e-12:
        keep.pop(-2)
    return wp[np.asarray(keep, dtype=np.int64)]


def subsample_knots(waypoints: np.ndarray, max_knots: int) -> np.ndarray:
    """Reduce knots by arc length (keeps endpoints, avoids large joint-space gaps)."""
    wp = collapse_near_duplicate_knots(np.asarray(waypoints, dtype=np.float64))
    n = len(wp)
    mk = max(2, int(max_knots))
    if n <= mk:
        return wp

    dist = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dist[i] = dist[i - 1] + float(np.max(np.abs(wp[i] - wp[i - 1])))
    if dist[-1] <= 1e-9:
        return wp[:mk]

    targets = np.linspace(0.0, dist[-1], mk)
    idx = np.searchsorted(dist, targets, side="left")
    idx = np.clip(idx, 0, n - 1)
    idx[0] = 0
    idx[-1] = n - 1
    return wp[np.unique(idx)]


def _clamped_knot_velocities(
    knots: np.ndarray,
    knot_times: np.ndarray,
    max_vel_deg_s: float,
) -> np.ndarray:
    n, dof = knots.shape
    vmax = max(1.0, float(max_vel_deg_s))
    vel = np.zeros((n, dof), dtype=np.float64)
    if n < 2:
        return vel

    for i in range(1, n - 1):
        dt0 = knot_times[i] - knot_times[i - 1]
        dt1 = knot_times[i + 1] - knot_times[i]
        step0 = float(np.max(np.abs(knots[i] - knots[i - 1])))
        step1 = float(np.max(np.abs(knots[i + 1] - knots[i])))
        # Near-stationary knots must keep v≈0; otherwise quintic overshoots massively.
        if step0 < 0.05 or step1 < 0.05:
            continue
        if dt0 > 1e-9 and dt1 > 1e-9:
            v0 = (knots[i] - knots[i - 1]) / dt0
            v1 = (knots[i + 1] - knots[i]) / dt1
            vel[i] = np.clip(0.5 * (v0 + v1), -vmax, vmax)
    return vel


def smooth_and_retime_waypoints(
    waypoints_deg: np.ndarray,
    *,
    sample_dt: float = 0.05,
    max_vel_deg_s: float = 25.0,
    max_acc_deg_s2: float = 12.0,
    max_knots: int = 80,
    pin_endpoints: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Quintic spline + retime + uniform resample for robot streaming.

    Returns ``(positions_deg, velocities_deg_s, info_dict)``.
    """
    wp = np.asarray(waypoints_deg, dtype=np.float64)
    if len(wp) < 2:
        return wp.copy(), np.zeros_like(wp), {"n_in": len(wp), "n_out": len(wp)}

    q_start = wp[0].copy()
    q_goal = wp[-1].copy()

    knots = subsample_knots(wp, max_knots)
    spline = build_quintic_spline(
        knots,
        max_vel_deg_s=max_vel_deg_s,
        max_acc_deg_s2=max_acc_deg_s2,
    )
    spline, time_scale, peak_v, peak_a = retime_spline_to_limits(
        spline,
        max_vel_deg_s=max_vel_deg_s,
        max_acc_deg_s2=max_acc_deg_s2,
        sample_dt=min(0.02, sample_dt),
    )

    times, pos, vel, _acc = spline.resample(sample_dt, max_points=5000)
    if pin_endpoints:
        pos[0] = q_start
        pos[-1] = q_goal
        vel[0] = 0.0
        vel[-1] = 0.0

    info = {
        "n_in": len(wp),
        "n_knots": len(knots),
        "n_out": len(pos),
        "duration_s": float(times[-1]) if len(times) else 0.0,
        "time_scale": time_scale,
        "peak_vel_deg_s": peak_v,
        "peak_acc_deg_s2": peak_a,
        "max_vel_limit": max_vel_deg_s,
        "max_acc_limit": max_acc_deg_s2,
        "sample_dt_s": sample_dt,
    }
    return pos, vel, info


def summarize_smooth_info(info: dict) -> str:
    knots = info.get("n_knots", info["n_in"])
    return (
        f"smooth: {info['n_in']} plan pts -> {knots} knots -> {info['n_out']} samples, "
        f"T={info['duration_s']:.2f}s, scale={info['time_scale']:.2f}, "
        f"peak |v|={info['peak_vel_deg_s']:.1f} deg/s, "
        f"peak |a|={info['peak_acc_deg_s2']:.1f} deg/s^2"
    )
