"""Light temporal smoothing for live sensor grasp + holder axis."""

from __future__ import annotations

import math


class KeypointEMA:
    """EMA on grasp UV + direction tip (stabilizes live jitter)."""

    def __init__(self, *, alpha: float = 0.35) -> None:
        self.alpha = float(alpha)
        self._kpts: list[list[float]] | None = None

    def reset(self) -> None:
        self._kpts = None

    def update(self, kpts: list[list[float]] | None) -> list[list[float]] | None:
        if not kpts or len(kpts) < 2:
            return kpts
        cur = [[float(kpts[0][0]), float(kpts[0][1])], [float(kpts[1][0]), float(kpts[1][1])]]
        if self._kpts is None:
            self._kpts = cur
            return [list(p) for p in self._kpts]
        a = self.alpha
        # Smooth grasp
        g = [
            a * cur[0][0] + (1.0 - a) * self._kpts[0][0],
            a * cur[0][1] + (1.0 - a) * self._kpts[0][1],
        ]
        # Smooth axis via unit direction (avoid averaging across 180° jumps badly)
        def _unit(p0, p1):
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            n = math.hypot(dx, dy)
            if n < 1e-6:
                return 1.0, 0.0
            return dx / n, dy / n

        cx, cy = _unit(cur[0], cur[1])
        px, py = _unit(self._kpts[0], self._kpts[1])
        sx = a * cx + (1.0 - a) * px
        sy = a * cy + (1.0 - a) * py
        sn = math.hypot(sx, sy) or 1.0
        sx, sy = sx / sn, sy / sn
        tip_len = max(20.0, math.hypot(cur[1][0] - cur[0][0], cur[1][1] - cur[0][1]))
        tip = [g[0] + tip_len * sx, g[1] + tip_len * sy]
        self._kpts = [g, tip]
        return [list(p) for p in self._kpts]
