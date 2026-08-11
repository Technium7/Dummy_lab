"""Temporal filter: contact gate → phase (TOUCH/FLOW) → SETTLED after flow stops."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque

from high_level_jobs.epoxy_app.vision.model import PHASE_NAMES


class VisionState(str, Enum):
    NOT_TOUCHING = "NOT_TOUCHING"
    TOUCHING = "TOUCHING"
    FLOWING = "FLOWING"
    SETTLED = "SETTLED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class TemporalFilterConfig:
    contact_threshold: float = 0.50
    phase_min_conf: float = 0.40
    contact_consecutive: int = 2
    flow_consecutive: int = 2
    settled_consecutive: int = 3
    uncertainty_low: float = 0.40
    uncertainty_high: float = 0.60
    uncertainty_max_frames: int = 30

    @classmethod
    def from_dict(cls, cfg: dict) -> "TemporalFilterConfig":
        t = cfg.get("thresholds", cfg)
        return cls(
            contact_threshold=float(t.get("contact", 0.50)),
            phase_min_conf=float(t.get("phase_min_conf", 0.40)),
            contact_consecutive=int(t.get("contact_consecutive", 2)),
            flow_consecutive=int(t.get("flow_consecutive", 2)),
            settled_consecutive=int(t.get("settled_consecutive", 3)),
            uncertainty_low=float(t.get("uncertainty_low", 0.40)),
            uncertainty_high=float(t.get("uncertainty_high", 0.60)),
            uncertainty_max_frames=int(t.get("uncertainty_max_frames", 30)),
        )


class ConsecutiveGate:
    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self._count = 0
        self._latched = False

    def update(self, raw: bool) -> bool:
        if raw:
            self._count += 1
            if self._count >= self.n:
                self._latched = True
        else:
            self._count = 0
            self._latched = False
        return self._latched

    def reset(self) -> None:
        self._count = 0
        self._latched = False


class TemporalVisionFilter:
    """Phase model is binary TOUCHING/FLOWING; SETTLED = touch after flow was seen."""

    def __init__(self, cfg: TemporalFilterConfig | None = None) -> None:
        self.cfg = cfg or TemporalFilterConfig()
        self.contact_gate = ConsecutiveGate(self.cfg.contact_consecutive)
        self.flow_gate = ConsecutiveGate(self.cfg.flow_consecutive)
        self.settled_gate = ConsecutiveGate(self.cfg.settled_consecutive)
        self._saw_flow = False
        self._uncertain_streak = 0
        self.history: Deque[tuple[float, int, float]] = deque(maxlen=64)

    def reset(self) -> None:
        self.contact_gate.reset()
        self.flow_gate.reset()
        self.settled_gate.reset()
        self._saw_flow = False
        self._uncertain_streak = 0
        self.history.clear()

    def update(
        self,
        contact_p: float,
        phase_probs: list[float] | tuple[float, ...] | None = None,
        *,
        flow_p: float | None = None,
    ) -> VisionState:
        # Legacy: contact_p, flow_p → synthesize 2-class phase
        if phase_probs is None and flow_p is not None:
            phase_probs = (1.0 - float(flow_p), float(flow_p))

        if phase_probs is None:
            phase_probs = (1.0, 0.0)

        # Accept old 3-class vectors: merge SETTLED into TOUCHING
        if len(phase_probs) >= 3:
            touch = float(phase_probs[0]) + float(phase_probs[2])
            flow = float(phase_probs[1])
            phase_probs = (touch, flow)
        elif len(phase_probs) == 1:
            phase_probs = (1.0 - float(phase_probs[0]), float(phase_probs[0]))

        phase_idx = int(max(range(len(phase_probs)), key=lambda i: phase_probs[i]))
        phase_conf = float(phase_probs[phase_idx])
        self.history.append((float(contact_p), phase_idx, phase_conf))

        lo, hi = self.cfg.uncertainty_low, self.cfg.uncertainty_high
        if lo <= contact_p <= hi:
            self._uncertain_streak += 1
        else:
            self._uncertain_streak = 0
        if self._uncertain_streak >= self.cfg.uncertainty_max_frames:
            return VisionState.UNCERTAIN

        contact = self.contact_gate.update(contact_p >= self.cfg.contact_threshold)
        if not contact:
            self.flow_gate.reset()
            self.settled_gate.reset()
            self._saw_flow = False
            return VisionState.NOT_TOUCHING

        if phase_conf < self.cfg.phase_min_conf:
            # Unclear phase: if we already flowed, keep counting settle on non-flow
            if self._saw_flow and self.settled_gate.update(True):
                return VisionState.SETTLED
            return VisionState.TOUCHING

        name = PHASE_NAMES[phase_idx] if phase_idx < len(PHASE_NAMES) else "TOUCHING"
        flowing = self.flow_gate.update(name == "FLOWING")
        if flowing:
            self._saw_flow = True
            self.settled_gate.reset()
            return VisionState.FLOWING

        # Not flowing while in contact
        if self._saw_flow:
            if self.settled_gate.update(name == "TOUCHING"):
                return VisionState.SETTLED
        else:
            self.settled_gate.reset()

        return VisionState.TOUCHING
