from __future__ import annotations

import sys
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[4]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.vision.temporal_filter import (
    ConsecutiveGate,
    TemporalFilterConfig,
    TemporalVisionFilter,
    VisionState,
)


def test_consecutive_gate():
    g = ConsecutiveGate(3)
    assert g.update(True) is False
    assert g.update(True) is False
    assert g.update(True) is True
    assert g.update(False) is False


def test_contact_gate():
    f = TemporalVisionFilter(
        TemporalFilterConfig(contact_threshold=0.5, contact_consecutive=2, uncertainty_max_frames=999)
    )
    assert f.update(0.2, (1, 0)) == VisionState.NOT_TOUCHING
    assert f.update(0.9, (1, 0)) == VisionState.NOT_TOUCHING  # need 2
    assert f.update(0.9, (1, 0)) == VisionState.TOUCHING


def test_flow_then_settled_binary_phase():
    f = TemporalVisionFilter(
        TemporalFilterConfig(
            contact_threshold=0.5,
            contact_consecutive=1,
            flow_consecutive=2,
            settled_consecutive=2,
            phase_min_conf=0.3,
            uncertainty_max_frames=999,
        )
    )
    assert f.update(0.9, (0.9, 0.1)) == VisionState.TOUCHING
    assert f.update(0.9, (0.1, 0.9)) == VisionState.TOUCHING  # need 2 flow
    assert f.update(0.9, (0.1, 0.9)) == VisionState.FLOWING
    assert f.update(0.9, (0.9, 0.1)) == VisionState.TOUCHING  # need 2 settle
    assert f.update(0.9, (0.9, 0.1)) == VisionState.SETTLED
