from __future__ import annotations

import sys
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[4]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.vision.state_machine import (
    ControlState,
    ControllerConfig,
    EpoxyApplicationController,
    MockRobot,
)
from high_level_jobs.epoxy_app.vision.temporal_filter import VisionState


def _ctl(**kwargs):
    robot = MockRobot(z=0.02)
    defaults = dict(
        min_z_step=0.001,
        max_downward_travel=0.01,
        flow_check_delay=0.1,
        max_settling_time=1.5,
        lift_distance=0.005,
        retract_distance=0.05,
    )
    defaults.update(kwargs)
    return EpoxyApplicationController(robot, ControllerConfig(**defaults)), robot


def test_approach_then_retract_axis_separated():
    # With Y retract: Z then Y
    ctl, robot = _ctl(flow_check_delay=0.0, lift_distance=0.010, retract_distance=0.05)
    ctl.step(VisionState.NOT_TOUCHING, now=0.0)
    ctl.step(VisionState.TOUCHING, now=0.1)
    ctl.step(VisionState.FLOWING, now=0.2)
    r = ctl.step(VisionState.SETTLED, now=0.3)
    assert r.control_state == ControlState.RETRACT
    z0, y0, x0 = robot.z, robot.y, robot.x
    ctl.step(VisionState.SETTLED, now=0.4)
    assert abs(robot.z - (z0 + 0.010)) < 1e-9
    assert abs(robot.y - y0) < 1e-12
    assert abs(robot.x - x0) < 1e-12
    r = ctl.step(VisionState.SETTLED, now=0.5)
    assert r.done and r.control_state == ControlState.DONE
    assert abs(robot.y - (y0 + 0.05)) < 1e-9
    assert abs(robot.x - x0) < 1e-12


def test_retract_z_only_when_retract_distance_zero():
    ctl, robot = _ctl(flow_check_delay=0.0, lift_distance=0.010, retract_distance=0.0)
    ctl.step(VisionState.TOUCHING, now=0.0)
    ctl.step(VisionState.FLOWING, now=0.1)
    r = ctl.step(VisionState.SETTLED, now=0.2)
    assert r.control_state == ControlState.RETRACT
    z0, y0 = robot.z, robot.y
    r = ctl.step(VisionState.SETTLED, now=0.3)
    assert r.done and r.control_state == ControlState.DONE
    assert abs(robot.z - (z0 + 0.010)) < 1e-9
    assert abs(robot.y - y0) < 1e-12
    assert "Z done" in r.message

def test_camera_failure_errors():
    ctl, _ = _ctl()
    r = ctl.step(VisionState.NOT_TOUCHING, camera_ok=False, now=0.0)
    assert r.control_state == ControlState.ERROR


def test_settling_timeout():
    ctl, _ = _ctl(flow_check_delay=0.0, max_settling_time=2.0)
    ctl.step(VisionState.TOUCHING, now=0.0)
    ctl.step(VisionState.FLOWING, now=0.1)
    ctl.step(VisionState.TOUCHING, now=0.2)
    r = ctl.step(VisionState.TOUCHING, now=2.2)
    assert r.control_state == ControlState.RETRACT
    assert ctl.settling_timed_out


def test_check_flow_holds_on_raw_flow_probability():
    ctl, robot = _ctl(flow_check_delay=0.0, min_z_step=0.001)
    ctl.step(VisionState.TOUCHING, now=0.0)
    z0 = robot.z
    # Delay expired but high raw flow_p → hold, do not creep
    r = ctl.step(VisionState.TOUCHING, flow_probability=0.8, now=1.0)
    assert r.control_state == ControlState.CHECK_FLOW
    assert abs(robot.z - z0) < 1e-12
    assert "holding" in r.message


def test_approach_flowing_goes_direct_to_flowing():
    ctl, _ = _ctl(flow_check_delay=0.0)
    r = ctl.step(VisionState.FLOWING, now=0.0)
    assert r.control_state == ControlState.FLOWING
    r = ctl.step(VisionState.FLOWING, now=0.1)
    assert r.control_state == ControlState.SETTLING


def test_check_flow_zero_time_bug():
    """_t_check_flow_enter == 0.0 must not be treated as missing."""
    ctl, robot = _ctl(flow_check_delay=0.2, min_z_step=0.001)
    ctl.step(VisionState.TOUCHING, now=0.0)
    z0 = robot.z
    ctl.step(VisionState.TOUCHING, now=0.05)
    assert abs(robot.z - z0) < 1e-12
    r = ctl.step(VisionState.TOUCHING, now=0.25)
    assert r.moved_z < 0
