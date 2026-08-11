"""Deterministic epoxy application FSM. Vision never commands the robot directly."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

from high_level_jobs.epoxy_app.vision.temporal_filter import VisionState


class ControlState(str, Enum):
    APPROACH = "APPROACH"
    CONTACT = "CONTACT"
    CHECK_FLOW = "CHECK_FLOW"
    FLOWING = "FLOWING"
    SETTLING = "SETTLING"
    RETRACT = "RETRACT"
    DONE = "DONE"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class RobotMotion(Protocol):
    def move_z_relative(self, distance: float) -> None: ...
    def move_x_relative(self, distance: float) -> None: ...
    def move_y_relative(self, distance: float) -> None: ...
    def stop(self) -> None: ...
    def get_position(self) -> tuple[float, float, float]: ...


@dataclass
class ControllerConfig:
    min_z_step: float = 0.00012
    approach_step_period: float = 1.50
    flow_check_delay: float = 2.00
    post_move_vision_delay: float = 0.50
    flow_hold_probability: float = 0.55
    max_settling_time: float = 2.0
    lift_distance: float = 0.010
    retract_distance: float = 0.05
    max_downward_travel: float = 0.015
    max_z_steps: int = 400
    max_approach_time: float = 60.0
    max_flow_detect_time: float = 25.0

    @classmethod
    def from_dict(cls, cfg: dict) -> "ControllerConfig":
        c = cfg.get("control", cfg)
        return cls(
            min_z_step=float(c.get("min_z_step", 0.00012)),
            approach_step_period=float(c.get("approach_step_period", 1.50)),
            flow_check_delay=float(c.get("flow_check_delay", 2.00)),
            post_move_vision_delay=float(c.get("post_move_vision_delay", 0.50)),
            flow_hold_probability=float(c.get("flow_hold_probability", 0.55)),
            max_settling_time=float(c.get("max_settling_time", 2.0)),
            lift_distance=float(c.get("lift_distance", 0.010)),
            retract_distance=float(c.get("retract_distance", 0.05)),
            max_downward_travel=float(c.get("max_downward_travel", 0.015)),
            max_z_steps=int(c.get("max_z_steps", 400)),
            max_approach_time=float(c.get("max_approach_time", 60.0)),
            max_flow_detect_time=float(c.get("max_flow_detect_time", 25.0)),
        )


@dataclass
class StepResult:
    control_state: ControlState
    vision_state: VisionState
    moved_z: float = 0.0
    message: str = ""
    done: bool = False


@dataclass
class EpoxyApplicationController:
    robot: RobotMotion
    config: ControllerConfig = field(default_factory=ControllerConfig)
    # Pause vision during Z creeps; settle tip; re-arm detector (set by epoxy_apply).
    before_z_move: Callable[[], None] | None = None
    after_z_move: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        self.state = ControlState.APPROACH
        self.error_message = ""
        self.settling_timed_out = False
        self._start_z: float | None = None
        self._z_steps = 0
        self._downward_travel = 0.0
        self._t_approach0: float | None = None
        self._t_contact0: float | None = None
        self._t_flow0: float | None = None
        self._t_check_flow_enter: float | None = None
        self._t_last_z_move: float | None = None
        self._retract_phase = 0
        self._estop = False
        self._vision_live = True

    def reset(self) -> None:
        before, after = self.before_z_move, self.after_z_move
        self.__post_init__()
        self.before_z_move = before
        self.after_z_move = after

    def set_vision_live(self, live: bool) -> None:
        self._vision_live = bool(live)

    def emergency_stop(self, reason: str = "operator estop") -> None:
        self._estop = True
        self.robot.stop()
        self.state = ControlState.ERROR
        self.error_message = reason

    def _enter_error(self, msg: str) -> StepResult:
        self.robot.stop()
        self.state = ControlState.ERROR
        self.error_message = msg
        return StepResult(self.state, VisionState.UNCERTAIN, message=msg, done=True)

    def _move_down_one_step(self) -> float:
        dz = -abs(self.config.min_z_step)
        remaining = self.config.max_downward_travel - self._downward_travel
        if remaining <= 0:
            raise RuntimeError("no remaining downward travel budget")
        if abs(dz) > remaining:
            dz = -remaining
        if self.before_z_move is not None:
            self.before_z_move()
        try:
            self.robot.move_z_relative(dz)
        finally:
            if self.after_z_move is not None:
                self.after_z_move()
        self._z_steps += 1
        self._downward_travel += abs(dz)
        return dz

    def step(
        self,
        vision_state: VisionState,
        *,
        camera_ok: bool = True,
        inference_ok: bool = True,
        vision_live: bool | None = None,
        flow_probability: float = 0.0,
        now: float | None = None,
    ) -> StepResult:
        now = time.perf_counter() if now is None else now
        live = self._vision_live if vision_live is None else bool(vision_live)
        if self.state in (ControlState.DONE, ControlState.ERROR):
            return StepResult(self.state, vision_state, done=True, message=self.error_message)

        if self._start_z is None:
            self._start_z = self.robot.get_position()[2]
            self._t_approach0 = now

        if self._estop:
            return self._enter_error(self.error_message or "emergency stop")

        if not camera_ok or not inference_ok:
            if self.state in (ControlState.APPROACH, ControlState.CONTACT, ControlState.CHECK_FLOW):
                return self._enter_error("camera failure" if not camera_ok else "inference failure")

        if (
            live
            and vision_state == VisionState.UNCERTAIN
            and self.state in (ControlState.APPROACH, ControlState.CHECK_FLOW)
        ):
            return self._enter_error("vision uncertain too long")

        if self._downward_travel > self.config.max_downward_travel + 1e-12:
            return self._enter_error("max downward travel exceeded")
        if self._z_steps > self.config.max_z_steps:
            return self._enter_error("max Z steps exceeded")
        if (
            self.state == ControlState.APPROACH
            and self._t_approach0 is not None
            and now - self._t_approach0 > self.config.max_approach_time
        ):
            return self._enter_error("max approach time exceeded")
        if (
            self.state in (ControlState.CONTACT, ControlState.CHECK_FLOW)
            and self._t_contact0 is not None
            and now - self._t_contact0 > self.config.max_flow_detect_time
        ):
            return self._enter_error("max flow detection time exceeded")

        if self.state == ControlState.APPROACH:
            # Only trust contact/flow while tip is still (vision armed after creep settle).
            if live and vision_state == VisionState.FLOWING:
                self.robot.stop()
                self._t_contact0 = now
                self._t_flow0 = now
                self.state = ControlState.FLOWING
                return StepResult(self.state, vision_state, message="approach flow -> FLOWING")
            if live and vision_state in (
                VisionState.TOUCHING,
                VisionState.SETTLED,
            ):
                self.robot.stop()
                self._t_contact0 = now
                self._t_check_flow_enter = now
                self.state = ControlState.CHECK_FLOW
                return StepResult(self.state, vision_state, message="contact -> CHECK_FLOW")
            if (
                self._t_last_z_move is not None
                and now - self._t_last_z_move < self.config.approach_step_period
            ):
                return StepResult(
                    self.state,
                    vision_state,
                    message=(
                        f"post-creep judge ({now - self._t_last_z_move:.3f}s)"
                        if live
                        else "creeping / vision paused"
                    ),
                )
            try:
                dz = self._move_down_one_step()
            except Exception as exc:  # noqa: BLE001
                return self._enter_error(f"Z move failed: {exc}")
            self._t_last_z_move = time.perf_counter()
            return StepResult(self.state, vision_state, moved_z=dz, message="approach step")

        if self.state == ControlState.CONTACT:
            self.state = ControlState.CHECK_FLOW
            self._t_check_flow_enter = now

        if self.state == ControlState.CHECK_FLOW:
            if live and vision_state == VisionState.FLOWING:
                self.robot.stop()
                self.state = ControlState.FLOWING
                self._t_flow0 = now
                return StepResult(self.state, vision_state, message="flow detected")
            # Raw phase may already say FLOWING before the temporal gate latches —
            # hold still; do not creep further into the pad.
            if live and float(flow_probability) >= self.config.flow_hold_probability:
                return StepResult(
                    self.state,
                    vision_state,
                    message=f"flow signal {float(flow_probability):.2f}; holding",
                )
            enter = self._t_check_flow_enter
            elapsed = now - (enter if enter is not None else now)
            if elapsed < self.config.flow_check_delay:
                return StepResult(
                    self.state,
                    vision_state,
                    message=(
                        f"waiting for flow ({elapsed:.3f}s)"
                        if live
                        else "creeping / vision paused"
                    ),
                )
            try:
                dz = self._move_down_one_step()
            except Exception as exc:  # noqa: BLE001
                return self._enter_error(f"Z move failed: {exc}")
            self._t_check_flow_enter = time.perf_counter()
            return StepResult(
                self.state, vision_state, moved_z=dz, message="no flow; crept one step"
            )

        if self.state == ControlState.FLOWING:
            self.state = ControlState.SETTLING

        if self.state == ControlState.SETTLING:
            assert self._t_flow0 is not None
            elapsed = now - self._t_flow0
            if vision_state == VisionState.SETTLED:
                self.state = ControlState.RETRACT
                self._retract_phase = 0
                return StepResult(self.state, vision_state, message="settled -> retract")
            if elapsed >= self.config.max_settling_time:
                self.settling_timed_out = True
                self.state = ControlState.RETRACT
                self._retract_phase = 0
                return StepResult(
                    self.state, vision_state, message="settling timeout -> retract"
                )
            return StepResult(self.state, vision_state, message="settling")

        if self.state == ControlState.RETRACT:
            # Contact/flow called robot.stop() for a hold — never leave abort latched.
            clear = getattr(self.robot, "clear_abort", None)
            if callable(clear):
                clear()
            try:
                if self._retract_phase == 0:
                    # +Z lift; optional Y slide only if retract_distance != 0.
                    dz = abs(self.config.lift_distance)
                    self.robot.move_z_relative(dz)
                    self._retract_phase = 1
                    dy = float(self.config.retract_distance)
                    if abs(dy) < 1e-12:
                        self.state = ControlState.DONE
                        return StepResult(
                            self.state,
                            vision_state,
                            moved_z=dz,
                            message=f"retract {dz * 1000:.0f} mm Z done",
                            done=True,
                        )
                    return StepResult(
                        self.state,
                        vision_state,
                        moved_z=dz,
                        message=f"lift {dz * 1000:.0f} mm Z",
                    )
                if self._retract_phase == 1:
                    dy = float(self.config.retract_distance)
                    self.robot.move_y_relative(dy)
                    self._retract_phase = 2
                    self.state = ControlState.DONE
                    return StepResult(
                        self.state, vision_state, message="retract Y done", done=True
                    )
            except Exception as exc:  # noqa: BLE001
                return self._enter_error(f"retract failed: {exc}")

        return self._enter_error(f"unhandled state {self.state}")


class MockRobot:
    """In-memory robot for FSM tests without hardware."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.02) -> None:
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.move_log: list[tuple[str, float]] = []

    def move_z_relative(self, distance: float) -> None:
        self.z += float(distance)
        self.move_log.append(("z", float(distance)))

    def move_x_relative(self, distance: float) -> None:
        self.x += float(distance)
        self.move_log.append(("x", float(distance)))

    def move_y_relative(self, distance: float) -> None:
        self.y += float(distance)
        self.move_log.append(("y", float(distance)))

    def stop(self) -> None:
        self.move_log.append(("stop", 0.0))

    def get_position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)
