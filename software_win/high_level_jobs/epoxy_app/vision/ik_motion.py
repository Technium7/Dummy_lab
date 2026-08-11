"""Cartesian relative steps via the same DLS velocity IK as manual_control.

Smoother than large teleop key holds: small ``step_linear_speed`` bursts until
the end-effector has moved approximately ``distance`` along X or Z.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import pybullet as p

from manual_control.ik_core import compute_dls_joint_velocity, get_end_link_pose_robot_frame
from manual_control.mapping import deg_speed_to_ascii_speed, sim_vel_rad_to_transformed_deg_s


@dataclass
class IkStepConfig:
    step_linear_speed: float = 0.00165
    ik_damping: float = 0.05
    rotation_weight: float = 8.0
    joint_vel_limit: float = 1.2
    joint_force: float = 2000.0
    sim_hz: float = 240.0
    # Must stay << min_z_step or tiny creeps are treated as already done.
    settle_tol: float = 1e-5
    max_step_time: float = 2.0
    # Ease in/out so long vision dwells don't restart as a hard jerk.
    accel_time: float = 0.20


class IkCartesianMover:
    """RobotMotion implementation backed by PyBullet Jacobian + stream intent."""

    def __init__(
        self,
        robot_id: int,
        end_link_idx: int,
        joint_indices: Sequence[int],
        *,
        robot_control=None,
        config: IkStepConfig | None = None,
        on_sim_step: Callable[[], None] | None = None,
        abort_check: Callable[[], bool] | None = None,
    ) -> None:
        self.robot_id = robot_id
        self.end_link_idx = end_link_idx
        self.joint_indices = list(joint_indices)
        self.robot_control = robot_control
        self.config = config or IkStepConfig()
        self.on_sim_step = on_sim_step
        self.abort_check = abort_check
        self._xyz = list(get_end_link_pose_robot_frame(robot_id, end_link_idx)[0])
        self._aborted = False

    def get_position(self) -> tuple[float, float, float]:
        pos, _ = get_end_link_pose_robot_frame(self.robot_id, self.end_link_idx)
        self._xyz = list(pos)
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def stop(self) -> None:
        """Hold current pose. Does not block later moves (use abort() for estop)."""
        self._stream_hold()

    def abort(self) -> None:
        """Emergency stop: hold now and reject further axis moves until clear_abort()."""
        self._aborted = True
        self._stream_hold()

    def clear_abort(self) -> None:
        self._aborted = False

    def _should_abort(self) -> bool:
        if self._aborted:
            return True
        if self.abort_check is not None and self.abort_check():
            self._aborted = True
            return True
        return False

    def move_z_relative(self, distance: float) -> None:
        self._move_axis(2, float(distance))

    def move_x_relative(self, distance: float) -> None:
        self._move_axis(0, float(distance))

    def move_y_relative(self, distance: float) -> None:
        self._move_axis(1, float(distance))

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _move_axis(self, axis: int, distance: float) -> None:
        if abs(distance) < 1e-9:
            return
        if self._should_abort():
            self._stream_hold()
            return
        start = self.get_position()
        target = start[axis] + distance
        sign = 1.0 if distance > 0 else -1.0
        speed = abs(self.config.step_linear_speed)
        dt = 1.0 / max(self.config.sim_hz, 1.0)
        t0 = time.perf_counter()
        accel_t = max(float(self.config.accel_time), 1e-3)
        # Allow long retract slides; tiny creeps still use the short default cap.
        max_t = max(
            float(self.config.max_step_time),
            abs(distance) / max(speed, 1e-6) + 2.0 * accel_t + 0.5,
        )
        # Keep settle tolerance well below the commanded distance so small
        # approach creeps are not skipped (tol ≈ commanded was eating 0.07 mm steps).
        tol = min(abs(self.config.settle_tol), max(abs(distance) * 0.15, 1e-6))
        decel_dist = max(speed * accel_t, abs(distance) * 0.4, 2.0 * tol)

        while True:
            if self._should_abort():
                break
            pos = self.get_position()
            err = target - pos[axis]
            if abs(err) <= tol or sign * err <= 0:
                break
            if time.perf_counter() - t0 > max_t:
                break

            elapsed = time.perf_counter() - t0
            ramp_up = self._smoothstep(elapsed / accel_t)
            ramp_down = self._smoothstep(abs(err) / decel_dist)
            v = speed * ramp_up * ramp_down
            # One-step overshoot clamp
            v = min(v, abs(err) / max(dt, 1e-3))
            if v < 1e-6:
                v = min(speed * 0.05, abs(err) / max(dt, 1e-3))

            twist = [0.0, 0.0, 0.0, 0.0, 0.0]
            twist[axis] = sign * v

            arm_joints = self.joint_indices[:5]
            qdot = compute_dls_joint_velocity(
                self.robot_id,
                self.end_link_idx,
                arm_joints,
                twist,
                self.config.ik_damping,
                self.config.rotation_weight,
                joint_weights=(1.15, 1.1, 1.05, 1.0, 1.0)[: len(arm_joints)],
            )
            commanded = [0.0] * len(self.joint_indices)
            if qdot is not None:
                for i, vel in enumerate(qdot):
                    if i < len(commanded):
                        commanded[i] = float(vel)
            lim = self.config.joint_vel_limit
            if lim > 0:
                commanded = [min(max(vv, -lim), lim) for vv in commanded]

            for j_idx, vel in zip(self.joint_indices, commanded):
                p.setJointMotorControl2(
                    self.robot_id,
                    j_idx,
                    p.VELOCITY_CONTROL,
                    targetVelocity=vel,
                    force=self.config.joint_force,
                )
            p.stepSimulation()
            if self.on_sim_step:
                self.on_sim_step()

            if self.robot_control is not None:
                arm_deg = [
                    float(p.getJointState(self.robot_id, j)[0]) * 180.0 / 3.141592653589793
                    for j in self.joint_indices[:6]
                ]
                # Prefer mapping helpers when available for firmware frame
                try:
                    from manual_control.mapping import sim_rad_to_transformed_deg

                    arm_deg = [
                        sim_rad_to_transformed_deg(i, float(p.getJointState(self.robot_id, j)[0]))
                        for i, j in enumerate(self.joint_indices[:6])
                    ]
                    vel_deg = [
                        sim_vel_rad_to_transformed_deg_s(i, commanded[i] if i < len(commanded) else 0.0)
                        for i in range(6)
                    ]
                except Exception:  # noqa: BLE001
                    vel_deg = [c * 180.0 / 3.141592653589793 for c in (commanded + [0.0] * 6)[:6]]
                speed_cmd = deg_speed_to_ascii_speed(vel_deg)
                self.robot_control.set_stream_intent(
                    target_deg=arm_deg[:6],
                    vel_deg_s=vel_deg[:6],
                    speed=speed_cmd,
                    moving_key_down=True,
                    home_latched=False,
                    timestamp=time.perf_counter(),
                )
            time.sleep(dt)

        self._stream_hold()

    def _stream_hold(self) -> None:
        for j_idx in self.joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                j_idx,
                p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=self.config.joint_force,
            )
        if self.robot_control is None:
            return
        try:
            from manual_control.mapping import sim_rad_to_transformed_deg

            arm_deg = [
                sim_rad_to_transformed_deg(i, float(p.getJointState(self.robot_id, j)[0]))
                for i, j in enumerate(self.joint_indices[:6])
            ]
        except Exception:  # noqa: BLE001
            arm_deg = [
                float(p.getJointState(self.robot_id, j)[0]) * 180.0 / 3.141592653589793
                for j in self.joint_indices[:6]
            ]
        self.robot_control.set_stream_intent(
            target_deg=arm_deg[:6],
            vel_deg_s=[0.0] * 6,
            speed=1,
            moving_key_down=False,
            home_latched=False,
            timestamp=time.perf_counter(),
        )
