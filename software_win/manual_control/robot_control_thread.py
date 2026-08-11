from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import serial

_SOFTWARE_DIR = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from manual_control.mapping import (
    HOME_JOINTS_DEG,
    clamp_firmware_joints_deg,
    home_stream_ascii_speed,
    max_ascii_speed,
)
from manual_control.robot_session import connect_robot_for_teleop, disconnect_robot, sync_target_to_current
from robot_ascii_interface import RobotAsciiInterface
from robot_config import find_dummy_robot_port


@dataclass
class RobotStreamIntent:
    target_deg: List[float]
    vel_deg_s: List[float]
    speed: float
    moving_key_down: bool
    home_latched: bool
    timestamp: float


class ThreadSafeRobotProxy:
    """Routes infrequent blocking queries through the robot I/O thread."""

    def __init__(self, thread: RobotControlThread) -> None:
        self._thread = thread

    def get_joint_pos(self) -> Optional[List[float]]:
        return self._thread.get_joint_pos_blocking()

    def set_stream_intent(
        self,
        *,
        target_deg: Sequence[float],
        vel_deg_s: Sequence[float],
        speed: float,
        moving_key_down: bool,
        home_latched: bool,
        timestamp: float,
    ) -> None:
        """Forward stream waypoints (used by epoxy apply / IkCartesianMover)."""
        self._thread.set_stream_intent(
            target_deg=target_deg,
            vel_deg_s=vel_deg_s,
            speed=speed,
            moving_key_down=moving_key_down,
            home_latched=home_latched,
            timestamp=timestamp,
        )


class RobotControlThread(threading.Thread):
    """Dedicated serial thread: waypoint streaming, hand/home, recovery."""

    _MAX_INTEGRATE_DT_FACTOR = 1.5
    _STREAM_VEL_SCALE = 0.92
    _SERIAL_FAIL_RECOVER = 15
    _HOME_FIRMWARE_WAIT_S = 3.0

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        timeout: float,
        control_dt_s: float,
        home_use_firmware: bool,
        sim_only: bool,
        debug_serial: bool = False,
        joint_current_limit_a: float = 0.0,
    ) -> None:
        super().__init__(name="robot-control", daemon=True)
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._control_dt_s = max(0.02, float(control_dt_s))
        self._home_use_firmware = home_use_firmware
        self._sim_only = sim_only
        self._debug_serial = debug_serial
        self._joint_current_limit_a = float(joint_current_limit_a)
        self._stop = threading.Event()
        self._intent_lock = threading.Lock()
        self._latest_intent: Optional[RobotStreamIntent] = None
        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._query_queue: queue.Queue[tuple[str, queue.Queue[object]]] = queue.Queue()
        self._robot: RobotAsciiInterface | None = None
        self._next_robot_tick = 0.0
        self._last_sent_deg: Optional[List[float]] = None
        self._home_stream_speed = home_stream_ascii_speed()
        self._hold_stream_speed = max(6.9, max_ascii_speed() * 0.25)
        self._ready = threading.Event()
        self._stream_send_count = 0
        self._stream_pose_deg: Optional[List[float]] = None
        self._last_stream_send_s = 0.0
        self._was_streaming = False
        self._serial_failures = 0
        self._home_busy = False
        self._drives_enabled = True

    @property
    def proxy(self) -> ThreadSafeRobotProxy:
        return ThreadSafeRobotProxy(self)

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout=timeout)

    def request_stop(self) -> None:
        self._stop.set()
        try:
            self._cmd_queue.put_nowait("stop")
        except Exception:
            pass

    def set_stream_intent(
        self,
        *,
        target_deg: Sequence[float],
        vel_deg_s: Sequence[float],
        speed: float,
        moving_key_down: bool,
        home_latched: bool,
        timestamp: float,
    ) -> None:
        intent = RobotStreamIntent(
            target_deg=[float(v) for v in target_deg[:6]],
            vel_deg_s=[float(v) for v in vel_deg_s[:6]],
            speed=float(speed),
            moving_key_down=bool(moving_key_down),
            home_latched=bool(home_latched),
            timestamp=float(timestamp),
        )
        with self._intent_lock:
            prev = self._latest_intent
            self._latest_intent = intent
        if moving_key_down and not (prev and prev.moving_key_down):
            self._stream_pose_deg = None
            self._last_stream_send_s = 0.0
            self._next_robot_tick = 0.0
        if moving_key_down and self._next_robot_tick > timestamp:
            self._next_robot_tick = 0.0

    def request_hand_open(self) -> None:
        if not self._sim_only:
            try:
                self._cmd_queue.put_nowait("hand_open")
            except Exception:
                pass

    def request_hand_close(self) -> None:
        if not self._sim_only:
            try:
                self._cmd_queue.put_nowait("hand_close")
            except Exception:
                pass

    def request_home_firmware(self) -> None:
        if not self._sim_only and self._home_use_firmware:
            self._home_busy = True
            self._stream_pose_deg = None
            self._last_stream_send_s = 0.0
            try:
                self._cmd_queue.put_nowait("home_firmware")
            except Exception:
                self._home_busy = False

    def request_disable(self) -> None:
        """Stop streaming, then send firmware ``!DISABLE`` (motors torque-off)."""
        if not self._sim_only:
            self._was_streaming = False
            self._stream_pose_deg = None
            self._last_stream_send_s = 0.0
            try:
                self._cmd_queue.put_nowait("disable")
            except Exception:
                pass

    def request_enable(self) -> None:
        """Re-arm drives with ``!START`` + CMDMODE 2 after a ``B`` disable."""
        if not self._sim_only:
            try:
                self._cmd_queue.put_nowait("enable")
            except Exception:
                pass

    def get_joint_pos_blocking(self, timeout: float = 0.35) -> Optional[List[float]]:
        if self._sim_only or self._robot is None:
            return None
        response: queue.Queue[object] = queue.Queue(maxsize=1)
        try:
            self._query_queue.put_nowait(("joint_pos", response))
        except Exception:
            return None
        try:
            result = response.get(timeout=timeout)
        except queue.Empty:
            return None
        return result if isinstance(result, list) else None

    def _current_intent(self) -> Optional[RobotStreamIntent]:
        with self._intent_lock:
            return self._latest_intent

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            if cmd == "stop":
                return
            self._run_command(cmd)

    def _drain_queries(self) -> None:
        if self._was_streaming:
            return
        try:
            kind, response = self._query_queue.get_nowait()
        except queue.Empty:
            return
        if kind != "joint_pos":
            return
        pos = None
        if self._robot is not None:
            try:
                pos = self._robot.get_joint_pos()
            except Exception:
                pos = None
        try:
            response.put_nowait(pos)
        except Exception:
            pass

    def _resume_stream(self) -> None:
        self._next_robot_tick = 0.0
        self._stream_pose_deg = None
        self._last_stream_send_s = 0.0
        self._was_streaming = False

    def _recover_serial(self) -> None:
        if self._sim_only:
            return
        print("[Robot] Serial error — reconnecting...", file=sys.stderr)
        disconnect_robot(self._robot)
        found = find_dummy_robot_port()
        if found:
            self._port = found
        self._robot = connect_robot_for_teleop(
            port=self._port,
            baudrate=self._baudrate,
            timeout=self._timeout,
            sim_only=False,
            debug_serial=self._debug_serial,
            joint_current_limit_a=self._joint_current_limit_a,
        )
        self._last_sent_deg = None
        self._drives_enabled = True
        self._resume_stream()
        self._serial_failures = 0

    def _hold_on_key_up(self, intent: RobotStreamIntent) -> None:
        self._stream_pose_deg = None
        self._last_stream_send_s = 0.0
        if self._robot is None:
            return
        hold = list(self._last_sent_deg) if self._last_sent_deg is not None else list(intent.target_deg[:6])
        try:
            hold = self._robot.hold_stream_pose(hold, self._hold_stream_speed)
            hold = clamp_firmware_joints_deg(hold)
            self._last_sent_deg = list(hold)
            if self._debug_serial:
                print(f"[Robot stream] key-up hold {[round(v, 2) for v in hold]}")
        except Exception as exc:
            print(f"[Robot stream] hold failed: {exc}", file=sys.stderr)
            self._serial_failures += 1

    def _run_home_firmware(self) -> None:
        if self._robot is None:
            self._home_busy = False
            return
        self._was_streaming = False
        try:
            if self._robot._session is not None:
                self._robot._session.discard_stream_acks(max_lines=512)
            resp = self._robot.home(max_wait_s=self._HOME_FIRMWARE_WAIT_S)
            if resp:
                print(f"[Robot] !HOME -> {resp}")
                self._last_sent_deg = clamp_firmware_joints_deg(HOME_JOINTS_DEG)
                self._robot.set_command_mode(2)
            else:
                print(
                    "[Robot] !HOME timed out — restoring CMDMODE 2",
                    file=sys.stderr,
                )
                self._last_sent_deg = None
                self._robot.ensure_cmdmode_stream()
            if self._robot._session is not None:
                self._robot._session.discard_stream_acks(max_lines=512)
            self._stream_pose_deg = None
            self._last_stream_send_s = 0.0
            self._next_robot_tick = 0.0
        except Exception as exc:
            print(f"[Robot] !HOME failed: {exc}", file=sys.stderr)
            self._last_sent_deg = None
            self._stream_pose_deg = None
            self._serial_failures += 1
        finally:
            self._home_busy = False

    def _run_disable(self) -> None:
        if self._robot is None:
            return
        self._was_streaming = False
        self._stream_pose_deg = None
        self._last_stream_send_s = 0.0
        try:
            # Sync target←current first so the next !START does not chase a stale
            # HOME target left over from T. Then torque-off the motors.
            sync_target_to_current(self._robot)
            resp = self._robot.disable()
            self._drives_enabled = False
            print(f"[Robot] !DISABLE -> {resp}")
        except Exception as exc:
            print(f"[Robot] !DISABLE failed: {exc}", file=sys.stderr)
            self._serial_failures += 1

    def _run_enable(self) -> None:
        if self._robot is None:
            return
        try:
            sync_target_to_current(self._robot)
            resp = self._robot.start()
            print(f"[Robot] !START -> {resp}")
            self._robot.set_command_mode(2)
            if self._robot._session is not None:
                self._robot._session.discard_stream_acks(max_lines=512)
            self._drives_enabled = True
            self._resume_stream()
        except Exception as exc:
            print(f"[Robot] !START (re-enable) failed: {exc}", file=sys.stderr)
            self._serial_failures += 1

    def _run_command(self, cmd: str) -> None:
        if self._robot is None:
            return
        was_streaming = self._was_streaming
        self._was_streaming = False
        try:
            if cmd == "hand_open":
                self._robot.hand_enable()
                print(f"[Robot] HAND_O -> {self._robot.hand_open()}")
            elif cmd == "hand_close":
                self._robot.hand_enable()
                print(f"[Robot] HAND_C -> {self._robot.hand_close()}")
            elif cmd == "home_firmware":
                self._run_home_firmware()
                return
            elif cmd == "disable":
                self._run_disable()
                return
            elif cmd == "enable":
                self._run_enable()
                return
        except Exception as exc:
            print(f"[Robot] {cmd} failed: {exc}")
            self._serial_failures += 1
        finally:
            if cmd not in ("home_firmware", "disable", "enable"):
                self._resume_stream()
                self._was_streaming = was_streaming

    def _seed_stream_pose(self, intent: RobotStreamIntent) -> None:
        if self._last_sent_deg is not None:
            seed = list(self._last_sent_deg)
        else:
            hw = self._robot.get_joint_pos() if self._robot is not None else None
            if hw is not None and len(hw) >= 6:
                seed = [float(v) for v in hw[:6]]
            else:
                seed = list(intent.target_deg[:6])
        self._stream_pose_deg = seed
        if self._debug_serial:
            print(f"[Robot stream] init pose {[round(v, 2) for v in seed]}")

    def _maybe_stream_waypoint(self, now: float) -> None:
        if self._robot is None or now < self._next_robot_tick or self._home_busy:
            return
        if not self._drives_enabled:
            return
        intent = self._current_intent()
        if intent is None or len(intent.target_deg) < 6:
            return

        stream_dt = self._control_dt_s
        streaming = intent.moving_key_down and not intent.home_latched
        if self._was_streaming and not streaming:
            self._was_streaming = False
            self._hold_on_key_up(intent)
            self._next_robot_tick = now + stream_dt
            return
        self._was_streaming = streaming

        if streaming:
            if self._stream_pose_deg is None:
                self._seed_stream_pose(intent)
            dt_send = (
                max(1e-6, now - self._last_stream_send_s) if self._last_stream_send_s > 0.0 else stream_dt
            )
            dt_send = min(dt_send, stream_dt * self._MAX_INTEGRATE_DT_FACTOR)
            for j in range(6):
                self._stream_pose_deg[j] += (
                    intent.vel_deg_s[j] * self._STREAM_VEL_SCALE * dt_send
                )
            robot_target_deg = self._stream_pose_deg
        elif intent.home_latched:
            robot_target_deg = intent.target_deg
        else:
            # Idle: key-up hold already sent the stop pose — do not chase sim (avoids snap-back).
            self._next_robot_tick = now + stream_dt
            return

        robot_target_deg = clamp_firmware_joints_deg(robot_target_deg)

        should_send = True

        if should_send:
            send_speed = self._home_stream_speed if intent.home_latched else intent.speed
            ok = self._robot.send_trajectory_point_async(
                robot_target_deg[0],
                robot_target_deg[1],
                robot_target_deg[2],
                robot_target_deg[3],
                robot_target_deg[4],
                robot_target_deg[5],
                send_speed,
            )
            if ok:
                self._stream_send_count += 1
                self._last_sent_deg = list(robot_target_deg)
                self._last_stream_send_s = now
                self._serial_failures = 0
                if self._debug_serial and self._stream_send_count % 50 == 0:
                    print(
                        f"[Robot stream] sent={self._stream_send_count} "
                        f"speed={send_speed:.1f} target={[round(v, 2) for v in robot_target_deg]}"
                    )
            else:
                self._serial_failures += 1
                if self._debug_serial:
                    print("[Robot stream] serial write failed")
                if self._serial_failures >= self._SERIAL_FAIL_RECOVER:
                    self._recover_serial()

        self._next_robot_tick = now + stream_dt

    def run(self) -> None:
        self._robot = connect_robot_for_teleop(
            port=self._port,
            baudrate=self._baudrate,
            timeout=self._timeout,
            sim_only=self._sim_only,
            debug_serial=self._debug_serial,
            joint_current_limit_a=self._joint_current_limit_a,
        )
        self._ready.set()
        poll_s = min(0.002, self._control_dt_s * 0.1)
        try:
            while not self._stop.is_set():
                try:
                    self._drain_commands()
                    self._drain_queries()
                    now = time.perf_counter()
                    self._maybe_stream_waypoint(now)
                except (OSError, serial.SerialException) as exc:
                    self._serial_failures += 1
                    if self._debug_serial:
                        print(f"[Robot stream] serial error: {exc}")
                    if self._serial_failures >= self._SERIAL_FAIL_RECOVER:
                        try:
                            self._recover_serial()
                        except Exception as recover_exc:
                            print(f"[Robot] reconnect failed: {recover_exc}", file=sys.stderr)
                            time.sleep(0.5)
                    else:
                        time.sleep(0.02)
                except Exception as exc:
                    print(f"[Robot stream] error: {exc}", file=sys.stderr)
                    self._serial_failures += 1
                    time.sleep(0.02)
                time.sleep(poll_s)
        finally:
            try:
                disconnect_robot(self._robot, already_disabled=not self._drives_enabled)
            except Exception:
                pass
            self._robot = None
