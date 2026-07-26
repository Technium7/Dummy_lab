#!/usr/bin/env python3
"""
Robot ASCII Protocol Interface
Provides a simple Python interface to send ASCII commands to the DummyRobot over COM3.

Run as a script for a raw ASCII REPL (type full lines such as ``#OFFSET_J 1``)::

    python -m robot_ascii_interface.robot_ascii_interface --port COM3
"""

import argparse
import serial
import sys
import time
from pathlib import Path

_SOFTWARE_DIR = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))
from manual_control.mapping import clamp_firmware_joints_deg
from robot_ascii_interface.serial_session import SerialLineSession
from robot_config import default_robot_port


class RobotAsciiInterface:
    def __init__(self, port: str | None = None, baudrate=115200, timeout=1.0, *, debug_serial: bool = False):
        """
        Initialize serial connection to robot.
        
        Args:
            port: Serial port name (default: platform-specific; see robot_config.default_robot_port)
            baudrate: Baud rate (default: 115200)
            timeout: Read timeout in seconds (default: 1.0)
            debug_serial: Log every command sent and response/drained bytes received
        """
        self.port = port if port is not None else default_robot_port()
        self.baudrate = baudrate
        self.timeout = timeout
        self.debug_serial = bool(debug_serial)
        self.ser = None
        self._stream_tx_count = 0
        self._session: SerialLineSession | None = None

    def _log_tx(self, cmd: str) -> None:
        if not self.debug_serial:
            return
        if cmd.startswith("&"):
            self._stream_tx_count += 1
            print(f"[Serial TX stream #{self._stream_tx_count}] {cmd}")
        else:
            print(f"[Serial TX] {cmd}")

    def _log_rx_text(self, text: str, *, source: str = "response") -> None:
        if not self.debug_serial or not text:
            return
        for line in text.replace("\x00", "").splitlines():
            cleaned = line.strip()
            if cleaned:
                print(f"[Serial RX {source}] {cleaned}")

    def _log_drained(self, raw: bytes) -> None:
        if not self.debug_serial or not raw:
            return
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = repr(raw)
        self._log_rx_text(text, source=f"drain {len(raw)}B")

    def drain_rx(self, max_bytes: int = 4096, *, max_time_s: float = 0.008) -> int:
        """Discard unread stream ack lines (legacy API for callers outside teleop)."""
        if self._session is None:
            return 0
        return self._session.discard_stream_acks(max_lines=max(1, max_bytes // 8))

    def _log_rx_line(self, line: str, source: str) -> None:
        self._log_rx_text(line, source=source)

    def connect(self):
        """Open serial connection."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
                rtscts=False,
                dsrdtr=False,
            )
            SerialLineSession.configure_linux_port(self.port, self.ser)
            self._session = SerialLineSession(
                self.ser,
                debug_serial=self.debug_serial,
                log_tx=self._log_tx,
                log_rx=self._log_rx_line,
            )
            self._session.start()
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
            return False

    def disconnect(self):
        """Close serial connection."""
        if self._session is not None:
            self._session.stop()
            self._session = None
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected")

    def _require_session(self) -> SerialLineSession:
        if self._session is None:
            raise serial.SerialException("Serial session not connected")
        return self._session

    def send_command_async(self, cmd: str) -> bool:
        """Write a command without waiting for a response."""
        if self._session is None:
            return False
        return self._session.send_fire_and_forget(cmd)

    def send_command(self, cmd, wait_ms=200, *, extra_read_sec: float = 0.0, max_read_s: float | None = None):
        """
        Send a command and get response.
        """
        if self._session is None:
            print("Not connected")
            return None
        try:
            return self._session.send_command(
                cmd,
                wait_ms=wait_ms,
                extra_read_sec=extra_read_sec,
                max_read_s=max_read_s,
            )
        except (serial.SerialException, UnicodeDecodeError) as e:
            print(f"Error reading response: {e}")
            return None

    def hold_stream_pose(
        self, hold_joints: list[float] | tuple[float, ...], speed: float
    ) -> list[float]:
        """
        Hold current pose after teleop keys release.

        One hold waypoint at the last commanded pose — no ``!STOP``/``!START`` or ``#GETJPOS``.
        """
        if self._session is None:
            return [float(v) for v in hold_joints[:6]]
        j = clamp_firmware_joints_deg([float(v) for v in hold_joints[:6]])
        self._session.discard_stream_acks(max_lines=512)
        self.send_trajectory_point_async(j[0], j[1], j[2], j[3], j[4], j[5], speed)
        return j
    
    # Control Commands
    def start(self):
        """Enable robot."""
        resp = self.send_command('!START')
        return resp
    
    def stop(self):
        """Emergency stop."""
        if self._session is not None:
            self._session.flush_rx_queue()
        resp = self.send_command("!STOP", wait_ms=120)
        return resp
    
    def ensure_cmdmode_stream(self) -> None:
        """Restore interruptible ``&`` streaming without ``!STOP``/``!START``."""
        if self._session is not None:
            self._session.discard_stream_acks(max_lines=512)
        self.send_command("#CMDMODE 2", wait_ms=80, max_read_s=0.5)
        if self._session is not None:
            self._session.discard_stream_acks(max_lines=64)

    def home(self, wait_ms: int = 200, *, extra_read_sec: float = 45.0, max_wait_s: float | None = None):
        """Home all joints to (0, 0, 90, 0, 0, 0). Firmware blocks until motion completes."""
        if self._session is None:
            return None
        timeout_s = max_wait_s if max_wait_s is not None else max(30.0, extra_read_sec + wait_ms / 1000.0)
        try:
            return self._session.send_home_command(max_wait_s=timeout_s)
        except (serial.SerialException, UnicodeDecodeError):
            return None

    def calibrate_home_offset(self, wait_ms: int = 8000):
        """
        Run firmware global home-offset calibration (firmware: !CALIBRATION -> CalibrateHomeOffset).
        Typical response line: ``calibration ok``. Increase wait_ms if the routine takes longer.
        """
        return self.send_command('!CALIBRATION', wait_ms=wait_ms)

    def apply_joint_home_offset(self, joint: int, wait_ms: int = 500):
        """
        Use the motor's current position as its zero/home reference (firmware: #OFFSET_J).

        In the reference firmware this command is handled on the USB ASCII path only (not the
        trimmed UART4 # handler).

        Args:
            joint: Motor index 1..7 (matches firmware motorJ[node]; 1..6 are arm joints).

        Returns:
            Response string from the controller, or None on timeout.
        """
        j = int(joint)
        if j < 1 or j > 7:
            raise ValueError("joint must be in 1..7")
        return self.send_command(f'#OFFSET_J {j}', wait_ms=wait_ms)

    def set_joint_current_limit(self, joint: int, current_a: float, wait_ms: int = 80):
        """
        Set a motor current limit in amps (firmware: ``#I_LIMIT_J``).

        Args:
            joint: Motor index 1..7 (1..6 are arm joints).
            current_a: Current limit in amperes (typical stock value is 1.0 A).
        """
        j = int(joint)
        if j < 1 or j > 7:
            raise ValueError("joint must be in 1..7")
        return self.send_command(
            f"#I_LIMIT_J {j} {float(current_a):g}",
            wait_ms=wait_ms,
            max_read_s=0.5,
        )

    def set_arm_current_limits(self, current_a: float, *, joints: range = range(1, 7), wait_ms: int = 80):
        """Apply the same current limit (A) to arm joints J1..J6 (or custom range)."""
        results = []
        for j in joints:
            resp = self.set_joint_current_limit(j, current_a, wait_ms=wait_ms)
            results.append((j, resp))
            time.sleep(0.05)
        return results

    def reset(self):
        """Reset to rest position."""
        resp = self.send_command('!RESET', wait_ms=2000)
        return resp
    
    def disable(self):
        """Disable motors."""
        resp = self.send_command('!DISABLE')
        return resp

    def hand_open(self):
        """Open the gripper (firmware: !HAND_O)."""
        return self.send_command('!HAND_O')

    def hand_close(self):
        """Close the gripper (firmware: !HAND_C)."""
        return self.send_command('!HAND_C')

    def hand_enable(self):
        """Enable gripper driver (firmware: !HAND_EN)."""
        return self.send_command('!HAND_EN')

    # Query Commands
    def get_joint_pos(self):
        """Get current joint positions (J1-J6 in degrees)."""
        if self._session is None:
            return None
        try:
            return self._session.read_joint_pos(wait_ms=60, max_read_s=0.5)
        except (serial.SerialException, UnicodeDecodeError):
            return None
    
    def get_linear_pos(self):
        """Get current linear position (X, Y, Z, A, B, C)."""
        resp = self.send_command('#GETLPOS')
        if resp and resp.startswith('ok'):
            try:
                values = resp.split()[1:]  # Skip 'ok'
                if len(values) < 6:
                    return None
                pose = [float(v) for v in values[:6]]
                return pose
            except (ValueError, IndexError):
                return None
        return None
    
    def set_command_mode(self, mode):
        """
        Set command mode.
        
        Args:
            mode: 1=Sequential, 2=Interruptable, 3=Trajectory, 4=Tuning
        """
        resp = self.send_command(f'#CMDMODE {mode}')
        return resp
    
    # Motion Commands
    def send_trajectory_point_async(self, j1, j2, j3, j4, j5, j6, speed):
        """Send a single trajectory waypoint WITHOUT waiting for a response."""
        if self._session is None:
            return False

        cmd = f"&{j1:.3f},{j2:.3f},{j3:.3f},{j4:.3f},{j5:.3f},{j6:.3f},{speed:.1f}"
        ok = self._session.send_stream_point(cmd)
        if not ok and self.debug_serial:
            print("[Serial TX stream write failed]")
        return ok

    def send_trajectory_point(self, j1, j2, j3, j4, j5, j6, speed, timeout_sec=30):
        """
        Send a single trajectory waypoint and block until robot confirms 'ok'.
        """
        if self._session is None:
            return False

        cmd = f"&{j1:.3f},{j2:.3f},{j3:.3f},{j4:.3f},{j5:.3f},{j6:.3f},{speed:.1f}"
        self._session.send_fire_and_forget(cmd)

        start = time.time()
        while time.time() - start < timeout_sec:
            line = self._session.wait_line(0.08)
            if line and "ok" in line:
                return True
        return False

    def run_trajectory(self, start_joints, end_joints,
                       n_points=30, min_speed=5, max_speed=40,
                       send_interval_ms=50, timeout_sec=60):
        """
        Execute a smooth point-to-point trajectory using CMDMODE 3.
        Velocity follows a bell (sine) curve: ramps up, cruises, ramps down.

        Args:
            start_joints : list of 6 floats – starting joint angles (degrees)
            end_joints   : list of 6 floats – target  joint angles (degrees)
            n_points     : number of waypoints (more = smoother, but higher latency)
            min_speed    : speed at start/end of trajectory (degrees/s)
            max_speed    : peak speed at midpoint (degrees/s)
            timeout_sec  : per-waypoint timeout

        Returns:
            True if all waypoints were acknowledged, False otherwise
        """
        import math

        print(f"Running trajectory: {n_points} waypoints, speed {min_speed}→{max_speed}→{min_speed}")

        for i in range(1, n_points + 1):
            t = i / n_points  # 0 → 1

            # Bell-shaped speed profile (sine arch: 0 → 1 → 0)
            speed = min_speed + (max_speed - min_speed) * math.sin(math.pi * t)
            speed = max(min_speed, speed)

            # Linear position interpolation
            joints = [
                start_joints[j] + (end_joints[j] - start_joints[j]) * t
                for j in range(6)
            ]

            pct = int(t * 100)
            print(f"  [{pct:3d}%] J1={joints[0]:6.2f} spd={speed:.1f}", end='  ')

            # Send WITHOUT waiting for response (async) to avoid braking
            ok = self.send_trajectory_point_async(
                joints[0], joints[1], joints[2],
                joints[3], joints[4], joints[5],
                speed
            )

            if ok:
                print("sent")
            else:
                print("SEND FAILED")
                return False

            # Delay before next waypoint to maintain streaming
            time.sleep(send_interval_ms / 1000.0)

        # All waypoints sent; wait for final motion to complete
        print("  Waiting for trajectory to complete...")
        time.sleep(2.0)
        return True

    def move_joint(self, j1, j2, j3, j4, j5, j6, speed=None, wait_ms=500):
        """
        Move to joint position.
        
        Args:
            j1-j6: Joint angles in degrees
            speed: Optional speed (0-100, degree/s)
            wait_ms: Wait time before reading response
        """
        if speed is not None:
            cmd = f'&{j1},{j2},{j3},{j4},{j5},{j6},{speed}'
        else:
            cmd = f'&{j1},{j2},{j3},{j4},{j5},{j6}'
        
        resp = self.send_command(cmd, wait_ms=wait_ms)
        return resp
    
    def move_linear(self, x, y, z, a, b, c, speed=None, wait_ms=500):
        """
        Move to linear position (if supported by robot).
        
        Args:
            x, y, z: Position in mm
            a, b, c: Orientation in degrees
            speed: Optional speed
            wait_ms: Wait time before reading response
        """
        if speed is not None:
            cmd = f'@{x},{y},{z},{a},{b},{c},{speed}'
        else:
            cmd = f'@{x},{y},{z},{a},{b},{c}'
        
        resp = self.send_command(cmd, wait_ms=wait_ms)
        return resp
    
    def wait_for_motion_complete(self, timeout_sec=30, poll_interval_ms=100, tolerance=0.5):
        """
        Wait for robot motion to complete by polling position until stable.
        
        Args:
            timeout_sec: Maximum time to wait (seconds)
            poll_interval_ms: Poll interval (milliseconds)
            tolerance: Position change threshold to detect motion (degrees)
            
        Returns:
            True if motion completed, False if timeout
        """
        import time
        
        start_time = time.time()
        last_pos = None
        stable_count = 0
        stable_threshold = 3  # Require 3 consecutive stable readings
        
        while time.time() - start_time < timeout_sec:
            pos = self.get_joint_pos()
            
            if pos is None or len(pos) < 6:
                time.sleep(poll_interval_ms / 1000.0)
                continue
            
            # Check if position changed significantly
            if last_pos is None or len(last_pos) < 6:
                last_pos = pos
                stable_count = 0
            else:
                # Calculate max difference between current and last position
                try:
                    max_diff = max(abs(pos[i] - last_pos[i]) for i in range(6))
                    
                    if max_diff < tolerance:
                        stable_count += 1
                        if stable_count >= stable_threshold:
                            return True  # Motion complete
                    else:
                        stable_count = 0  # Reset counter if motion detected
                        last_pos = pos
                except (TypeError, ValueError):
                    # If comparison fails, skip this poll
                    time.sleep(poll_interval_ms / 1000.0)
                    continue
            
            time.sleep(poll_interval_ms / 1000.0)
        
        return False  # Timeout
    
    def interactive_shell(self, default_wait_ms: int = 200):
        """
        Read lines from stdin and send each non-empty line verbatim over serial (raw ASCII REPL).

        Type full protocol strings, for example ``#OFFSET_J 1`` or ``!START``. Use ``help`` / ``exit`` as built-ins.
        """
        print("Robot ASCII Interface — raw command REPL")
        print("Enter full lines exactly as on the wire (e.g. #OFFSET_J 1, !START, #GETJPOS).")
        print("Type help for examples, exit to quit.")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("robot> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    break
                
                if user_input.lower() == 'help':
                    print("""
Available Commands:
  Control:
    !START              - Enable robo
    !STOP               - Emergency stop
    !HOME               - Home position
    !CALIBRATION        - Calibrate home offsets (global routine)
    !RESET              - Rest position
    !DISABLE            - Disable motors

  Home offset (per motor, USB path):
    #OFFSET_J <1-7>     - Set current position as home for that joint motor

  Motor tuning:
    #I_LIMIT_J <1-7> <A> - Set motor current limit in amperes

  Query:
    #GETJPOS            - Get joint positions
    #GETLPOS            - Get linear position
    #CMDMODE <mode>     - Set mode (1-4)
    
  Motion (joint):
    &j1,j2,j3,j4,j5,j6[,speed]
    Example: &0,0,90,0,0,0,20
    
  Motion (linear):
    @x,y,z,a,b,c[,speed]
    Example: @100,200,300,0,0,0

  Hand (gripper):
    !HAND_O             - Open hand
    !HAND_C             - Close hand
""")
                    continue
                
                resp = self.send_command(user_input, wait_ms=default_wait_ms)
                if resp:
                    print(f"Response: {resp}")
                else:
                    print("No response (timeout)")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open serial and type raw ASCII robot commands (e.g. #OFFSET_J 1, !START)."
    )
    parser.add_argument(
        "--port",
        type=str,
        default=default_robot_port(),
        help="Serial port (env ROBOT_PORT; Linux default /dev/ttyACM0 or auto-detect)",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate")
    parser.add_argument("--timeout", type=float, default=1.0, help="Serial read timeout (seconds)")
    parser.add_argument(
        "--send-wait-ms",
        type=int,
        default=200,
        metavar="MS",
        help="Delay before reading the response after each line (increase if you see timeouts)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    robot = RobotAsciiInterface(port=args.port, baudrate=args.baudrate, timeout=args.timeout)
    if not robot.connect():
        sys.exit(1)
    try:
        robot.interactive_shell(default_wait_ms=args.send_wait_ms)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
