from __future__ import annotations

import threading
import time
from typing import List, Optional, Sequence

try:
    import serial
except ImportError as exc:
    raise SystemExit("Missing dependency: pyserial. Install with: pip install pyserial") from exc


class AsciiRobotClient:
    """ASCII protocol client with a single lock for thread-safe command IO."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.4) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser = None
        self._io_lock = threading.Lock()

    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
            )
            return True
        except serial.SerialException:
            self._ser = None
            return False

    def disconnect(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def _send_command(self, cmd: str, wait_ms: int = 120) -> Optional[str]:
        if not self._ser or not self._ser.is_open:
            return None

        with self._io_lock:
            try:
                self._ser.write((cmd + "\n").encode("utf-8"))
                self._ser.flush()
                time.sleep(max(0.0, wait_ms) / 1000.0)

                deadline = time.time() + max(self.timeout, 0.25)
                responses: List[str] = []
                while time.time() < deadline:
                    raw = self._ser.readline()
                    if not raw:
                        if responses:
                            break
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip().replace("\x00", "")
                    if line:
                        responses.append(line)
                    time.sleep(0.01)
                    if self._ser.in_waiting == 0 and responses:
                        break
                if not responses:
                    return None
                return responses[-1]
            except serial.SerialException:
                return None

    def start(self) -> Optional[str]:
        return self._send_command("!START", wait_ms=150)

    def stop(self) -> Optional[str]:
        return self._send_command("!STOP", wait_ms=150)

    def reset(self) -> Optional[str]:
        return self._send_command("!RESET", wait_ms=300)

    def set_command_mode(self, mode: int) -> Optional[str]:
        return self._send_command(f"#CMDMODE {int(mode)}", wait_ms=120)

    def move_joint(self, joints_deg: Sequence[float], speed: float, wait_ms: int = 400) -> Optional[str]:
        cmd = "&{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.2f}".format(
            float(joints_deg[0]),
            float(joints_deg[1]),
            float(joints_deg[2]),
            float(joints_deg[3]),
            float(joints_deg[4]),
            float(joints_deg[5]),
            float(speed),
        )
        return self._send_command(cmd, wait_ms=wait_ms)

    def get_joint_pos(self) -> Optional[List[float]]:
        resp = self._send_command("#GETJPOS", wait_ms=80)
        if not resp or not resp.startswith("ok"):
            return None

        parts = resp.split()
        if len(parts) < 7:
            return None

        try:
            return [float(parts[i]) for i in range(1, 7)]
        except (ValueError, IndexError):
            return None

    def send_trajectory_point_async(self, joints_deg: Sequence[float], speed: float) -> bool:
        if not self._ser or not self._ser.is_open:
            return False

        cmd = "&{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.2f}".format(
            float(joints_deg[0]),
            float(joints_deg[1]),
            float(joints_deg[2]),
            float(joints_deg[3]),
            float(joints_deg[4]),
            float(joints_deg[5]),
            float(speed),
        )

        with self._io_lock:
            try:
                self._ser.write((cmd + "\n").encode("utf-8"))
                self._ser.flush()
                return True
            except serial.SerialException:
                return False
