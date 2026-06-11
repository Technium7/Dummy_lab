"""Robust Linux-friendly serial line I/O for the Dummy robot ASCII protocol."""

from __future__ import annotations

import platform
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, List, Optional

import serial


LogFn = Callable[[str, str], None]

# Immediate ``&`` replies: queue depth (e.g. ``15``) and/or motion ack (``ok``).
_STREAM_ACK_ONLY = re.compile(r"^(\d+|ok)$")
_STREAM_ACK_COMBINED = re.compile(r"^(\d+)ok$")
_JOINT_POS_LINE = re.compile(
    r"^ok\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)


def parse_joint_pos_line(line: str) -> Optional[List[float]]:
    """Parse ``ok j1 j2 j3 j4 j5 j6`` (#GETJPOS)."""
    match = _JOINT_POS_LINE.match(line.strip())
    if not match:
        return None
    return [float(match.group(i)) for i in range(1, 7)]


def split_rx_line(line: str) -> List[str]:
    """Split glued firmware replies like ``15ok`` into separate lines."""
    text = line.strip().replace("\x00", "")
    if not text:
        return []
    match = _STREAM_ACK_COMBINED.match(text)
    if match:
        return [match.group(1), "ok"]
    return [text]


def is_stream_ack_line(line: str) -> bool:
    """True for per-waypoint acks only — not ``ok 0.01 -75.17 ...`` (#GETJPOS)."""
    text = line.strip()
    if _STREAM_ACK_ONLY.match(text):
        return True
    return bool(_STREAM_ACK_COMBINED.match(text))


class SerialLineSession:
    """
    One RX reader thread + mutex-protected writes.

    Stream ``&`` acks are discarded proactively so they never fill the queue or
    block ``#GETJPOS`` / ``!START`` responses.
    """

    def __init__(
        self,
        ser: serial.Serial,
        *,
        debug_serial: bool = False,
        log_tx: LogFn | None = None,
        log_rx: LogFn | None = None,
    ) -> None:
        self._ser = ser
        self._debug_serial = debug_serial
        self._log_tx = log_tx
        self._log_rx = log_rx
        self._write_lock = threading.Lock()
        self._lines: Deque[str] = deque(maxlen=512)
        self._lines_cond = threading.Condition()
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None

    @staticmethod
    def configure_linux_port(port_path: str, ser: serial.Serial) -> None:
        if platform.system() != "Linux":
            return
        name = Path(port_path).name
        latency_path = Path(f"/sys/bus/usb-serial/devices/{name}/latency_timer")
        if latency_path.exists():
            try:
                latency_path.write_text("1")
            except OSError:
                pass
        try:
            import termios

            termios.tcflush(ser.fileno(), termios.TCIFLUSH)
        except (ImportError, OSError):
            pass

    def start(self) -> None:
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, name="serial-rx", daemon=True)
        self._reader.start()

    def stop(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None
        with self._lines_cond:
            self._lines.clear()
            self._lines_cond.notify_all()

    def _emit_tx(self, cmd: str) -> None:
        if self._log_tx is not None:
            self._log_tx(cmd)

    def _emit_rx(self, line: str, *, source: str) -> None:
        if self._log_rx is not None and not is_stream_ack_line(line):
            self._log_rx(line, source)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            if not self._ser.is_open:
                time.sleep(0.01)
                continue
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                decoded = raw.decode("utf-8", errors="ignore")
                for line in split_rx_line(decoded):
                    if not line:
                        continue
                    with self._lines_cond:
                        if is_stream_ack_line(line) and len(self._lines) >= 24:
                            continue
                        self._lines.append(line)
                        self._lines_cond.notify_all()
                    self._emit_rx(line, source="rx")
            except (serial.SerialException, OSError):
                time.sleep(0.01)

    def _write_line(self, cmd: str) -> None:
        self._emit_tx(cmd)
        self._ser.write((cmd + "\n").encode())
        self._ser.flush()

    def flush_rx_queue(self) -> int:
        """Drop all pending RX lines before a blocking command/exit."""
        with self._lines_cond:
            count = len(self._lines)
            self._lines.clear()
            return count

    def discard_stream_acks(self, *, max_lines: int = 256) -> int:
        removed = 0
        with self._lines_cond:
            while self._lines and removed < max_lines:
                head = self._lines[0]
                if is_stream_ack_line(head):
                    self._lines.popleft()
                    removed += 1
                else:
                    break
        return removed

    def wait_line(self, timeout_s: float) -> Optional[str]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._lines_cond:
            while not self._lines:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._lines_cond.wait(timeout=remaining)
            return self._lines.popleft()

    def send_command(self, cmd: str, *, wait_ms: int = 200, extra_read_sec: float = 0.0, max_read_s: float | None = None) -> Optional[str]:
        if not self._ser.is_open:
            return None
        with self._write_lock:
            self.flush_rx_queue()
            try:
                self._write_line(cmd)
            except (serial.SerialException, OSError):
                return None

            time.sleep(wait_ms / 1000.0)
            responses: list[str] = []
            if max_read_s is None:
                read_window_s = max(3.0, wait_ms / 1000.0 + 1.5) + extra_read_sec
            else:
                read_window_s = max(0.05, float(max_read_s)) + extra_read_sec
            deadline = time.monotonic() + read_window_s

            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                line = self.wait_line(min(0.08, remaining))
                if not line:
                    if responses:
                        break
                    continue
                if is_stream_ack_line(line):
                    continue
                responses.append(line)
                time.sleep(0.02)
                if cmd.startswith("#"):
                    break
                if cmd.startswith("!HAND_EN"):
                    extra = self.wait_line(0.05)
                    while extra and is_stream_ack_line(extra):
                        extra = self.wait_line(0.05)
                    if extra:
                        responses.append(extra)
                    break
                if cmd.startswith("!"):
                    break

            return responses[-1] if responses else None

    def send_home_command(self, *, max_wait_s: float = 120.0) -> Optional[str]:
        """Send ``!HOME`` and wait for the post-homing reply (firmware blocks until done)."""
        if not self._ser.is_open:
            return None
        with self._write_lock:
            self.flush_rx_queue()
            try:
                self._write_line("!HOME")
            except (serial.SerialException, OSError):
                return None

            deadline = time.monotonic() + max(1.0, float(max_wait_s))
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                line = self.wait_line(min(0.15, remaining))
                if not line or is_stream_ack_line(line):
                    continue
                return line.strip()
            return None

    def read_joint_pos(self, *, wait_ms: int = 60, max_read_s: float = 0.5) -> Optional[List[float]]:
        """Send ``#GETJPOS`` and return joints once a valid ``ok j1..j6`` line arrives."""
        if not self._ser.is_open:
            return None
        with self._write_lock:
            self.flush_rx_queue()
            try:
                self._write_line("#GETJPOS")
            except (serial.SerialException, OSError):
                return None

            time.sleep(wait_ms / 1000.0)
            deadline = time.monotonic() + max(0.05, float(max_read_s))
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                line = self.wait_line(min(0.08, remaining))
                if not line or is_stream_ack_line(line):
                    continue
                joints = parse_joint_pos_line(line)
                if joints is not None:
                    return joints
            return None

    def send_stream_point(self, cmd: str) -> bool:
        """Fire-and-forget ``&`` waypoint; always clears stale acks first."""
        if not self._ser.is_open:
            return False
        with self._write_lock:
            self.discard_stream_acks(max_lines=512)
            try:
                self._write_line(cmd)
                return True
            except (serial.SerialException, OSError):
                return False

    def send_fire_and_forget(self, cmd: str) -> bool:
        if not self._ser.is_open:
            return False
        with self._write_lock:
            self.discard_stream_acks(max_lines=512)
            try:
                self._write_line(cmd)
                return True
            except (serial.SerialException, OSError):
                return False
