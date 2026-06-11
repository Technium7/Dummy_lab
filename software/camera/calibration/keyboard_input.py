from __future__ import annotations

import queue
import time

import pybullet as p

from manual_control.ik_core import is_key_down


_MOVEMENT_KEYS = frozenset("wsadqehkujiy")


class DualWindowKeyboard:
    """
    Merge PyBullet and OpenCV key state so teleop works from either window.

    OpenCV only reports key-down events (with OS repeat while held). We treat a
    key as held until repeat stops for ``cv2_hold_timeout_s``.
    """

    def __init__(self, cv2_hold_timeout_s: float = 0.12) -> None:
        self._cv2_hold_timeout_s = cv2_hold_timeout_s
        self._cv2_last_seen: dict[str, float] = {}
        self._cv2_triggered: set[str] = set()

    def drain_cv2_queue(self, key_queue: queue.SimpleQueue[int]) -> None:
        while True:
            try:
                key_code = key_queue.get_nowait()
            except Exception:
                break
            self.ingest_cv2_key(key_code)

    def ingest_cv2_key(self, key_code: int) -> None:
        if key_code in (-1, 255):
            return
        try:
            ch = chr(int(key_code) & 0xFF).lower()
        except ValueError:
            return
        if not ch.isalnum():
            return
        now = time.perf_counter()
        self._cv2_last_seen[ch] = now
        self._cv2_triggered.add(ch)

    def _cv2_is_down(self, char: str) -> bool:
        last = self._cv2_last_seen.get(char.lower())
        if last is None:
            return False
        return (time.perf_counter() - last) <= self._cv2_hold_timeout_s

    def is_down(self, events: dict[int, int], char: str) -> bool:
        return is_key_down(events, char) or self._cv2_is_down(char)

    def was_triggered(self, events: dict[int, int], char: str) -> bool:
        lower = char.lower()
        if lower in self._cv2_triggered:
            self._cv2_triggered.discard(lower)
            return True
        upper = ord(char.upper())
        lower_code = ord(char.lower())
        pb = ((events.get(lower_code, 0) & p.KEY_WAS_TRIGGERED) != 0) or (
            (events.get(upper, 0) & p.KEY_WAS_TRIGGERED) != 0
        )
        return pb

    def any_movement_down(self, events: dict[int, int]) -> bool:
        return any(self.is_down(events, k) for k in _MOVEMENT_KEYS)

    def clear_triggers(self) -> None:
        self._cv2_triggered.clear()
