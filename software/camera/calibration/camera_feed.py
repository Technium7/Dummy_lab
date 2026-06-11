from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SharedCameraFrame:
    color_bgr: np.ndarray
    color_intrinsics: Any
    timestamp: float


class BackgroundCameraFeed:
    """Read RealSense frames on a worker thread so teleop is not blocked."""

    def __init__(self, streamer) -> None:
        self._streamer = streamer
        self._lock = threading.Lock()
        self._latest: SharedCameraFrame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="calibration-camera", daemon=True)
        self._thread.start()
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            if self._error is not None:
                raise self._error
            if self.latest() is not None:
                return
            time.sleep(0.02)
        raise TimeoutError("Camera feed did not produce a frame within 10 s")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self._streamer.read()
                color_bgr = frame.color_rgb[:, :, ::-1].copy()
                shared = SharedCameraFrame(
                    color_bgr=color_bgr,
                    color_intrinsics=frame.color_intrinsics,
                    timestamp=frame.timestamp,
                )
                with self._lock:
                    self._latest = shared
        except Exception as exc:
            self._error = exc

    def latest(self) -> SharedCameraFrame | None:
        with self._lock:
            if self._latest is None:
                return None
            return SharedCameraFrame(
                color_bgr=self._latest.color_bgr.copy(),
                color_intrinsics=self._latest.color_intrinsics,
                timestamp=self._latest.timestamp,
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
