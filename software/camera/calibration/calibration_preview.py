from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .camera_feed import BackgroundCameraFeed
from .card_config import CalibrationCardConfig
from .chessboard import detect_board_pose


@dataclass
class PreviewOverlay:
    board_status: str = "waiting for first capture (press C)"
    estimate_text: str = "need 3 valid samples for solve"
    sample_count: int = 0
    valid_count: int = 0


class CalibrationPreviewThread:
    """Camera overlay UI on a worker thread so teleop stays at full sim rate."""

    def __init__(
        self,
        *,
        camera_feed: BackgroundCameraFeed,
        card_cfg: CalibrationCardConfig,
        key_queue: queue.SimpleQueue[int],
        capture_queue: queue.SimpleQueue[bool],
        overlay_provider: Callable[[], PreviewOverlay],
        detect_interval_s: float = 0.5,
    ) -> None:
        self._camera_feed = camera_feed
        self._card_cfg = card_cfg
        self._key_queue = key_queue
        self._capture_queue = capture_queue
        self._overlay_provider = overlay_provider
        self._detect_interval_s = detect_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_board = None
        self._latest_board_lock = threading.Lock()

    def start(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit(
                "opencv-python is required for camera calibration GUI. "
                "Install with: pip install -r requirements-vision.txt"
            ) from exc
        self._cv2 = cv2
        self._thread = threading.Thread(target=self._run, name="calibration-preview", daemon=True)
        self._thread.start()

    @property
    def latest_board_observation(self):
        with self._latest_board_lock:
            return self._latest_board

    def _run(self) -> None:
        cv2 = self._cv2
        cv2.namedWindow("Camera calibration", cv2.WINDOW_NORMAL)
        last_detect_s = 0.0
        cached_preview = None

        while not self._stop.is_set():
            shared = self._camera_feed.latest()
            if shared is None:
                key = cv2.waitKey(20)
                if key not in (-1, 255):
                    self._key_queue.put(int(key) & 0xFF)
                continue

            overlay = self._overlay_provider()
            now = time.perf_counter()
            if cached_preview is None or (now - last_detect_s) >= self._detect_interval_s:
                board = detect_board_pose(shared.color_bgr, self._card_cfg, shared.color_intrinsics)
                with self._latest_board_lock:
                    self._latest_board = board
                cached_preview = board.annotated_bgr if board.annotated_bgr is not None else shared.color_bgr.copy()
                last_detect_s = now

            preview = cached_preview.copy()
            cv2.putText(
                preview,
                "C=capture  O/P=gripper  T=home  (keys: this window or PyBullet)",
                (12, preview.shape[0] - 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                preview,
                f"samples {overlay.sample_count} valid {overlay.valid_count}",
                (12, preview.shape[0] - 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                overlay.board_status,
                (12, preview.shape[0] - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                overlay.estimate_text,
                (12, preview.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Camera calibration", preview)
            key = cv2.waitKey(20)
            if key not in (-1, 255):
                code = int(key) & 0xFF
                self._key_queue.put(code)
                if code in (ord("c"), ord("C")):
                    try:
                        self._capture_queue.put_nowait(True)
                    except Exception:
                        pass

        cv2.destroyAllWindows()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
