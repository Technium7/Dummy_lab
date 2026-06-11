from __future__ import annotations

import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parents[2]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from high_level_jobs.subtasks.grab_sensor.capture import RealSenseStreamer

from .card_config import CalibrationCardConfig
from .chessboard import detect_board_pose
from .convergence_plot import ConvergencePlotWindow, save_convergence_trend
from .hand_eye import HandEyeCalibrator
from .shared_state import SharedCalibrationBus


def _default_shot_name() -> str:
    return datetime.now(timezone.utc).strftime("shot_%Y%m%d_%H%M%S")


class CalibrationCvThread(threading.Thread):
    """
    Camera preview, checkerboard detection, capture, and hand-eye solve.

    Reads joint snapshots from the control thread; never touches PyBullet or the robot.
    """

    def __init__(
        self,
        bus: SharedCalibrationBus,
        *,
        card_cfg: CalibrationCardConfig,
        out_dir: Path,
        camera_serial: str,
        camera_width: int,
        camera_height: int,
        camera_fps: int,
        detect_interval_s: float = 0.4,
        capture_min_interval_s: float = 0.35,
    ) -> None:
        super().__init__(name="calibration-cv", daemon=True)
        self._bus = bus
        self._card_cfg = card_cfg
        self._out_dir = out_dir
        self._camera_serial = camera_serial
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._camera_fps = camera_fps
        self._detect_interval_s = detect_interval_s
        self._capture_min_interval_s = capture_min_interval_s
        self._calibrator = HandEyeCalibrator(out_dir=out_dir)
        self._plot_gui = ConvergencePlotWindow()
        self._board_status = "waiting for first capture (press C)"
        self._estimate_text = "need 3 valid samples for solve"
        self._error: BaseException | None = None

    @property
    def calibrator(self) -> HandEyeCalibrator:
        return self._calibrator

    @property
    def error(self) -> BaseException | None:
        return self._error

    def run(self) -> None:
        try:
            self._run_cv_loop()
        except BaseException as exc:
            self._error = exc
            self._bus.stop.set()
            raise

    def _run_cv_loop(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit(
                "opencv-python is required for camera calibration GUI. "
                "Install with: pip install -r requirements-vision.txt"
            ) from exc

        if not self._plot_gui.start():
            print("[Calibration] matplotlib not available; convergence plot disabled.")

        streamer = RealSenseStreamer(
            width=self._camera_width,
            height=self._camera_height,
            fps=self._camera_fps,
            serial=self._camera_serial,
            warmup=15,
        )
        streamer.start()

        cv2.namedWindow("Camera calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera calibration", self._camera_width, self._camera_height)
        print(
            "[CV] Preview thread running. Press C in camera or PyBullet window to capture. "
            "Joint pose is sampled from the control thread."
        )

        last_detect_s = 0.0
        last_capture_s = 0.0
        cached_overlay = None
        last_snapshot = self._bus.read_joint_snapshot()

        try:
            while not self._bus.stop.is_set():
                frame = streamer.read()
                color_bgr = frame.color_rgb[:, :, ::-1].copy()
                intrinsics = frame.color_intrinsics
                last_snapshot = self._bus.read_joint_snapshot()

                now = time.perf_counter()
                if cached_overlay is None or (now - last_detect_s) >= self._detect_interval_s:
                    board = detect_board_pose(color_bgr, self._card_cfg, intrinsics)
                    cached_overlay = board.annotated_bgr if board.annotated_bgr is not None else color_bgr.copy()
                    last_detect_s = now

                preview = cached_overlay.copy()
                self._draw_overlay(preview, last_snapshot)
                cv2.imshow("Camera calibration", preview)
                key = cv2.waitKey(20)
                if key not in (-1, 255):
                    code = int(key) & 0xFF
                    self._bus.key_codes.put(code)
                    if code in (ord("c"), ord("C")) and (now - last_capture_s) >= self._capture_min_interval_s:
                        last_capture_s = now
                        self._capture_shot(color_bgr, intrinsics)

                self._drain_capture_snapshots(color_bgr, intrinsics)
        finally:
            result_path = self._calibrator.save_result()
            if result_path is not None:
                print(f"[Calibration] Saved result to {result_path}")
            history = [h.to_dict() for h in self._calibrator.history]
            json_path, png_path = save_convergence_trend(self._out_dir, history)
            if json_path is not None:
                print(f"[Calibration] Saved convergence history to {json_path}")
            if png_path is not None:
                print(f"[Calibration] Saved convergence plot to {png_path}")
            streamer.stop()
            cv2.destroyAllWindows()
            self._plot_gui.close()

    def _capture_shot(self, color_bgr: np.ndarray, intrinsics) -> None:
        self._bus.capture_signals.put(True)
        pose = self._wait_snapshot(timeout_s=0.5)
        if pose is None:
            print("[Calibration] capture timed out waiting for control thread pose")
            return
        stem = _default_shot_name()
        board = detect_board_pose(color_bgr, self._card_cfg, intrinsics)
        self._store_sample(stem, color_bgr, board, pose)

    def _drain_capture_snapshots(self, color_bgr: np.ndarray, intrinsics) -> None:
        """Handle captures triggered from the PyBullet window (C key)."""
        try:
            while True:
                snapshot = self._bus.capture_snapshots.get_nowait()
                stem = _default_shot_name()
                board = detect_board_pose(color_bgr, self._card_cfg, intrinsics)
                self._store_sample(stem, color_bgr, board, snapshot)
        except queue.Empty:
            pass

    def _wait_snapshot(self, timeout_s: float):
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline and not self._bus.stop.is_set():
            try:
                return self._bus.capture_snapshots.get(timeout=0.05)
            except queue.Empty:
                continue
        return None

    def _store_sample(self, stem: str, color_bgr: np.ndarray, board, snapshot) -> None:
        sample = self._calibrator.add_sample(
            joints_deg=snapshot.joints_deg,
            board=board,
            t_base_target=snapshot.t_base_target,
            color_bgr=color_bgr,
            stem=stem,
        )
        if sample.board_found:
            self._board_status = f"captured {stem}: board ok, reproj {sample.reprojection_error_px:.2f}px"
        else:
            self._board_status = f"captured {stem}: checkerboard not detected"
        print(f"[Calibration] {self._board_status}")
        self._update_estimate()

    def _draw_overlay(self, preview: np.ndarray, snapshot) -> None:
        import cv2

        keys = " ".join(snapshot.active_keys) if snapshot.active_keys else "(none)"
        cv2.putText(
            preview,
            f"joints (ctrl): {keys}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 180),
            1,
        )
        if snapshot.robot_target_deg:
            jtxt = " ".join(f"{v:+.1f}" for v in snapshot.robot_target_deg[:6])
            cv2.putText(
                preview,
                f"target deg: {jtxt}",
                (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 255, 180),
                1,
            )
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
            f"samples {self._calibrator.sample_count} valid {self._calibrator.valid_count}",
            (12, preview.shape[0] - 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            preview,
            self._board_status,
            (12, preview.shape[0] - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            preview,
            self._estimate_text,
            (12, preview.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

    def _update_estimate(self) -> None:
        estimate = self._calibrator.estimate()
        if estimate is not None:
            self._estimate_text = (
                f"cam dist {estimate.distance_m:.4f} m | "
                f"rpy {estimate.rpy_deg[0]:+.1f} {estimate.rpy_deg[1]:+.1f} {estimate.rpy_deg[2]:+.1f} deg | "
                f"spread {estimate.translation_spread_m * 1000:.1f} mm / {estimate.angle_spread_deg:.2f} deg"
            )
            self._plot_gui.update([h.to_dict() for h in self._calibrator.history])
        else:
            need = self._calibrator.samples_needed()
            self._estimate_text = f"need {need} more valid sample(s) for solve (have {self._calibrator.valid_count})"
