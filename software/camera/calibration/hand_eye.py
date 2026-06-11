from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from high_level_jobs.subtasks.grab_sensor.transforms import decompose_matrix

from .chessboard import BoardObservation, rvec_tvec_to_matrix
from .fk import rotation_translation_from_matrix


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for hand-eye calibration. "
            "Install with: pip install -r requirements-vision.txt"
        ) from exc
    return cv2


@dataclass
class CalibrationSample:
    index: int
    joints_deg: list[float]
    board_found: bool
    reprojection_error_px: float
    rvec: list[float] | None
    tvec: list[float] | None
    r_base_target: list[list[float]]
    t_base_target: list[float]
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationEstimate:
    sample_count: int
    translation_m: np.ndarray
    rpy_deg: np.ndarray
    distance_m: float
    angle_deg: float
    transform_robot_cam: np.ndarray
    translation_spread_m: float
    angle_spread_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "translation_m": self.translation_m.tolist(),
            "rpy_deg": self.rpy_deg.tolist(),
            "distance_m": float(self.distance_m),
            "angle_deg": float(self.angle_deg),
            "translation_spread_m": float(self.translation_spread_m),
            "angle_spread_deg": float(self.angle_spread_deg),
            "transform_robot_cam": self.transform_robot_cam.tolist(),
        }


def _rotation_angle_deg(rot: np.ndarray) -> float:
    trace = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


_MIN_HAND_EYE_SAMPLES = 3


class HandEyeCalibrator:
    """Eye-to-hand solver: fixed camera observes a checkerboard on the moving gripper."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir.resolve()
        self.samples_dir = self.out_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self._samples: list[CalibrationSample] = []
        self._valid_r_base_gripper: list[np.ndarray] = []
        self._valid_t_base_gripper: list[np.ndarray] = []
        self._valid_r_target_cam: list[np.ndarray] = []
        self._valid_t_target_cam: list[np.ndarray] = []
        self._history: list[CalibrationEstimate] = []

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def valid_count(self) -> int:
        return len(self._valid_r_base_gripper)

    @property
    def history(self) -> list[CalibrationEstimate]:
        return list(self._history)

    def add_sample(
        self,
        *,
        joints_deg: Sequence[float],
        board: BoardObservation,
        t_base_target: np.ndarray,
        color_bgr: np.ndarray,
        stem: str,
    ) -> CalibrationSample:
        idx = len(self._samples) + 1
        r_bt, t_bt = rotation_translation_from_matrix(t_base_target)
        sample = CalibrationSample(
            index=idx,
            joints_deg=[float(v) for v in joints_deg[:6]],
            board_found=board.found,
            reprojection_error_px=float(board.reprojection_error_px),
            rvec=board.rvec.tolist() if board.rvec is not None else None,
            tvec=board.tvec.tolist() if board.tvec is not None else None,
            r_base_target=r_bt.tolist(),
            t_base_target=t_bt.tolist(),
        )
        self._samples.append(sample)

        sample_dir = self.samples_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta_path = sample_dir / f"{stem}.meta.json"
        meta_path.write_text(json.dumps(sample.to_dict(), indent=2), encoding="utf-8")

        try:
            import cv2

            cv2.imwrite(str(sample_dir / f"{stem}.color.png"), color_bgr)
            if board.annotated_bgr is not None:
                cv2.imwrite(str(sample_dir / f"{stem}.annotated.png"), board.annotated_bgr)
        except Exception:
            pass

        if board.found and board.rvec is not None and board.tvec is not None:
            t_target_cam = rvec_tvec_to_matrix(board.rvec, board.tvec)
            r_tc, t_tc = rotation_translation_from_matrix(t_target_cam)
            self._valid_r_base_gripper.append(r_bt)
            self._valid_t_base_gripper.append(t_bt)
            self._valid_r_target_cam.append(r_tc)
            self._valid_t_target_cam.append(t_tc)
            self._append_estimate()

        return sample

    def _append_estimate(self) -> None:
        estimate = self.estimate()
        if estimate is not None:
            self._history.append(estimate)

    def samples_needed(self) -> int:
        return max(0, _MIN_HAND_EYE_SAMPLES - self.valid_count)

    def estimate(self) -> CalibrationEstimate | None:
        n = len(self._valid_r_base_gripper)
        if n < _MIN_HAND_EYE_SAMPLES:
            return None

        cv2 = _require_cv2()
        try:
            r_cam_base, t_cam_base = cv2.calibrateHandEye(
                self._valid_r_base_gripper,
                self._valid_t_base_gripper,
                self._valid_r_target_cam,
                self._valid_t_target_cam,
                method=cv2.CALIB_HAND_EYE_TSAI,
            )
        except cv2.error:
            return None
        t_robot_cam = np.eye(4, dtype=np.float64)
        t_robot_cam[:3, :3] = np.asarray(r_cam_base, dtype=np.float64)
        t_robot_cam[:3, 3] = np.asarray(t_cam_base, dtype=np.float64).reshape(3)

        trans, rpy_rad = decompose_matrix(t_robot_cam)
        rpy_deg = np.degrees(rpy_rad)
        distance_m = float(np.linalg.norm(trans))
        angle_deg = _rotation_angle_deg(t_robot_cam[:3, :3])

        trans_spread, angle_spread = self._bootstrap_spread()
        return CalibrationEstimate(
            sample_count=n,
            translation_m=trans,
            rpy_deg=rpy_deg,
            distance_m=distance_m,
            angle_deg=angle_deg,
            transform_robot_cam=t_robot_cam,
            translation_spread_m=trans_spread,
            angle_spread_deg=angle_spread,
        )

    def _bootstrap_spread(self) -> tuple[float, float]:
        n = len(self._valid_r_base_gripper)
        if n < _MIN_HAND_EYE_SAMPLES + 1:
            return 0.0, 0.0

        cv2 = _require_cv2()
        translations: list[np.ndarray] = []
        angles: list[float] = []
        for i in range(n):
            r_bg = self._valid_r_base_gripper[:i] + self._valid_r_base_gripper[i + 1 :]
            t_bg = self._valid_t_base_gripper[:i] + self._valid_t_base_gripper[i + 1 :]
            r_tc = self._valid_r_target_cam[:i] + self._valid_r_target_cam[i + 1 :]
            t_tc = self._valid_t_target_cam[:i] + self._valid_t_target_cam[i + 1 :]
            if len(r_bg) < _MIN_HAND_EYE_SAMPLES:
                continue
            try:
                r_cb, t_cb = cv2.calibrateHandEye(r_bg, t_bg, r_tc, t_tc, method=cv2.CALIB_HAND_EYE_TSAI)
            except cv2.error:
                continue
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = np.asarray(r_cb, dtype=np.float64)
            mat[:3, 3] = np.asarray(t_cb, dtype=np.float64).reshape(3)
            translations.append(mat[:3, 3].copy())
            angles.append(_rotation_angle_deg(mat[:3, :3]))

        if len(translations) < 2:
            return 0.0, 0.0
        trans_stack = np.vstack(translations)
        trans_spread = float(np.linalg.norm(np.std(trans_stack, axis=0)))
        angle_spread = float(np.std(angles))
        return trans_spread, angle_spread

    def save_result(self, path: Path | None = None) -> Path | None:
        estimate = self.estimate()
        if estimate is None:
            return None
        out = path or (self.out_dir / "camera_to_robot.json")
        payload = {
            "method": "eye_to_hand_hand_eye",
            "sample_count": estimate.sample_count,
            "camera_to_robot": {
                "translation_m": estimate.translation_m.tolist(),
                "rpy_deg": estimate.rpy_deg.tolist(),
            },
            "metrics": {
                "distance_m": estimate.distance_m,
                "angle_deg": estimate.angle_deg,
                "translation_spread_m": estimate.translation_spread_m,
                "angle_spread_deg": estimate.angle_spread_deg,
            },
            "samples": [s.to_dict() for s in self._samples],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out
