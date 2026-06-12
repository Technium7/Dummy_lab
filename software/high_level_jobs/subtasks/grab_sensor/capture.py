from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

try:
    import pyrealsense2 as rs
except ImportError as exc:
    rs = None  # type: ignore
    _RS_IMPORT_ERROR = exc
else:
    _RS_IMPORT_ERROR = None


@dataclass
class CaptureResult:
    color_rgb: np.ndarray
    depth_m: np.ndarray
    depth_scale: float
    color_intrinsics: Any
    depth_intrinsics: Any
    device_name: str
    device_serial: str


@dataclass
class AlignedFrame:
    """Color image with depth aligned to the same pixel grid."""

    color_rgb: np.ndarray
    depth_m: np.ndarray
    depth_scale: float
    color_intrinsics: Any
    device_name: str
    device_serial: str
    timestamp: float


def _require_realsense():
    if rs is None:
        raise ImportError(
            "pyrealsense2 is required for grab_sensor capture. "
            "Install with: pip install pyrealsense2"
        ) from _RS_IMPORT_ERROR


def _intrinsics_to_dict(intr: Any) -> dict[str, float | int]:
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
    }


def _color_sensor(device: Any) -> Any | None:
    try:
        return device.first_color_sensor()
    except Exception:
        return None


def lock_realsense_color_exposure(
    color_sensor: Any,
    *,
    exposure_us: float | None = None,
    lock_white_balance: bool = True,
) -> dict[str, float]:
    """Disable auto exposure (and optionally WB); return locked values."""
    _require_realsense()
    locked: dict[str, float] = {}
    if color_sensor is None:
        return locked

    if color_sensor.supports(rs.option.enable_auto_exposure):
        if exposure_us is None:
            exposure_us = float(color_sensor.get_option(rs.option.exposure))
        if color_sensor.supports(rs.option.gain):
            locked["gain"] = float(color_sensor.get_option(rs.option.gain))
        color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
        if "gain" in locked:
            color_sensor.set_option(rs.option.gain, locked["gain"])
        color_sensor.set_option(rs.option.exposure, float(exposure_us))
        locked["exposure_us"] = float(exposure_us)

    if lock_white_balance and color_sensor.supports(rs.option.enable_auto_white_balance):
        wb = float(color_sensor.get_option(rs.option.white_balance))
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
        color_sensor.set_option(rs.option.white_balance, wb)
        locked["white_balance"] = wb

    return locked


def restore_realsense_color_auto(color_sensor: Any | None) -> None:
    """Re-enable RealSense color auto exposure and white balance."""
    if color_sensor is None:
        return
    _require_realsense()
    try:
        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
    except Exception:
        pass
    try:
        if color_sensor.supports(rs.option.enable_auto_white_balance):
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0)
    except Exception:
        pass


class RealSenseStreamer:
    """Continuous RealSense capture with depth aligned to color."""

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str = "",
        warmup: int = 15,
        lock_exposure: bool = False,
        exposure_us: float | None = None,
        lock_white_balance: bool = True,
    ) -> None:
        _require_realsense()
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.warmup = warmup
        self.lock_exposure = lock_exposure
        self.exposure_us = exposure_us
        self.lock_white_balance = lock_white_balance
        self._pipeline: Any = None
        self._align: Any = None
        self._depth_scale = 0.001
        self._device_name = ""
        self._device_serial = ""
        self._color_intrinsics: Any = None
        self._color_sensor: Any = None
        self.color_exposure_locked: dict[str, float] | None = None

    def start(self) -> None:
        self._pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        profile = self._pipeline.start(config)
        device = profile.get_device()
        self._device_name = device.get_info(rs.camera_info.name)
        self._device_serial = device.get_info(rs.camera_info.serial_number)
        depth_sensor = device.first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        self._align = rs.align(rs.stream.color)

        for _ in range(max(0, self.warmup)):
            self._pipeline.wait_for_frames()

        self._color_sensor = _color_sensor(device)
        if self.lock_exposure:
            self.color_exposure_locked = lock_realsense_color_exposure(
                self._color_sensor,
                exposure_us=self.exposure_us,
                lock_white_balance=self.lock_white_balance,
            )
            # One frame for the locked settings to settle.
            self._pipeline.wait_for_frames()

    def read(self) -> AlignedFrame:
        if self._pipeline is None:
            raise RuntimeError("RealSenseStreamer not started")
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Incomplete aligned frame")

        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = color_bgr[:, :, ::-1].copy()
        depth_m = depth_raw.astype(np.float64) * self._depth_scale
        self._color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        return AlignedFrame(
            color_rgb=color_rgb,
            depth_m=depth_m,
            depth_scale=self._depth_scale,
            color_intrinsics=self._color_intrinsics,
            device_name=self._device_name,
            device_serial=self._device_serial,
            timestamp=time.time(),
        )

    def frames(self) -> Iterator[AlignedFrame]:
        while True:
            yield self.read()

    def stop(self) -> None:
        if self._color_sensor is not None and self.color_exposure_locked:
            restore_realsense_color_auto(self._color_sensor)
        self._color_sensor = None
        self.color_exposure_locked = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


def capture_realsense_frame(
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup: int = 15,
    serial: str = "",
) -> CaptureResult:
    streamer = RealSenseStreamer(width=width, height=height, fps=fps, serial=serial, warmup=warmup)
    streamer.start()
    try:
        frame = streamer.read()
        return CaptureResult(
            color_rgb=frame.color_rgb,
            depth_m=frame.depth_m,
            depth_scale=frame.depth_scale,
            color_intrinsics=frame.color_intrinsics,
            depth_intrinsics=frame.color_intrinsics,
            device_name=frame.device_name,
            device_serial=frame.device_serial,
        )
    finally:
        streamer.stop()


def _dict_to_intrinsics(data: dict[str, float | int]) -> Any:
    class _Intr:
        pass

    intr = _Intr()
    intr.width = int(data["width"])
    intr.height = int(data["height"])
    intr.fx = float(data["fx"])
    intr.fy = float(data["fy"])
    intr.ppx = float(data["ppx"])
    intr.ppy = float(data["ppy"])
    return intr


_DEFAULT_D415_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 610.814697265625,
    "fy": 610.5267944335938,
    "ppx": 314.0621337890625,
    "ppy": 234.38771057128906,
}


def load_capture_from_files(
    color_path: Path,
    depth_npy_path: Path | None = None,
    depth_scale: float = 0.001,
) -> CaptureResult:
    color_rgb = np.asarray(Image.open(color_path).convert("RGB"))
    meta_path = color_path.parent / f"{color_path.name.split('.color')[0]}.meta.json"
    if not meta_path.is_file():
        meta_path = color_path.with_suffix(".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        intr_data = meta.get("intrinsics", _DEFAULT_D415_INTRINSICS)
    else:
        intr_data = _DEFAULT_D415_INTRINSICS

    if depth_npy_path and depth_npy_path.is_file():
        depth_raw = np.load(depth_npy_path)
        if np.issubdtype(depth_raw.dtype, np.floating):
            valid = depth_raw[np.isfinite(depth_raw) & (depth_raw > 0)]
            if valid.size and float(np.median(valid)) < 5.0:
                depth_m = depth_raw.astype(np.float64)
            else:
                depth_m = depth_raw.astype(np.float64) * depth_scale
        else:
            depth_m = depth_raw.astype(np.float64) * depth_scale
    else:
        depth_png = color_path.with_suffix(".depth.png")
        if not depth_png.is_file():
            raise FileNotFoundError(f"No depth file for {color_path}")
        depth_raw = np.asarray(Image.open(depth_png))
        depth_m = depth_raw.astype(np.float64) * depth_scale

    intr = _dict_to_intrinsics(intr_data)

    return CaptureResult(
        color_rgb=color_rgb,
        depth_m=depth_m,
        depth_scale=depth_scale,
        color_intrinsics=intr,
        depth_intrinsics=intr,
        device_name="file",
        device_serial="file",
    )


def save_debug_capture(capture: CaptureResult, out_dir: Path, stem: str = "capture") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(capture.color_rgb).save(out_dir / f"{stem}.color.png")
    depth_mm = (capture.depth_m * 1000.0).astype(np.uint16)
    Image.fromarray(depth_mm, mode="I;16").save(out_dir / f"{stem}.depth.png")
    np.save(out_dir / f"{stem}.depth.npy", depth_mm)
    meta = {
        "intrinsics": _intrinsics_to_dict(capture.color_intrinsics),
        "device_name": capture.device_name,
        "device_serial": capture.device_serial,
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_labeled_sample(
    out_dir: Path,
    sample_id: str,
    frame: AlignedFrame,
    grasp_uv: tuple[int, int],
    *,
    note: str = "",
) -> Path:
    """Save RGB, aligned depth, and grasp pixel label for NN training."""
    out_dir = out_dir.resolve()
    images_dir = out_dir / "images"
    depth_dir = out_dir / "depth"
    labels_dir = out_dir / "labels"
    previews_dir = out_dir / "previews"
    for d in (images_dir, depth_dir, labels_dir, previews_dir):
        d.mkdir(parents=True, exist_ok=True)

    u, v = int(grasp_uv[0]), int(grasp_uv[1])
    h, w = frame.color_rgb.shape[:2]
    depth_at = float(frame.depth_m[v, u]) if 0 <= v < h and 0 <= u < w else 0.0
    valid_depth = depth_at > 1e-4

    image_path = images_dir / f"{sample_id}.png"
    depth_path = depth_dir / f"{sample_id}.npy"
    label_path = labels_dir / f"{sample_id}.json"
    preview_path = previews_dir / f"{sample_id}.png"

    Image.fromarray(frame.color_rgb).save(image_path)
    np.save(depth_path, frame.depth_m.astype(np.float32))

    label = {
        "id": sample_id,
        "grasp_u": u,
        "grasp_v": v,
        "grasp_uv": [u, v],
        "width": w,
        "height": h,
        "depth_m_at_grasp": depth_at if valid_depth else None,
        "valid_depth": valid_depth,
        "note": note,
        "timestamp": frame.timestamp,
        "device_name": frame.device_name,
        "device_serial": frame.device_serial,
        "intrinsics": _intrinsics_to_dict(frame.color_intrinsics),
        "files": {
            "image": str(image_path.relative_to(out_dir)).replace("\\", "/"),
            "depth": str(depth_path.relative_to(out_dir)).replace("\\", "/"),
        },
    }
    label_path.write_text(json.dumps(label, indent=2), encoding="utf-8")

    manifest = out_dir / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(label, separators=(",", ":")) + "\n")

    _save_preview(preview_path, frame.color_rgb, grasp_uv, sample_id)
    return label_path


def _save_preview(path: Path, color_rgb: np.ndarray, grasp_uv: tuple[int, int], sample_id: str) -> None:
    try:
        import cv2
    except ImportError:
        Image.fromarray(color_rgb).save(path)
        return

    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    u, v = grasp_uv
    cv2.drawMarker(bgr, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(bgr, (u, v), 10, (0, 0, 255), 2)
    cv2.putText(bgr, sample_id, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(path), bgr)


def next_sample_id(out_dir: Path) -> str:
    labels_dir = out_dir / "labels"
    if not labels_dir.is_dir():
        return "000001"
    existing = sorted(p.stem for p in labels_dir.glob("*.json"))
    if not existing:
        return "000001"
    last = int(existing[-1])
    return f"{last + 1:06d}"
