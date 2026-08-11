"""Intel RealSense D415 backend for Windows grab_sensor (RGB + aligned depth)."""

from __future__ import annotations

import atexit
import time
import weakref
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from camera.base import CameraBackend, CameraInfo

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    rs = None  # type: ignore
    _RS_IMPORT_ERROR = exc
else:
    _RS_IMPORT_ERROR = None


def _require_realsense() -> None:
    if rs is None:
        raise ImportError(
            "pyrealsense2 is required for RealSense on Windows. "
            "Install Intel RealSense SDK +: pip install pyrealsense2"
        ) from _RS_IMPORT_ERROR


def list_realsense_devices() -> list[CameraInfo]:
    """Enumerate connected RealSense devices (empty if SDK missing)."""
    if rs is None:
        return []
    out: list[CameraInfo] = []
    ctx = rs.context()
    for i, device in enumerate(ctx.query_devices()):
        try:
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
        except Exception:
            continue
        out.append(
            CameraInfo(
                backend="realsense",
                name=name,
                index=i,
                serial=serial,
                model=name,
            )
        )
    return out


def _list_serials() -> list[str]:
    return [d.serial for d in list_realsense_devices() if d.serial]


def hardware_reset_realsense(serial: str = "", *, settle_s: float = 2.5) -> bool:
    _require_realsense()
    ctx = rs.context()
    target = None
    for device in ctx.query_devices():
        try:
            sn = device.get_info(rs.camera_info.serial_number)
            if not serial or sn == serial:
                target = device
                break
        except Exception:
            continue
    if target is None:
        return bool(_list_serials())
    try:
        target.hardware_reset()
    except Exception:
        pass
    time.sleep(settle_s)
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        serials = _list_serials()
        if serial:
            if serial in serials:
                return True
        elif serials:
            return True
        time.sleep(0.25)
    return False


def _wait_for_frames(pipeline: Any, *, timeout_ms: int = 10000, retries: int = 3) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(0.25 * (attempt + 1))
    devices = _list_serials()
    hint = f" Connected serials: {devices}." if devices else " No RealSense devices found."
    raise RuntimeError(
        f"RealSense frame did not arrive within {timeout_ms} ms.{hint}"
    ) from last_exc


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


@dataclass
class AlignedFrame:
    color_rgb: np.ndarray
    depth_m: np.ndarray
    depth_scale: float
    color_intrinsics: Any
    device_name: str
    device_serial: str
    timestamp: float


_active_streamers: weakref.WeakSet = weakref.WeakSet()


def _release_all() -> None:
    for s in list(_active_streamers):
        try:
            s.stop()
        except Exception:
            pass


atexit.register(_release_all)


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
        self._started = False
        _active_streamers.add(self)

    def __enter__(self) -> "RealSenseStreamer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                self._start_pipeline()
                self._started = True
                return
            except RuntimeError as exc:
                last_exc = exc
                self.stop()
                if attempt + 1 < 3:
                    print(f"[RealSense] start failed ({exc}); hardware reset…", flush=True)
                    hardware_reset_realsense(self.serial)
        raise last_exc  # type: ignore[misc]

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
            time.sleep(0.15)
        self._started = False

    def _start_pipeline(self) -> None:
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
            _wait_for_frames(self._pipeline)
        self._color_sensor = _color_sensor(device)
        if self.lock_exposure:
            self.color_exposure_locked = lock_realsense_color_exposure(
                self._color_sensor,
                exposure_us=self.exposure_us,
                lock_white_balance=self.lock_white_balance,
            )
            _wait_for_frames(self._pipeline)

    def read(self, *, with_depth: bool = True) -> AlignedFrame:
        if self._pipeline is None:
            raise RuntimeError("RealSenseStreamer not started")
        frames = _wait_for_frames(self._pipeline)
        if with_depth:
            aligned = self._align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                raise RuntimeError("Incomplete aligned frame")
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
            color_bgr = np.asanyarray(color_frame.get_data())
            color_rgb = color_bgr[:, :, ::-1].copy()
            depth_m = depth_raw.astype(np.float64) * self._depth_scale
        else:
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Incomplete color frame")
            color_bgr = np.asanyarray(color_frame.get_data())
            color_rgb = color_bgr[:, :, ::-1].copy()
            depth_m = np.zeros((self.height, self.width), dtype=np.float64)
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


class RealSenseCamera(CameraBackend):
    """``CameraBackend`` wrapper (BGR grabs) around ``RealSenseStreamer``."""

    def __init__(
        self,
        *,
        serial: str = "",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup: int = 10,
    ) -> None:
        self._serial = serial
        self._width = width
        self._height = height
        self._fps = fps
        self._warmup = warmup
        self._streamer: RealSenseStreamer | None = None
        self._info = CameraInfo(
            backend="realsense",
            name="RealSense",
            serial=serial,
            model="D415",
        )

    def open(self) -> None:
        self._streamer = RealSenseStreamer(
            width=self._width,
            height=self._height,
            fps=self._fps,
            serial=self._serial,
            warmup=self._warmup,
        )
        self._streamer.start()
        self._info = CameraInfo(
            backend="realsense",
            name=self._streamer._device_name or "RealSense",
            serial=self._streamer._device_serial or self._serial,
            model="D415",
        )

    def close(self) -> None:
        if self._streamer is not None:
            self._streamer.stop()
            self._streamer = None

    def grab_bgr(self) -> Optional[np.ndarray]:
        if self._streamer is None:
            return None
        frame = self._streamer.read(with_depth=False)
        return frame.color_rgb[:, :, ::-1].copy()

    def grab_rgb_depth(self) -> tuple[np.ndarray, np.ndarray]:
        if self._streamer is None:
            raise RuntimeError("RealSenseCamera not open")
        frame = self._streamer.read(with_depth=True)
        return frame.color_rgb, frame.depth_m

    @property
    def info(self) -> CameraInfo:
        return self._info

    @property
    def streamer(self) -> RealSenseStreamer | None:
        return self._streamer
