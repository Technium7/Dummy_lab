from __future__ import annotations

import atexit
import gc
import json
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from .base_ellipse import BaseEllipse, draw_base_ellipse

try:
    import pyrealsense2 as rs
except ImportError as exc:
    rs = None  # type: ignore
    _RS_IMPORT_ERROR = exc
else:
    _RS_IMPORT_ERROR = None


def validate_color_image_file(color_path: Path) -> None:
    """Raise ``OSError`` if ``color_path`` is missing or not a readable RGB image."""
    path = color_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Color capture not found: {path}")
    if path.stat().st_size < 1024:
        raise OSError(f"Color capture looks truncated ({path.stat().st_size} bytes): {path}")
    try:
        with Image.open(path) as im:
            im.load()
            if im.size[0] < 32 or im.size[1] < 32:
                raise OSError(f"Color capture has invalid size {im.size}: {path}")
    except OSError:
        raise
    except Exception as exc:
        raise OSError(f"Color capture is not a valid image: {path}") from exc


def _atomic_save_image(path: Path, image: Image.Image, *, format: str = "PNG") -> None:
    """Write image atomically so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        image.save(tmp, format=format)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


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


def _list_realsense_devices() -> list[str]:
    _require_realsense()
    ctx = rs.context()
    serials: list[str] = []
    for device in ctx.query_devices():
        try:
            serials.append(device.get_info(rs.camera_info.serial_number))
        except Exception:
            continue
    return serials


def hardware_reset_realsense(
    serial: str = "",
    *,
    settle_s: float = 2.5,
    rediscover_timeout_s: float = 12.0,
) -> bool:
    """USB hardware reset (same effect as unplug/replug). Returns True if device reappeared."""
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
        return bool(_list_realsense_devices())

    try:
        target.hardware_reset()
    except Exception:
        pass

    time.sleep(settle_s)
    deadline = time.monotonic() + rediscover_timeout_s
    while time.monotonic() < deadline:
        serials = _list_realsense_devices()
        if serial:
            if serial in serials:
                return True
        elif serials:
            return True
        time.sleep(0.25)
    return False


def _wait_for_frames(
    pipeline: Any,
    *,
    timeout_ms: int = 10000,
    retries: int = 3,
    context: str = "frame",
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(0.25 * (attempt + 1))
    devices = _list_realsense_devices()
    hint = f" Connected serials: {devices}." if devices else " No RealSense devices found."
    raise RuntimeError(
        f"RealSense {context} did not arrive within {timeout_ms} ms after {retries} tries.{hint} "
        "Another process may hold the camera (e.g. a stopped live_label from Ctrl+Z). "
        "Run `jobs -l`, then `kill %N` or `fg` and quit. "
        "If the camera is stuck, retry (auto hardware-reset) or unplug/replug USB. "
        "Or run with --offline to use saved debug/grab_sensor.color.png."
    ) from last_exc


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


_REALSENSE_STOP_SETTLE_S = 0.4
_active_realsense_streamers: weakref.WeakSet = weakref.WeakSet()


def _release_all_realsense_streamers() -> None:
    """Best-effort release for any streamer still registered (e.g. Ctrl+C during live_label)."""
    for streamer in list(_active_realsense_streamers):
        try:
            streamer.stop()
        except Exception:
            pass


atexit.register(_release_all_realsense_streamers)


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
        self.frame_timeout_ms = 10000
        self.frame_retries = 3
        self.start_retries = 3
        self.reset_on_start_retry = True
        self.auto_reconnect_on_read = True
        self._pipeline: Any = None
        self._align: Any = None
        self._depth_scale = 0.001
        self._device_name = ""
        self._device_serial = ""
        self._color_intrinsics: Any = None
        self._color_sensor: Any = None
        self.color_exposure_locked: dict[str, float] | None = None
        self._started = False
        self._stop_settle_s = _REALSENSE_STOP_SETTLE_S
        _active_realsense_streamers.add(self)

    def __enter__(self) -> RealSenseStreamer:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    def start(self) -> None:
        last_exc: Exception | None = None
        for attempt in range(max(1, self.start_retries)):
            try:
                self._start_pipeline()
                self._started = True
                return
            except RuntimeError as exc:
                last_exc = exc
                self.stop()
                if attempt + 1 < self.start_retries and self.reset_on_start_retry:
                    print(
                        f"[RealSense] start failed ({exc}); hardware reset and retry "
                        f"({attempt + 2}/{self.start_retries})…",
                        flush=True,
                    )
                    hardware_reset_realsense(self.serial)
        raise last_exc  # type: ignore[misc]

    def reconnect(self, *, reset_hardware: bool = True) -> None:
        """Stop, optionally USB-reset the device, and reopen the pipeline."""
        self.stop()
        if reset_hardware:
            print("[RealSense] Reconnecting (hardware reset)…", flush=True)
            hardware_reset_realsense(self.serial)
        self.start()

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
            _wait_for_frames(
                self._pipeline,
                timeout_ms=self.frame_timeout_ms,
                retries=self.frame_retries,
                context="warmup frame",
            )

        self._color_sensor = _color_sensor(device)
        if self.lock_exposure:
            self.color_exposure_locked = lock_realsense_color_exposure(
                self._color_sensor,
                exposure_us=self.exposure_us,
                lock_white_balance=self.lock_white_balance,
            )
            _wait_for_frames(
                self._pipeline,
                timeout_ms=self.frame_timeout_ms,
                retries=self.frame_retries,
                context="exposure-settle frame",
            )

    def read(self, *, with_depth: bool = True) -> AlignedFrame:
        try:
            return self._read_once(with_depth=with_depth)
        except RuntimeError as exc:
            if not self.auto_reconnect_on_read:
                raise
            print(f"[RealSense] frame lost ({exc}); reconnecting…", flush=True)
            self.reconnect(reset_hardware=True)
            return self._read_once(with_depth=with_depth)

    def _read_once(self, *, with_depth: bool = True) -> AlignedFrame:
        if self._pipeline is None:
            raise RuntimeError("RealSenseStreamer not started")
        frames = _wait_for_frames(
            self._pipeline,
            timeout_ms=self.frame_timeout_ms,
            retries=self.frame_retries,
            context="color frame" if not with_depth else "aligned frame",
        )
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

    def frames(self) -> Iterator[AlignedFrame]:
        while True:
            yield self.read()

    def stop(self) -> None:
        if not self._started and self._pipeline is None:
            _active_realsense_streamers.discard(self)
            return

        had_pipeline = self._pipeline is not None
        if self._color_sensor is not None and self.color_exposure_locked:
            restore_realsense_color_auto(self._color_sensor)
        self._color_sensor = None
        self.color_exposure_locked = None
        self._align = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._started = False
        _active_realsense_streamers.discard(self)

        if had_pipeline and self._stop_settle_s > 0.0:
            time.sleep(self._stop_settle_s)
            gc.collect()
            print("[RealSense] camera released.", flush=True)


def capture_realsense_frame(
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup: int = 15,
    serial: str = "",
    frame_timeout_ms: int = 10000,
    start_retries: int = 3,
) -> CaptureResult:
    last_exc: Exception | None = None
    for attempt in range(max(1, start_retries)):
        streamer = RealSenseStreamer(
            width=width,
            height=height,
            fps=fps,
            serial=serial,
            warmup=warmup,
        )
        streamer.frame_timeout_ms = frame_timeout_ms
        streamer.start_retries = start_retries
        streamer.reset_on_start_retry = True
        streamer.auto_reconnect_on_read = False
        try:
            with streamer:
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
        except RuntimeError as exc:
            last_exc = exc
            if attempt + 1 < start_retries:
                print(
                    f"[RealSense] capture retry {attempt + 2}/{start_retries} after failure: {exc}",
                    flush=True,
                )
                hardware_reset_realsense(serial)
    raise last_exc  # type: ignore[misc]


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
    validate_color_image_file(color_path)
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
        from .grasp_point_net import _load_depth_m

        depth_m = _load_depth_m(depth_npy_path).astype(np.float64)
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
    _atomic_save_image(out_dir / f"{stem}.color.png", Image.fromarray(capture.color_rgb))
    depth_mm = (capture.depth_m * 1000.0).astype(np.uint16)
    _atomic_save_image(out_dir / f"{stem}.depth.png", Image.fromarray(depth_mm, mode="I;16"), format="PNG")
    np.save(out_dir / f"{stem}.depth.npy", capture.depth_m.astype(np.float32))
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
    base_uv: tuple[int, int] | None = None,
    base_ellipse: BaseEllipse | None = None,
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
    if base_ellipse is not None:
        bu, bv = base_ellipse.center_uv
        depth_at_base = float(frame.depth_m[bv, bu]) if 0 <= bv < h and 0 <= bu < w else 0.0
        valid_base_depth = depth_at_base > 1e-4
        label.update(
            {
                "base_u": bu,
                "base_v": bv,
                "base_uv": [bu, bv],
                "base_ellipse": base_ellipse.as_dict(),
                "depth_m_at_base": depth_at_base if valid_base_depth else None,
                "valid_base_depth": valid_base_depth,
            }
        )
    elif base_uv is not None:
        bu, bv = int(base_uv[0]), int(base_uv[1])
        depth_at_base = float(frame.depth_m[bv, bu]) if 0 <= bv < h and 0 <= bu < w else 0.0
        valid_base_depth = depth_at_base > 1e-4
        label.update(
            {
                "base_u": bu,
                "base_v": bv,
                "base_uv": [bu, bv],
                "depth_m_at_base": depth_at_base if valid_base_depth else None,
                "valid_base_depth": valid_base_depth,
            }
        )
    label_path.write_text(json.dumps(label, indent=2), encoding="utf-8")

    manifest = out_dir / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(label, separators=(",", ":")) + "\n")

    _save_preview(
        preview_path,
        frame.color_rgb,
        grasp_uv,
        sample_id,
        base_uv=base_ellipse.center_uv if base_ellipse is not None else base_uv,
        base_ellipse=base_ellipse,
    )
    return label_path


def update_sample_base_label(
    label_path: Path,
    base_uv: tuple[int, int] | None = None,
    *,
    base_ellipse: BaseEllipse | None = None,
    data_dir: Path | None = None,
) -> Path:
    """Add or update sensor-base ellipse on an existing grasp label JSON."""
    label_path = label_path.resolve()
    data = json.loads(label_path.read_text(encoding="utf-8"))
    out_dir = (data_dir or label_path.parent.parent).resolve()

    if base_ellipse is not None:
        bu, bv = base_ellipse.center_uv
    elif base_uv is not None:
        bu, bv = int(base_uv[0]), int(base_uv[1])
        base_ellipse = BaseEllipse.from_center_uv((bu, bv))
    else:
        raise ValueError("update_sample_base_label requires base_ellipse or base_uv")
    h = int(data.get("height", 480))
    w = int(data.get("width", 640))

    depth_at_base: float | None = None
    valid_base_depth = False
    depth_rel = data.get("files", {}).get("depth")
    if depth_rel:
        depth_path = out_dir / str(depth_rel)
        if depth_path.is_file():
            depth_m = np.load(depth_path)
            if np.issubdtype(depth_m.dtype, np.integer):
                depth_m = depth_m.astype(np.float32) * 0.001
            else:
                depth_m = depth_m.astype(np.float32)
            if 0 <= bv < depth_m.shape[0] and 0 <= bu < depth_m.shape[1]:
                depth_at_base = float(depth_m[bv, bu])
                valid_base_depth = depth_at_base > 1e-4

    data["base_u"] = bu
    data["base_v"] = bv
    data["base_uv"] = [bu, bv]
    data["base_ellipse"] = base_ellipse.as_dict()
    data["depth_m_at_base"] = depth_at_base if valid_base_depth else None
    data["valid_base_depth"] = valid_base_depth
    label_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    image_rel = data.get("files", {}).get("image")
    if image_rel:
        image_path = out_dir / str(image_rel)
        if image_path.is_file():
            color_rgb = np.asarray(Image.open(image_path).convert("RGB"))
            grasp_uv = (
                int(data.get("grasp_u", data.get("grasp_uv", [0, 0])[0])),
                int(data.get("grasp_v", data.get("grasp_uv", [0, 0])[1])),
            )
            preview_path = out_dir / "previews" / f"{data.get('id', label_path.stem)}.png"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            _save_preview(
                preview_path,
                color_rgb,
                grasp_uv,
                str(data.get("id", label_path.stem)),
                base_ellipse=base_ellipse,
            )

    return label_path


def _save_preview(
    path: Path,
    color_rgb: np.ndarray,
    grasp_uv: tuple[int, int],
    sample_id: str,
    *,
    base_uv: tuple[int, int] | None = None,
    base_ellipse: BaseEllipse | None = None,
) -> None:
    try:
        import cv2
    except ImportError:
        Image.fromarray(color_rgb).save(path)
        return

    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    u, v = grasp_uv
    cv2.drawMarker(bgr, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(bgr, (u, v), 10, (0, 0, 255), 2)
    cv2.putText(bgr, "grasp", (u + 12, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if base_ellipse is not None:
        draw_base_ellipse(bgr, base_ellipse)
    elif base_uv is not None:
        bu, bv = base_uv
        cv2.drawMarker(bgr, (bu, bv), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 24, 2)
        cv2.circle(bgr, (bu, bv), 10, (0, 255, 0), 2)
        cv2.putText(bgr, "base", (bu + 12, bv + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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
