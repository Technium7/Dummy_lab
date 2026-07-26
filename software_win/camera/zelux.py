"""Thorlabs Zelux (TLCamera / Scientific Camera) backend for Windows.

Preferred stack:
  1. Install ThorCam from Thorlabs (provides native DLLs).
  2. ``pip install pylablib`` (wraps thorlabs_tsi_camera_sdk.dll).

Optional official SDK:
  Install ``thorlabs_tsi_sdk`` from the ThorCam Python Toolkit zip, then set
  ``THORLABS_TSI_DLL_PATH`` or place DLLs on PATH.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from camera.base import CameraBackend, CameraInfo
from camera.controls import ControlLimits
from camera.display import to_bgr_u8

_DEFAULT_DLL_CANDIDATES = (
    Path(r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam"),
    Path(r"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support"),
    Path(os.environ.get("THORLABS_TSI_DLL_PATH", "")),
)


def _configure_pylablib_dlls() -> None:
    env = os.environ.get("THORLABS_TSI_DLL_PATH", "").strip()
    if env:
        try:
            import pylablib as pll  # type: ignore

            pll.par["devices/dlls/thorlabs_tlcam"] = env
        except Exception:
            pass
        return
    for candidate in _DEFAULT_DLL_CANDIDATES:
        if candidate and candidate.exists():
            try:
                import pylablib as pll  # type: ignore

                pll.par["devices/dlls/thorlabs_tlcam"] = str(candidate)
                return
            except Exception:
                continue


def list_zelux_devices() -> List[CameraInfo]:
    """List Thorlabs TLCamera devices visible to pylablib."""
    _configure_pylablib_dlls()
    try:
        from pylablib.devices import Thorlabs  # type: ignore
    except ImportError:
        return []
    infos: List[CameraInfo] = []
    try:
        serials = list(Thorlabs.list_cameras_tlcam())
    except Exception:
        return []
    for i, serial in enumerate(serials):
        infos.append(
            CameraInfo(
                backend="zelux",
                name=f"Thorlabs TLCamera {serial}",
                index=i,
                serial=str(serial),
                model="Zelux/TLCamera",
            )
        )
    return infos


class ZeluxCamera(CameraBackend):
    """Live grabber for a Thorlabs Zelux / TLCamera device via pylablib."""

    def __init__(
        self,
        serial: str | None = None,
        *,
        exposure_s: float = 0.01,
        gain: float = 0.0,
        index: int = 0,
    ) -> None:
        self._serial = (serial or "").strip() or None
        self._exposure_s = float(exposure_s)
        self._gain = float(gain)
        self._index = int(index)
        self._cam = None
        self._limits = ControlLimits(
            exposure_ms_min=0.1,
            exposure_ms_max=1000.0,
            gain_min=0.0,
            gain_max=48.0,
        )
        self._info = CameraInfo(
            backend="zelux",
            name="Thorlabs Zelux",
            index=self._index,
            serial=self._serial or "",
            model="Zelux",
        )

    @property
    def info(self) -> CameraInfo:
        return self._info

    def get_control_limits(self) -> ControlLimits:
        return self._limits

    def get_exposure_ms(self) -> float:
        if self._cam is not None:
            try:
                return float(self._cam.get_exposure()) * 1000.0
            except Exception:
                pass
        return self._exposure_s * 1000.0

    def set_exposure_ms(self, exposure_ms: float) -> None:
        exposure_ms = max(self._limits.exposure_ms_min, float(exposure_ms))
        self._exposure_s = exposure_ms / 1000.0
        if self._cam is None:
            return
        self._cam.set_exposure(self._exposure_s)

    def get_gain(self) -> float:
        if self._cam is not None:
            try:
                return float(self._cam.get_gain())
            except Exception:
                pass
        return self._gain

    def set_gain(self, gain: float) -> None:
        lo, hi = self._limits.gain_min, self._limits.gain_max
        self._gain = max(lo, min(hi, float(gain)))
        if self._cam is None:
            return
        self._cam.set_gain(self._gain, truncate=True)

    def disable_auto_exposure_gain(self) -> None:
        # TLCamera via pylablib is manual once exposure/gain are set.
        return None

    def open(self) -> None:
        _configure_pylablib_dlls()
        try:
            from pylablib.devices import Thorlabs  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pylablib is required for Zelux. Install ThorCam, then:\n"
                "  pip install pylablib"
            ) from exc

        serial = self._serial
        if serial is None:
            serials = list(Thorlabs.list_cameras_tlcam())
            if not serials:
                raise RuntimeError(
                    "No Thorlabs TLCamera found. Is the Zelux plugged in and "
                    "ThorCam installed?"
                )
            if self._index < 0 or self._index >= len(serials):
                raise RuntimeError(f"Zelux index {self._index} out of range ({len(serials)} cameras)")
            serial = str(serials[self._index])

        self._cam = Thorlabs.ThorlabsTLCamera(serial=serial)
        try:
            g_lo, g_hi = self._cam.get_gain_range()
            self._limits = ControlLimits(
                exposure_ms_min=self._limits.exposure_ms_min,
                exposure_ms_max=self._limits.exposure_ms_max,
                gain_min=float(g_lo),
                gain_max=float(g_hi),
            )
        except Exception:
            pass
        try:
            self._cam.set_exposure(self._exposure_s)
            self._cam.set_gain(self._gain, truncate=True)
        except Exception:
            pass
        # Color Zelux (if present) -> RGB; mono stays 2D.
        try:
            if hasattr(self._cam, "set_color_format"):
                self._cam.set_color_format("rgb")
        except Exception:
            pass
        # pylablib TLCamera: setup_acquisition(nframes=...) only (no mode=).
        self._cam.setup_acquisition(nframes=50)
        self._cam.start_acquisition()
        self._info = CameraInfo(
            backend="zelux",
            name=f"Thorlabs Zelux {serial}",
            index=self._index,
            serial=str(serial),
            model="Zelux",
        )

    def grab_bgr(self) -> Optional[np.ndarray]:
        if self._cam is None:
            return None
        try:
            ok = self._cam.wait_for_frame(timeout=0.5)
            if ok is False:
                return None
            frame = None
            if hasattr(self._cam, "read_newest_image"):
                frame = self._cam.read_newest_image()
            if frame is None:
                frame = self._cam.read_oldest_image()
            if frame is None:
                return None
            rgb = frame.ndim == 3
            return to_bgr_u8(frame, rgb=rgb)
        except Exception:
            return None

    def close(self) -> None:
        cam = self._cam
        self._cam = None
        if cam is None:
            return
        try:
            cam.stop_acquisition()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass


class DemoZeluxCamera(CameraBackend):
    """Synthetic Zelux feed for UI development without hardware."""

    def __init__(self, *, width: int = 640, height: int = 480) -> None:
        self._w = width
        self._h = height
        self._t0 = time.time()
        self._exposure_ms = 10.0
        self._gain = 0.0
        self._info = CameraInfo(backend="zelux-demo", name="Zelux demo", model="demo")

    @property
    def info(self) -> CameraInfo:
        return self._info

    def get_control_limits(self) -> ControlLimits:
        return ControlLimits(exposure_ms_min=0.1, exposure_ms_max=500.0, gain_min=0.0, gain_max=48.0)

    def get_exposure_ms(self) -> float:
        return self._exposure_ms

    def set_exposure_ms(self, exposure_ms: float) -> None:
        self._exposure_ms = float(exposure_ms)

    def get_gain(self) -> float:
        return self._gain

    def set_gain(self, gain: float) -> None:
        self._gain = float(gain)

    def open(self) -> None:
        self._t0 = time.time()

    def grab_bgr(self) -> Optional[np.ndarray]:
        t = time.time() - self._t0
        yy, xx = np.mgrid[0 : self._h, 0 : self._w]
        # Fake brightness from exposure/gain so sliders are visible in --demo.
        amp = min(1.0, (self._exposure_ms / 20.0) * (1.0 + self._gain / 24.0))
        phase = (xx + yy) * 0.05 + t * 4.0
        gray = ((np.sin(phase) * 0.5 + 0.5) * 220 * amp).astype(np.uint8)
        frame = np.stack([gray, np.roll(gray, 20, axis=1), np.roll(gray, 40, axis=0)], axis=-1)
        return frame

    def close(self) -> None:
        return None
