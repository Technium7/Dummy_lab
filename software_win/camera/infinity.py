"""Lumenera / Teledyne Infinity 2-1RM backend for Windows.

Uses the LuCam API via ``lucam`` (Infinity Capture drivers). OpenCV DirectShow
is only used when explicitly requested — never as a silent fallback, because
index 0 is usually a different camera.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from camera.base import CameraBackend, CameraInfo
from camera.controls import ControlLimits
from camera.display import to_bgr_u8

# Infinity 2-1C raw8 Bayer; BGGR is the common Lumenera layout.
_DEFAULT_BAYER = "BG"


def list_infinity_devices() -> List[CameraInfo]:
    """List Infinity cameras via lucam (preferred), else OpenCV indices."""
    infos = _list_lucam()
    if infos:
        return infos
    return _list_opencv_guesses()


def _list_lucam() -> List[CameraInfo]:
    try:
        from lucam import LucamEnumCameras  # type: ignore
    except ImportError:
        return []
    out: List[CameraInfo] = []
    try:
        versions = list(LucamEnumCameras())
    except Exception:
        return []
    for i, version in enumerate(versions):
        serial = str(
            getattr(version, "serialnumber", None)
            or getattr(version, "serial", None)
            or getattr(version, "serialNumber", None)
            or ""
        )
        model = str(getattr(version, "camera", None) or getattr(version, "model", None) or "Infinity")
        out.append(
            CameraInfo(
                backend="infinity-lucam",
                name=f"Infinity #{i + 1}",
                index=i + 1,
                serial=serial,
                model=model,
            )
        )
    return out


def _list_opencv_guesses() -> List[CameraInfo]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return []
    out: List[CameraInfo] = []
    for index in range(0, 8):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        out.append(
            CameraInfo(
                backend="infinity-opencv",
                name=f"OpenCV camera #{index}",
                index=index,
                model="DirectShow",
            )
        )
        cap.release()
    return out


class InfinityCamera(CameraBackend):
    """Live grabber for Infinity 2-1RM via lucam FastFrames."""

    def __init__(
        self,
        *,
        camera_number: int = 1,
        opencv_index: int | None = None,
        exposure_ms: float = 25.0,
        gain: float = 1.0,
        prefer_opencv: bool = False,
        bayer: str = _DEFAULT_BAYER,
    ) -> None:
        self._camera_number = int(camera_number)
        self._opencv_index = opencv_index
        self._exposure_ms = float(exposure_ms)
        self._gain = float(gain)
        self._prefer_opencv = bool(prefer_opencv)
        self._bayer = (bayer or _DEFAULT_BAYER).upper()
        self._lucam = None
        self._cap = None
        self._last_error = ""
        self._limits = ControlLimits(
            exposure_ms_min=0.1,
            exposure_ms_max=500.0,
            gain_min=1.0,
            gain_max=20.0,
        )
        self._info = CameraInfo(
            backend="infinity",
            name="Infinity 2-1RM",
            index=self._camera_number,
            model="Infinity 2-1RM",
        )

    @property
    def info(self) -> CameraInfo:
        return self._info

    def get_control_limits(self) -> ControlLimits:
        return self._limits

    def get_exposure_ms(self) -> float:
        if self._lucam is not None:
            try:
                value, _flags = self._lucam.GetProperty("exposure")
                return float(value)
            except Exception:
                pass
        return self._exposure_ms

    def set_exposure_ms(self, exposure_ms: float) -> None:
        exposure_ms = max(self._limits.exposure_ms_min, min(self._limits.exposure_ms_max, float(exposure_ms)))
        self._exposure_ms = exposure_ms
        if self._lucam is not None:
            self._write_lucam_exposure_gain(rearm=True)
            return
        if self._cap is not None:
            self._set_opencv_exposure(exposure_ms)

    def get_gain(self) -> float:
        if self._lucam is not None:
            try:
                value, _flags = self._lucam.GetProperty("gain")
                return float(value)
            except Exception:
                pass
        if self._cap is not None:
            try:
                g = float(self._cap.get(self._cv2().CAP_PROP_GAIN))
                if g > 0:
                    return g
            except Exception:
                pass
        return self._gain

    def set_gain(self, gain: float) -> None:
        gain = max(self._limits.gain_min, min(self._limits.gain_max, float(gain)))
        self._gain = gain
        if self._lucam is not None:
            self._write_lucam_exposure_gain(rearm=True)
            return
        if self._cap is not None:
            try:
                self._cap.set(self._cv2().CAP_PROP_GAIN, float(gain))
            except Exception:
                pass

    def _write_lucam_exposure_gain(self, *, rearm: bool) -> None:
        cam = self._lucam
        if cam is None:
            return
        try:
            cam.SetProperty("exposure", float(self._exposure_ms), flags=0)
        except Exception as exc:
            self._last_error = f"set exposure: {exc}"
        try:
            cam.SetProperty("still_exposure", float(self._exposure_ms), flags=0)
        except Exception:
            pass
        try:
            cam.SetProperty("gain", float(self._gain), flags=0)
        except Exception as exc:
            self._last_error = f"set gain: {exc}"
        try:
            cam.SetProperty("still_gain", float(self._gain), flags=0)
        except Exception:
            pass
        if rearm:
            self._rearm_fastframes()

    def _stop_fastframes(self) -> None:
        cam = self._lucam
        if cam is None:
            return
        try:
            if getattr(cam, "_fastframe", None) is not None:
                cam.DisableFastFrames()
        except Exception:
            pass
        try:
            cam._fastframe = None
        except Exception:
            pass

    def _start_fastframes(self) -> None:
        cam = self._lucam
        if cam is None:
            return
        self._stop_fastframes()
        snap = cam.default_snapshot()
        snap.exposure = float(self._exposure_ms)
        snap.gain = float(self._gain)
        cam.EnableFastFrames(snap)

    def _rearm_fastframes(self) -> None:
        """Re-enable FastFrames so snapshot exposure/gain take effect.

        Lucam's EnableFastFrames() copies exposure/gain into the snapshot
        struct once; SetProperty alone does not update an active FastFrames
        session.
        """
        if self._lucam is None:
            return
        # Only rearm when already capturing; open() starts FastFrames itself.
        if getattr(self._lucam, "_fastframe", None) is None:
            return
        try:
            self._start_fastframes()
        except Exception as exc:
            self._last_error = f"rearm fastframes: {exc}"

    def disable_auto_exposure_gain(self) -> None:
        if self._lucam is not None:
            try:
                if hasattr(self._lucam, "ContinuousAutoExposureDisable"):
                    self._lucam.ContinuousAutoExposureDisable()
            except Exception:
                pass
            # Write props only — do not EnableFastFrames here (avoids Busy on open).
            self._write_lucam_exposure_gain(rearm=False)
            return
        if self._cap is not None:
            cv2 = self._cv2()
            for auto_val in (0.25, 0.0, 1.0):
                try:
                    self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_val)
                except Exception:
                    pass
            self._set_opencv_exposure(self._exposure_ms)
            try:
                self._cap.set(cv2.CAP_PROP_GAIN, float(self._gain))
            except Exception:
                pass

    def _cv2(self):
        import cv2  # type: ignore

        return cv2

    def _set_opencv_exposure(self, exposure_ms: float) -> None:
        if self._cap is None:
            return
        cv2 = self._cv2()
        try:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_ms))
        except Exception:
            pass
        try:
            import math

            seconds = max(exposure_ms, 0.1) / 1000.0
            self._cap.set(cv2.CAP_PROP_EXPOSURE, math.log2(seconds))
        except Exception:
            pass

    def open(self) -> None:
        if self._prefer_opencv or self._opencv_index is not None:
            self._open_opencv(self._opencv_index if self._opencv_index is not None else 0)
            return

        if self._try_open_lucam():
            return

        # Do NOT silently open OpenCV #0 — that is almost never the Infinity.
        detail = self._last_error or "unknown Lucam error"
        raise RuntimeError(
            "Could not open Infinity via Lucam.\n"
            f"  Detail: {detail}\n"
            "  Check: Infinity Capture drivers installed, camera USB connected,\n"
            "  Infinity Capture app closed (exclusive USB), `pip install lucam`.\n"
            "  Or force DirectShow with --infinity-opencv-index N (usually wrong)."
        )

    def _try_open_lucam(self) -> bool:
        try:
            from lucam import Lucam, LucamNumCameras  # type: ignore
        except ImportError as exc:
            self._last_error = f"lucam not installed ({exc})"
            return False

        try:
            n = int(LucamNumCameras())
        except Exception as exc:
            self._last_error = f"LucamNumCameras failed: {exc}"
            return False
        if n < 1:
            self._last_error = "LucamNumCameras() == 0 (no Infinity / Lumenera camera)"
            return False
        if self._camera_number < 1 or self._camera_number > n:
            self._last_error = f"camera_number={self._camera_number} out of range 1..{n}"
            return False

        cam = None
        try:
            cam = Lucam(self._camera_number)
        except Exception as exc:
            self._last_error = f"Lucam({self._camera_number}) failed: {exc}"
            # Half-constructed Lucam objects can raise noisy __del__; drop ref.
            cam = None
            return False

        self._lucam = cam
        try:
            # Clear any leftover FastFrames/stream state from a previous crash.
            self._stop_fastframes()
            try:
                if hasattr(cam, "StreamVideoControl"):
                    cam.StreamVideoControl("stop_streaming")
            except Exception:
                pass
            self.disable_auto_exposure_gain()
            self._start_fastframes()
        except Exception as exc:
            self._last_error = f"Lucam setup failed: {exc}"
            self._close_lucam_safe()
            return False

        serial = ""
        model = "Infinity 2-1RM"
        try:
            version = cam.QueryVersion()
            serial = str(getattr(version, "serialnumber", "") or "")
        except Exception:
            pass
        try:
            # Model string from __str__ / camera id path is optional.
            camid = cam.GetCameraId()
            model = f"Infinity (id={hex(int(camid))})"
        except Exception:
            pass

        self._info = CameraInfo(
            backend="infinity-lucam",
            name=f"Infinity #{self._camera_number}",
            index=self._camera_number,
            serial=serial,
            model=model,
        )
        self._last_error = ""
        return True

    def _open_opencv(self, index: int) -> None:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(index))
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open camera index {index}. "
                "Prefer Lucam for Infinity 2-1RM (omit --infinity-opencv-index)."
            )
        self._cap = cap
        self._limits = ControlLimits(
            exposure_ms_min=0.1,
            exposure_ms_max=500.0,
            gain_min=0.0,
            gain_max=64.0,
        )
        self.disable_auto_exposure_gain()
        self._info = CameraInfo(
            backend="infinity-opencv",
            name=f"OpenCV camera #{index}",
            index=int(index),
            model="DirectShow (not Lucam)",
        )

    def grab_bgr(self) -> Optional[np.ndarray]:
        if self._lucam is not None:
            return self._grab_lucam()
        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
            return to_bgr_u8(frame, rgb=False)
        return None

    def _grab_lucam(self) -> Optional[np.ndarray]:
        cam = self._lucam
        assert cam is not None
        try:
            if getattr(cam, "_fastframe", None) is not None:
                frame = cam.TakeFastFrame()
            elif getattr(cam, "_streaming", None) is not None:
                frame = cam.TakeVideo(1)
                if isinstance(frame, np.ndarray) and frame.ndim == 4:
                    frame = frame[0]
            else:
                frame = cam.TakeSnapshot()
            if frame is None:
                return None
            if frame.ndim == 2:
                return _bayer_to_bgr(frame, self._bayer)
            return to_bgr_u8(frame, rgb=True)
        except Exception as exc:
            self._last_error = f"grab: {exc}"
            return None

    def close(self) -> None:
        self._close_lucam_safe()
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _close_lucam_safe(self) -> None:
        lucam = self._lucam
        self._lucam = None
        if lucam is None:
            return
        # Satisfy Lucam.__del__ assertions: no display window / streaming / fastframes.
        try:
            if getattr(lucam, "_displaying_window", False) and hasattr(lucam, "DestroyDisplayWindow"):
                lucam.DestroyDisplayWindow()
        except Exception:
            pass
        try:
            if getattr(lucam, "_streaming", None) is not None:
                lucam.StreamVideoControl("stop_streaming")
        except Exception:
            pass
        try:
            if getattr(lucam, "_fastframe", None) is not None:
                lucam.DisableFastFrames()
        except Exception:
            pass
        # Clear flags even if API calls failed, so __del__ does not assert.
        try:
            lucam._displaying_window = False
            lucam._streaming = None
            lucam._fastframe = None
        except Exception:
            pass
        try:
            lucam.CameraClose()
        except Exception:
            pass
        try:
            lucam._handle = 0
        except Exception:
            pass


def _bayer_to_bgr(frame: np.ndarray, bayer: str) -> np.ndarray:
    """Demosaic Infinity raw8 Bayer to BGR uint8."""
    u8 = frame
    if u8.dtype != np.uint8:
        max_v = float(np.max(u8)) if u8.size else 1.0
        if max_v <= 0:
            max_v = 1.0
        u8 = np.clip(u8.astype(np.float32) * (255.0 / max_v), 0, 255).astype(np.uint8)
    try:
        import cv2  # type: ignore

        code_map = {
            "BG": cv2.COLOR_BayerBG2BGR,
            "GB": cv2.COLOR_BayerGB2BGR,
            "RG": cv2.COLOR_BayerRG2BGR,
            "GR": cv2.COLOR_BayerGR2BGR,
        }
        code = code_map.get(bayer.upper()[:2], cv2.COLOR_BayerBG2BGR)
        return cv2.cvtColor(u8, code)
    except Exception:
        return to_bgr_u8(u8, rgb=False)


class DemoInfinityCamera(CameraBackend):
    """Synthetic Infinity feed for UI development without hardware."""

    def __init__(self, *, width: int = 640, height: int = 480) -> None:
        self._w = width
        self._h = height
        self._t0 = time.time()
        self._exposure_ms = 25.0
        self._gain = 1.0
        self._info = CameraInfo(backend="infinity-demo", name="Infinity demo", model="demo")

    @property
    def info(self) -> CameraInfo:
        return self._info

    def get_control_limits(self) -> ControlLimits:
        return ControlLimits(exposure_ms_min=0.1, exposure_ms_max=500.0, gain_min=1.0, gain_max=20.0)

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
        amp = min(1.0, (self._exposure_ms / 20.0) * (self._gain / 4.0))
        r = ((xx / max(self._w - 1, 1)) * 255 * amp).astype(np.uint8)
        g = ((yy / max(self._h - 1, 1)) * 255 * amp).astype(np.uint8)
        b = ((np.sin(t * 2.0 + xx * 0.02) * 0.5 + 0.5) * 255 * amp).astype(np.uint8)
        return np.stack([b, g, r], axis=-1)

    def close(self) -> None:
        return None
