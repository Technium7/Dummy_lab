"""Record epoxy teleop demos with Zelux + Infinity RGB and robot state.

Same control flow as insert_sensor:
  R = start/stop saving (live camera windows stay open under --record-demos)
  C = save pending episode
  O/P = gripper open/close marker
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from camera.base import CameraBackend
from camera.display import label_frame, require_cv2, resize_max
from camera.infinity import InfinityCamera
from camera.zelux import ZeluxCamera
from manual_control.mapping import sim_vel_rad_to_transformed_deg_s
from manual_control.teleop_tick import TeleopTickResult

from .config_loader import EpoxyConfig, InfinityCamConfig, ZeluxCamConfig

_PREVIEW_TITLES = {
    "zelux": "epoxy Zelux side",
    "zelux_front": "epoxy Zelux front",
    "infinity": "epoxy Infinity",
}
_PREVIEW_MAX_WIDTH = 960


class _RecordPhase(Enum):
    IDLE = auto()
    RECORDING = auto()
    PENDING = auto()


@dataclass
class _RecordSnapshot:
    joint_pos_deg: list[float]
    commanded_velocity: list[float]
    twist: list[float]
    home_latched: bool
    gripper: float


@dataclass
class _GrabbedFrame:
    role: str
    timestamp: float
    color_rgb: np.ndarray
    device_label: str = ""


def _key_triggered(events: dict[int, int], char: str) -> bool:
    import pybullet as p

    lower = ord(char.lower())
    upper = ord(char.upper())
    return ((events.get(lower, 0) & p.KEY_WAS_TRIGGERED) != 0) or (
        (events.get(upper, 0) & p.KEY_WAS_TRIGGERED) != 0
    )


def color_filename(stem: str, role: str) -> str:
    return f"{stem}_{role}.jpg"


def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame_bgr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR frame, got shape {arr.shape}")
    return arr[:, :, ::-1].copy()


def _open_zelux(cfg: ZeluxCamConfig) -> CameraBackend:
    cam = ZeluxCamera(
        serial=cfg.serial or None,
        exposure_s=max(cfg.exposure_ms, 0.1) / 1000.0,
        gain=cfg.gain,
        index=cfg.index,
    )
    cam.open()
    return cam


def _open_infinity(cfg: InfinityCamConfig) -> CameraBackend:
    cam = InfinityCamera(
        camera_number=cfg.camera_number,
        opencv_index=cfg.opencv_index,
        exposure_ms=cfg.exposure_ms,
        gain=cfg.gain,
        prefer_opencv=cfg.prefer_opencv or cfg.opencv_index is not None,
        bayer=cfg.bayer,
    )
    cam.open()
    return cam


def _save_frame_job(job: dict[str, Any]) -> None:
    staging_dir = Path(str(job["staging_dir"]))
    stem = str(job["stem"])
    for cam in job["cameras"]:
        role = str(cam["role"])
        Image.fromarray(cam["color_rgb"]).save(
            staging_dir / color_filename(stem, role),
            quality=90,
        )
    (staging_dir / f"{stem}.frame.json").write_text(
        json.dumps(job["payload"], indent=2),
        encoding="utf-8",
    )


def _disk_writer_main(job_queue: queue.Queue, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            job = job_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if job is None:
            break
        try:
            _save_frame_job(job)
        except Exception as exc:
            print(f"[epoxy_app] disk writer error: {exc}", flush=True)


@dataclass
class EpisodeRecorder:
    cfg: EpoxyConfig
    dataset_dir: Path
    own_cameras: bool = True
    task_name: str = "epoxy_application"
    episode_prefix: str = "episode"
    _staging_dir: Path | None = field(default=None, init=False)
    _frame_idx: int = field(default=0, init=False)
    _phase: _RecordPhase = field(default=_RecordPhase.IDLE, init=False)
    _gripper_state: float = field(default=0.0, init=False)
    _snapshot: list[_RecordSnapshot | None] = field(default_factory=lambda: [None], init=False)
    _recording_active: threading.Event = field(default_factory=threading.Event, init=False)
    _cmd_queue: queue.Queue[str] = field(default_factory=queue.Queue, init=False)
    _service_stop: threading.Event = field(default_factory=threading.Event, init=False)
    _service_thread: threading.Thread | None = field(default=None, init=False)
    _phase_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _next_snapshot_s: float = field(default=0.0, init=False)
    _disk_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=8), init=False)
    _next_save_s: float = field(default=0.0, init=False)
    _ingest_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _camera_labels: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.dataset_dir = self.dataset_dir.resolve()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._service_thread = threading.Thread(
            target=self._service_main,
            name="epoxy-record-service",
            daemon=True,
        )
        self._service_thread.start()

    @property
    def is_recording(self) -> bool:
        with self._phase_lock:
            return self._phase is _RecordPhase.RECORDING

    @property
    def has_pending(self) -> bool:
        with self._phase_lock:
            return self._phase is _RecordPhase.PENDING

    def _set_phase(self, phase: _RecordPhase) -> None:
        with self._phase_lock:
            self._phase = phase
        if phase is _RecordPhase.RECORDING:
            self._recording_active.set()
        else:
            self._recording_active.clear()

    def _staging_root(self) -> Path:
        return self.dataset_dir / "_staging"

    def _discard_staging(self) -> None:
        staging = self._staging_root()
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)
        self._staging_dir = None
        self._frame_idx = 0
        self._set_phase(_RecordPhase.IDLE)

    def _begin_staging(self) -> None:
        self._discard_staging()
        self._staging_dir = self._staging_root() / "current"
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._frame_idx = 0
        roles = self.cfg.camera_roles()
        cameras_meta: list[dict[str, Any]] = [
            {
                "role": "zelux",
                "serial": self.cfg.zelux.serial,
                "index": self.cfg.zelux.index,
            },
            {
                "role": "zelux_front",
                "serial": self.cfg.zelux_front.serial,
                "index": self.cfg.zelux_front.index,
            },
            {
                "role": "infinity",
                "camera_number": self.cfg.infinity.camera_number,
                "opencv_index": self.cfg.infinity.opencv_index,
            },
        ]
        if self._camera_labels:
            cameras_meta = [
                {"role": role, "label": self._camera_labels.get(role, "")}
                for role in roles
            ]
        meta = {
            "created": time.time(),
            "record_hz": self.cfg.record_hz,
            "task": self.task_name,
            "num_cameras": len(cameras_meta),
            "cameras": cameras_meta,
        }
        (self._staging_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._next_save_s = 0.0

    def _next_episode_dir(self) -> Path:
        prefix = self.episode_prefix.strip() or "episode"
        existing = sorted(p for p in self.dataset_dir.glob(f"{prefix}_*") if p.is_dir())
        if not existing:
            return self.dataset_dir / f"{prefix}_000001"
        last = int(existing[-1].name.split("_")[-1])
        return self.dataset_dir / f"{prefix}_{last + 1:06d}"

    def _enqueue_cmd(self, cmd: str) -> None:
        try:
            self._cmd_queue.put_nowait(cmd)
        except queue.Full:
            pass

    def _toggle_recording(self) -> None:
        with self._phase_lock:
            phase = self._phase
        if phase is _RecordPhase.RECORDING:
            self._set_phase(_RecordPhase.PENDING)
            print(
                f"[epoxy_app] Recording stopped ({self._frame_idx} frames). "
                "C = save  |  R = discard and record again (live view stays open)",
                flush=True,
            )
            return

        if phase is _RecordPhase.PENDING:
            print("[epoxy_app] Discarding pending recording.", flush=True)
            self._discard_staging()

        self._begin_staging()
        self._set_phase(_RecordPhase.RECORDING)
        print(
            "[epoxy_app] Recording...  (saving frames; live view + teleop unchanged)",
            flush=True,
        )

    def _commit(self) -> None:
        with self._phase_lock:
            phase = self._phase
            staging_dir = self._staging_dir
            frame_idx = self._frame_idx
        if phase is not _RecordPhase.PENDING or staging_dir is None:
            print("[epoxy_app] Nothing to save (stop recording with R first).", flush=True)
            return
        if frame_idx <= 0:
            print("[epoxy_app] Empty recording — discarded.", flush=True)
            self._discard_staging()
            return

        dest = self._next_episode_dir()
        if dest.exists():
            raise RuntimeError(f"Episode path already exists: {dest}")
        staging_dir.rename(dest)
        self._staging_root().mkdir(parents=True, exist_ok=True)
        print(f"[epoxy_app] Saved {frame_idx} frames -> {dest}", flush=True)
        self._staging_dir = None
        self._frame_idx = 0
        self._set_phase(_RecordPhase.IDLE)

    def shutdown(self) -> None:
        self._service_stop.set()
        self._enqueue_cmd("shutdown")
        if self._service_thread is not None:
            self._service_thread.join(timeout=12.0)
        with self._phase_lock:
            if self._phase is _RecordPhase.RECORDING:
                self._set_phase(_RecordPhase.PENDING)
            if self._phase in (_RecordPhase.PENDING, _RecordPhase.RECORDING):
                if self._frame_idx > 0:
                    print("[epoxy_app] Discarding unsaved recording.", flush=True)
                self._discard_staging()

    def request_toggle(self) -> None:
        self._enqueue_cmd("toggle")

    def request_commit(self) -> None:
        self._enqueue_cmd("commit")

    def set_camera_labels(self, labels: dict[str, str]) -> None:
        self._camera_labels = {str(k): str(v) for k, v in labels.items()}

    def note_teleop_tick(self, tick: TeleopTickResult, *, now: float) -> None:
        """Publish robot snapshot while recording (call from teleop after_tick)."""
        if not self._recording_active.is_set():
            return
        snap_period = 1.0 / max(self.cfg.record_hz, 1.0)
        if now >= self._next_snapshot_s:
            self._snapshot[0] = self._build_snapshot(tick)
            self._next_snapshot_s = now + snap_period

    def ingest_bgr_frames(
        self,
        frames_bgr: dict[str, np.ndarray],
        *,
        device_labels: dict[str, str] | None = None,
    ) -> None:
        """Save a synchronized frame pair when recording (external camera owners)."""
        if self.own_cameras or not self._recording_active.is_set():
            return
        if "zelux" not in frames_bgr:
            return
        now = time.perf_counter()
        period = 1.0 / max(self.cfg.record_hz, 1.0)
        with self._ingest_lock:
            if now < self._next_save_s:
                return
            self._next_save_s = now + period
        snap = self._snapshot[0]
        if snap is None:
            return
        labels = device_labels or self._camera_labels
        t_wall = time.time()
        frames_by_role = {
            role: _GrabbedFrame(
                role=role,
                timestamp=t_wall,
                color_rgb=_bgr_to_rgb(np.asarray(bgr)),
                device_label=str(labels.get(role, "")),
            )
            for role, bgr in frames_bgr.items()
        }
        job = self._make_frame_job(snap, frames_by_role)
        if job is None:
            return
        try:
            self._disk_queue.put_nowait(job)
            self._frame_idx += 1
        except queue.Full:
            pass

    def _build_snapshot(self, tick: TeleopTickResult) -> _RecordSnapshot:
        vel = [float(v) for v in tick.commanded_velocity]
        while len(vel) < 6:
            vel.append(0.0)
        return _RecordSnapshot(
            joint_pos_deg=[float(v) for v in tick.robot_target_deg[:6]],
            commanded_velocity=vel,
            twist=[float(v) for v in tick.twist],
            home_latched=bool(tick.home_latched),
            gripper=self._gripper_state,
        )

    def _make_frame_job(
        self,
        snap: _RecordSnapshot,
        frames_by_role: dict[str, _GrabbedFrame],
    ) -> dict[str, Any] | None:
        if self._staging_dir is None or "zelux" not in frames_by_role:
            return None
        joint_vel_deg_s = [
            sim_vel_rad_to_transformed_deg_s(i, float(snap.commanded_velocity[i]))
            for i in range(6)
        ]
        stem = f"{self._frame_idx:06d}"
        primary = frames_by_role["zelux"]
        cam_payload = [
            {
                "role": role,
                "label": frame.device_label,
                "t": float(frame.timestamp),
            }
            for role, frame in frames_by_role.items()
        ]
        payload = {
            "t": float(primary.timestamp),
            "joint_pos_deg": snap.joint_pos_deg,
            "joint_vel_deg_s": joint_vel_deg_s,
            "gripper": snap.gripper,
            "hand_open": snap.gripper < -0.5,
            "hand_close": snap.gripper > 0.5,
            "action_joint_vel_rad_s": list(snap.commanded_velocity[:6]),
            "action_gripper": snap.gripper,
            "twist": snap.twist,
            "home_latched": snap.home_latched,
            "cameras": cam_payload,
        }
        return {
            "staging_dir": str(self._staging_dir),
            "stem": stem,
            "cameras": [
                {
                    "role": role,
                    "color_rgb": np.asarray(frame.color_rgb, dtype=np.uint8),
                }
                for role, frame in frames_by_role.items()
            ],
            "payload": payload,
        }

    def _service_main(self) -> None:
        disk_stop = threading.Event()
        disk_queue = self._disk_queue
        disk_thread = threading.Thread(
            target=_disk_writer_main,
            args=(disk_queue, disk_stop),
            name="epoxy-disk-writer",
            daemon=True,
        )
        disk_thread.start()

        if not self.own_cameras:
            try:
                while not self._service_stop.is_set():
                    try:
                        cmd = self._cmd_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if cmd == "shutdown":
                        break
                    if cmd == "toggle":
                        self._toggle_recording()
                        continue
                    if cmd == "commit":
                        self._commit()
                        continue
            finally:
                try:
                    disk_queue.put_nowait(None)
                except queue.Full:
                    pass
                disk_stop.set()
                disk_thread.join(timeout=8.0)
            return

        cameras: dict[str, CameraBackend] = {}
        period = 1.0 / max(self.cfg.record_hz, 1.0)
        next_save_s = 0.0
        preview_hz = max(self.cfg.record_hz, 15.0)
        preview_period = 1.0 / preview_hz
        next_preview_s = 0.0
        cv2 = None
        windows_ready = False

        def _destroy_windows() -> None:
            nonlocal windows_ready
            if cv2 is not None and windows_ready:
                try:
                    for title in _PREVIEW_TITLES.values():
                        cv2.destroyWindow(title)
                except Exception:
                    pass
                try:
                    cv2.waitKey(1)
                except Exception:
                    pass
            windows_ready = False

        def _stop_cameras() -> None:
            nonlocal cameras
            _destroy_windows()
            for role, cam in list(cameras.items()):
                try:
                    cam.close()
                except Exception as exc:
                    print(f"[epoxy_app] close {role} error: {exc}", flush=True)
            cameras = {}

        def _ensure_windows() -> None:
            nonlocal cv2, windows_ready
            if windows_ready:
                return
            if cv2 is None:
                cv2 = require_cv2()
            for title in _PREVIEW_TITLES.values():
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            windows_ready = True

        def _show_preview(role: str, frame_bgr: np.ndarray, *, recording: bool) -> None:
            if cv2 is None:
                return
            _ensure_windows()
            title = _PREVIEW_TITLES[role]
            status = "REC" if recording else "LIVE"
            shown = label_frame(cv2, frame_bgr, f"{status} | {role}")
            shown = resize_max(cv2, shown, _PREVIEW_MAX_WIDTH)
            cv2.imshow(title, shown)

        def _start_cameras() -> bool:
            nonlocal cameras, next_save_s, next_preview_s
            if cameras:
                return True
            try:
                print("[epoxy_app] Opening Zelux side + front + Infinity…", flush=True)
                cameras["zelux"] = _open_zelux(self.cfg.zelux)
            except Exception as exc:
                _stop_cameras()
                print(f"[epoxy_app] Camera open failed (teleop unchanged): {exc}", flush=True)
                print(
                    "[epoxy_app] Camera tips: close ThorCam / Infinity Capture GUIs, "
                    "unplug/replug, then: python software_win/camera/list_devices.py",
                    flush=True,
                )
                return False
            try:
                cameras["zelux_front"] = _open_zelux(self.cfg.zelux_front)
            except Exception as exc:
                print(f"[epoxy_app] Zelux front open failed (side continues): {exc}", flush=True)
            try:
                cameras["infinity"] = _open_infinity(self.cfg.infinity)
            except Exception as exc:
                print(f"[epoxy_app] Infinity open failed (Zelux continues): {exc}", flush=True)
            _ensure_windows()
            labels = " | ".join(cam.info.label() for cam in cameras.values())
            print(f"[epoxy_app] Live view ready: {labels}", flush=True)
            now = time.perf_counter()
            next_save_s = now
            next_preview_s = now
            return True

        def _sync_meta_labels() -> None:
            if self._staging_dir is None or not cameras:
                return
            meta_path = self._staging_dir / "meta.json"
            if not meta_path.is_file():
                return
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["cameras"] = [
                {"role": role, "label": cameras[role].info.label()}
                for role in cameras
            ]
            meta["num_cameras"] = len(meta["cameras"])
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Open cameras immediately so --record-demos always shows live windows.
        _start_cameras()

        try:
            while not self._service_stop.is_set():
                try:
                    cmd = self._cmd_queue.get(timeout=0.01)
                except queue.Empty:
                    cmd = None

                if cmd == "shutdown":
                    break
                if cmd == "toggle":
                    was_recording = self.is_recording
                    self._toggle_recording()
                    if self.is_recording and not was_recording:
                        if not _start_cameras():
                            self._discard_staging()
                            print(
                                "[epoxy_app] Recording cancelled — cameras not available.",
                                flush=True,
                            )
                        else:
                            _sync_meta_labels()
                    continue
                if cmd == "commit":
                    self._commit()
                    continue

                if not cameras:
                    time.sleep(0.05)
                    continue

                now = time.perf_counter()
                if now < next_preview_s:
                    time.sleep(min(0.005, next_preview_s - now))
                    continue
                next_preview_s = now + preview_period

                recording = self._recording_active.is_set()
                frames_by_role: dict[str, _GrabbedFrame] = {}
                try:
                    for role, cam in cameras.items():
                        bgr = cam.grab_bgr()
                        if bgr is None:
                            raise RuntimeError(f"{role} returned no frame")
                        _show_preview(role, bgr, recording=recording)
                        frames_by_role[role] = _GrabbedFrame(
                            role=role,
                            timestamp=time.time(),
                            color_rgb=_bgr_to_rgb(bgr),
                            device_label=cam.info.label(),
                        )
                    if cv2 is not None:
                        cv2.waitKey(1)
                except Exception as exc:
                    print(f"[epoxy_app] capture error: {exc}", flush=True)
                    continue

                if not recording:
                    continue
                if now < next_save_s:
                    continue
                next_save_s = now + period

                snap = self._snapshot[0]
                if snap is None:
                    continue

                job = self._make_frame_job(snap, frames_by_role)
                if job is None:
                    continue
                try:
                    disk_queue.put_nowait(job)
                    self._frame_idx += 1
                except queue.Full:
                    pass
        finally:
            _stop_cameras()
            try:
                disk_queue.put_nowait(None)
            except queue.Full:
                pass
            disk_stop.set()
            disk_thread.join(timeout=8.0)

    def after_tick(
        self,
        tick: TeleopTickResult,
        *,
        now: float,
        events: dict[int, int],
    ) -> bool:
        if _key_triggered(events, "r"):
            self._enqueue_cmd("toggle")
        elif _key_triggered(events, "c"):
            self._enqueue_cmd("commit")
        elif _key_triggered(events, "o"):
            self._gripper_state = -1.0
        elif _key_triggered(events, "p"):
            self._gripper_state = 1.0

        if self._recording_active.is_set():
            snap_period = 1.0 / max(self.cfg.record_hz, 1.0)
            if now >= self._next_snapshot_s:
                self._snapshot[0] = self._build_snapshot(tick)
                self._next_snapshot_s = now + snap_period
        return True

    def before_tick(self, keys: Any) -> None:
        del keys


class ManualControlRecordHook:
    """Teleop hook: control loop only publishes a snapshot; I/O runs elsewhere."""

    needs_joint_telemetry = False

    def __init__(self, recorder: EpisodeRecorder) -> None:
        self.recorder = recorder

    def before_tick(self, keys: Any) -> None:
        self.recorder.before_tick(keys)

    def after_tick(
        self,
        tick: TeleopTickResult,
        *,
        robot: Any,
        now: float,
        events: dict[int, int],
    ) -> bool:
        del robot
        return self.recorder.after_tick(tick, now=now, events=events)


def create_manual_control_record_hook(
    cfg: EpoxyConfig | None = None,
) -> tuple[ManualControlRecordHook, EpisodeRecorder]:
    if cfg is None:
        from .config_loader import load_epoxy_config

        cfg = load_epoxy_config()
    recorder = EpisodeRecorder(cfg=cfg, dataset_dir=cfg.dataset_dir)
    z = cfg.zelux
    zf = cfg.zelux_front
    inf = cfg.infinity
    print(
        "[epoxy_app] Demo recording armed (live Zelux side + front + Infinity).\n"
        f"  Zelux side:  serial={z.serial or 'auto'} index={z.index} "
        f"exp={z.exposure_ms}ms gain={z.gain}  → *_zelux.jpg\n"
        f"  Zelux front: serial={zf.serial or 'auto'} index={zf.index} "
        f"exp={zf.exposure_ms}ms gain={zf.gain}  → *_zelux_front.jpg\n"
        f"  Infinity: #={inf.camera_number} opencv={inf.opencv_index} "
        f"exp={inf.exposure_ms}ms gain={inf.gain}\n"
        f"  Dataset: {cfg.dataset_dir}\n"
        "  R = start/stop saving  |  C = save episode  |  O/P = gripper\n"
        "  Focus PyBullet for R/C keys; camera windows show LIVE / REC.",
        flush=True,
    )
    return ManualControlRecordHook(recorder), recorder
