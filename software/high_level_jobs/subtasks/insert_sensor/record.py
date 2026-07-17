"""Record teleop demonstrations with multi-camera RGB-D and robot state."""

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

from manual_control.mapping import sim_vel_rad_to_transformed_deg_s
from manual_control.teleop_tick import TeleopTickResult

from .cameras import (
    OVERHEAD_ROLE,
    CameraSpec,
    color_filename,
    depth_filename,
    make_streamer,
    resolve_camera_serials,
    stop_streamer,
)
from .config_loader import InsertSensorConfig


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


def _key_triggered(events: dict[int, int], char: str) -> bool:
    import pybullet as p

    lower = ord(char.lower())
    upper = ord(char.upper())
    return ((events.get(lower, 0) & p.KEY_WAS_TRIGGERED) != 0) or (
        (events.get(upper, 0) & p.KEY_WAS_TRIGGERED) != 0
    )


def _save_frame_job(job: dict[str, Any]) -> None:
    staging_dir = Path(str(job["staging_dir"]))
    stem = str(job["stem"])
    for cam in job["cameras"]:
        role = str(cam["role"])
        Image.fromarray(cam["color_rgb"]).save(staging_dir / color_filename(stem, role))
        np.save(staging_dir / depth_filename(stem, role), cam["depth_m"])
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
            print(f"[insert_sensor] disk writer error: {exc}", flush=True)


@dataclass
class EpisodeRecorder:
    cfg: InsertSensorConfig
    dataset_dir: Path
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
    _resolved_cameras: list[CameraSpec] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.dataset_dir = self.dataset_dir.resolve()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._service_thread = threading.Thread(
            target=self._service_main,
            name="insert-sensor-record-service",
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
        record_cams = self._resolved_cameras or list(self.cfg.record_cameras())
        meta = {
            "created": time.time(),
            "record_hz": self.cfg.record_hz,
            "num_cameras": len(record_cams),
            "policy_num_cameras": self.cfg.num_cameras,
            "cameras": [{"role": c.role, "serial": c.serial} for c in record_cams],
        }
        (self._staging_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _next_episode_dir(self) -> Path:
        existing = sorted(p for p in self.dataset_dir.glob("episode_*") if p.is_dir())
        if not existing:
            return self.dataset_dir / "episode_000001"
        last = int(existing[-1].name.split("_")[-1])
        return self.dataset_dir / f"episode_{last + 1:06d}"

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
            self._enqueue_cmd("stop_io")
            print(
                f"[insert_sensor] Recording stopped ({self._frame_idx} frames). "
                "C = save  |  R = discard and record again",
                flush=True,
            )
            return

        if phase is _RecordPhase.PENDING:
            print("[insert_sensor] Discarding pending recording.", flush=True)
            self._discard_staging()

        self._begin_staging()
        self._set_phase(_RecordPhase.RECORDING)
        self._enqueue_cmd("start_io")
        n = len(self.cfg.record_cameras())
        print(
            f"[insert_sensor] Recording...  "
            f"(opening {n} RealSense camera(s) on background thread; robot teleop unchanged)",
            flush=True,
        )

    def _commit(self) -> None:
        with self._phase_lock:
            phase = self._phase
            staging_dir = self._staging_dir
            frame_idx = self._frame_idx
        if phase is not _RecordPhase.PENDING or staging_dir is None:
            print("[insert_sensor] Nothing to save (stop recording with R first).", flush=True)
            return
        if frame_idx <= 0:
            print("[insert_sensor] Empty recording — discarded.", flush=True)
            self._discard_staging()
            return

        dest = self._next_episode_dir()
        if dest.exists():
            raise RuntimeError(f"Episode path already exists: {dest}")
        staging_dir.rename(dest)
        self._staging_root().mkdir(parents=True, exist_ok=True)
        print(f"[insert_sensor] Saved {frame_idx} frames -> {dest.name}", flush=True)
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
                    print("[insert_sensor] Discarding unsaved recording.", flush=True)
                self._discard_staging()

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
        frames_by_role: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self._staging_dir is None:
            return None
        overhead = frames_by_role.get(OVERHEAD_ROLE)
        if overhead is None:
            return None
        joint_vel_deg_s = [
            sim_vel_rad_to_transformed_deg_s(i, float(snap.commanded_velocity[i]))
            for i in range(6)
        ]
        stem = f"{self._frame_idx:06d}"
        cam_payload = [
            {
                "role": role,
                "serial": getattr(frame, "device_serial", ""),
                "t": float(frame.timestamp),
            }
            for role, frame in frames_by_role.items()
        ]
        payload = {
            "t": float(overhead.timestamp),
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
                    "depth_m": np.asarray(frame.depth_m, dtype=np.float32),
                }
                for role, frame in frames_by_role.items()
            ],
            "payload": payload,
        }

    def _service_main(self) -> None:
        disk_stop = threading.Event()
        disk_queue: queue.Queue = queue.Queue(maxsize=4)
        disk_thread = threading.Thread(
            target=_disk_writer_main,
            args=(disk_queue, disk_stop),
            name="insert-sensor-disk-writer",
            daemon=True,
        )
        disk_thread.start()

        streamers: dict[str, Any] = {}
        period = 1.0 / max(self.cfg.record_hz, 1.0)
        next_capture_s = 0.0

        def _stop_streamers() -> None:
            nonlocal streamers
            for role, streamer in list(streamers.items()):
                stop_streamer(streamer, role=role)
            streamers = {}

        def _abort_recording_no_camera(reason: str) -> None:
            nonlocal streamers
            streamers = {}
            self._discard_staging()
            print(f"[insert_sensor] Recording cancelled (teleop unchanged): {reason}", flush=True)
            print(
                "[insert_sensor] Camera tips: free stuck processes (`jobs -l`), "
                "unplug/replug cameras, use separate USB ports from the robot serial. "
                "List devices: python -m high_level_jobs.subtasks.insert_sensor --list-cameras",
                flush=True,
            )

        def _start_streamers() -> None:
            nonlocal streamers, next_capture_s
            _stop_streamers()
            try:
                wanted = list(self.cfg.record_cameras())
                self.cfg.assert_cameras_pinned(for_recording=True)
                resolved = resolve_camera_serials(
                    wanted,
                    require_pinned=self.cfg.require_camera_serials,
                )
                self._resolved_cameras = resolved
                if self._staging_dir is not None and (self._staging_dir / "meta.json").is_file():
                    meta = json.loads((self._staging_dir / "meta.json").read_text(encoding="utf-8"))
                    meta["num_cameras"] = len(resolved)
                    meta["policy_num_cameras"] = self.cfg.num_cameras
                    meta["cameras"] = [{"role": c.role, "serial": c.serial} for c in resolved]
                    (self._staging_dir / "meta.json").write_text(
                        json.dumps(meta, indent=2), encoding="utf-8"
                    )
                for cam in resolved:
                    streamers[cam.role] = make_streamer(
                        cam.serial,
                        role=cam.role,
                        exposure_us=cam.exposure_us,
                    )
            except Exception as exc:
                _stop_streamers()
                _abort_recording_no_camera(str(exc))
                return
            next_capture_s = time.perf_counter()

        try:
            while not self._service_stop.is_set():
                try:
                    cmd = self._cmd_queue.get(timeout=0.01)
                except queue.Empty:
                    cmd = None

                if cmd == "shutdown":
                    break
                if cmd == "toggle":
                    self._toggle_recording()
                    continue
                if cmd == "commit":
                    self._commit()
                    continue
                if cmd == "start_io":
                    _start_streamers()
                    continue
                if cmd == "stop_io":
                    _stop_streamers()
                    continue

                if not self._recording_active.is_set() or not streamers:
                    continue

                now = time.perf_counter()
                if now < next_capture_s:
                    time.sleep(min(0.005, next_capture_s - now))
                    continue
                next_capture_s = now + period

                snap = self._snapshot[0]
                if snap is None:
                    continue

                frames_by_role: dict[str, Any] = {}
                try:
                    for role, streamer in streamers.items():
                        frames_by_role[role] = streamer.read()
                except Exception as exc:
                    print(f"[insert_sensor] capture error: {exc}", flush=True)
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
            _stop_streamers()
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
    cfg: InsertSensorConfig | None = None,
) -> tuple[ManualControlRecordHook, EpisodeRecorder]:
    if cfg is None:
        from .config_loader import default_config_path, load_insert_sensor_config

        cfg = load_insert_sensor_config(default_config_path())
    recorder = EpisodeRecorder(cfg=cfg, dataset_dir=cfg.dataset_dir)
    cams = cfg.record_cameras()
    cam_lines = ", ".join(f"{c.role}={c.serial or 'auto'}" for c in cams)
    print(
        f"[insert_sensor] Demo recording armed ({len(cams)} camera(s) OFF until R; "
        f"policy still uses {cfg.num_cameras}).\n"
        f"  Cameras: {cam_lines}\n"
        "  Teleop/control: identical to manual_control (recording is a background thread).\n"
        "  Use separate USB ports for robot serial and each RealSense.\n"
        "  R = start/stop RGB-D capture  |  C = save  |  O/P = gripper",
        flush=True,
    )
    return ManualControlRecordHook(recorder), recorder


def run_record_session(
    cfg: InsertSensorConfig,
    *,
    urdf_path: Path,
    extra_search_paths: list[Path],
    end_link_hint: str,
    port: str,
    baudrate: int,
    sim_only: bool = False,
    **teleop_kwargs: Any,
) -> None:
    from manual_control.controller_runtime import launch_controller

    hook, recorder = create_manual_control_record_hook(cfg)
    launch_controller(
        urdf_path=urdf_path,
        fixed_base=True,
        extra_search_paths=extra_search_paths,
        end_link_hint=end_link_hint,
        linear_speed=float(teleop_kwargs.get("linear_speed", 0.015)),
        angular_speed_deg=float(teleop_kwargs.get("angular_speed_deg", 10.0)),
        joint_vel_limit=float(teleop_kwargs.get("joint_vel_limit", 0.0)),
        sim_hz=float(teleop_kwargs.get("sim_hz", 480.0)),
        speed_factor=float(teleop_kwargs.get("speed_factor", 1.0)),
        rotation_weight=float(teleop_kwargs.get("rotation_weight", 8.0)),
        joint_force=float(teleop_kwargs.get("joint_force", 2000.0)),
        ik_damping=float(teleop_kwargs.get("ik_damping", 0.05)),
        port=port,
        baudrate=baudrate,
        timeout=float(teleop_kwargs.get("timeout", 1.0)),
        control_dt_ms=float(teleop_kwargs.get("control_dt_ms", 70.0)),
        sim_only=sim_only,
        debug_serial=bool(teleop_kwargs.get("debug_serial", False)),
        joint_current_limit_a=float(teleop_kwargs.get("joint_current_limit_a", 1.2)),
        teleop_hook=hook,
        on_shutdown=recorder.shutdown,
    )
