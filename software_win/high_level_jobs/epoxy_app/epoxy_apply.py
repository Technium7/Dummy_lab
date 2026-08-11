"""Epoxy vision apply on top of manual_control teleop.

Jog the tip to the start pose with normal teleop keys, then start the FSM.

::

    cd C:\\Users\\PQE_339_ThinkPad\\Documents\\Dummy_lab
    conda activate .\\.conda
    $env:PYTHONPATH = \"$PWD\\software_win\"
    python -m high_level_jobs.epoxy_app.epoxy_apply
    python -m high_level_jobs.epoxy_app.epoxy_apply --sim-only

Keys (focus the PyBullet window):
  G  start apply from the current pose
  X  emergency stop — freeze at the current position
  R  start/stop recording apply run → dataset/record_result
  C  save pending recording
  (WASD / QE / … — normal teleop jog while idle)

OpenCV windows ``epoxy_apply Zelux`` (predictions) and ``epoxy_apply Infinity`` (live view).
  x  emergency stop (same as PyBullet X when a camera window has focus)
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import torch

_SOFTWARE_WIN = Path(__file__).resolve().parents[2]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from camera.display import label_frame, require_cv2, resize_max
from camera.infinity import InfinityCamera
from camera.zelux import ZeluxCamera
from high_level_jobs.epoxy_app.config_loader import load_epoxy_config
from high_level_jobs.epoxy_app.record import EpisodeRecorder
from high_level_jobs.epoxy_app.vision.config_util import load_vision_config, resolve_repo_path
from high_level_jobs.epoxy_app.vision.ik_motion import IkCartesianMover, IkStepConfig
from high_level_jobs.epoxy_app.vision.infer_crop import maybe_load_tip_model
from high_level_jobs.epoxy_app.vision.model import load_checkpoint
from high_level_jobs.epoxy_app.vision.run_apply import VisionEngine
from high_level_jobs.epoxy_app.vision.state_machine import (
    ControlState,
    ControllerConfig,
    EpoxyApplicationController,
)
from high_level_jobs.epoxy_app.vision.temporal_filter import TemporalFilterConfig, VisionState
from manual_control.ik_core import get_controllable_joints
from manual_control.teleop_runtime import TeleopLaunchConfig, run_teleop_session
from manual_control.teleop_tick import KeyInput, PyBulletKeyInput, TeleopTickResult
from manual_control.urdf_tools import build_package_map, choose_description, discover_descriptions, materialize_urdf
from robot_config import DEFAULT_ROBOT, default_robot_port, repo_root

_ZELUX_TITLE = "epoxy_apply Zelux"
_INFINITY_TITLE = "epoxy_apply Infinity"
_PREVIEW_MAX_WIDTH = 960
_MOVEMENT_KEYS = frozenset("wsadqehkujiy")


def _key_triggered(events: dict[int, int], char: str) -> bool:
    lower = ord(char.lower())
    upper = ord(char.upper())
    return ((events.get(lower, 0) & p.KEY_WAS_TRIGGERED) != 0) or (
        (events.get(upper, 0) & p.KEY_WAS_TRIGGERED) != 0
    )


class ApplyAwareKeyInput:
    """Block teleop jog while the epoxy FSM owns the arm."""

    def __init__(self, inner: KeyInput, session: "EpoxyApplySession") -> None:
        self._inner = inner
        self._session = session

    def is_down(self, events: dict[int, int], char: str) -> bool:
        if self._session.blocks_teleop and char.lower() in _MOVEMENT_KEYS:
            return False
        return self._inner.is_down(events, char)

    def was_triggered(self, events: dict[int, int], char: str) -> bool:
        return self._inner.was_triggered(events, char)

    def any_movement_down(self, events: dict[int, int]) -> bool:
        if self._session.blocks_teleop:
            return False
        return self._inner.any_movement_down(events)


class EpoxyApplySession:
    """Zelux vision thread + teleop hook that runs the apply FSM."""

    needs_joint_telemetry = False

    def __init__(
        self,
        *,
        vision_engine: VisionEngine,
        epoxy_cfg,
        control_cfg: ControllerConfig,
        ik_cfg: IkStepConfig,
        sim_hz: float,
        recorder: EpisodeRecorder | None = None,
    ) -> None:
        self.vision = vision_engine
        self.epoxy_cfg = epoxy_cfg
        self.control_cfg = control_cfg
        self.ik_cfg = ik_cfg
        self.ik_cfg.sim_hz = float(sim_hz)
        self.recorder = recorder

        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {
            "vision_state": VisionState.NOT_TOUCHING,
            "contact_probability": 0.0,
            "flow_probability": 0.0,
            "phase_name": "-",
            "tip_xy": None,
            "tip_conf": 0.0,
            "used_crop": False,
            "crop_size": 0,
            "inference_ok": True,
            "camera_ok": False,
            "ctrl_state": "IDLE",
            "ctrl_message": "jog tip, then G to start",
            "recording": False,
            "vision_paused": False,
        }
        self._estop = threading.Event()
        self._vision_paused = threading.Event()
        self._applying = False
        self._hold_after_estop = False

        self._robot_id: int | None = None
        self._end_link_idx: int | None = None
        self._joint_indices: list[int] = []
        self._mover: IkCartesianMover | None = None
        self._controller: EpoxyApplicationController | None = None
        self._robot_control = None

        self._stop = threading.Event()
        self._vision_thread = threading.Thread(
            target=self._vision_main,
            name="epoxy-apply-vision",
            daemon=True,
        )

    @property
    def blocks_teleop(self) -> bool:
        return self._applying or self._hold_after_estop

    def bind_scene(self, robot_id: int, end_link_idx: int) -> None:
        self._robot_id = int(robot_id)
        self._end_link_idx = int(end_link_idx)
        joint_indices, *_rest = get_controllable_joints(robot_id)
        self._joint_indices = list(joint_indices)
        print(
            f"[epoxy_apply] Scene ready robot_id={robot_id} end_link={end_link_idx} "
            f"joints={len(self._joint_indices)}",
            flush=True,
        )

    def start(self) -> None:
        self._vision_thread.start()

    def shutdown(self) -> None:
        self._estop.set()
        self._stop.set()
        if self._mover is not None:
            self._mover.abort()
        if self.recorder is not None:
            self.recorder.shutdown()
        if self._vision_thread.is_alive():
            self._vision_thread.join(timeout=8.0)

    def _ensure_mover(self, robot_control) -> bool:
        if self._robot_id is None or self._end_link_idx is None or not self._joint_indices:
            print("[epoxy_apply] Scene not ready yet.", flush=True)
            return False
        self._robot_control = robot_control
        if self._mover is None:
            self._mover = IkCartesianMover(
                self._robot_id,
                self._end_link_idx,
                self._joint_indices,
                robot_control=robot_control,
                config=self.ik_cfg,
                abort_check=self._estop.is_set,
            )
        else:
            self._mover.robot_control = robot_control
        if self._controller is None:
            self._controller = EpoxyApplicationController(
                self._mover,
                self.control_cfg,
                before_z_move=self._on_before_z_move,
                after_z_move=self._on_after_z_move,
            )
        return True

    def _on_before_z_move(self) -> None:
        """Tip is about to creep — freeze detector so motion ≠ FLOWING."""
        self._vision_paused.set()
        if self._controller is not None:
            self._controller.set_vision_live(False)
        self._publish(vision_paused=True, ctrl_message="creeping (vision paused)")

    def _on_after_z_move(self) -> None:
        """Wait for tip to settle, then re-arm vision.

        Hard-reset the temporal filter after approach creeps (motion ≠ contact).
        During CHECK_FLOW keep filter memory so a flow streak can latch across waits.
        """
        delay = float(self.control_cfg.post_move_vision_delay)
        t_end = time.perf_counter() + max(delay, 0.0)
        while time.perf_counter() < t_end:
            if self._estop.is_set() or self._stop.is_set():
                break
            time.sleep(0.02)
        in_check_flow = (
            self._controller is not None
            and self._controller.state == ControlState.CHECK_FLOW
        )
        if not in_check_flow:
            self.vision.reset()
        self._vision_paused.clear()
        if self._controller is not None:
            self._controller.set_vision_live(True)
        self._publish(vision_paused=False, ctrl_message="vision armed")

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def _publish(self, **kwargs: Any) -> None:
        with self._lock:
            self._latest.update(kwargs)

    def _request_estop(self, reason: str = "operator estop") -> None:
        self._estop.set()
        self._hold_after_estop = True
        self._applying = False
        if self._mover is not None:
            self._mover.abort()
        if self._controller is not None:
            self._controller.emergency_stop(reason)
        self._publish(ctrl_state="ESTOP", ctrl_message=reason)
        print(f"[epoxy_apply] ESTOP — holding current pose ({reason})", flush=True)

    def _start_apply(self) -> None:
        if self._applying:
            print("[epoxy_apply] Already applying.", flush=True)
            return
        if self._robot_control is None and self._mover is None:
            # mover is created on first after_tick with robot proxy
            pass
        if self._mover is None or self._controller is None:
            print(
                "[epoxy_apply] Wait one teleop tick for robot binding, then press G again.",
                flush=True,
            )
            return
        self._estop.clear()
        self._hold_after_estop = False
        self._vision_paused.clear()
        self._mover.clear_abort()
        self.vision.reset()
        self._controller.reset()
        self._controller.set_vision_live(True)
        self._applying = True
        self._publish(ctrl_state="APPROACH", ctrl_message="apply started", vision_paused=False)
        print("[epoxy_apply] Apply started from current pose. X = estop.", flush=True)

    def before_tick(self, keys: KeyInput) -> None:
        del keys

    def after_tick(
        self,
        tick: TeleopTickResult,
        *,
        robot: Any,
        now: float,
        events: dict[int, int],
    ) -> bool:
        if robot is not None:
            self._ensure_mover(robot)

        if self.recorder is not None:
            if _key_triggered(events, "r"):
                self.recorder.request_toggle()
            elif _key_triggered(events, "c"):
                self.recorder.request_commit()
            elif _key_triggered(events, "o"):
                self.recorder._gripper_state = -1.0
            elif _key_triggered(events, "p"):
                self.recorder._gripper_state = 1.0
            self.recorder.note_teleop_tick(tick, now=now)
            self._publish(recording=self.recorder.is_recording)

        if _key_triggered(events, "x"):
            self._request_estop("key X")
            return True
        if _key_triggered(events, "g"):
            # After estop, first G only releases the hold so you can re-jog.
            if self._hold_after_estop:
                self._hold_after_estop = False
                self._estop.clear()
                if self._mover is not None:
                    self._mover.clear_abort()
                if self._controller is not None:
                    self._controller.reset()
                self._publish(ctrl_state="IDLE", ctrl_message="estop cleared — jog, then G")
                print("[epoxy_apply] Cleared estop hold — teleop jog enabled. G again to apply.", flush=True)
                return True
            self._start_apply()
            return True

        if not self._applying or self._controller is None:
            return True

        snap = self._snapshot()
        if self._estop.is_set():
            self._applying = False
            return True

        step = self._controller.step(
            snap["vision_state"],
            camera_ok=bool(snap.get("camera_ok", False)),
            inference_ok=bool(snap.get("inference_ok", True)),
            vision_live=not self._vision_paused.is_set(),
            flow_probability=float(snap.get("flow_probability", 0.0) or 0.0),
        )
        self._publish(
            ctrl_state=step.control_state.value,
            ctrl_message=step.message or "",
        )
        if step.done:
            self._applying = False
            print(
                f"[epoxy_apply] Finished: {step.control_state.value} — {step.message}",
                flush=True,
            )
            if step.control_state == ControlState.ERROR:
                self._hold_after_estop = True
        return True

    def _vision_main(self) -> None:
        cv2 = require_cv2()
        zelux: ZeluxCamera | None = None
        infinity: InfinityCamera | None = None
        z = self.epoxy_cfg.zelux
        inf = self.epoxy_cfg.infinity
        try:
            print("[epoxy_apply] Opening Zelux + Infinity…", flush=True)
            zelux = ZeluxCamera(
                serial=z.serial or None,
                exposure_s=max(z.exposure_ms, 0.1) / 1000.0,
                gain=z.gain,
                index=z.index,
            )
            zelux.open()
            cv2.namedWindow(_ZELUX_TITLE, cv2.WINDOW_NORMAL)
            print(f"[epoxy_apply] Zelux live: {zelux.info.label()}", flush=True)
            if self.recorder is not None:
                self.recorder.set_camera_labels({"zelux": zelux.info.label()})
        except Exception as exc:  # noqa: BLE001
            print(f"[epoxy_apply] Zelux open failed: {exc}", flush=True)
            self._publish(camera_ok=False, ctrl_message=f"camera fail: {exc}")
            return

        try:
            infinity = InfinityCamera(
                camera_number=inf.camera_number,
                opencv_index=inf.opencv_index,
                exposure_ms=inf.exposure_ms,
                gain=inf.gain,
                prefer_opencv=inf.prefer_opencv or inf.opencv_index is not None,
                bayer=inf.bayer,
            )
            infinity.open()
            cv2.namedWindow(_INFINITY_TITLE, cv2.WINDOW_NORMAL)
            print(f"[epoxy_apply] Infinity live: {infinity.info.label()}", flush=True)
            if self.recorder is not None:
                labels = dict(self.recorder._camera_labels)
                labels["infinity"] = infinity.info.label()
                self.recorder.set_camera_labels(labels)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[epoxy_apply] Infinity open failed (Zelux+apply continue): {exc}",
                flush=True,
            )
            infinity = None

        try:
            while not self._stop.is_set():
                try:
                    frame = zelux.grab_bgr()
                except Exception as exc:  # noqa: BLE001
                    self._publish(camera_ok=False, inference_ok=False)
                    print(f"[epoxy_apply] Zelux grab error: {exc}", flush=True)
                    time.sleep(0.05)
                    continue
                if frame is None:
                    self._publish(camera_ok=False)
                    time.sleep(0.01)
                    continue

                paused = self._vision_paused.is_set()
                ctrl = self._snapshot()
                if paused:
                    # Live preview only — do not run cascade/filter while tip is moving.
                    vs = ctrl.get("vision_state", VisionState.NOT_TOUCHING)
                    result = {
                        "vision_state": vs,
                        "contact_probability": ctrl.get("contact_probability", 0.0),
                        "flow_probability": ctrl.get("flow_probability", 0.0),
                        "phase_name": "paused",
                        "tip_xy": ctrl.get("tip_xy"),
                        "tip_conf": ctrl.get("tip_conf", 0.0),
                        "used_crop": ctrl.get("used_crop", False),
                        "crop_size": ctrl.get("crop_size", 0),
                        "inference_ok": True,
                    }
                else:
                    result = self.vision.process_bgr(frame)
                    ctrl = self._snapshot()

                recording = bool(ctrl.get("recording")) or (
                    self.recorder is not None and self.recorder.is_recording
                )
                overlay = self._draw_overlay(
                    cv2, frame, result, ctrl, recording=recording, vision_paused=paused
                )
                shown = resize_max(cv2, overlay, _PREVIEW_MAX_WIDTH)
                cv2.imshow(_ZELUX_TITLE, shown)

                inf_bgr = None
                if infinity is not None:
                    try:
                        inf_bgr = infinity.grab_bgr()
                        if inf_bgr is not None:
                            status = str(ctrl.get("ctrl_state", "IDLE"))
                            tag = "REC" if recording else ("HOLD" if paused else "LIVE")
                            inf_shown = label_frame(cv2, inf_bgr, f"{tag} | {status}")
                            inf_shown = resize_max(cv2, inf_shown, _PREVIEW_MAX_WIDTH)
                            cv2.imshow(_INFINITY_TITLE, inf_shown)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[epoxy_apply] Infinity grab error: {exc}", flush=True)

                if self.recorder is not None and recording and inf_bgr is not None:
                    self.recorder.ingest_bgr_frames(
                        {"zelux": frame, "infinity": inf_bgr},
                        device_labels=self.recorder._camera_labels,
                    )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("x"), ord("X")):
                    self._request_estop("opencv X")

                if not paused:
                    self._publish(
                        vision_state=result["vision_state"],
                        contact_probability=result["contact_probability"],
                        flow_probability=result["flow_probability"],
                        phase_name=result.get("phase_name", "-"),
                        tip_xy=result.get("tip_xy"),
                        tip_conf=result.get("tip_conf", 0.0),
                        used_crop=result.get("used_crop", False),
                        crop_size=result.get("crop_size", 0),
                        inference_ok=result.get("inference_ok", True),
                        camera_ok=True,
                        recording=recording,
                        vision_paused=False,
                    )
                else:
                    self._publish(
                        camera_ok=True,
                        recording=recording,
                        vision_paused=True,
                        phase_name="paused",
                    )
        finally:
            for cam in (zelux, infinity):
                if cam is None:
                    continue
                try:
                    cam.close()
                except Exception:  # noqa: BLE001
                    pass
            for title in (_ZELUX_TITLE, _INFINITY_TITLE):
                try:
                    cv2.destroyWindow(title)
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _draw_overlay(
        cv2,
        frame: np.ndarray,
        result: dict[str, Any],
        ctrl: dict[str, Any],
        *,
        recording: bool = False,
        vision_paused: bool = False,
    ) -> np.ndarray:
        overlay = frame.copy()
        tip = result.get("tip_xy")
        if tip is not None and not vision_paused:
            tx, ty = int(round(tip[0])), int(round(tip[1]))
            cv2.drawMarker(
                overlay,
                (tx, ty),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
            if result.get("used_crop"):
                # Square ROI in full-frame pixels (matches crop_around_tip / tip.crop_size)
                half = int(result.get("crop_size") or 256) // 2
                x0, y0 = max(0, tx - half), max(0, ty - half)
                x1, y1 = tx + half, ty + half
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 255), 1)

        applying = ctrl.get("ctrl_state", "IDLE")
        vs = result.get("vision_state", VisionState.NOT_TOUCHING)
        vs_txt = vs.value if hasattr(vs, "value") else str(vs)
        if vision_paused:
            vs_txt = "PAUSED (creeping)"
        lines = [
            f"vision {vs_txt}",
            f"contact {result['contact_probability']:.3f}  "
            f"flow {result.get('flow_probability', 0):.3f}  "
            f"phase {result.get('phase_name', '-')}",
            f"tip_conf {result.get('tip_conf', 0):.2f} crop={int(bool(result.get('used_crop')))}",
            f"ctrl {applying}  {ctrl.get('ctrl_message', '')}",
            "G=apply  X=estop  R=rec  C=save",
        ]
        for i, text in enumerate(lines):
            y = 28 + i * 26
            cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        mode = "REC" if recording else ("HOLD" if vision_paused else "LIVE")
        tag = f"{mode} | {applying}"
        return label_frame(cv2, overlay, tag)


def _build_vision_engine(args: argparse.Namespace) -> tuple[VisionEngine, dict]:
    vcfg = load_vision_config(args.vision_config)
    ckpt_path = resolve_repo_path(args.checkpoint or vcfg["paths"]["best_checkpoint"])
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train first, or pass --checkpoint."
        )
    device_s = args.device or vcfg.get("inference", {}).get("device", "cuda")
    if device_s == "cuda" and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)
    model, ckpt = load_checkpoint(ckpt_path, device=device, cfg=vcfg)
    tip_cfg = dict(vcfg.get("tip", {}))
    tip_model = maybe_load_tip_model(vcfg, device, cascade_ckpt=ckpt)
    if tip_cfg.get("enabled") and tip_model is None:
        print("[epoxy_apply] Tip enabled but no tip weights — full-frame fallback.", flush=True)
        tip_cfg["enabled"] = False
    return VisionEngine(
        model,
        device,
        int(vcfg["model"]["image_size"]),
        TemporalFilterConfig.from_dict(vcfg),
        contact_threshold=float(vcfg["thresholds"]["contact"]),
        tip_model=tip_model,
        tip_cfg=tip_cfg,
    ), vcfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Epoxy apply: manual_control teleop + vision FSM (G start, X estop)."
    )
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT)
    parser.add_argument("--fixed-base", action="store_true", default=True)
    parser.add_argument("--floating-base", action="store_true")
    parser.add_argument("--end-link", type=str, default="")
    parser.add_argument("--linear-speed", type=float, default=0.015)
    parser.add_argument("--angular-speed-deg", type=float, default=10.0)
    parser.add_argument("--joint-vel-limit", type=float, default=0.0)
    parser.add_argument("--sim-hz", type=float, default=480.0)
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--rotation-weight", type=float, default=8.0)
    parser.add_argument("--joint-force", type=float, default=2000.0)
    parser.add_argument("--ik-damping", type=float, default=0.05)
    parser.add_argument("--port", type=str, default=default_robot_port())
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--control-dt-ms", type=float, default=70.0)
    parser.add_argument("--home-firmware", action="store_true")
    parser.add_argument("--sim-only", action="store_true")
    parser.add_argument("--debug-serial", action="store_true")
    parser.add_argument("--joint-current-limit-a", type=float, default=0.0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--vision-config", default=None)
    parser.add_argument("--epoxy-config", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    fixed_base = args.fixed_base and not args.floating_base

    engine, vcfg = _build_vision_engine(args)
    epoxy_cfg = load_epoxy_config(args.epoxy_config)
    control_cfg = ControllerConfig.from_dict(vcfg)
    c = vcfg.get("control", {})
    ik_cfg = IkStepConfig(
        step_linear_speed=float(c.get("step_linear_speed", 0.00165)),
        ik_damping=float(c.get("ik_damping", args.ik_damping)),
        rotation_weight=float(c.get("rotation_weight", args.rotation_weight)),
        joint_vel_limit=float(c.get("joint_vel_limit", 1.2)),
        joint_force=float(c.get("joint_force", args.joint_force)),
        settle_tol=float(c.get("settle_tol", 1e-5)),
        accel_time=float(c.get("accel_time", 0.20)),
        max_step_time=float(c.get("max_step_time", 2.0)),
        sim_hz=float(args.sim_hz),
    )

    recorder = EpisodeRecorder(
        cfg=epoxy_cfg,
        dataset_dir=epoxy_cfg.record_result_dir,
        own_cameras=False,
        task_name="epoxy_apply_result",
        episode_prefix="result",
    )
    session = EpoxyApplySession(
        vision_engine=engine,
        epoxy_cfg=epoxy_cfg,
        control_cfg=control_cfg,
        ik_cfg=ik_cfg,
        sim_hz=float(args.sim_hz),
        recorder=recorder,
    )

    candidates = discover_descriptions(root)
    selected = choose_description(candidates, args.robot, root)
    package_map = build_package_map(root)
    urdf_path, temp_file = materialize_urdf(selected, package_map)
    print(f"[epoxy_apply] Selected robot: {selected}")
    print(f"[epoxy_apply] Loading URDF: {urdf_path}")
    if selected.suffix.lower() == ".xacro":
        urdf_text = urdf_path.read_text(encoding="utf-8")
        if len(re.findall(r"<joint\b", urdf_text)) == 0:
            print("Warning: resolved URDF has 0 joints. IK control will not work.")
    search_paths = {root, selected.parent, urdf_path.parent, *package_map.values()}

    print(
        "[epoxy_apply] Teleop + vision armed.\n"
        "  Jog tip above the pad with WASD/QE (PyBullet focus).\n"
        "  G = start apply   |   X = emergency stop (hold pose)\n"
        "  R = start/stop recording  |  C = save → record_result\n"
        f"  Record dir: {epoxy_cfg.record_result_dir}\n"
        "  Live cameras: 'epoxy_apply Zelux' (predictions) + 'epoxy_apply Infinity'.",
        flush=True,
    )
    session.start()

    key_input = ApplyAwareKeyInput(PyBulletKeyInput(), session)
    try:
        run_teleop_session(
            TeleopLaunchConfig(
                urdf_path=urdf_path,
                fixed_base=fixed_base,
                extra_search_paths=search_paths,
                end_link_hint=args.end_link,
                linear_speed=args.linear_speed,
                angular_speed_deg=args.angular_speed_deg,
                joint_vel_limit=args.joint_vel_limit,
                sim_hz=args.sim_hz,
                speed_factor=args.speed_factor,
                rotation_weight=args.rotation_weight,
                joint_force=args.joint_force,
                ik_damping=args.ik_damping,
                port=args.port,
                baudrate=args.baudrate,
                timeout=args.timeout,
                control_dt_ms=args.control_dt_ms,
                home_use_firmware=args.home_firmware,
                sim_only=args.sim_only,
                debug_serial=args.debug_serial,
                joint_current_limit_a=args.joint_current_limit_a,
            ),
            keys=key_input,
            hook=session,
            on_scene_ready=session.bind_scene,
            on_shutdown=session.shutdown,
        )
    finally:
        session.shutdown()
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
