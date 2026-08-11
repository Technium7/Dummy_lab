"""Real-time Zelux inference + epoxy FSM (MockRobot or IkCartesianMover).

::

    # Vision-only dry run (no robot): MockRobot + live Zelux
    python -m high_level_jobs.epoxy_app.vision.run_apply --mock --checkpoint ...

    # FSM smoke without camera/weights
    python -m high_level_jobs.epoxy_app.vision.run_apply --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.config_loader import load_epoxy_config
from high_level_jobs.epoxy_app.vision.config_util import load_vision_config, resolve_repo_path
from high_level_jobs.epoxy_app.vision.infer_crop import infer_frame_cascade, maybe_load_tip_model
from high_level_jobs.epoxy_app.vision.model import load_checkpoint
from high_level_jobs.epoxy_app.vision.preprocess import bgr_to_gray
from high_level_jobs.epoxy_app.vision.state_machine import (
    ControllerConfig,
    EpoxyApplicationController,
    MockRobot,
)
from high_level_jobs.epoxy_app.vision.temporal_filter import TemporalFilterConfig, TemporalVisionFilter
from high_level_jobs.epoxy_app.vision.tip_track import TipTrackConfig, TipTracker


class VisionEngine:
    """Cascade: tip crop (optional) → contact → phase → temporal filter."""

    def __init__(
        self,
        model,
        device,
        image_size: int,
        filter_cfg: TemporalFilterConfig,
        contact_threshold: float = 0.5,
        tip_model=None,
        tip_cfg: dict | None = None,
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.image_size = image_size
        self.contact_threshold = contact_threshold
        self.filter = TemporalVisionFilter(filter_cfg)
        self.tip_model = tip_model
        self.tip_cfg = tip_cfg or {"enabled": False}
        self.tip_tracker = TipTracker(TipTrackConfig.from_tip_cfg(self.tip_cfg))
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self.filter.reset()
        self.tip_tracker.reset()
        self._prev = None

    @torch.inference_mode()
    def process_bgr(self, frame_bgr: np.ndarray) -> dict:
        gray = bgr_to_gray(frame_bgr)
        try:
            result = infer_frame_cascade(
                model=self.model,
                tip_model=self.tip_model,
                gray=gray,
                prev_gray=self._prev,
                device=self.device,
                image_size=self.image_size,
                contact_thr=self.contact_threshold,
                tip_cfg=self.tip_cfg,
                tip_tracker=self.tip_tracker,
            )
            self._prev = gray
            contact_p = result["contact_probability"]
            phase_probs = result["phase_probs"]
            phase_name = result["phase_name"]
            ok = True
        except Exception:  # noqa: BLE001
            self._prev = gray
            contact_p, phase_probs, phase_name, ok = 0.5, [1.0, 0.0], "-", False
            result = {"tip_xy": None, "tip_conf": 0.0, "used_crop": False, "crop_size": 0}
        state = self.filter.update(contact_p, phase_probs)
        flow_p = float(phase_probs[1]) if len(phase_probs) > 1 else 0.0
        return {
            "contact_probability": contact_p,
            "flow_probability": flow_p,
            "phase_name": phase_name,
            "phase_probs": phase_probs,
            "tip_xy": result.get("tip_xy"),
            "tip_conf": result.get("tip_conf", 0.0),
            "used_crop": result.get("used_crop", False),
            "crop_size": int(result.get("crop_size") or self.tip_cfg.get("crop_size", 256)),
            "vision_state": state,
            "inference_ok": ok,
        }


def _open_zelux(epoxy_cfg):
    from camera.zelux import ZeluxCamera

    z = epoxy_cfg.zelux
    cam = ZeluxCamera(serial=z.serial or None, index=z.index)
    cam.open()
    if z.exposure_ms:
        try:
            cam.set_exposure_ms(z.exposure_ms)
        except Exception:  # noqa: BLE001
            pass
    return cam


def run_smoke() -> int:
    robot = MockRobot(z=0.02)
    ctl = EpoxyApplicationController(robot, ControllerConfig(flow_check_delay=0.0, max_settling_time=0.05))
    from high_level_jobs.epoxy_app.vision.temporal_filter import VisionState

    for _ in range(3):
        print(ctl.step(VisionState.NOT_TOUCHING, now=0.0))
    print(ctl.step(VisionState.TOUCHING, now=0.1))
    print(ctl.step(VisionState.FLOWING, now=0.2))
    print(ctl.step(VisionState.TOUCHING, now=0.3))
    print(ctl.step(VisionState.SETTLED, now=0.4))
    print(ctl.step(VisionState.SETTLED, now=0.5))
    print(ctl.step(VisionState.SETTLED, now=0.6))
    print("pose", robot.get_position(), "log", robot.move_log)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Epoxy apply: vision + FSM")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mock", action="store_true", help="MockRobot + live/file Zelux")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--vision-config", default=None)
    parser.add_argument("--epoxy-config", default=None)
    parser.add_argument("--video", default=None, help="Zelux video/file instead of camera")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if args.smoke:
        return run_smoke()

    vcfg = load_vision_config(args.vision_config)
    epoxy = load_epoxy_config(args.epoxy_config)
    ckpt_path = resolve_repo_path(args.checkpoint or vcfg["paths"]["best_checkpoint"])
    if not ckpt_path.exists():
        print(
            f"Checkpoint not found: {ckpt_path}\n"
            "Label episodes and train first, or use --smoke.",
            file=sys.stderr,
        )
        return 1

    device_s = args.device or vcfg.get("inference", {}).get("device", "cuda")
    if device_s == "cuda" and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)
    model, ckpt = load_checkpoint(ckpt_path, device=device, cfg=vcfg)
    tip_cfg = dict(vcfg.get("tip", {}))
    tip_model = maybe_load_tip_model(vcfg, device, cascade_ckpt=ckpt)
    if tip_cfg.get("enabled") and tip_model is None:
        print("Tip enabled but no tip checkpoint — full-frame fallback.", file=sys.stderr)
        tip_cfg["enabled"] = False
    engine = VisionEngine(
        model,
        device,
        int(vcfg["model"]["image_size"]),
        TemporalFilterConfig.from_dict(vcfg),
        contact_threshold=float(vcfg["thresholds"]["contact"]),
        tip_model=tip_model,
        tip_cfg=tip_cfg,
    )

    if not args.mock:
        print(
            "Hardware IK apply is started from the teleop session (IkCartesianMover).\n"
            "For now use --mock to validate vision+FSM with Zelux, or --smoke.",
            file=sys.stderr,
        )
        # Fall through to mock for safety until integrated into teleop_runtime hook
        args.mock = True

    robot = MockRobot(z=0.02)
    ctl = EpoxyApplicationController(robot, ControllerConfig.from_dict(vcfg))

    cam = None
    cap = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Cannot open video {args.video}", file=sys.stderr)
            return 1
    else:
        try:
            cam = _open_zelux(epoxy)
        except Exception as exc:  # noqa: BLE001
            print(f"Zelux open failed: {exc}", file=sys.stderr)
            return 1

    win = "epoxy_apply"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Running — q quit, e estop")
    try:
        while True:
            if cam is not None:
                frame = cam.grab_bgr()
                camera_ok = frame is not None
            else:
                ok, frame = cap.read()
                camera_ok = bool(ok and frame is not None)
                if not camera_ok:
                    break

            if not camera_ok:
                ctl.emergency_stop("camera failure")
                break

            result = engine.process_bgr(frame)
            step = ctl.step(
                result["vision_state"],
                camera_ok=True,
                inference_ok=result["inference_ok"],
            )
            overlay = frame.copy()
            if result.get("tip_xy") is not None:
                tx, ty = int(round(result["tip_xy"][0])), int(round(result["tip_xy"][1]))
                cv2.drawMarker(
                    overlay, (tx, ty), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2
                )
            lines = [
                f"vision {result['vision_state'].value}",
                f"contact {result['contact_probability']:.3f}  "
                f"phase {result.get('phase_name', '-')}",
                f"tip_conf {result.get('tip_conf', 0):.2f} crop={int(bool(result.get('used_crop')))}",
                f"ctrl {step.control_state.value}  {step.message}",
            ]
            for i, text in enumerate(lines):
                y = 28 + i * 26
                cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                ctl.emergency_stop("key e")
            if step.done:
                print("Done:", step.control_state, step.message)
                time.sleep(0.5)
                break
    finally:
        if cam is not None:
            cam.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
