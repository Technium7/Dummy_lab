#!/usr/bin/env python3
"""
Live GUI: show current YOLO + grasp/base CNN predictions, correct and save.

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.live_review
  python -m high_level_jobs.subtasks.grab_sensor.detection.train_model --review

Keys:
  G                 grasp mode — left-click grasp point (red)
  B                 base mode — left-drag ellipse box; drag handles to adjust
  Y / Enter / S     save current grasp + base (NN or corrected)
  R                 clear overrides — back to live NN proposals
  T                 retrain model on updated dataset
  O                 save grasp_overlay.png from this live frame (same NN as above)
  Q / Esc           quit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parents[4]

from ..config_loader import apply_runtime_calibrations, default_config_path, load_grab_sensor_config
from ..grasp import build_grasp_targets
from ..models import GrabSensorConfig
from .base_ellipse import BaseEllipse, draw_base_ellipse, draw_label_base, draw_label_grasp
from ..table_position import NN_BASE_BGR
from .capture import AlignedFrame, CaptureResult, RealSenseStreamer, next_sample_id, save_base_labeled_sample, save_labeled_sample
from .cv2_gui import PANEL_H, init_cv2_window, require_cv2_gui
from .ellipse_editor import EllipseEditor, draw_ellipse_editor
from .grasp_point_net import default_model_path
from .train_model import make_train_kwargs, run_training
from .visualize import save_grasp_overlay
from .yolo_grasp import infer_yolo_grasp_points

LabelMode = Literal["grasp", "base"]


@dataclass
class NnProposal:
    bbox_xy: tuple[int, int, int, int] | None = None
    yolo_conf: float | None = None
    grasp_uv: tuple[int, int] | None = None
    base_uv: tuple[int, int] | None = None
    base_ellipse: BaseEllipse | None = None
    error: str | None = None


def _predict_nn(frame: AlignedFrame, cfg: GrabSensorConfig) -> NnProposal:
    if cfg.yolo is None or cfg.grasp_net is None:
        return NnProposal(error="Config missing yolo or grasp_net")
    try:
        nn = infer_yolo_grasp_points(frame.color_rgb, frame.depth_m, cfg)
        return NnProposal(
            bbox_xy=nn.yolo.bbox_xy,
            yolo_conf=nn.yolo.confidence,
            grasp_uv=nn.grasp_uv,
            base_uv=nn.base_uv,
            base_ellipse=nn.base_ellipse,
        )
    except Exception as exc:
        return NnProposal(error=str(exc))


def _export_grasp_overlay(
    frame: AlignedFrame,
    cfg: GrabSensorConfig,
    out_path: Path,
) -> tuple[int, int] | None:
    """Render grasp_overlay.png from the current live frame (same NN path as markers)."""
    from .yolo_grasp import detect_with_yolo_grasp

    capture = CaptureResult(
        color_rgb=frame.color_rgb,
        depth_m=frame.depth_m,
        depth_scale=frame.depth_scale,
        color_intrinsics=frame.color_intrinsics,
        depth_intrinsics=frame.color_intrinsics,
        device_name=frame.device_name,
        device_serial=frame.device_serial,
    )
    result = detect_with_yolo_grasp(capture, cfg)
    targets = build_grasp_targets(cfg, result.detection, vision_debug=result.debug)
    save_grasp_overlay(capture, result.mask, result.detection, targets, cfg, out_path)
    return result.base_uv


def _effective_uv(
    nn_uv: tuple[int, int] | None,
    override_uv: tuple[int, int] | None,
) -> tuple[int, int] | None:
    return override_uv if override_uv is not None else nn_uv


def _mode_label(mode: LabelMode) -> str:
    return "GRASP" if mode == "grasp" else "BASE"


def _seed_base_editor(editor: EllipseEditor, proposal: NnProposal) -> None:
    if editor.is_valid():
        return
    if proposal.base_ellipse is not None:
        editor.set_ellipse(proposal.base_ellipse)
    elif proposal.base_uv is not None:
        editor.set_ellipse(BaseEllipse.from_center_uv(proposal.base_uv))


def _draw_review_ui(
    cv2,
    frame_rgb: np.ndarray,
    proposal: NnProposal,
    override_grasp: tuple[int, int] | None,
    editor: EllipseEditor,
    *,
    label_mode: LabelMode,
    saved_count: int,
    status: str,
    mode: str,
    predict_base: bool,
    base_point_only: bool = False,
    override_base: tuple[int, int] | None = None,
) -> np.ndarray:
    if base_point_only:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        if proposal.bbox_xy is not None:
            x, y, bw, bh = proposal.bbox_xy
            cv2.rectangle(bgr, (x, y), (x + bw, y + bh), (0, 255, 255), 2)

        if proposal.base_uv is not None and override_base is None:
            u, v = proposal.base_uv
            cv2.drawMarker(bgr, (u, v), (255, 255, 0), cv2.MARKER_CROSS, 22, 2)
            cv2.circle(bgr, (u, v), 12, (255, 255, 0), 2)
            cv2.putText(
                bgr,
                f"NN base ({u}, {v})",
                (min(u + 14, w - 220), max(v - 14, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if override_base is not None:
            draw_label_base(bgr, override_base[0], override_base[1])

        panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        lines = [
            "Left-click=fix base  Y/S=save  R=reset  T=retrain  Q=quit",
            f"{mode}  |  saved {saved_count}  |  {status}",
            "Yellow box=YOLO  cyan=NN base  green=your edit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (10, 28 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46 if i == 0 else 0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        return np.vstack([panel, bgr])

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    editing_base = label_mode == "base" or editor.is_valid()

    if proposal.bbox_xy is not None:
        x, y, bw, bh = proposal.bbox_xy
        cv2.rectangle(bgr, (x, y), (x + bw, y + bh), (0, 255, 255), 2)

    if proposal.grasp_uv is not None and override_grasp is None:
        u, v = proposal.grasp_uv
        cv2.drawMarker(bgr, (u, v), (255, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(bgr, (u, v), 12, (255, 255, 0), 2)
        cv2.putText(
            bgr,
            f"NN grasp ({u}, {v})",
            (min(u + 14, w - 220), max(v - 14, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if (
        predict_base
        and not editing_base
        and proposal.base_ellipse is not None
    ):
        draw_base_ellipse(bgr, proposal.base_ellipse)
        bu, bv = proposal.base_ellipse.center_uv
        cv2.putText(
            bgr,
            f"NN base ({bu}, {bv})",
            (min(bu + 14, w - 220), min(bv + 28, h - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            NN_BASE_BGR,
            1,
            cv2.LINE_AA,
        )
    elif predict_base and not editing_base and proposal.base_uv is not None:
        bu, bv = proposal.base_uv
        draw_base_ellipse(bgr, BaseEllipse.from_center_uv((bu, bv)))
        cv2.putText(
            bgr,
            f"NN base ({bu}, {bv})",
            (min(bu + 14, w - 220), min(bv + 28, h - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            NN_BASE_BGR,
            1,
            cv2.LINE_AA,
        )

    if override_grasp is not None:
        draw_label_grasp(bgr, override_grasp[0], override_grasp[1])

    if editing_base:
        draw_ellipse_editor(bgr, editor.ellipse, preview_ellipse=editor.preview_ellipse)

    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    mode_color = (0, 120, 255) if label_mode == "grasp" else (0, 220, 0)
    cv2.putText(
        panel,
        f"Mode: {_mode_label(label_mode)}",
        (w - 140, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        mode_color,
        1,
        cv2.LINE_AA,
    )
    lines = [
        "G=grasp (left-click)  B=base (left-drag ellipse)  |  Y/S=save  R=reset  T=retrain  O=overlay  Q=quit",
        f"{mode}  |  saved {saved_count}  |  {status}",
        "Yellow box=YOLO  cyan/green=NN  red/green=your edits",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (10, 28 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46 if i == 0 else 0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([panel, bgr])


def run_live_review_gui(
    *,
    data_dir: Path,
    cfg: GrabSensorConfig,
    model_path: Path | None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    serial: str = "",
    retrain_epochs: int = 120,
    overlay_out: Path | None = None,
    base_point_only: bool = False,
    window_title: str | None = None,
) -> int:
    cv2 = require_cv2_gui()
    overlay_path = (overlay_out or (_SOFTWARE_DIR / "grasp_overlay.png")).resolve()
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if cfg.yolo is None or not cfg.yolo.model_path.is_file():
        raise SystemExit(f"YOLO model not found: {cfg.yolo.model_path if cfg.yolo else None}")

    base_path = cfg.grasp_net.base_model_path if cfg.grasp_net else None
    predict_base = base_path is not None and base_path.is_file()

    if base_point_only:
        if not predict_base:
            raise SystemExit(f"Base model not found: {base_path}")
        print(f"Top review — YOLO: {cfg.yolo.model_path}")
        print(f"Top review — base: {base_path}")
    else:
        if model_path is None or not model_path.is_file():
            raise SystemExit(f"Grasp model not found: {model_path}")
        model_path = model_path.resolve()
        print(f"Loaded grasp model from {model_path}")
        if predict_base:
            print(f"Loaded base model from {base_path}")
        else:
            print("No separate base_point.pt — using legacy combined grasp model if available")
        print(f"YOLO: {cfg.yolo.model_path}")

    editor = EllipseEditor()
    state = {
        "proposal": NnProposal(),
        "override_grasp": None,
        "override_base": None,
        "label_mode": "grasp",
        "mode": "propose",
        "saved_count": len(list((data_dir / "labels").glob("*.json")))
        if (data_dir / "labels").is_dir()
        else 0,
        "status": "Live NN — Y/S to confirm or left-click to fix",
        "predict_every": 3,
        "frame_idx": 0,
    }

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H:
            return
        uv = (int(x), int(y - PANEL_H))
        if base_point_only:
            if event == cv2.EVENT_LBUTTONDOWN:
                state["override_base"] = uv
                state["mode"] = "edit"
                state["status"] = f"Base ({uv[0]}, {uv[1]}) — Y/S to save"
            return

        label_mode: LabelMode = state["label_mode"]

        if label_mode == "grasp":
            if event == cv2.EVENT_LBUTTONDOWN:
                state["override_grasp"] = uv
                state["mode"] = "edit"
                state["status"] = f"Grasp ({uv[0]}, {uv[1]}) — B for base or Y/S to save"
            return

        msg = editor.on_mouse(event, uv, cv2)
        if msg:
            state["status"] = msg
            state["mode"] = "edit"

    window = window_title or (
        "Top review (YOLO + base point)" if base_point_only else "Live review (NN grasp + base)"
    )
    init_cv2_window(cv2, window, width, height + PANEL_H, on_mouse)

    streamer = RealSenseStreamer(width=width, height=height, fps=fps, serial=serial, warmup=5)
    print(f"Camera live. Dataset: {data_dir}", flush=True)
    streamer.start()
    print("Camera ready.", flush=True)

    def _ui_kwargs(frame_rgb: np.ndarray) -> dict:
        return dict(
            cv2=cv2,
            frame_rgb=frame_rgb,
            proposal=state["proposal"],
            override_grasp=state["override_grasp"],
            editor=editor,
            label_mode=state["label_mode"],
            saved_count=state["saved_count"],
            status=state["status"],
            mode=state["mode"],
            predict_base=predict_base,
            base_point_only=base_point_only,
            override_base=state["override_base"],
        )

    def save_sample(frame: AlignedFrame, note: str) -> None:
        proposal = state["proposal"]
        if base_point_only:
            base_uv = _effective_uv(proposal.base_uv, state["override_base"])
            if base_uv is None:
                state["status"] = "No base to save — wait for NN or left-click to fix"
                return
            save_frame = streamer.read(with_depth=True)
            sid = next_sample_id(data_dir)
            label_path = save_base_labeled_sample(
                data_dir,
                sid,
                save_frame,
                base_uv,
                note=note,
            )
            state["saved_count"] += 1
            state["override_base"] = None
            state["mode"] = "propose"
            state["status"] = f"Saved {sid} ({note}) — adjust scene for next"
            print(f"Saved {label_path} ({note})")
            return

        grasp_uv = _effective_uv(proposal.grasp_uv, state["override_grasp"])
        if grasp_uv is None:
            state["status"] = "No grasp to save — wait for NN or press G and click to fix"
            return

        base_ellipse = editor.ellipse if editor.is_valid() else proposal.base_ellipse
        if predict_base and base_ellipse is None:
            base_uv = _effective_uv(proposal.base_uv, None)
            if base_uv is not None:
                base_ellipse = BaseEllipse.from_center_uv(base_uv)
        if predict_base and base_ellipse is None:
            state["status"] = "No base to save — wait for NN or press B and draw ellipse"
            return

        save_frame = streamer.read(with_depth=True)
        sid = next_sample_id(data_dir)
        label_path = save_labeled_sample(
            data_dir,
            sid,
            save_frame,
            grasp_uv,
            base_ellipse=base_ellipse,
            note=note,
        )
        state["saved_count"] += 1
        state["override_grasp"] = None
        editor.clear()
        state["label_mode"] = "grasp"
        state["mode"] = "propose"
        state["status"] = f"Saved {sid} ({note}) — adjust scene for next"
        print(f"Saved {label_path} ({note})")

    def retrain() -> None:
        nonlocal predict_base
        state["status"] = "Retraining…"
        cv2.imshow(window, _draw_review_ui(**_ui_kwargs(frame.color_rgb)))
        cv2.waitKey(1)
        yolo_path = cfg.yolo.model_path if cfg.yolo else None
        if base_point_only:
            from .top_detection.train_model import make_train_kwargs as top_train_kwargs
            from .top_detection.train_model import run_training as top_run_training

            result = top_run_training(
                data_dir,
                base_path=base_path,
                train_kwargs=top_train_kwargs(
                    epochs=retrain_epochs,
                    yolo_model_path=yolo_path if yolo_path and yolo_path.is_file() else None,
                    crop_mode=True,
                    use_depth=True,
                ),
                verbose=False,
            )
            err = result.best_val_base_px_error or result.best_val_px_error
            state["status"] = f"Retrained — base ~ {err:.1f}px"
            print(state["status"])
            return

        results = run_training(
            data_dir,
            grasp_path=model_path,
            base_path=base_path or (model_path.parent / "base_point.pt"),
            train_grasp=True,
            train_base=predict_base or (base_path is not None and base_path.parent.exists()),
            train_kwargs=make_train_kwargs(
                epochs=retrain_epochs,
                yolo_model_path=yolo_path if yolo_path and yolo_path.is_file() else None,
                crop_mode=True,
                use_depth=True,
            ),
            verbose=False,
        )
        err = ""
        if results.grasp is not None:
            err = f"grasp ~ {results.grasp.best_val_px_error:.1f}px"
        if results.base is not None:
            base_err = results.base.best_val_base_px_error or results.base.best_val_px_error
            err += f"  base ~ {base_err:.1f}px"
            predict_base = True
        state["status"] = f"Retrained — val {err.strip()}"
        print(state["status"])

    try:
        while True:
            need_predict = state["mode"] == "propose" and state["frame_idx"] % state["predict_every"] == 0
            frame = streamer.read(with_depth=need_predict)
            state["frame_idx"] += 1

            if need_predict:
                proposal = _predict_nn(frame, cfg)
                state["proposal"] = proposal
                if proposal.error:
                    state["status"] = f"NN error: {proposal.error}"
                elif state["mode"] == "propose":
                    conf = f" conf={proposal.yolo_conf:.2f}" if proposal.yolo_conf is not None else ""
                    state["status"] = f"NN proposal{conf} — Y/S confirm or G/B to fix"

            display = _draw_review_ui(**_ui_kwargs(frame.color_rgb))
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if not base_point_only:
                if key in (ord("g"), ord("G")):
                    state["label_mode"] = "grasp"
                    state["status"] = "Grasp mode — left-click grasp point"
                if key in (ord("b"), ord("B")):
                    state["label_mode"] = "base"
                    _seed_base_editor(editor, state["proposal"])
                    state["status"] = "Base mode — left-drag box over counter, adjust handles"
            if key in (ord("r"), ord("R")):
                state["override_grasp"] = None
                state["override_base"] = None
                editor.clear()
                if not base_point_only:
                    state["label_mode"] = "grasp"
                state["mode"] = "propose"
                state["status"] = "Reset — live NN proposals"
            if key in (ord("t"), ord("T")):
                retrain()
            if not base_point_only and key in (ord("o"), ord("O")):
                frame_overlay = streamer.read(with_depth=True)
                try:
                    base_uv = _export_grasp_overlay(frame_overlay, cfg, overlay_path)
                    state["status"] = (
                        f"Overlay saved {overlay_path.name}"
                        + (f"  NN base {base_uv}" if base_uv else "")
                    )
                    print(f"[live_review] grasp overlay saved: {overlay_path}")
                    if base_uv:
                        print(f"[live_review] NN base_uv={base_uv}")
                except Exception as exc:
                    state["status"] = f"Overlay failed: {exc}"
            if key in (ord("y"), ord("Y"), ord("s"), ord("S"), 13):
                note = "corrected_nn" if state["mode"] == "edit" else "confirmed_nn"
                save_sample(frame, note=note)

    finally:
        streamer.stop()
        cv2.destroyAllWindows()

    print(f"Done. {state['saved_count']} samples in {data_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent / "side_detection" / "dataset" / "grasp_keypoints"
    p = argparse.ArgumentParser(description="Live review: NN grasp+base proposals, correct and save.")
    p.add_argument("--data-dir", type=Path, default=default_data)
    p.add_argument("--config", type=Path, default=None, help="grab_sensor config (default_grasp.yaml)")
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default="")
    p.add_argument("--retrain-epochs", type=int, default=60)
    p.add_argument(
        "--train-first",
        action="store_true",
        help="Train (or retrain) on existing labels before opening the GUI",
    )
    p.add_argument(
        "--base-point-only",
        action="store_true",
        help="Top/overhead mode: YOLO + base point only (no grasp/ellipse)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    cfg = apply_runtime_calibrations(
        load_grab_sensor_config((args.config or default_config_path()).resolve()),
        {},
    )
    base_path = cfg.grasp_net.base_model_path if cfg.grasp_net else None

    if args.base_point_only:
        if args.train_first or base_path is None or not base_path.is_file():
            print("Training base model before review…")
            yolo_path = cfg.yolo.model_path if cfg.yolo else None
            from .top_detection.train_model import make_train_kwargs as top_train_kwargs
            from .top_detection.train_model import run_training as top_run_training

            try:
                top_run_training(
                    data_dir,
                    base_path=base_path,
                    train_kwargs=top_train_kwargs(
                        epochs=args.retrain_epochs,
                        yolo_model_path=yolo_path if yolo_path and yolo_path.is_file() else None,
                        crop_mode=True,
                        use_depth=True,
                    ),
                )
            except ValueError as exc:
                print(exc)
                print(
                    "Collect labels: "
                    "python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_label"
                )
                return 1

        if base_path is None or not base_path.is_file():
            print(f"No base model at {base_path}. Collect labels and train first.")
            return 1

        return run_live_review_gui(
            data_dir=data_dir,
            cfg=cfg,
            model_path=None,
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial=args.serial.strip(),
            retrain_epochs=args.retrain_epochs,
            base_point_only=True,
        )

    model_path = (
        args.model_path or (cfg.grasp_net.model_path if cfg.grasp_net else default_model_path(data_dir))
    )
    if model_path is not None:
        model_path = model_path.resolve()

    if args.train_first or not model_path.is_file():
        print("Training models before review…")
        yolo_path = cfg.yolo.model_path if cfg.yolo else None
        base_path = cfg.grasp_net.base_model_path if cfg.grasp_net else None
        run_training(
            data_dir,
            grasp_path=model_path,
            base_path=base_path,
            train_grasp=True,
            train_base=base_path is not None,
            train_kwargs=make_train_kwargs(
                epochs=args.retrain_epochs,
                yolo_model_path=yolo_path if yolo_path and yolo_path.is_file() else None,
                crop_mode=True,
                use_depth=True,
            ),
        )

    if not model_path.is_file():
        print(f"No model at {model_path}. Collect labels with live_label and train first.")
        return 1

    return run_live_review_gui(
        data_dir=data_dir,
        cfg=cfg,
        model_path=model_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial.strip(),
        retrain_epochs=args.retrain_epochs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
