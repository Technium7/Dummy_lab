#!/usr/bin/env python3
"""
Live GUI: show NN grasp proposal, confirm or reset to collect more labels.

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.review_grasp
  python -m high_level_jobs.subtasks.grab_sensor.train_grasp_model --review

Keys:
  Y / Enter    confirm proposed grasp and save to dataset
  Click        override grasp point manually (red)
  S            save manual click
  R            reset — reject proposal, click manually to collect more data
  T            retrain model on updated dataset
  Q / Esc      quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .capture import RealSenseStreamer, next_sample_id, save_labeled_sample
from .grasp_point_net import default_model_path, load_grasp_model, predict_grasp_uv, train_grasp_model
from .label_grasp import PANEL_H, _require_cv2_gui


def _draw_review_ui(
    cv2,
    frame_rgb: np.ndarray,
    proposed_uv: tuple[int, int] | None,
    manual_uv: tuple[int, int] | None,
    *,
    saved_count: int,
    status: str,
    mode: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    if proposed_uv is not None and manual_uv is None:
        u, v = proposed_uv
        cv2.drawMarker(bgr, (u, v), (255, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(bgr, (u, v), 12, (255, 255, 0), 2)
        cv2.putText(
            bgr,
            f"proposed ({u}, {v})",
            (min(u + 14, w - 200), max(v - 14, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if manual_uv is not None:
        u, v = manual_uv
        cv2.drawMarker(bgr, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(bgr, (u, v), 12, (0, 0, 255), 2)
        cv2.putText(
            bgr,
            f"manual ({u}, {v})",
            (min(u + 14, w - 180), max(v + 18, 36)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    lines = [
        "Y/Enter=confirm NN  |  Click+S=manual save  |  R=reset  |  T=retrain  |  Q=quit",
        f"Mode: {mode}  |  Saved: {saved_count}  |  {status}",
        "Cyan=network proposal. Red=manual override after reset.",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (10, 28 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 if i == 0 else 0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([panel, bgr])


def run_review_gui(
    *,
    data_dir: Path,
    model_path: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    serial: str = "",
    retrain_epochs: int = 120,
) -> int:
    cv2 = _require_cv2_gui()
    data_dir = data_dir.resolve()
    model_path = model_path.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    model, meta = load_grasp_model(model_path)
    print(f"Loaded model from {model_path}")
    if meta.get("best_val_px_error") is not None:
        print(f"  val error ~ {meta['best_val_px_error']:.1f}px")

    state = {
        "proposed_uv": None,
        "manual_uv": None,
        "mode": "propose",
        "saved_count": len(list((data_dir / "labels").glob("*.json")))
        if (data_dir / "labels").is_dir()
        else 0,
        "status": "Live NN proposal — Y to confirm, R to reset",
        "predict_every": 3,
        "frame_idx": 0,
    }

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            state["manual_uv"] = (int(x), int(y - PANEL_H))
            state["mode"] = "manual"
            state["status"] = f"Manual grasp {state['manual_uv']} — S to save, R to reset"

    window = "Grasp review (NN proposal)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, width, height + PANEL_H)
    cv2.setMouseCallback(window, on_mouse)

    streamer = RealSenseStreamer(width=width, height=height, fps=fps, serial=serial, warmup=15)
    print(f"Camera live. Dataset: {data_dir}")
    streamer.start()

    def save_grasp(frame, grasp_uv: tuple[int, int], note: str) -> None:
        sid = next_sample_id(data_dir)
        label_path = save_labeled_sample(data_dir, sid, frame, grasp_uv, note=note)
        state["saved_count"] += 1
        state["proposed_uv"] = None
        state["manual_uv"] = None
        state["mode"] = "propose"
        state["status"] = f"Saved {sid} — adjust scene; new proposal follows"
        print(f"Saved {label_path} ({note})")

    def retrain() -> None:
        state["status"] = "Retraining…"
        display = _draw_review_ui(
            cv2,
            frame.color_rgb,
            state["proposed_uv"],
            state["manual_uv"],
            saved_count=state["saved_count"],
            status=state["status"],
            mode=state["mode"],
        )
        cv2.imshow(window, display)
        cv2.waitKey(1)
        result = train_grasp_model(data_dir, model_path=model_path, epochs=retrain_epochs)
        nonlocal model
        model, meta = load_grasp_model(model_path)
        state["status"] = f"Retrained — val err ~ {result.best_val_px_error:.1f}px"
        print(state["status"])

    try:
        while True:
            frame = streamer.read()
            state["frame_idx"] += 1

            if state["mode"] == "propose" and state["frame_idx"] % state["predict_every"] == 0:
                state["proposed_uv"] = predict_grasp_uv(model, frame.color_rgb)

            display = _draw_review_ui(
                cv2,
                frame.color_rgb,
                state["proposed_uv"],
                state["manual_uv"],
                saved_count=state["saved_count"],
                status=state["status"],
                mode=state["mode"],
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                state["proposed_uv"] = None
                state["manual_uv"] = None
                state["mode"] = "manual"
                state["status"] = "Reset — click grasp point, S to save more data"
            if key in (ord("t"), ord("T")):
                retrain()
            if key in (ord("y"), ord("Y"), 13):
                if state["mode"] != "propose" or state["proposed_uv"] is None:
                    state["status"] = "Nothing to confirm — wait for cyan proposal or press R"
                    continue
                save_grasp(frame, state["proposed_uv"], note="confirmed_nn")
            if key in (ord("s"), ord("S")):
                if state["manual_uv"] is None:
                    state["status"] = "Click a grasp point first (after R), then S"
                    continue
                save_grasp(frame, state["manual_uv"], note="manual_label")

    finally:
        streamer.stop()
        cv2.destroyAllWindows()

    print(f"Done. {state['saved_count']} samples in {data_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent / "dataset" / "grasp_keypoints"
    p = argparse.ArgumentParser(description="Review NN grasp proposals on live D415 feed.")
    p.add_argument("--data-dir", type=Path, default=default_data)
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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    model_path = (args.model_path or default_model_path(data_dir)).resolve()

    if args.train_first or not model_path.is_file():
        print("Training model before review…")
        train_grasp_model(data_dir, model_path=model_path, epochs=args.retrain_epochs)

    if not model_path.is_file():
        print(f"No model at {model_path}. Collect labels and train first.")
        return 1

    return run_review_gui(
        data_dir=data_dir,
        model_path=model_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial.strip(),
        retrain_epochs=args.retrain_epochs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
