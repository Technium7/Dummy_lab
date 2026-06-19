#!/usr/bin/env python3
"""
Interactive grasp + sensor-base labeling for D415 RGB (live preview).

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.live_label
  python -m high_level_jobs.subtasks.grab_sensor.detection.live_label --out-dir dataset/grasp_keypoints

Keys:
  G                 grasp mode — left-click grasp point (red)
  B                 base mode — left-drag ellipse box; drag handles to adjust
  S / Enter         save frame + labels (both required by default)
  C                 clear grasp + base ellipse
  D                 delete last saved sample
  Q / Esc           quit

Requires: pyrealsense2, opencv-python (with GUI — not headless)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from .base_ellipse import draw_label_grasp
from .capture import RealSenseStreamer, next_sample_id, save_labeled_sample
from .cv2_gui import PANEL_H, init_cv2_window, require_cv2_gui
from .ellipse_editor import EllipseEditor, draw_ellipse_editor

LabelMode = Literal["grasp", "base"]


def _require_cv2_gui():
    return require_cv2_gui()


def _mode_label(mode: LabelMode) -> str:
    return "GRASP" if mode == "grasp" else "BASE"


def _draw_ui(
    cv2,
    frame_rgb: np.ndarray,
    grasp_uv: tuple[int, int] | None,
    editor: EllipseEditor,
    *,
    mode: LabelMode,
    saved_count: int,
    status: str,
    require_base: bool,
) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, w = bgr.shape[:2]

    if grasp_uv is not None:
        draw_label_grasp(bgr, grasp_uv[0], grasp_uv[1])

    draw_ellipse_editor(bgr, editor.ellipse, preview_ellipse=editor.preview_ellipse)

    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    mode_color = (0, 120, 255) if mode == "grasp" else (0, 220, 0)
    cv2.putText(
        panel,
        f"Mode: {_mode_label(mode)}",
        (w - 140, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        mode_color,
        1,
        cv2.LINE_AA,
    )
    base_hint = " + base ellipse" if require_base else ""
    lines = [
        "G=grasp mode (left-click)  B=base mode (left-drag)  S=save  C=clear  D=undo  Q=quit",
        f"Saved: {saved_count}  |  {status}",
        f"Label grasp{base_hint}: press G or B to switch mode, then click/drag on image.",
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


def _delete_last_sample(out_dir: Path) -> str:
    labels_dir = out_dir / "labels"
    if not labels_dir.is_dir():
        return "Nothing to delete"
    labels = sorted(labels_dir.glob("*.json"))
    if not labels:
        return "Nothing to delete"

    last = labels[-1]
    data = json.loads(last.read_text(encoding="utf-8"))
    sid = data.get("id", last.stem)

    for key in ("image", "depth"):
        rel = data.get("files", {}).get(key)
        if rel:
            p = out_dir / rel
            if p.is_file():
                p.unlink()
    preview = out_dir / "previews" / f"{sid}.png"
    if preview.is_file():
        preview.unlink()
    last.unlink()

    manifest = out_dir / "manifest.jsonl"
    if manifest.is_file():
        lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            try:
                if json.loads(lines[-1]).get("id") == sid:
                    lines.pop()
            except json.JSONDecodeError:
                pass
        manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return f"Deleted sample {sid}"


def run_live_label_gui(
    *,
    out_dir: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    serial: str = "",
    require_base: bool = True,
) -> int:
    cv2 = _require_cv2_gui()
    out_dir.mkdir(parents=True, exist_ok=True)

    editor = EllipseEditor()
    state: dict = {
        "mode": "grasp",
        "grasp_uv": None,
        "saved_count": len(list((out_dir / "labels").glob("*.json"))) if (out_dir / "labels").is_dir() else 0,
        "status": "Grasp mode — left-click grasp point (press B for base ellipse)",
    }

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H:
            return
        uv = (int(x), int(y - PANEL_H))
        mode: LabelMode = state["mode"]

        if mode == "grasp":
            if event == cv2.EVENT_LBUTTONDOWN:
                state["grasp_uv"] = uv
                state["status"] = f"Grasp ({uv[0]}, {uv[1]}) — press B to label base or S to save"
            return

        msg = editor.on_mouse(event, uv, cv2)
        if msg:
            state["status"] = msg

    window = "Live label (grasp + base)"
    init_cv2_window(cv2, window, width, height + PANEL_H, on_mouse)

    streamer = RealSenseStreamer(width=width, height=height, fps=fps, serial=serial, warmup=5)
    print(f"Opening camera… Saving to {out_dir.resolve()}", flush=True)
    streamer.start()
    print("Camera ready.", flush=True)

    try:
        while True:
            frame = streamer.read(with_depth=False)
            display = _draw_ui(
                cv2,
                frame.color_rgb,
                state["grasp_uv"],
                editor,
                mode=state["mode"],
                saved_count=state["saved_count"],
                status=state["status"],
                require_base=require_base,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("g"), ord("G")):
                state["mode"] = "grasp"
                state["status"] = "Grasp mode — left-click grasp point"
            if key in (ord("b"), ord("B")):
                state["mode"] = "base"
                state["status"] = "Base mode — left-drag box over counter, adjust handles"
            if key in (ord("c"), ord("C")):
                state["grasp_uv"] = None
                editor.clear()
                state["status"] = "Cleared — press G or B to label"
            if key in (ord("d"), ord("D")):
                msg = _delete_last_sample(out_dir)
                state["saved_count"] = len(list((out_dir / "labels").glob("*.json")))
                state["status"] = msg
                print(msg)
            if key in (ord("s"), ord("S"), 13):
                if state["grasp_uv"] is None:
                    state["status"] = "Press G, then left-click grasp point before saving"
                    continue
                if require_base and not editor.is_valid():
                    state["status"] = "Press B, left-drag ellipse over counter, then adjust handles"
                    continue
                state["status"] = "Saving (capturing depth)…"
                display = _draw_ui(
                    cv2,
                    frame.color_rgb,
                    state["grasp_uv"],
                    editor,
                    mode=state["mode"],
                    saved_count=state["saved_count"],
                    status=state["status"],
                    require_base=require_base,
                )
                cv2.imshow(window, display)
                cv2.waitKey(1)
                save_frame = streamer.read(with_depth=True)
                sid = next_sample_id(out_dir)
                label_path = save_labeled_sample(
                    out_dir,
                    sid,
                    save_frame,
                    state["grasp_uv"],
                    base_ellipse=editor.ellipse,
                )
                state["saved_count"] += 1
                state["grasp_uv"] = None
                editor.clear()
                state["mode"] = "grasp"
                state["status"] = f"Saved {sid} — press G to label next grasp"
                print(f"Saved {label_path}")

    finally:
        streamer.stop()
        cv2.destroyAllWindows()

    print(f"Done. {state['saved_count']} samples in {out_dir.resolve()}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_out = Path(__file__).resolve().parent / "dataset" / "grasp_keypoints"
    p = argparse.ArgumentParser(description="Live D415 GUI: label grasp + sensor base and save training samples.")
    p.add_argument("--out-dir", type=Path, default=default_out, help="Dataset output directory")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default="", help="RealSense serial (default: first device)")
    p.add_argument(
        "--grasp-only",
        action="store_true",
        help="Save grasp point only (skip base label requirement)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_live_label_gui(
        out_dir=args.out_dir.resolve(),
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial.strip(),
        require_base=not args.grasp_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
