#!/usr/bin/env python3
"""
Relabel elliptical sensor-base counters on saved grasp_keypoints images (offline).

Use this to add or fix base ellipse labels on existing dataset images — no camera.
For new live captures (grasp + base together), use live_label instead.

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.label_base
  python -m high_level_jobs.subtasks.grab_sensor.detection.label_base --data-dir dataset/grasp_keypoints

Keys:
  Left drag         draw ellipse box; drag handles to move/resize/rotate
  S / Enter         save and advance
  N / P             next / previous image
  C                 clear base ellipse
  U                 jump to first sample missing full ellipse
  Q / Esc           quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .base_ellipse import BaseEllipse, draw_label_grasp, parse_label_base_ellipse
from .capture import update_sample_base_label
from .ellipse_editor import EllipseEditor, draw_ellipse_editor
from .grasp_point_net import list_labeled_samples
from .cv2_gui import PANEL_H, init_cv2_window, require_cv2_gui as _require_cv2_gui


def _has_full_ellipse(label_path: Path) -> bool:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    ell = parse_label_base_ellipse(data)
    if ell is None:
        return False
    stored = data.get("base_ellipse") or {}
    return "axis_a_px" in stored and "axis_b_px" in stored


def _load_sample_entries(data_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for sample in list_labeled_samples(data_dir):
        label_path = sample.get("label_path") or (data_dir / "labels" / f"{sample['id']}.json")
        label_path = Path(label_path)
        data = json.loads(label_path.read_text(encoding="utf-8"))
        rgb = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        entries.append(
            {
                "id": sample["id"],
                "label_path": label_path,
                "image_rgb": rgb,
                "grasp_uv": sample["grasp_uv"],
                "base_ellipse": parse_label_base_ellipse(data),
                "has_ellipse": _has_full_ellipse(label_path),
            }
        )
    return entries


def _draw_label_ui(
    cv2,
    frame_rgb: np.ndarray,
    grasp_uv: tuple[int, int],
    editor: EllipseEditor,
    *,
    sample_id: str,
    index: int,
    total: int,
    labeled_count: int,
    status: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, w = bgr.shape[:2]

    gu, gv = grasp_uv
    draw_label_grasp(bgr, gu, gv)
    draw_ellipse_editor(bgr, editor.ellipse, preview_ellipse=editor.preview_ellipse)

    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    lines = [
        "Left-drag ellipse + handles  S=save  N/P=nav  U=needs ellipse  C=clear  Q=quit",
        f"{sample_id}  [{index + 1}/{total}]  full ellipse: {labeled_count}/{total}  |  {status}",
        "Offline relabel. Red=grasp. Green ellipse + box handles = counter base (center shown).",
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


def run_label_base_gui(
    *,
    data_dir: Path,
    start_unlabeled: bool = True,
) -> int:
    cv2 = _require_cv2_gui()
    data_dir = data_dir.resolve()

    entries = _load_sample_entries(data_dir)
    if not entries:
        print(f"No labeled grasp samples found in {data_dir}")
        print("Collect grasp labels first: python -m high_level_jobs.subtasks.grab_sensor.detection.live_label")
        return 1

    idx = 0
    if start_unlabeled:
        for i, entry in enumerate(entries):
            if not entry["has_ellipse"]:
                idx = i
                break

    editor = EllipseEditor()
    state = {
        "idx": idx,
        "status": "Left-drag box over counter, then adjust handles",
    }

    def labeled_count() -> int:
        return sum(1 for e in entries if e["has_ellipse"])

    def refresh_from_entry() -> None:
        entry = entries[state["idx"]]
        ell: BaseEllipse | None = entry.get("base_ellipse")
        if ell is not None and entry["has_ellipse"]:
            editor.set_ellipse(ell)
        else:
            editor.clear()

    refresh_from_entry()

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H:
            return
        uv = (int(x), int(y - PANEL_H))
        msg = editor.on_mouse(event, uv, cv2)
        if msg:
            state["status"] = msg

    window = "Relabel base ellipse (offline)"
    entry0 = entries[idx]
    h, w = entry0["image_rgb"].shape[:2]
    init_cv2_window(cv2, window, w, h + PANEL_H, on_mouse)

    print(f"Offline relabel: {data_dir}")
    print(f"  {len(entries)} samples, {labeled_count()} with full ellipse labels")

    try:
        while True:
            entry = entries[state["idx"]]
            display = _draw_label_ui(
                cv2,
                entry["image_rgb"],
                entry["grasp_uv"],
                editor,
                sample_id=entry["id"],
                index=state["idx"],
                total=len(entries),
                labeled_count=labeled_count(),
                status=state["status"],
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 255:
                continue
            if key in (ord("c"), ord("C")):
                editor.clear()
                state["status"] = "Base ellipse cleared"
            if key in (ord("n"), ord("N")):
                state["idx"] = min(len(entries) - 1, state["idx"] + 1)
                refresh_from_entry()
                state["status"] = f"Sample {entries[state['idx']]['id']}"
            if key in (ord("p"), ord("P")):
                state["idx"] = max(0, state["idx"] - 1)
                refresh_from_entry()
                state["status"] = f"Sample {entries[state['idx']]['id']}"
            if key in (ord("u"), ord("U")):
                for i, e in enumerate(entries):
                    if not e["has_ellipse"]:
                        state["idx"] = i
                        refresh_from_entry()
                        state["status"] = f"Jumped to {e['id']} (needs ellipse)"
                        break
                else:
                    state["status"] = "All samples have full ellipse labels"
            if key in (ord("s"), ord("S"), 13):
                if not editor.is_valid():
                    state["status"] = "Draw ellipse (left-drag) and adjust handles before saving"
                    continue
                label_path = update_sample_base_label(
                    entry["label_path"],
                    base_ellipse=editor.ellipse,
                    data_dir=data_dir,
                )
                entry["base_ellipse"] = editor.ellipse
                entry["has_ellipse"] = True
                state["status"] = f"Saved ellipse for {entry['id']}"
                print(f"Saved {label_path}")
                if state["idx"] < len(entries) - 1:
                    state["idx"] += 1
                    refresh_from_entry()
                    state["status"] = "Relabel next image"
    finally:
        cv2.destroyAllWindows()

    print(f"Done. {labeled_count()}/{len(entries)} samples have full ellipse labels.")
    return 0 if labeled_count() >= 2 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent / "dataset" / "grasp_keypoints"
    p = argparse.ArgumentParser(
        description="Offline relabel: elliptical sensor base on saved grasp dataset images.",
    )
    p.add_argument("--data-dir", type=Path, default=default_data, help="grasp_keypoints dataset directory")
    p.add_argument(
        "--start-first",
        action="store_true",
        help="Start at first sample instead of first missing ellipse",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_label_base_gui(
        data_dir=args.data_dir.resolve(),
        start_unlabeled=not args.start_first,
    )


if __name__ == "__main__":
    raise SystemExit(main())
