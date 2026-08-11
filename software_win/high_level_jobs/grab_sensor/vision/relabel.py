#!/usr/bin/env python3
"""Offline review + relabel of saved grab_sensor frames (no camera).

Usage:
  python -m high_level_jobs.grab_sensor.vision.relabel
  python -m high_level_jobs.grab_sensor.vision.relabel --start 5

Keys:
  A / D              previous / next frame
  1 / 2 / 3          class = sensor / post / other
  Left-click         sensor (default): update CONTACT only on nearest 2-pt sensor
                     (keeps holder-dir tip). Or full 2-click draft after E/R / F.
                     post/other: center
  F                  toggle full 2-click sensor draft (add new / both clicks)
  N                  finish current draft object
  R / C              clear *all* objects on this frame
  E                  clear *sensor* objects only (keep post/other)
  S / Enter          save overwritten label + preview
  X                  delete this sample from disk
  Q / Esc            quit
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.grab_sensor.paths import DATASET_DIR  # noqa: E402
from high_level_jobs.grab_sensor.vision.dataset_io import (  # noqa: E402
    bbox_from_keypoints,
    delete_sample,
    iter_labels,
    load_label,
    update_sample_objects,
)
from high_level_jobs.grab_sensor.vision.live_label import (  # noqa: E402
    PANEL_H,
    _draw_object,
    _sensor_click_status,
)
from high_level_jobs.grab_sensor.vision.sensor_geometry import (  # noqa: E402
    SENSOR_NUM_KPTS,
    axis_deg_from_sensor_kpts,
)


def _require_cv2():
    import cv2

    return cv2


def _nearest_sensor_idx(objects: list[dict], pt: list[float]) -> int | None:
    best_i: int | None = None
    best_d = 1e18
    for i, obj in enumerate(objects):
        if str(obj.get("class", "")).lower() != "sensor":
            continue
        kpts = obj.get("kpts") or []
        if len(kpts) < SENSOR_NUM_KPTS:
            continue
        c = kpts[0]
        d = (float(c[0]) - pt[0]) ** 2 + (float(c[1]) - pt[1]) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _refresh_sensor_obj(obj: dict, *, width: int, height: int) -> None:
    kpts = [[float(p[0]), float(p[1])] for p in (obj.get("kpts") or [])[:SENSOR_NUM_KPTS]]
    obj["kpts"] = kpts
    obj["bbox_xyxy"] = bbox_from_keypoints(kpts, class_name="sensor", width=width, height=height)
    obj["axis_deg"] = axis_deg_from_sensor_kpts(kpts)
    obj["bbox_source"] = "manual"


def _finish_object(state: dict, *, width: int, height: int) -> str:
    cur = state["current"]
    cls = cur.get("class") or state["class"]
    kpts = list(cur.get("kpts") or [])
    if cls == "sensor":
        if len(kpts) < SENSOR_NUM_KPTS:
            return f"Sensor needs {SENSOR_NUM_KPTS} keypoints (contact + along holder)"
        kpts2 = [[float(p[0]), float(p[1])] for p in kpts[:SENSOR_NUM_KPTS]]
        bbox = bbox_from_keypoints(kpts2, class_name="sensor", width=width, height=height)
        obj = {
            "class": "sensor",
            "bbox_xyxy": bbox,
            "kpts": kpts2,
            "axis_deg": axis_deg_from_sensor_kpts(kpts2, bbox_xyxy=bbox),
            "bbox_source": "auto_from_kpts",
        }
    else:
        if len(kpts) < 1:
            return f"{cls} needs 1 center keypoint"
        kpt = [[float(kpts[0][0]), float(kpts[0][1])]]
        obj = {
            "class": cls,
            "bbox_xyxy": bbox_from_keypoints(kpt, class_name=cls, width=width, height=height),
            "kpts": kpt,
            "bbox_source": "auto_from_kpts",
        }
    state["objects"].append(obj)
    state["current"] = {"class": state["class"], "kpts": []}
    return f"Added {cls} (total {len(state['objects'])})"


def _draw_ui(cv2, rgb: np.ndarray, state: dict, *, idx: int, total: int, sid: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    for obj in state["objects"]:
        _draw_object(cv2, bgr, obj)

    cur = state.get("current") or {}
    kpts = list(cur.get("kpts") or [])
    if kpts:
        tmp = {
            "class": cur.get("class") or state["class"],
            "kpts": kpts,
        }
        need = SENSOR_NUM_KPTS if tmp["class"] == "sensor" else 1
        if len(kpts) >= need:
            try:
                tmp["bbox_xyxy"] = bbox_from_keypoints(
                    kpts[:need], class_name=tmp["class"], width=w, height=h
                )
            except ValueError:
                pass
        _draw_object(cv2, bgr, tmp)

    mode = "FULL-2click" if state.get("full_draft") else "CONTACT-only"
    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (35, 35, 35)
    lines = [
        f"A/D=nav  click={mode}  F=toggle  E=drop sensors  N=finish  S=save  X=del  Q=quit",
        f"[{idx + 1}/{total}] {sid}  class={state['class']}  objs={len(state['objects'])}  "
        f"draft={len(kpts)}  |  {state['status']}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (8, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    return np.vstack([panel, bgr])


def run_relabel(*, data_dir: Path, start_index: int = 0) -> int:
    cv2 = _require_cv2()
    labels = iter_labels(data_dir)
    if not labels:
        print(f"No labels in {data_dir / 'labels'}")
        return 1

    idx = int(np.clip(start_index, 0, len(labels) - 1))
    state: dict = {
        "class": "sensor",
        "objects": [],
        "current": {"class": "sensor", "kpts": []},
        "status": "Click = update CONTACT only (keeps holder dir). F=full 2-click.",
        "dirty": False,
        "frame_wh": (640, 480),
        "full_draft": False,
    }

    def load_idx(i: int) -> tuple[Path, dict, np.ndarray]:
        path = labels[i]
        data = load_label(path)
        img_rel = (data.get("files") or {}).get("image")
        if not img_rel:
            raise FileNotFoundError(f"No image in {path}")
        rgb = np.asarray(Image.open(data_dir / img_rel).convert("RGB"))
        h, w = rgb.shape[:2]
        state["frame_wh"] = (w, h)
        state["objects"] = copy.deepcopy(data.get("objects") or [])
        state["current"] = {"class": state["class"], "kpts": []}
        state["dirty"] = False
        state["full_draft"] = False
        n_s = sum(1 for o in state["objects"] if str(o.get("class", "")).lower() == "sensor")
        state["status"] = (
            f"Loaded {n_s} sensor(s) — click to set CONTACT only, S save "
            "(F=full 2-click / E=drop sensors)"
        )
        return path, data, rgb

    label_path, data, rgb = load_idx(idx)

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H or event != cv2.EVENT_LBUTTONDOWN:
            return
        uy, ux = int(y - PANEL_H), int(x)
        pt = [float(ux), float(uy)]
        w, h = state["frame_wh"]
        if "current" not in state or state["current"] is None:
            state["current"] = {"class": state["class"], "kpts": []}
        state["current"]["class"] = state["class"]
        kpts = list(state["current"].get("kpts") or [])
        cls = state["class"]

        if cls == "sensor" and not state.get("full_draft") and len(kpts) == 0:
            # One-click contact touch-up on nearest existing 2-pt sensor
            si = _nearest_sensor_idx(state["objects"], pt)
            if si is not None:
                obj = state["objects"][si]
                old = list(obj.get("kpts") or [])
                obj["kpts"] = [pt, [float(old[1][0]), float(old[1][1])]]
                _refresh_sensor_obj(obj, width=w, height=h)
                state["dirty"] = True
                state["status"] = (
                    f"Updated CONTACT on sensor[{si}] "
                    f"(kept holder dir) axis={obj['axis_deg']:.1f} — S to save"
                )
                return
            state["status"] = "No 2-pt sensor to update — press E then F for full 2-click, or F"
            return

        if cls == "sensor":
            if len(kpts) >= SENSOR_NUM_KPTS:
                state["status"] = "Already 2 sensor kpts — press N to finish object"
                return
            kpts.append(pt)
            state["current"]["kpts"] = kpts
            state["status"] = _sensor_click_status(len(kpts), kpts)
            state["dirty"] = True
        else:
            state["current"]["kpts"] = [pt]
            state["status"] = f"{cls} center set — N to finish object"
            state["dirty"] = True

    window = "grab_sensor relabel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print(f"Offline relabel: {data_dir.resolve()}  ({len(labels)} samples)")
    print("Default: click updates CONTACT only (keeps holder direction). F = full 2-click.")

    try:
        while True:
            sid = str(data.get("id", label_path.stem))
            display = _draw_ui(cv2, rgb, state, idx=idx, total=len(labels), sid=sid)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 255 or key == 0xFF:
                continue

            w, h = state["frame_wh"]

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("a"), ord("A"), ord("p"), ord("P"), 81):
                idx = (idx - 1) % len(labels)
                label_path, data, rgb = load_idx(idx)
            elif key in (ord("d"), ord("D"), 83):
                idx = (idx + 1) % len(labels)
                label_path, data, rgb = load_idx(idx)
            elif key in (ord("f"), ord("F")):
                state["full_draft"] = not bool(state.get("full_draft"))
                state["current"] = {"class": state["class"], "kpts": []}
                state["status"] = (
                    "FULL 2-click: contact then along holder, N"
                    if state["full_draft"]
                    else "CONTACT-only: click nearest sensor contact, S save"
                )
            elif key == ord("1"):
                state["class"] = "sensor"
                state["current"] = {"class": "sensor", "kpts": []}
                state["status"] = "Class=sensor — click contact (or F for full 2-click)"
            elif key == ord("2"):
                state["class"] = "post"
                state["current"] = {"class": "post", "kpts": []}
                state["full_draft"] = True
                state["status"] = "Class=post — click 1 center, then N"
            elif key == ord("3"):
                state["class"] = "other"
                state["current"] = {"class": "other", "kpts": []}
                state["full_draft"] = True
                state["status"] = "Class=other — click 1 center, then N"
            elif key in (ord("n"), ord("N")):
                state["status"] = _finish_object(state, width=w, height=h)
                state["dirty"] = True
            elif key in (ord("r"), ord("R"), ord("c"), ord("C")):
                state["objects"] = []
                state["current"] = {"class": state["class"], "kpts": []}
                state["full_draft"] = True
                state["dirty"] = True
                state["status"] = "Cleared — FULL mode: contact then along holder, N, S"
            elif key in (ord("e"), ord("E")):
                kept = [o for o in state["objects"] if str(o.get("class", "")).lower() != "sensor"]
                n_drop = len(state["objects"]) - len(kept)
                state["objects"] = kept
                state["class"] = "sensor"
                state["current"] = {"class": "sensor", "kpts": []}
                state["full_draft"] = True
                state["dirty"] = True
                state["status"] = (
                    f"Dropped {n_drop} sensor(s) — FULL: contact then along holder, N, S"
                )
            elif key in (ord("s"), ord("S"), 13):
                kpts = list(state["current"].get("kpts") or [])
                need = SENSOR_NUM_KPTS if state["class"] == "sensor" else 1
                if len(kpts) >= need:
                    msg = _finish_object(state, width=w, height=h)
                    if "needs" in msg:
                        state["status"] = msg
                        continue
                data = update_sample_objects(
                    data_dir, label_path, list(state["objects"]), color_rgb=rgb
                )
                prev = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for obj in state["objects"]:
                    _draw_object(cv2, prev, obj)
                (data_dir / "previews").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(data_dir / "previews" / f"{sid}.png"), prev)
                state["dirty"] = False
                state["status"] = f"Saved {sid} ({len(state['objects'])} objects)"
            elif key in (ord("x"), ord("X")):
                msg = delete_sample(data_dir, label_path)
                labels = iter_labels(data_dir)
                if not labels:
                    print(msg)
                    print("No samples left.")
                    break
                idx = min(idx, len(labels) - 1)
                label_path, data, rgb = load_idx(idx)
                state["status"] = msg
    finally:
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline review/relabel grab_sensor dataset")
    p.add_argument("--out", type=Path, default=DATASET_DIR)
    p.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    args = p.parse_args(argv)
    return run_relabel(data_dir=args.out, start_index=args.start)


if __name__ == "__main__":
    raise SystemExit(main())
