#!/usr/bin/env python3
"""Interactive keypoint labeling for top-view D415 (Windows grab_sensor).

No manual boxes — YOLO boxes are auto-built from keypoints at save/export time.

Classes:
  sensor  — 2 keypoints: (0) holder–sensor contact, (1) direction along holder
            (toward free space / grasp). Grasp = contact + fixed mm along holder.
  post    — 1 center keypoint
  other   — 1 center keypoint

Keys:
  1/2/3           class = sensor / post / other
  Left-click      add keypoint
  N               finish current object
  S / Enter       save frame
  C               clear
  D               delete last saved sample
  Q / Esc         quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from camera.realsense import RealSenseStreamer  # noqa: E402
from high_level_jobs.grab_sensor.paths import DATASET_DIR  # noqa: E402
from high_level_jobs.grab_sensor.vision.dataset_io import (  # noqa: E402
    bbox_from_keypoints,
    delete_sample,
    iter_labels,
    save_labeled_sample,
)
from high_level_jobs.grab_sensor.vision.sensor_geometry import (  # noqa: E402
    DEFAULT_GRASP_OFFSET_PX,
    SENSOR_CONTACT_IDX,
    SENSOR_NUM_KPTS,
    axis_deg_from_sensor_kpts,
    contact_axis_geometry,
    grasp_uv_from_sensor_kpts,
)

PANEL_H = 96
CLASS_COLORS = {
    "sensor": (0, 200, 255),
    "post": (80, 200, 80),
    "other": (200, 160, 40),
}


def _require_cv2():
    import cv2

    return cv2


def axis_deg_from_kpts_compat(kpts: list) -> float:
    return axis_deg_from_sensor_kpts(kpts)


def _sensor_need() -> int:
    return SENSOR_NUM_KPTS


def _draw_object(
    cv2,
    bgr: np.ndarray,
    obj: dict,
    *,
    show_holder_pt: bool = True,
    show_grasp_pt: bool = True,
) -> None:
    cls = obj.get("class", "other")
    color = CLASS_COLORS.get(cls, (200, 200, 200))
    bbox = obj.get("bbox_xyxy")
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 1)
        cv2.putText(bgr, cls, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    kpts = obj.get("kpts") or []
    for i, uv in enumerate(kpts):
        if cls == "sensor" and i != SENSOR_CONTACT_IDX and not show_holder_pt:
            continue
        u, v = int(uv[0]), int(uv[1])
        marker = (
            cv2.MARKER_TILTED_CROSS
            if (cls == "sensor" and i == SENSOR_CONTACT_IDX)
            else cv2.MARKER_CROSS
        )
        cv2.drawMarker(bgr, (u, v), color, marker, 16, 2)
        if cls == "sensor" and not show_holder_pt:
            continue
        label = "C" if (cls == "sensor" and i == 0) else ("H" if cls == "sensor" and i == 1 else str(i))
        cv2.putText(bgr, label, (u + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if cls == "sensor" and len(kpts) >= SENSOR_NUM_KPTS:
        geo = contact_axis_geometry(kpts)
        cv2.arrowedLine(
            bgr,
            (int(geo.contact_uv[0]), int(geo.contact_uv[1])),
            (int(geo.dir_uv[0]), int(geo.dir_uv[1])),
            (0, 0, 255),
            2,
            tipLength=0.25,
        )
        if show_grasp_pt:
            gx, gy = grasp_uv_from_sensor_kpts(
                kpts, offset_px=DEFAULT_GRASP_OFFSET_PX, bbox_xyxy=bbox
            )
            cv2.circle(bgr, (int(gx), int(gy)), 7, (0, 0, 255), 2)


def _draw_ui(cv2, frame_rgb: np.ndarray, state: dict) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    for obj in state["objects"]:
        _draw_object(cv2, bgr, obj)
    cur = state["current"]
    if cur.get("kpts"):
        tmp = dict(cur)
        kpts = cur.get("kpts") or []
        need = _sensor_need() if state["class"] == "sensor" else 1
        if len(kpts) >= need:
            try:
                tmp["bbox_xyxy"] = bbox_from_keypoints(
                    kpts[:need], class_name=state["class"], width=w, height=h
                )
            except ValueError:
                pass
        _draw_object(cv2, bgr, tmp)

    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    cls = state["class"]
    lines = [
        "1=sensor 2=post 3=other | sensor: contact then along holder | N=next | S=save | Q=quit",
        f"Class={cls}  objects={len(state['objects'])}  saved={state['saved_count']}  |  {state['status']}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (8, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    return np.vstack([panel, bgr])


def _finish_current(state: dict, *, frame_wh: tuple[int, int]) -> str:
    cur = state["current"]
    cls = cur.get("class") or state["class"]
    kpts = cur.get("kpts") or []
    w, h = frame_wh
    if cls == "sensor":
        if len(kpts) < SENSOR_NUM_KPTS:
            return "Sensor needs 2 keypoints (contact + direction along holder)"
        kpts2 = [[float(p[0]), float(p[1])] for p in kpts[:SENSOR_NUM_KPTS]]
        bbox = bbox_from_keypoints(kpts2, class_name="sensor", width=w, height=h)
        obj = {
            "class": "sensor",
            "bbox_xyxy": bbox,
            "kpts": kpts2,
            "axis_deg": axis_deg_from_sensor_kpts(kpts2),
            "bbox_source": "auto_from_kpts",
        }
    else:
        if len(kpts) < 1:
            return f"{cls} needs 1 center keypoint"
        kpt = [[float(kpts[0][0]), float(kpts[0][1])]]
        obj = {
            "class": cls,
            "bbox_xyxy": bbox_from_keypoints(kpt, class_name=cls, width=w, height=h),
            "kpts": kpt,
            "bbox_source": "auto_from_kpts",
        }
    state["objects"].append(obj)
    state["current"] = {"class": state["class"], "kpts": []}
    return f"Added {cls} (total {len(state['objects'])})"


def _sensor_click_status(n: int, kpts: list, *, bbox: list | None = None) -> str:
    _ = bbox
    if n == 1:
        return "Contact set — click along holder toward grasp/free space (2/2)"
    ang = axis_deg_from_sensor_kpts(kpts)
    gx, gy = grasp_uv_from_sensor_kpts(kpts, offset_px=DEFAULT_GRASP_OFFSET_PX)
    return f"Sensor contact+holder axis={ang:.1f} deg grasp~({gx:.0f},{gy:.0f}) — N to finish"


def run_live_label(
    *,
    out_dir: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    serial: str = "",
) -> int:
    cv2 = _require_cv2()
    out_dir.mkdir(parents=True, exist_ok=True)
    state: dict = {
        "class": "sensor",
        "objects": [],
        "current": {"class": "sensor", "kpts": []},
        "saved_count": len(iter_labels(out_dir)),
        "status": "Sensor: click contact, then along holder (toward grasp)",
        "frame_wh": (width, height),
    }

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H:
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        uy, ux = int(y - PANEL_H), int(x)
        pt = (float(ux), float(uy))
        kpts = state["current"].setdefault("kpts", [])
        cls = state["class"]
        state["current"]["class"] = cls
        if cls == "sensor":
            if len(kpts) < SENSOR_NUM_KPTS:
                kpts.append(list(pt))
                state["status"] = _sensor_click_status(len(kpts), kpts)
            else:
                state["status"] = "Already 2 sensor kpts — press N"
        else:
            state["current"]["kpts"] = [list(pt)]
            state["status"] = f"{cls} center set — N to finish"

    window = "grab_sensor live_label"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, width, height + PANEL_H)
    cv2.setMouseCallback(window, on_mouse)

    streamer = RealSenseStreamer(width=width, height=height, fps=fps, serial=serial, warmup=5)
    print(f"Opening D415… Saving to {out_dir.resolve()}", flush=True)
    streamer.start()
    print("Camera ready. Sensor = contact + direction along holder.", flush=True)

    try:
        while True:
            frame = streamer.read(with_depth=True)
            h, w = frame.color_rgb.shape[:2]
            state["frame_wh"] = (w, h)
            display = _draw_ui(cv2, frame.color_rgb, state)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord("1"):
                state["class"] = "sensor"
                state["current"]["class"] = "sensor"
                state["status"] = "Class=sensor (contact + along holder)"
            elif key == ord("2"):
                state["class"] = "post"
                state["current"]["class"] = "post"
                state["status"] = "Class=post (1 click)"
            elif key == ord("3"):
                state["class"] = "other"
                state["current"]["class"] = "other"
                state["status"] = "Class=other (1 click)"
            elif key in (ord("n"), ord("N")):
                state["status"] = _finish_current(state, frame_wh=state["frame_wh"])
            elif key in (ord("c"), ord("C")):
                state["current"] = {"class": state["class"], "kpts": []}
                state["objects"] = []
                state["status"] = "Cleared"
            elif key in (ord("d"), ord("D")):
                labels = iter_labels(out_dir)
                if labels:
                    state["status"] = delete_sample(out_dir, labels[-1])
                    state["saved_count"] = len(iter_labels(out_dir))
                else:
                    state["status"] = "Nothing to delete"
            elif key in (ord("s"), ord("S"), 13):
                kpts = state["current"].get("kpts") or []
                need = _sensor_need() if state["class"] == "sensor" else 1
                if len(kpts) >= need:
                    msg = _finish_current(state, frame_wh=state["frame_wh"])
                    if msg.startswith("Sensor") or "needs" in msg:
                        state["status"] = msg
                        continue
                if not state["objects"]:
                    state["status"] = "No objects to save"
                    continue
                intr = {}
                if frame.color_intrinsics is not None:
                    intr = {
                        "fx": float(frame.color_intrinsics.fx),
                        "fy": float(frame.color_intrinsics.fy),
                        "ppx": float(frame.color_intrinsics.ppx),
                        "ppy": float(frame.color_intrinsics.ppy),
                        "width": int(frame.color_intrinsics.width),
                        "height": int(frame.color_intrinsics.height),
                    }
                rec = save_labeled_sample(
                    out_dir,
                    color_rgb=frame.color_rgb,
                    depth_m=frame.depth_m,
                    objects=list(state["objects"]),
                    intrinsics=intr,
                )
                prev = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
                for obj in state["objects"]:
                    _draw_object(cv2, prev, obj)
                (out_dir / "previews").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / "previews" / f"{rec['id']}.png"), prev)
                state["objects"] = []
                state["current"] = {"class": state["class"], "kpts": []}
                state["saved_count"] = len(iter_labels(out_dir))
                state["status"] = f"Saved {rec['id']}"
    finally:
        streamer.stop()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Keypoint-only labeling for Windows grab_sensor")
    p.add_argument("--out", type=Path, default=DATASET_DIR)
    p.add_argument("--serial", type=str, default="")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args(argv)
    return run_live_label(
        out_dir=args.out,
        width=args.width,
        height=args.height,
        serial=args.serial.strip(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
