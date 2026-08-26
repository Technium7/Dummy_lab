#!/usr/bin/env python3
"""Live detection GUI with optional correct-and-save into the dataset.

Shows YOLO + keypoint-head results on the live D415 stream. Freeze a frame to
label like live_label / relabel (sensor: contact + along holder) and save.

Usage:
  python -m high_level_jobs.grab_sensor.vision.live_detect
  python -m high_level_jobs.grab_sensor.vision.live_detect --serial <D415>

Keys:
  Live mode
    Space           freeze: keep NN posts/other; clear sensors for manual 2-pt label
    A               accept/save NN detections as a labeled sample (no edit)
    Q / Esc         quit

  Frozen (edit) mode — same as relabel for sensors
    1 / 2 / 3       class = sensor / post / other
    Left-click      sensor: contact then along holder; post/other: center
    N               finish current object
    E               drop sensor objects only (keep posts)
    R / C           clear all objects on this frame (stay frozen)
    S / Enter       save corrected labels for this frozen frame
    X               delete last saved sample
    Space           resume live without saving
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from camera.realsense import RealSenseStreamer  # noqa: E402
from high_level_jobs.grab_sensor.config_loader import (  # noqa: E402
    apply_runtime_calibrations,
    default_config_path,
    load_grab_sensor_config,
)
from high_level_jobs.grab_sensor.paths import DATASET_DIR  # noqa: E402
from high_level_jobs.grab_sensor.vision.dataset_io import (  # noqa: E402
    bbox_from_keypoints,
    delete_sample,
    iter_labels,
    save_labeled_sample,
)
from high_level_jobs.grab_sensor.vision.infer import DetectedObject, infer_scene  # noqa: E402
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


def _det_to_label_obj(det: DetectedObject, *, width: int, height: int) -> dict:
    cls = det.class_name
    kpts = [[float(p[0]), float(p[1])] for p in (det.kpts or [])]
    if cls == "sensor" and len(kpts) >= SENSOR_NUM_KPTS:
        bbox = list(det.bbox_xyxy) if det.bbox_xyxy else bbox_from_keypoints(
            kpts[:SENSOR_NUM_KPTS], class_name="sensor", width=width, height=height
        )
        return {
            "class": "sensor",
            "bbox_xyxy": [float(v) for v in bbox],
            "kpts": kpts[:SENSOR_NUM_KPTS],
            "axis_deg": float(
                det.axis_deg
                if det.axis_deg is not None
                else axis_deg_from_sensor_kpts(kpts[:SENSOR_NUM_KPTS], bbox_xyxy=bbox)
            ),
            "bbox_source": "nn",
            "score": float(det.score),
        }
    if not kpts and det.center_uv is not None:
        kpts = [[float(det.center_uv[0]), float(det.center_uv[1])]]
    if not kpts:
        # fall back to bbox center
        b = det.bbox_xyxy
        kpts = [[(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5]]
    bbox = list(det.bbox_xyxy) if det.bbox_xyxy else bbox_from_keypoints(
        kpts[:1], class_name=cls, width=width, height=height
    )
    return {
        "class": cls,
        "bbox_xyxy": [float(v) for v in bbox],
        "kpts": kpts[:1] if cls != "sensor" else kpts,
        "bbox_source": "nn",
        "score": float(det.score),
    }


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
            "bbox_source": "manual",
        }
    else:
        if len(kpts) < 1:
            return f"{cls} needs 1 center keypoint"
        kpt = [[float(kpts[0][0]), float(kpts[0][1])]]
        obj = {
            "class": cls,
            "bbox_xyxy": bbox_from_keypoints(kpt, class_name=cls, width=width, height=height),
            "kpts": kpt,
            "bbox_source": "manual",
        }
    state["objects"].append(obj)
    state["current"] = {"class": state["class"], "kpts": []}
    return f"Added {cls} (total {len(state['objects'])})"


def _draw_ui(
    cv2,
    rgb: np.ndarray,
    *,
    objects: list[dict],
    current: dict | None,
    class_name: str,
    frozen: bool,
    status: str,
    saved_count: int,
    infer_ms: float | None,
    err: str | None,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    for obj in objects:
        _draw_object(cv2, bgr, obj, show_holder_pt=False, show_grasp_pt=False)
    if current and current.get("kpts"):
        tmp = {
            "class": current.get("class") or class_name,
            "kpts": list(current["kpts"]),
        }
        need = SENSOR_NUM_KPTS if tmp["class"] == "sensor" else 1
        if len(tmp["kpts"]) >= need:
            try:
                tmp["bbox_xyxy"] = bbox_from_keypoints(
                    tmp["kpts"][:need], class_name=tmp["class"], width=w, height=h
                )
            except ValueError:
                pass
        _draw_object(cv2, bgr, tmp, show_holder_pt=False, show_grasp_pt=False)

    mode = "FROZEN-edit" if frozen else "LIVE"
    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (50, 30, 30) if frozen else (30, 45, 30)
    ms = f"{infer_ms:.0f}ms" if infer_ms is not None else "-"
    line0 = (
        "Space=freeze+seed NN | A=accept-save NN | Q=quit"
        if not frozen
        else "1=sensor  E=drop sensors  click contact+holder  N=finish  R=clear  S=save  Space=live"
    )
    line1 = f"{mode}  class={class_name}  objs={len(objects)}  saved={saved_count}  infer={ms}  |  {status}"
    if err:
        line1 = f"{mode}  ERROR: {err[:70]}"
    for i, line in enumerate((line0, line1)):
        cv2.putText(panel, line, (8, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    return np.vstack([panel, bgr])


def run_live_detect(
    *,
    out_dir: Path,
    serial: str = "",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    config_path: Path | None = None,
    infer_every: int = 1,
) -> int:
    cv2 = _require_cv2()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = apply_runtime_calibrations(load_grab_sensor_config(config_path or default_config_path()))

    state: dict = {
        "frozen": False,
        "class": "sensor",
        "objects": [],
        "current": {"class": "sensor", "kpts": []},
        "status": "Live detection — Space to freeze & edit, A to accept-save",
        "saved_count": len(iter_labels(out_dir)),
        "freeze_rgb": None,
        "freeze_depth": None,
        "freeze_intr": {},
        "live_objects": [],
        "infer_ms": None,
        "err": None,
        "frame_i": 0,
    }

    def on_mouse(event, x, y, _flags, _param):
        if not state["frozen"] or y < PANEL_H or event != cv2.EVENT_LBUTTONDOWN:
            return
        uy, ux = int(y - PANEL_H), int(x)
        pt = [float(ux), float(uy)]
        state["current"]["class"] = state["class"]
        kpts = list(state["current"].get("kpts") or [])
        cls = state["class"]
        if cls == "sensor":
            if len(kpts) >= SENSOR_NUM_KPTS:
                state["status"] = "Already 2 sensor kpts — press N"
                return
            kpts.append(pt)
            state["current"]["kpts"] = kpts
            state["status"] = _sensor_click_status(len(kpts), kpts)
        else:
            state["current"]["kpts"] = [pt]
            state["status"] = f"{cls} center set — N to finish"

    window = "grab_sensor live_detect"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, width, height + PANEL_H)
    cv2.setMouseCallback(window, on_mouse)

    streamer = RealSenseStreamer(width=width, height=height, fps=fps, serial=serial, warmup=8)
    print(f"Opening D415… dataset={out_dir.resolve()}", flush=True)
    streamer.start()
    print("Live detect ready.", flush=True)

    try:
        while True:
            if not state["frozen"]:
                frame = streamer.read(with_depth=True)
                rgb = frame.color_rgb
                depth = frame.depth_m
                h, w = rgb.shape[:2]
                state["frame_i"] += 1
                if state["frame_i"] % max(1, infer_every) == 0:
                    t0 = time.perf_counter()
                    try:
                        dets = infer_scene(rgb, depth, cfg)
                        state["live_objects"] = [
                            _det_to_label_obj(d, width=w, height=h) for d in dets
                        ]
                        state["err"] = None
                    except Exception as exc:
                        state["live_objects"] = []
                        state["err"] = str(exc)
                    state["infer_ms"] = (time.perf_counter() - t0) * 1000.0
                objects = state["live_objects"]
                current = None
            else:
                rgb = state["freeze_rgb"]
                depth = state["freeze_depth"]
                h, w = rgb.shape[:2]
                objects = state["objects"]
                current = state["current"]

            display = _draw_ui(
                cv2,
                rgb,
                objects=objects,
                current=current,
                class_name=state["class"],
                frozen=state["frozen"],
                status=state["status"],
                saved_count=state["saved_count"],
                infer_ms=state["infer_ms"],
                err=state["err"],
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (255, 0xFF):
                continue

            if key in (ord("q"), ord("Q"), 27):
                break

            if not state["frozen"]:
                if key == ord(" "):
                    frame = streamer.read(with_depth=True)
                    rgb = frame.color_rgb
                    depth = frame.depth_m
                    h, w = rgb.shape[:2]
                    try:
                        dets = infer_scene(rgb, depth, cfg)
                        seeded = [_det_to_label_obj(d, width=w, height=h) for d in dets]
                        state["err"] = None
                    except Exception as exc:
                        seeded = []
                        state["err"] = str(exc)
                    state["frozen"] = True
                    state["freeze_rgb"] = rgb.copy()
                    state["freeze_depth"] = depth.copy()
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
                    state["freeze_intr"] = intr
                    # Keep NN posts/other; sensors labeled manually like live_label/relabel
                    kept = [
                        o
                        for o in copy.deepcopy(seeded)
                        if str(o.get("class", "")).lower() != "sensor"
                    ]
                    n_drop = len(seeded) - len(kept)
                    state["objects"] = kept
                    state["class"] = "sensor"
                    state["current"] = {"class": "sensor", "kpts": []}
                    state["status"] = (
                        f"Frozen — kept {len(kept)} non-sensor (dropped {n_drop} NN sensor). "
                        "Click contact, then along holder, N, S"
                    )
                elif key in (ord("a"), ord("A")):
                    h, w = rgb.shape[:2]
                    objs = list(state["live_objects"])
                    if not objs:
                        state["status"] = "No NN detections to accept"
                        continue
                    rec = save_labeled_sample(
                        out_dir,
                        color_rgb=rgb,
                        depth_m=depth,
                        objects=objs,
                        intrinsics={},
                    )
                    prev = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    for obj in objs:
                        _draw_object(cv2, prev, obj, show_holder_pt=False, show_grasp_pt=False)
                    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_dir / "previews" / f"{rec['id']}.png"), prev)
                    state["saved_count"] = len(iter_labels(out_dir))
                    state["status"] = f"Accepted NN sample {rec['id']}"
                continue

            # ---- frozen edit mode ----
            h, w = state["freeze_rgb"].shape[:2]
            if key == ord(" "):
                state["frozen"] = False
                state["objects"] = []
                state["current"] = {"class": state["class"], "kpts": []}
                state["status"] = "Resumed live"
            elif key in (ord("e"), ord("E")):
                kept = [
                    o
                    for o in state["objects"]
                    if str(o.get("class", "")).lower() != "sensor"
                ]
                n_drop = len(state["objects"]) - len(kept)
                state["objects"] = kept
                state["class"] = "sensor"
                state["current"] = {"class": "sensor", "kpts": []}
                state["status"] = (
                    f"Dropped {n_drop} sensor(s), kept {len(kept)} — "
                    "click contact then along holder, N, S"
                )
            elif key in (ord("r"), ord("R"), ord("c"), ord("C")):
                state["objects"] = []
                state["class"] = "sensor"
                state["current"] = {"class": "sensor", "kpts": []}
                state["status"] = (
                    "Cleared frame — click contact then along holder, N, S (Space=live)"
                )
            elif key == ord("1"):
                state["class"] = "sensor"
                state["current"] = {"class": "sensor", "kpts": []}
                state["status"] = "Class=sensor — contact then along holder, then N"
            elif key == ord("2"):
                state["class"] = "post"
                state["current"] = {"class": "post", "kpts": []}
                state["status"] = "Class=post — click center, then N"
            elif key == ord("3"):
                state["class"] = "other"
                state["current"] = {"class": "other", "kpts": []}
                state["status"] = "Class=other — click center, then N"
            elif key in (ord("n"), ord("N")):
                state["status"] = _finish_object(state, width=w, height=h)
            elif key in (ord("s"), ord("S"), 13):
                kpts = list(state["current"].get("kpts") or [])
                need = SENSOR_NUM_KPTS if state["class"] == "sensor" else 1
                if len(kpts) >= need:
                    msg = _finish_object(state, width=w, height=h)
                    if "needs" in msg:
                        state["status"] = msg
                        continue
                if not state["objects"]:
                    state["status"] = "No objects to save"
                    continue
                rec = save_labeled_sample(
                    out_dir,
                    color_rgb=state["freeze_rgb"],
                    depth_m=state["freeze_depth"],
                    objects=list(state["objects"]),
                    intrinsics=state.get("freeze_intr") or {},
                )
                prev = cv2.cvtColor(state["freeze_rgb"], cv2.COLOR_RGB2BGR)
                for obj in state["objects"]:
                    _draw_object(cv2, prev, obj, show_holder_pt=False, show_grasp_pt=False)
                (out_dir / "previews").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / "previews" / f"{rec['id']}.png"), prev)
                state["saved_count"] = len(iter_labels(out_dir))
                state["frozen"] = False
                state["objects"] = []
                state["current"] = {"class": state["class"], "kpts": []}
                state["status"] = f"Saved {rec['id']} — live again"
            elif key in (ord("x"), ord("X")):
                labels = iter_labels(out_dir)
                if labels:
                    state["status"] = delete_sample(out_dir, labels[-1])
                    state["saved_count"] = len(iter_labels(out_dir))
                else:
                    state["status"] = "Nothing to delete"
    finally:
        streamer.stop()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live detect + correct/save grab_sensor labels")
    p.add_argument("--out", type=Path, default=DATASET_DIR)
    p.add_argument("--serial", type=str, default="")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--infer-every", type=int, default=1, help="Run NN every N frames")
    args = p.parse_args(argv)
    return run_live_detect(
        out_dir=args.out,
        serial=args.serial.strip(),
        width=args.width,
        height=args.height,
        config_path=args.config,
        infer_every=args.infer_every,
    )


if __name__ == "__main__":
    raise SystemExit(main())
