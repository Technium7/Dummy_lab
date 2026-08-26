"""Scene inference: YOLO (or cascade) boxes → sensor/center keypoint heads."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from high_level_jobs.grab_sensor.models import GrabSensorConfig, VisionModelsConfig
from high_level_jobs.grab_sensor.vision.crop_features import build_sensor_crop_chw
from high_level_jobs.grab_sensor.vision.dataset_io import CLASS_NAMES
from high_level_jobs.grab_sensor.vision.models import (
    FULL_SIZE,
    CenterHead,
    ClassNet,
    ProposalDetector,
    SensorHead,
)
from high_level_jobs.grab_sensor.vision.sensor_geometry import (
    SENSOR_NUM_KPTS,
    axis_deg_from_sensor_kpts,
    kpts_from_grasp_axis,
)
from high_level_jobs.grab_sensor.vision.temporal_smooth import KeypointEMA
from high_level_jobs.grab_sensor.vision.train_detector import _letterbox
from high_level_jobs.grab_sensor.vision.train_heads import _crop_resize


@dataclass
class DetectedObject:
    class_name: str
    bbox_xyxy: list[float]
    score: float
    kpts: list[list[float]] = field(default_factory=list)
    axis_deg: float | None = None
    center_uv: tuple[float, float] | None = None


def _nms_peaks(heat: np.ndarray, *, max_n: int, thresh: float) -> list[tuple[int, int, float]]:
    h, w = heat.shape
    peaks: list[tuple[int, int, float]] = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            v = float(heat[y, x])
            if v < thresh:
                continue
            if v >= heat[y - 1 : y + 2, x - 1 : x + 2].max():
                peaks.append((x, y, v))
    peaks.sort(key=lambda t: t[2], reverse=True)
    return peaks if max_n <= 0 else peaks[:max_n]


def _load_ckpt(model: torch.nn.Module, path: Path, device: torch.device) -> bool:
    if not path.is_file():
        return False
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        print(f"[warn] skip incompatible checkpoint {path.name}: {exc}")
        return False
    model.to(device).eval()
    return True


@lru_cache(maxsize=4)
def _load_yolo(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("pip install ultralytics") from exc
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {path}\n"
            "Train with: python -m high_level_jobs.grab_sensor.vision.train_yolo"
        )
    return YOLO(str(path))


def _yolo_boxes(
    color_rgb: np.ndarray,
    vision: VisionModelsConfig,
) -> list[tuple[list[float], float, str]]:
    import cv2

    if vision.yolo_model_path is None:
        raise FileNotFoundError("vision.yolo_model_path is not set")
    model = _load_yolo(str(vision.yolo_model_path.resolve()))
    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(
        bgr,
        conf=float(vision.confidence),
        iou=0.45,
        verbose=False,
    )
    if not results:
        return []
    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names or {}
    out: list[tuple[list[float], float, str]] = []
    h, w = color_rgb.shape[:2]
    for box in boxes:
        cls_id = int(box.cls.item())
        name = str(names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else cls_id))
        name = name.lower()
        if name not in CLASS_NAMES:
            continue
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = [
            float(np.clip(x1, 0, w - 1)),
            float(np.clip(y1, 0, h - 1)),
            float(np.clip(x2, 0, w - 1)),
            float(np.clip(y2, 0, h - 1)),
        ]
        out.append((bbox, conf, name))
    out.sort(key=lambda t: t[1], reverse=True)
    limit = int(vision.max_proposals)
    return out if limit <= 0 else out[:limit]


class CascadeSceneDetector:
    def __init__(self, vision: VisionModelsConfig, device: str | None = None) -> None:
        self.vision = vision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.proposal = ProposalDetector()
        self.class_net = ClassNet(pretrained=False)
        self.sensor_head = SensorHead()
        self.center_head = CenterHead()
        self.has_proposal = _load_ckpt(self.proposal, vision.proposal_model_path, self.device)
        self.has_class = _load_ckpt(self.class_net, vision.class_model_path, self.device)
        self.has_sensor = _load_ckpt(self.sensor_head, vision.sensor_head_path, self.device)
        self.has_center = _load_ckpt(self.center_head, vision.center_head_path, self.device)
        self._sensor_ema = KeypointEMA(alpha=0.35)

    def _proposals_from_heatmap(self, rgb: np.ndarray) -> list[tuple[list[float], float]]:
        canvas, scale, pad_x, pad_y = _letterbox(rgb, FULL_SIZE)
        x = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
        with torch.no_grad():
            heat_logits, size_map = self.proposal(x)
            heat = torch.sigmoid(heat_logits)[0, 0].cpu().numpy()
            size_map = size_map[0].cpu().numpy()
        peaks = _nms_peaks(heat, max_n=self.vision.max_proposals, thresh=self.vision.confidence)
        boxes: list[tuple[list[float], float]] = []
        h_img, w_img = rgb.shape[:2]
        for gx, gy, score in peaks:
            log_w = float(size_map[0, gy, gx])
            log_h = float(size_map[1, gy, gx])
            bw = float(np.exp(log_w) * FULL_SIZE) / max(scale, 1e-6)
            bh = float(np.exp(log_h) * FULL_SIZE) / max(scale, 1e-6)
            cx = ((gx + 0.5) * 8.0 - pad_x) / max(scale, 1e-6)
            cy = ((gy + 0.5) * 8.0 - pad_y) / max(scale, 1e-6)
            x0 = float(np.clip(cx - bw * 0.5, 0, w_img - 1))
            y0 = float(np.clip(cy - bh * 0.5, 0, h_img - 1))
            x1 = float(np.clip(cx + bw * 0.5, 0, w_img - 1))
            y1 = float(np.clip(cy + bh * 0.5, 0, h_img - 1))
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                boxes.append(([x0, y0, x1, y1], score))
        return boxes

    def _apply_heads(
        self,
        color_rgb: np.ndarray,
        bbox: list[float],
        class_name: str,
        score: float,
        depth_m: np.ndarray | None = None,
    ) -> DetectedObject:
        obj = DetectedObject(class_name=class_name, bbox_xyxy=list(bbox), score=score)

        if class_name == "sensor" and self.has_sensor:
            # Large YOLO crop — head predicts grasp UV + (cos θ, sin θ)
            chw, origin = build_sensor_crop_chw(
                color_rgb, depth_m, bbox, pad=self.vision.crop_padding_px
            )
            xt = torch.from_numpy(chw)[None].to(self.device)
            x0, y0, x1, y1 = origin
            bw = max(1.0, x1 - x0)
            bh = max(1.0, y1 - y0)
            with torch.no_grad():
                pred = self.sensor_head(xt)[0].cpu().numpy()
            gu = float(np.clip(pred[0], 0, 1)) * bw + x0
            gv = float(np.clip(pred[1], 0, 1)) * bh + y0
            cx, cy = float(pred[2]), float(pred[3])
            n = float(np.hypot(cx, cy))
            if n < 1e-6:
                cx, cy = 1.0, 0.0
            else:
                cx, cy = cx / n, cy / n
            axis_deg = float(np.degrees(np.arctan2(cy, cx)))
            # kpts[0]=contact, kpts[1]=synthetic tip along holder
            kpts = kpts_from_grasp_axis((gu, gv), axis_deg, length_px=40.0)
            smoothed = self._sensor_ema.update(kpts)
            if smoothed is not None:
                kpts = smoothed
                axis_deg = axis_deg_from_sensor_kpts(kpts)
            obj.kpts = kpts
            obj.bbox_xyxy = list(bbox)
            obj.axis_deg = float(axis_deg)
            obj.center_uv = (float(kpts[0][0]), float(kpts[0][1]))  # contact
            return obj

        # Non-sensor: reset EMA so a later sensor starts fresh
        if class_name != "sensor":
            pass

        crop, origin = _crop_resize(color_rgb, bbox, pad=self.vision.crop_padding_px)
        xt = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
        x0, y0, x1, y1 = origin
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)

        if class_name in ("post", "other") and self.has_center:
            with torch.no_grad():
                pred = self.center_head(xt)[0].cpu().numpy()
            u = float(np.clip(pred[0], 0, 1)) * bw + x0
            v = float(np.clip(pred[1], 0, 1)) * bh + y0
            obj.kpts = [[u, v]]
            obj.center_uv = (u, v)
        else:
            cx = (bbox[0] + bbox[2]) * 0.5
            cy = (bbox[1] + bbox[3]) * 0.5
            obj.center_uv = (cx, cy)
            obj.kpts = [[cx, cy]]
            if class_name == "sensor":
                obj.axis_deg = 0.0
        return obj

    def infer_scene(self, color_rgb: np.ndarray, depth_m: np.ndarray | None = None) -> list[DetectedObject]:
        backend = str(self.vision.detector_backend).lower()
        results: list[DetectedObject] = []

        if backend == "yolo":
            saw_sensor = False
            for bbox, score, class_name in _yolo_boxes(color_rgb, self.vision):
                if class_name == "sensor":
                    saw_sensor = True
                results.append(
                    self._apply_heads(color_rgb, bbox, class_name, score, depth_m=depth_m)
                )
            if not saw_sensor:
                self._sensor_ema.reset()
            results.sort(key=lambda o: o.score, reverse=True)
            return results

        if not (self.has_proposal and self.has_class):
            raise FileNotFoundError(
                "Missing proposal/class weights (cascade backend). "
                "Prefer YOLO: set vision.detector_backend: yolo and train_yolo, "
                f"or train cascade under {self.vision.proposal_model_path.parent}"
            )
        saw_sensor = False
        for bbox, score in self._proposals_from_heatmap(color_rgb):
            crop, _origin = _crop_resize(color_rgb, bbox, pad=self.vision.crop_padding_px)
            xt = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
            with torch.no_grad():
                logits = self.class_net(xt)[0]
                probs = torch.softmax(logits, dim=0)
                cls_i = int(torch.argmax(probs).item())
                cls_score = float(probs[cls_i].item())
            class_name = CLASS_NAMES[cls_i]
            combined = float(score * cls_score)
            if combined < self.vision.confidence * 0.5:
                continue
            if class_name == "sensor":
                saw_sensor = True
            results.append(
                self._apply_heads(color_rgb, bbox, class_name, combined, depth_m=depth_m)
            )
        if not saw_sensor:
            self._sensor_ema.reset()
        results.sort(key=lambda o: o.score, reverse=True)
        return results


@lru_cache(maxsize=4)
def _get_scene_detector(
    proposal: str,
    class_m: str,
    sensor: str,
    center: str,
    yolo: str,
    backend: str,
    confidence: float,
    crop_pad: int,
    max_prop: int,
) -> CascadeSceneDetector:
    vision = VisionModelsConfig(
        detector_backend=backend,  # type: ignore[arg-type]
        proposal_model_path=Path(proposal),
        class_model_path=Path(class_m),
        sensor_head_path=Path(sensor),
        center_head_path=Path(center),
        yolo_model_path=Path(yolo),
        confidence=confidence,
        crop_padding_px=crop_pad,
        max_proposals=max_prop,
    )
    return CascadeSceneDetector(vision)


def infer_scene(
    color_rgb: np.ndarray,
    depth_m: np.ndarray | None,
    cfg: GrabSensorConfig,
) -> list[DetectedObject]:
    v = cfg.vision
    det = _get_scene_detector(
        str(v.proposal_model_path),
        str(v.class_model_path),
        str(v.sensor_head_path),
        str(v.center_head_path),
        str(v.yolo_model_path),
        str(v.detector_backend),
        float(v.confidence),
        int(v.crop_padding_px),
        int(v.max_proposals),
    )
    return det.infer_scene(color_rgb, depth_m)


def objects_to_debug(objects: list[DetectedObject]) -> dict[str, Any]:
    return {
        "vision_mode": "yolo_cascade",
        "objects": [
            {
                "class": o.class_name,
                "bbox_xyxy": o.bbox_xyxy,
                "score": o.score,
                "kpts": o.kpts,
                "axis_deg": o.axis_deg,
                "center_uv": list(o.center_uv) if o.center_uv else None,
            }
            for o in objects
        ],
    }
