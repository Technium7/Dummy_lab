# Top/overhead detection (YOLO + base point)

Overhead D415 view for `grab_sensor`. Grasp and base coincide in image space — only a **single base point** is labeled and predicted (no grasp CNN, no ellipse).

## Commands (from `software/`)

```bash
# 1. Label base points (live camera)
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_label

# 1b. Fix labels on saved images (offline — no camera)
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.relabel

# 2. Train YOLO (exports bbox from base clicks)
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.train_yolo_model --export-first

# 3. Train base point CNN (YOLO crop + RGBD, 2 outputs)
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.train_model --crop --depth --epochs 120

# 4. Review NN on live camera — confirm or fix bad predictions
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_review
# (same camera/GUI code path as side_detection.live_review, with --base-point-only)
```

### Labeling keys

**live_label:** left-click = base point, **S** save, **C** clear, **D** undo, **Q** quit.

**relabel:** offline on saved images. Left-click = base point, **S** save, **N/P** navigate, **Q** quit.

**live_review:** cyan = NN base, green = your edit. **Y/S** save, **R** reset, **T** retrain, **Q** quit.

## Models

| File | Role |
|------|------|
| `dataset/base_keypoints/model/sensor_yolo.pt` | YOLO sensor detector |
| `dataset/base_keypoints/model/base_point.pt` | Base point CNN (2 outputs, not ellipse) |

Production config (`config/default_grasp.yaml`) uses `grasp_net.base_only: true` and these paths.

## Layout

```
top_detection/
  dataset/
    base_keypoints/
      images/ depth/ labels/ previews/ manifest.jsonl
      model/ sensor_yolo.pt  base_point.pt  runs/
    sensor_yolo/             # exported YOLO train/val
```

See also `side_detection/` for the legacy oblique view (grasp + base ellipse).
