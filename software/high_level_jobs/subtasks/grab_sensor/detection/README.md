# Grab sensor — vision & training

Camera capture, YOLO sensor detection, grasp/base CNN, labeling GUIs, and model training. Runtime inference is called from `grab_sensor.pipeline`; motion planning and execution stay in the parent package.

## Layout

```
detection/
  capture.py              # RealSense D415 capture (live + offline)
  segment.py / pose.py    # Legacy depth-segment + PCA pose
  yolo_detect.py          # YOLO wrapper
  yolo_grasp.py           # YOLO → crop RGBD → grasp + base CNN
  grasp_point_net.py      # CNN definition, train, predict
  visualize.py            # grasp_overlay.png debug drawing
  live_label.py           # Live GUI: label grasp + base manually
  live_review.py          # Live GUI: NN proposals, correct and save
  label_base.py           # Offline relabel: base ellipse on saved dataset images
  review_grasp.py         # Live GUI: confirm / override NN grasp proposals
  export_yolo_dataset.py  # Bootstrap YOLO labels from grasp clicks
  train_yolo_model.py     # Train YOLO sensor detector
  train_grasp_model.py    # Train grasp + base CNNs (separate nets)
  train_base_model.py     # Train base net only
  dataset/                # Local training data (not in git; see dataset/README.md)
```

## Data collection & training

All commands from `software/`:

```bash
pip install opencv-python pyrealsense2 torch ultralytics

# 1. Label grasp + base (live camera)
python -m high_level_jobs.subtasks.grab_sensor.detection.live_label

# 1b. Relabel base ellipse on saved images (offline — no camera)
python -m high_level_jobs.subtasks.grab_sensor.detection.label_base

# 3. Train YOLO sensor detector
python -m high_level_jobs.subtasks.grab_sensor.detection.export_yolo_dataset
python -m high_level_jobs.subtasks.grab_sensor.detection.train_yolo_model --export-first

# 4. Train grasp + base CNNs (separate networks, YOLO crop + RGBD)
python -m high_level_jobs.subtasks.grab_sensor.detection.train_grasp_model --crop --depth --epochs 120
# Or base only:
python -m high_level_jobs.subtasks.grab_sensor.detection.train_base_model --crop --depth --epochs 120

# 5. Review NN on live camera — confirm or fix bad predictions
python -m high_level_jobs.subtasks.grab_sensor.detection.live_review
```

### Labeling keys

**live_label:** **G** = grasp mode (left-click). **B** = base mode (left-drag ellipse + handles). **S** save.

**live_review:** cyan/green = NN grasp/base on live feed. Left/right-click to fix, **Y**/**S** save, **R** reset to NN, **T** retrain.

**label_base:** offline relabel. Left-drag ellipse box + handles; **S** save, **N**/**P** navigate.

## Models

Default weights (override in `config/default_grasp.yaml`):

- `detection/dataset/grasp_keypoints/model/sensor_yolo.pt` — YOLO detector
- `detection/dataset/grasp_keypoints/model/grasp_point.pt` — grasp CNN
- `detection/dataset/grasp_keypoints/model/base_point.pt` — base CNN (separate)

Models must exist before running `vision_mode: yolo_grasp` in the grab pipeline.

## Overlay legend (`grasp_overlay.png`)

| Marker | Meaning |
|--------|---------|
| Yellow box | YOLO sensor region |
| Red **GRASP (NN)** | Grasp CNN pixel |
| Green **BASE (NN)** | Sensor base on table (when model predicts base) |
| Cyan **object center** | 3D grasp point reprojected |
| Magenta **robot IK** | Robot target reprojected — offset from red means fix `camera_to_robot` |
