# Windows flat grab_sensor

Top-view **D415** → YOLO boxes → keypoint heads → perpendicular grasp (yaw from sensor kpts) → iLQR (+ SCP posts).

## Setup

```powershell
$env:PYTHONPATH = "$PWD\software_win"
pip install pyrealsense2 ultralytics   # + Intel RealSense SDK
```

Table calib: `software_win/camera/calibration_table/runs/*/table_calib.json` (falls back to `software/camera/...`).

## Label (keypoints only)

```powershell
python -m high_level_jobs.grab_sensor.vision.live_label --serial <D415_SN>
python -m high_level_jobs.grab_sensor.vision.live_review
```

| Class | Clicks |
|-------|--------|
| `sensor` (1) | **2 keypoints**: (0) holder–sensor **contact**, (1) along **holder** (toward grasp); table grasp = contact + `grasp_offset_from_holder_mm` |
| `post` (2) | **1 center**; bbox auto (tight) |
| `other` (3) | **1 center**; bbox auto |

Keys: `1/2/3` class · click · `N` next object · `S` save · `D` undo · `Q` quit.

## Live detect + correct/save

```powershell
python -m high_level_jobs.grab_sensor.vision.live_detect --serial <D415>
```

- **Live:** NN boxes/keypoints overlaid continuously  
- **Space:** freeze — keeps NN posts/other; clears sensors so you label **contact → along holder**  
- **Frozen:** click · `N` finish · `E` drop sensors · `R` clear frame · `S` save · `Space` resume live  
- **A:** accept current NN detections and save as a new sample (no edit) 

```powershell
# View only
python -m high_level_jobs.grab_sensor.vision.live_review

# Edit keypoints on saved frames (no camera)
python -m high_level_jobs.grab_sensor.vision.relabel
python -m high_level_jobs.grab_sensor.vision.relabel --start 10
```

Relabel: **default click = update CONTACT only** (keeps existing holder-dir tip). `F` = full 2-click. `E` drop sensors · `S` save.

## Train

```powershell
# YOLO boxes (sensor / post / other) — exports auto-bboxes from keypoints
python -m high_level_jobs.grab_sensor.vision.train_yolo --epochs 60

# Keypoint heads on YOLO crops (sensor head: RGB + depth + edge)
python -m high_level_jobs.grab_sensor.vision.train_heads --epochs 60
```

Weights: `vision/dataset/model/sensor_yolo.pt`, `sensor_head.pt`, `center_head.pt`.

Default config: `vision.detector_backend: yolo`.

## Run

```powershell
python -m high_level_jobs.grab_sensor --detect-only --visualize
python -m high_level_jobs.grab_sensor --offline --plan-gui
python -m high_level_jobs.grab_sensor --ik-gui
python -m high_level_jobs.grab_sensor --execute --visualize --port COM3

# Continuous session: grab_sensor saves motion; put_sensor reverses it
python -m high_level_jobs.session --port COM3
```

Config: [`config/default_grasp.yaml`](config/default_grasp.yaml).
Session records: [`../session/records/`](../session/records/).
