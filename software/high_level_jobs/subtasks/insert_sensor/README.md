# Insert sensor — imitation learning to place sensor back in holder

Learn a visuomotor policy for tight insertion using **RGB-D**, **robot state**, a **pretrained base keypoint** (from `grab_sensor` vision), and **teleop demonstrations**. No manual geometry labels.

## Architecture

| Component | Role |
|-----------|------|
| Perception encoder | RGB-D CNN + MLP(keypoint) + MLP(state) → fused features |
| Temporal encoder | GRU (or Transformer) over history `H` → latent `z_t` |
| Policy head | BC: `a_t = π(z_t)` vs expert teleop action |
| World model | `(z_t, a_t) → z_{t+1}` with stopgrad target from encoder |
| Smoothness | `‖z_t − z_{t−1}‖²` regularization |

The base CNN (`base_point.pt`) is **not** trained here — it only supplies a noisy hole-base observation in robot frame.

## Workflow

All commands from `software/`:

```bash
# 1. Record demonstrations via manual_control (recommended; auto-detects robot USB port)
python manual_control/manual_control_app.py --record-demos

# Recording workflow:
#   Teleop to grasp sensor / align (not recorded)
#   R = start recording (RGB-D + joints + O/P gripper)
#   Insert, open hand (O), retreat, T = home (stream waypoints, same as plain teleop)
#   R = stop recording
#   C = save episode  |  R again = discard and re-record

# Or standalone insert_sensor teleop (same R/C keys):
python -m high_level_jobs.subtasks.insert_sensor --record

# 2. Check recordings
python -m high_level_jobs.subtasks.insert_sensor --check
python -m high_level_jobs.subtasks.insert_sensor --check --episode episode_000001 --frame 0

# 3. Label base keypoints offline (default; keeps teleop from stalling)
python -m high_level_jobs.subtasks.insert_sensor --label-keypoints

# 4. Train policy
python -m high_level_jobs.subtasks.insert_sensor --train

# 5. Deploy (camera loop; wire robot streaming as needed)
python -m high_level_jobs.subtasks.insert_sensor --deploy --model \
  high_level_jobs/subtasks/insert_sensor/runs/latest/policy.pt
```

## Config

Primary file: `config/default_insert_sensor.yaml`

| Key | Purpose |
|-----|---------|
| `grasp_config` | Reuses `grab_sensor` vision + table calib for base keypoint |
| `dataset_dir` | Teleop episodes (`episode_*/`) |
| `model_path` | Trained policy checkpoint |
| `history_len` | Temporal window `H` |
| `lambda_world`, `lambda_smooth` | Auxiliary loss weights |

## Episode format

```
dataset/episodes/episode_000001/
  meta.json
  000000.color.png
  000000.depth.npy
  000000.frame.json   # joint state, expert action, keypoint vector
```

## Module map

| Module | Role |
|--------|------|
| `base_keypoint.py` | YOLO + `base_point.pt` → robot-frame hole base |
| `model.py` | Policy + world model architecture |
| `dataset.py` | Episode loader + normalization |
| `record.py` | Teleop hook + RealSense logging |
| `train.py` | IL training loop |
| `deploy.py` | Closed-loop inference |

## Dependencies

Same as `grab_sensor` (`pyrealsense2`, `torch`, `opencv-python`, `pybullet`) plus teleop stack in `manual_control/`.
