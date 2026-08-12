# software_win

Windows-first copy of the Dummy robot teleop / planning stack, plus epoxy-oriented cameras.

Use this tree on the ThinkPad instead of `software/` when running:

- `manual_control`
- `movement_sim` (PyBullet controllers only)
- `robot_ascii_interface`
- `trajectory_planner`
- `trajectory_run`
- `camera` (Thorlabs Zelux + Infinity 2-1RM + RealSense D415 for grab_sensor)
- `high_level_jobs/grab_sensor` (flat top-view grasp)

Linux RealSense / YOLO grab_sensor / insert_sensor jobs stay under `software/`. Windows flat grab_sensor lives here.

## Setup

```powershell
cd C:\Users\PQE_339_ThinkPad\Documents\Dummy_lab
conda activate .\.conda   # or your Windows venv
pip install -r software_win\requirements.txt
```

Camera vendor SDKs (one-time):

1. **Thorlabs Zelux** — install [ThorCam](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam), then `pip install pylablib`
2. **Infinity 2-1RM** — install Infinity Capture / Lumenera drivers, then `pip install lucam`  
   (OpenCV DirectShow fallback is available if lucam cannot open the device)

## Run robot stack

Always put `software_win` on `PYTHONPATH` (scripts also insert it automatically):

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python software_win\robot_ascii_interface\robot_ascii_interface.py --port COM3
python software_win\manual_control\manual_control_app.py --port COM3
python software_win\manual_control\manual_control_app.py --sim-only
python software_win\manual_control\manual_control_app.py --record-demos
python software_win\trajectory_planner\plan_and_demo.py
python software_win\trajectory_run\run_planned_trajectory.py --dry-run
```

Serial defaults: `ROBOT_PORT` → USB VID/PID `1209:0D32` → `COM3`.

### Record epoxy demos

```powershell
python software_win\manual_control\manual_control_app.py --record-demos
```

Focus the PyBullet window. **R** start/stop saving · **C** save episode · **O/P** gripper.
Zelux + Infinity open immediately as live OpenCV windows (`LIVE` / `REC` overlay).
Episodes land in `software_win/high_level_jobs/epoxy_app/dataset/episodes/episode_######/`.
Camera exposure/serials: `software_win/high_level_jobs/epoxy_app/config/default_epoxy.yaml`.

### Review recorded episodes

Dual Zelux + Infinity player with one shared scrubber / play-pause bar:

```powershell
python software_win\high_level_jobs\epoxy_app\view_episode.py
python software_win\high_level_jobs\epoxy_app\view_episode.py --episode 1 --play
```

Space play/pause · Left/Right step · Home/End · `[` `]` speed · L loop.

Export Zelux as MP4 (also available via **Save Zelux video** in the viewer):

```powershell
python software_win\high_level_jobs\epoxy_app\view_episode.py --episode 1 --export-zelux
python software_win\high_level_jobs\epoxy_app\view_episode.py --episode 1 --export-zelux path\to\out.mp4
```

Default output: `episode_######/episode_######_zelux.mp4` at the episode `record_hz`.

### Flat grab_sensor (D415)

```powershell
pip install pyrealsense2 ultralytics   # + Intel RealSense SDK
python -m high_level_jobs.grab_sensor.vision.live_label --serial <D415>
python -m high_level_jobs.grab_sensor.vision.train_yolo
python -m high_level_jobs.grab_sensor.vision.train_heads
python -m high_level_jobs.grab_sensor --detect-only --visualize
python -m high_level_jobs.grab_sensor --ik-gui
```

Label keypoints only (boxes auto). See `software_win/high_level_jobs/grab_sensor/README.md`.

### Vision model + Z contact/flow control

Native to `epoxy_app` (labels on episodes; motion via manual_control DLS IK):

```powershell
$env:PYTHONPATH = "$PWD\software_win"
pip install -r software_win\high_level_jobs\epoxy_app\vision\requirements.txt

# Label episode 1 Zelux frames in place -> episode_000001/labels.json
python -m high_level_jobs.epoxy_app.vision.label_tool --episode 1

# After ~10-15 labeled episodes:
python -m high_level_jobs.epoxy_app.vision.train
python -m high_level_jobs.epoxy_app.vision.run_apply --smoke

# Real robot: teleop jog, then G=apply / X=estop (Zelux prediction overlay)
python -m high_level_jobs.epoxy_app.epoxy_apply
```

See `software_win/high_level_jobs/epoxy_app/vision/README.md`.

## Cameras (epoxy_app)

```powershell
python software_win\camera\list_devices.py
python software_win\camera\live_view.py --demo
python software_win\camera\live_view.py
python software_win\camera\live_zelux.py
python software_win\camera\live_infinity.py --opencv-index 0
```

`live_view.py` opens **three** OpenCV windows (Zelux front + Zelux side + Infinity). Press **Q** / **Esc** to quit. Pin with `--front-serial` / `--side-serial` after `list_devices.py`.

## Windows notes vs Linux `software/`

| Topic | Behavior |
|-------|----------|
| Serial | Same ASCII protocol; Linux `termios` / latency_timer is skipped on Windows |
| URDF paths | `$(find …)` / `package://` rewritten with POSIX separators for PyBullet |
| `movement_sim/Dummy_ros2_description` | ROS2/Gazebo/RViz — not used on Windows; PyBullet `joint_controller` / `end_controller` are |
| Gamepad | Not used; teleop is PyBullet keyboard |
| Robot meshes | Still loaded from repo `robot_design/` via `robot_config.repo_root()` |

## Relation to epoxy_app

`software_win/camera` exposes `ZeluxCamera` and `InfinityCamera` with a shared `grab_bgr()` API.
`software_win/high_level_jobs/epoxy_app/record.py` uses them from `manual_control --record-demos`.
