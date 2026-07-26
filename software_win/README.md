# software_win

Windows-first copy of the Dummy robot teleop / planning stack, plus epoxy-oriented cameras.

Use this tree on the ThinkPad instead of `software/` when running:

- `manual_control`
- `movement_sim` (PyBullet controllers only)
- `robot_ascii_interface`
- `trajectory_planner`
- `trajectory_run`
- `camera` (Thorlabs Zelux + Infinity 2-1RM)

Linux RealSense / grab_sensor / insert_sensor jobs stay under `software/`.

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
python software_win\trajectory_planner\plan_and_demo.py
python software_win\trajectory_run\run_planned_trajectory.py --dry-run
```

Serial defaults: `ROBOT_PORT` → USB VID/PID `1209:0D32` → `COM3`.

## Cameras (for future epoxy_app)

```powershell
python software_win\camera\list_devices.py
python software_win\camera\dual_live_view.py --demo
python software_win\camera\dual_live_view.py
python software_win\camera\live_zelux.py
python software_win\camera\live_infinity.py --opencv-index 0
```

`dual_live_view.py` opens **two** OpenCV windows (Zelux + Infinity). Press **Q** / **Esc** to quit.

## Windows notes vs Linux `software/`

| Topic | Behavior |
|-------|----------|
| Serial | Same ASCII protocol; Linux `termios` / latency_timer is skipped on Windows |
| URDF paths | `$(find …)` / `package://` rewritten with POSIX separators for PyBullet |
| `movement_sim/Dummy_ros2_description` | ROS2/Gazebo/RViz — not used on Windows; PyBullet `joint_controller` / `end_controller` are |
| Gamepad | Not used; teleop is PyBullet keyboard |
| Robot meshes | Still loaded from repo `robot_design/` via `robot_config.repo_root()` |

## Relation to epoxy_app

`software_win/camera` exposes `ZeluxCamera` and `InfinityCamera` with a shared `grab_bgr()` API so a future `software_win/high_level_jobs/epoxy_app` can import them without reimplementing capture.
