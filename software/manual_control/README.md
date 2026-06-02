# Manual Control

Manual control for physical robot using:

- End-effector keyboard IK in PyBullet
- Joint transform from simulated joints to physical robot joints
- Async trajectory-point streaming to robot ASCII interface
- Separate status window with transformed joint display

## Code structure

Runtime implementation is split by function:

- `manual_control_app.py`: entrypoint and argument handling
- `controller_runtime.py`: main control loop (sim + robot streaming)
- `ik_core.py`: IK helpers and Jacobian solve
- `mapping.py`: joint transform and speed mapping
- `status_window.py`: instruction/status UI
- `urdf_tools.py`: URDF/Xacro discovery and materialization

Reference-only script:

- `test_end_controller_smooth_ct.py`

## Run

From the repo root with the local conda env (see `requirements.txt`):

```bash
conda activate ./.conda
python software/manual_control/manual_control_app.py
```

Serial port defaults: `ROBOT_PORT` env, auto-detect Dummy USB (1209:0d32), then `COM3` (Windows) or `/dev/ttyACM0` (Linux). Override with `--port`.

Linux udev (one-time, for non-root access):

```bash
sudo cp hardware/99-dummy-robot-serial.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Windows venv example:

```powershell
.\.venv\Scripts\python.exe software\manual_control\manual_control_app.py --port COM3
```

## Important options

- `--robot`: URDF/Xacro path
- `--sim-hz`: simulation/control loop rate
- `--control-dt-ms`: robot `&` waypoint period in ms (default **20**; lower = less lag, min 20)
- `--speed-factor`: sim time vs wall clock (default **1**; was 2 — faster sim made the robot trail the preview)
- `--linear-speed`: base linear command speed
- `--angular-speed-deg`: base angular command speed
- `--joint-vel-limit`: optional joint velocity clamp
- `--no-home-firmware`: `T` only streams HOME; default also runs `!HOME` in the background
- `--port`, `--baudrate`, `--timeout`: robot serial settings

### Reducing keyboard lag

Lag is mostly the robot stream running slower than the PyBullet preview. Defaults are tuned for tighter sync:

1. **`--control-dt-ms 20`** (already default) — ~50 Hz waypoints; try `15` if serial keeps up.
2. **`--speed-factor 1`** (already default) — do not run the sim faster than real time unless you accept more chase lag.
3. **`--linear-speed` / `--angular-speed-deg`** — end-effector speeds; unchanged accuracy, only feel.

Motion still ramps from 40% to 250% while keys are held (smoother start, less “crawl” on first press than before).

## Controls

- Linear: `W/S -> -Y/+Y`, `A/D -> -X/+X`, `Q/E -> +Z/-Z`
- Angular: `H/K -> +Wy/-Wy`, `U/J -> +Wx/-Wx`
- Home: `T` -> firmware `!HOME` + stream to `[0, 0, 90, 0, 0, 0]` at max speed (use `--no-home-firmware` for stream-only)

## Behavior notes

- Motion speed scales from 40% to 250% while keys are held.
- On key release, robot holds the last target position (prevents drift).
- While moving, robot targets are integrated from IK velocities at the waypoint rate (not raw sim snapshots), so accuracy tracks commanded twist even when the preview runs ahead.
- Pressing `T` latches HOME for ~25 s and streams at max speed; firmware homing runs in a background thread unless `--no-home-firmware`.

