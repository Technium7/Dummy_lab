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
- `--sim-hz`: PyBullet IK loop rate (default **480**)
- `--control-dt-ms`: robot `&` waypoint period in ms (default **20**; streams at ~10 ms while moving)
- `--speed-factor`: sim time vs wall clock (default **1** — keeps preview in sync)
- `--linear-speed`: base linear command speed
- `--angular-speed-deg`: base angular command speed
- `--joint-vel-limit`: optional joint velocity clamp
- `--no-home-firmware`: `T` only streams HOME; default also runs `!HOME` in the background
- `--port`, `--baudrate`, `--timeout`: robot serial settings

### Tuning

Control logic matches the stable Windows `manual_control` (commit `64acdd4`): W/S = Y, A/D = X, sim joint snapshot streamed to the robot. Adjust `JOINT_SIGNS` in `mapping.py` only if a **joint angle** is inverted vs firmware — do not add separate sim/robot sign layers.

Linux defaults already use `--control-dt-ms 20` and `--speed-factor 1`. For snappier response try `--control-dt-ms 15`.

## Controls

- Linear: `W/S -> -Y/+Y`, `A/D -> +X/-X`, `Q/E -> +Z/-Z`
- Angular: `H/K -> +Wy/-Wy`, `U/J -> +Wx/-Wx`
- Home: `T` -> firmware `!HOME` + stream to `[0, 0, 90, 0, 0, 0]` at max speed (use `--no-home-firmware` for stream-only)

## Behavior notes

- Motion speed scales from 40% to 250% while keys are held; first waypoint sends immediately on key press.
- On key release, robot holds the last target position (prevents drift).
- While moving, robot receives a sim joint target with one control-period velocity lookahead (larger per-waypoint delta so firmware does not crawl on tiny moves).
- Pressing `T` latches HOME for ~1.2 s; optional firmware `!HOME` runs in background unless `--no-home-firmware`.

