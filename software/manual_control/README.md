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

```bash
python software/manual_control/manual_control_app.py --port COM3
```

Windows venv example:

```bash
.\.venv\Scripts\python.exe software/manual_control/manual_control_app.py --port COM3
```

## Important options

- `--robot`: URDF/Xacro path
- `--sim-hz`: simulation/control loop rate
- `--control-dt-ms`: robot command loop period in ms (lower = faster updates)
- `--linear-speed`: base linear command speed
- `--angular-speed-deg`: base angular command speed
- `--joint-vel-limit`: optional joint velocity clamp
- `--port`, `--baudrate`, `--timeout`: robot serial settings

## Controls

- Linear: `W/S -> -Y/+Y`, `A/D -> -X/+X`, `Q/E -> +Z/-Z`
- Angular: `H/K -> +Wy/-Wy`, `U/J -> +Wx/-Wx`
- Home: `T` -> `[0, 0, 90, 0, 0, 0]`

## Behavior notes

- Motion speed scales from a low starting multiplier and increases while keys are held.
- On key release, robot holds the last target position (prevents drift).
- Pressing `T` latches HOME target briefly so stream updates do not override it.

