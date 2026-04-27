# Manual Control (Phase 1)

This folder integrates:
- End-effector IK teleop trajectory generation (from movement_sim/end_controller.py logic)
- Continuous smooth tracking and gating execution (from CLI-Tool/test_robot_smooth_ct.py logic)
- ASCII serial communication to robot

## Modules

- `control_solver.py`: PyBullet teleop + IK solver; outputs time-series joint targets/velocities.
- `motion_plan.py`: Continuous tracking planner with freeze/run gating and speed conversion.
- `ascii_comm.py`: ASCII protocol communication layer.
- `trajectory_schema.py`: Recorded trajectory data model and JSON persistence.
- `manual_control_app.py`: CLI entrypoint (`record`, `execute`, and `live`).

## Usage

Record teleop IK trajectory:

```bash
python software/manual_control/manual_control_app.py record \
  --root . \
  --robot software/movement_sim/Dummy_ros2_description/urdf/Dummy_ros2.xacro \
  --output software/manual_control/recorded_trajectory.json \
  --duration 30
```

Execute recorded trajectory on robot:

```bash
python software/manual_control/manual_control_app.py execute \
  --input software/manual_control/recorded_trajectory.json \
  --port COM5 \
  --control-dt-ms 150
```

Live teleop control on robot (no record step):

```bash
python software/manual_control/manual_control_app.py live \
  --root . \
  --robot software/manual_control/Dummy_ros2_description/urdf/Dummy_ros2.xacro \
  --port COM5 \
  --stream-hz 6.667
```

Quick start from `software/manual_control` (uses local default robot path):

```bash
python manual_control_app.py live --port COM5
```

By default, `live` moves the robot to home position once at startup.
To skip this behavior, add:

```bash
--no-home-first
```

If keyboard control is not captured, click inside the PyBullet window.
`live` also enables in-window twist sliders (`vx/vy/vz/wx/wy`) by default.
Disable sliders if needed:

```bash
--no-twist-sliders
```

During `live`, the simulator now mirrors robot joint feedback (`#GETJPOS`) so the PyBullet pose reflects what the robot reports.
Feedback poll rate is configurable with `--feedback-hz` (default: 8).

Default REST pose is `[0, -73, 180, 0, 0, 0]` (degrees).
Default HOME pose is `[0, 0, 90, 0, 0, 0]` (degrees).
At startup (`--home-first`), controller moves `REST -> HOME` and updates simulator pose from robot feedback during this transition.

## Notes

- `record` supports keyboard mapping from the original end-controller:
  - Linear: `W/S`, `A/D`, `Q/E`
  - Angular: `H/K`, `U/J`
- `execute` uses command mode 2 and asynchronous `&` trajectory streaming.
- `live` uses the same teleop key mapping as `record` and now runs: IK target -> `ContinuousTracker` -> asynchronous `&` streaming (same smoothing pipeline style as `CLI-Tool/test_robot_smooth_ct.py`).
- `live` homes once at startup by default (`--no-home-first` to disable).
- Startup protocol for `execute/live` is `STOP -> START -> CMDMODE 2`, matching `CLI-Tool/test_robot_smooth_ct.py`.
- Default `live` stream rate is 6.667 Hz (150 ms), matching `CLI-Tool/test_robot_smooth_ct.py` control period.
- Default `live` robot model path is `software/manual_control/Dummy_ros2_description/urdf/Dummy_ros2.xacro`.
- This is host-side phase-1 integration; for deterministic hard real-time execution, move the final trajectory interpolator to firmware C++.
