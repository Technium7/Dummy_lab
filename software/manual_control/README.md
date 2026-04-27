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
- `manual_control_app.py`: CLI entrypoint (`record` and `execute`).

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

## Notes

- `record` supports keyboard mapping from the original end-controller:
  - Linear: `W/S`, `A/D`, `Q/E`
  - Angular: `H/K`, `U/J`
- `execute` uses command mode 2 and asynchronous `&` trajectory streaming.
- This is host-side phase-1 integration; for deterministic hard real-time execution, move the final trajectory interpolator to firmware C++.
