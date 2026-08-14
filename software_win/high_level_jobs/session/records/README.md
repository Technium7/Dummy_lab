# Motion records (session)

JSON trajectories shared across skills (`grab_sensor`, `power_measure`, `return`, …).
Generated at runtime — do not commit large/debug copies.

## Layout

- `grab_sensor.json` / `power_measure.json` — last skill recordings (overwritten)
- `library/<name>.json` — **named reusable trajectories** for `traj_save` / `traj_replay`

Save into the library with `power_measure save_as=<name>` or `traj_save name=<name>`.
Replay outbound place with `traj_replay <name>`; return with `return name=<name>`.
