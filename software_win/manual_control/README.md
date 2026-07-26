# Manual Control (Windows)

Windows entrypoint lives under `software_win/`. Same teleop stack as Linux `software/manual_control`.

## Run

```powershell
conda activate .\.conda
$env:PYTHONPATH = "$PWD\software_win"
python software_win\manual_control\manual_control_app.py --port COM3
python software_win\manual_control\manual_control_app.py --sim-only
```

Serial defaults: `ROBOT_PORT` → auto-detect Dummy USB (`1209:0D32`) → `COM3`.

## Controls

- Linear: `W/S -> -Y/+Y`, `A/D -> +X/-X`, `Q/E -> +Z/-Z`
- Angular: `H/K -> +Wy/-Wy`, `U/J -> +Wx/-Wx`
- Home: `T`
- Gripper: `O` open / `P` close
- Drives: `B` disable · `N` enable

Focus the **PyBullet** window for keyboard input.

See the Linux package README under `software/manual_control/README.md` for the full option list.
