# Continuous session (`high_level_jobs.session`)

Long-lived Windows runner: **start the arm → WORK**, then accept skill commands
(or natural language / multi-step plans) without reconnecting each time.

Motion recordings live in [`records/`](records/) (gitignored payloads; keep the directory).
Named reusable trajectories live in [`records/library/`](records/library/).

## Quick start

```powershell
cd C:\Users\PQE_339_ThinkPad\Documents\Dummy_lab
$env:PYTHONPATH = "$PWD\software_win"
python -m high_level_jobs.session --port COM3
# optional: --camera-serial <D415>  --no-llm  --dry-run
```

At the prompt:

```text
session> grab_sensor
session> power_measure save_as=spot1
session> return name=spot1
session> wait 60
session> traj_replay spot1
session> return name=spot1
session> traj_list
session> llm measure power at the same spot 10 times and wait at work for 1min in between
session> quit
```

## Command semantics

| Command | Behavior | Recording |
|---------|----------|-----------|
| `grab_sensor` | Detect → grasp → lift → **WORK** (sensor held) | Saves `records/grab_sensor.json` for `put_sensor` |
| `put_sensor` | Reverse grab segments → open at grasp → **WORK** | Reads `grab_sensor.json` |
| `power_measure` | Live click → **place only** (WORK→insert) → **stay**; optional `save_as=<name>` | Saves `power_measure.json` + optional library entry |
| `power_optimization` | From current insert: PM100 search/climb → **stay** | Needs prior place extras; appends final peak to `records/power_logs/record_power.jsonl` |
| `record_power` | Settle (default 1s) at current pose; average PM100; append JSONL | `records/power_logs/record_power.jsonl` |
| `return` | Stream recorded return path to **WORK**; optional `name=<lib>` | Default `power_measure.json`, or library traj |
| `standby` | **Fresh plan** current joints → WORK | — |
| `wait` | Dwell at current pose (`wait 60` or `seconds=60`) | — |
| `traj_list` | List `records/library/*.json` | — |
| `traj_save` | Copy last `power_measure` into library as `name=` | Writes `library/<name>.json` |
| `traj_replay` | Replay library traj **WORK→insert** and stay | Reads library; refreshes `power_measure.json` |
| `manual` / `man` | Release port → `manual_control` teleop → reconnect | Pose may change; use `standby` after |
| `s` / `stop` | Immediate firmware `!STOP` (hold + soft-disable) | Next skill re-`!START`s |
| `epoxy_apply` | Stub for now | Use `python -m high_level_jobs.epoxy_app.epoxy_apply` |

Typical power flow: `grab_sensor` → `power_measure` → (`power_optimization`) → `return` → `put_sensor`.

Repeat the same place without re-clicking: `power_measure save_as=spot` → `return` → later `traj_replay spot` → `return`.

## Multi-step plans (LLM)

Free text / `llm …` is planned only by the LLM from a **capability catalog**
(what each subtask does). There is no keyword-plan fallback: bad JSON or an
incoherent plan raises and the REPL prints the error (robot stays connected).

Exact command tokens (`grab_sensor`, `wait 30`, `traj_replay spot`, …) still
bypass the LLM.

LLM plans are printed as a numbered list; confirm with `y` before they run
(use `--yes-llm` to skip the prompt).

Output shapes: `{"command":"..."}` or `{"steps":[...]}` or
`{"steps":[...],"then_repeat":{"times":N,"steps":[...]}}`.

## Layout

```text
software_win/high_level_jobs/session/
  __main__.py      CLI entry
  loop.py          connect + WORK + REPL + plan runner
  commands.py      command handlers
  motion_store.py  records + traj library
  llm_map.py       NL → plan (Ollama / OpenAI / keywords)
  llm_client.py    (package-level) Ollama / OpenAI clients
  records/         skill motion JSON
  records/library/ named reusable trajectories
  README.md        this file
```

## LLM input

- Exact tokens always win (`grab_sensor`, `wait 60`, `traj_replay spot`, …).
- `llm <utterance>` forces the mapper.
- Free text uses the mapper when LLM is enabled (default); `--no-llm` uses keyword / heuristic plans.
- Uses `high_level_jobs/llm_client.py` (`OLLAMA_*` or `OPENAI_*` env).

## Standalone CLIs (still available)

```powershell
python -m high_level_jobs.grab_sensor --execute --port COM3
python -m high_level_jobs.power_measure --execute --place-only --port COM3
python -m high_level_jobs.power_measure --execute --optimize-only --port COM3
```

One-shot `power_measure --execute` (no `--place-only`) runs place → optimize → return as a composed sequence.
