---
name: dummy-lab-platforms
description: >-
  Cross-machine Dummy_lab paths, hosts, storage, and conda layout for local
  ThinkPad vs Cornell unicorn/cluster. Use when working across platforms,
  SSH, /share/mehta, unicorn-login, cloning the repo elsewhere, or asking
  where demos/checkpoints live.
---

# Dummy lab platforms

## Machines

| Role | Host / path notes |
|------|-------------------|
| Local robot PC | ThinkPad; repo often `/home/jack/Desktop/Dummy_lab` |
| Login / transfer | `zx296@unicorn-login-01.coecis.cornell.edu` (short name may not resolve off VPN) |
| Large storage | Prefer `/share/mehta/zx296/...` — home/`~/Documents` can hit **disk quota** |

## Repo layout (same on every clone)

- Code root for Python modules: `software/` (add to `PYTHONPATH` / `cd software`)
- Conda env: typically `.conda` at repo root (`Dummy_lab/.conda`)
- Do **not** git-commit large demos or weights (see `.gitignore`)

## Data that stays local / share (not git)

| Content | Typical path under repo |
|---------|-------------------------|
| insert_sensor demos | `software/high_level_jobs/subtasks/insert_sensor/dataset/episodes/` |
| insert_sensor ACT ckpt | `.../insert_sensor/runs/latest/policy.pt` |
| insert_sensor diffusion | `.../insert_sensor/runs/diffusion/latest/policy.pt` |
| grab_sensor labels/weights | `.../grab_sensor/detection/*/dataset/` and `debug/` |

On cluster, demos often live at:

`/share/mehta/zx296/Dummy_lab/software/high_level_jobs/subtasks/insert_sensor/dataset/`

## Agent rules

1. Prefer `/share/mehta/...` for multi‑GB rsync destinations; warn if targeting `~/Documents`.
2. After clone on a new machine: use `software/` as cwd; activate repo `.conda` if present.
3. Never suggest committing `dataset/episodes`, `runs/`, or grab_sensor dataset blobs.
4. For job-specific train/deploy, read [insert-sensor](../insert-sensor/SKILL.md). For transfers, read [lab-data-sync](../lab-data-sync/SKILL.md).
