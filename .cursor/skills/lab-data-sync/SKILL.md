---
name: lab-data-sync
description: >-
  Rsync Dummy_lab demo datasets and checkpoints between local robot PC and
  Cornell unicorn/share storage. Use when transferring episodes, fixing
  partial rsync, disk quota exceeded, or syncing to /share/mehta.
---

# Lab data sync (rsync)

## Default hosts / paths

- **SSH**: `zx296@unicorn-login-01.coecis.cornell.edu` (use FQDN if short name fails DNS)
- **Local demos**: `/home/jack/Desktop/Dummy_lab/software/high_level_jobs/subtasks/insert_sensor/dataset/`
- **Remote (preferred)**: `/share/mehta/zx296/Dummy_lab/software/high_level_jobs/subtasks/insert_sensor/dataset/`
- **Avoid for multi‑GB**: `~/Documents/...` (quota → rsync code 23 / errno 122)

Exclude recording scratch: `--exclude='_staging/'`.

## Resume / finish a failed transfer

Partial transfer is normal. Re-run the **same** command with `--partial`; rsync skips complete files.

```bash
rsync -avhP --partial --exclude='_staging/' \
  /home/jack/Desktop/Dummy_lab/software/high_level_jobs/subtasks/insert_sensor/dataset/ \
  zx296@unicorn-login-01.coecis.cornell.edu:/share/mehta/zx296/Dummy_lab/software/high_level_jobs/subtasks/insert_sensor/dataset/
```

Dry-run first (list what would send):

```bash
rsync -avhn --exclude='_staging/' \
  LOCAL_DATASET/ \
  REMOTE_USER@HOST:REMOTE_DATASET/
```

## Quota / code 23

If logs show `Disk quota exceeded (122)`:

1. Check remote: `quota -s; df -h ~; du -sh ~/Documents /share/mehta/$USER 2>/dev/null`
2. Retarget destination to `/share/mehta/...`
3. Re-run rsync (only missing files transfer)

Last known local→home failure (historical): mid-`episode_000020` from `000102.side2.depth.npy` onward — fix by syncing that episode (or full tree) to `/share/mehta`.

## Subset sync (e.g. first 10 episodes)

```bash
rsync -avhP --partial \
  /home/jack/Desktop/Dummy_lab/software/high_level_jobs/subtasks/insert_sensor/dataset/episodes/episode_00000{1..9}/ \
  /home/jack/Desktop/Dummy_lab/software/high_level_jobs/subtasks/insert_sensor/dataset/episodes/episode_000010/ \
  zx296@unicorn-login-01.coecis.cornell.edu:/share/mehta/zx296/Dummy_lab/software/high_level_jobs/subtasks/insert_sensor/dataset/episodes/
```

## Agent rules

1. Always keep trailing slashes consistent (`dataset/` → `dataset/`).
2. Prefer `--partial` for large episode trees.
3. Do not invent hostnames; use FQDN from this skill or ask the user.
4. Platform context: [dummy-lab-platforms](../dummy-lab-platforms/SKILL.md).
