from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SOFTWARE_WIN = Path(__file__).resolve().parents[4]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.vision.dataset import (
    load_episode_samples,
    phase_samples,
    save_labels,
    stems_before_removing,
)


def _write_fake_episode(tmp: Path) -> Path:
    ep = tmp / "episode_000099"
    ep.mkdir(parents=True)
    stems = [f"{i:06d}" for i in range(6)]
    for s in stems:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        import cv2

        cv2.imwrite(str(ep / f"{s}_zelux.jpg"), img)
    frames = {
        "000000": {"contact": 0, "flow": 0, "state": "NOT_TOUCHING"},
        "000001": {"contact": 1, "flow": 0, "state": "TOUCHING"},
        "000002": {"contact": 1, "flow": 1, "state": "FLOWING"},
        "000003": {"contact": 1, "flow": 0, "state": "SETTLED"},
        "000004": {"contact": 0, "flow": 0, "removing": 1, "state": "REMOVING"},
        "000005": {"contact": 0, "flow": 0, "state": "NOT_TOUCHING"},
    }
    save_labels(ep, frames)
    return ep


def test_stems_cut_at_removing(tmp_path: Path):
    ep = _write_fake_episode(tmp_path)
    labels = json.loads((ep / "labels.json").read_text(encoding="utf-8"))["frames"]
    stems = [f"{i:06d}" for i in range(6)]
    kept = stems_before_removing(stems, labels)
    assert kept == ["000000", "000001", "000002", "000003"]


def test_load_cuts_at_removing_keeps_settled(tmp_path: Path):
    ep = _write_fake_episode(tmp_path)
    samples = load_episode_samples(ep, interpolate_tips=False)
    assert [s.stem for s in samples] == ["000000", "000001", "000002", "000003"]
    by_state = {s.state: s for s in samples}
    assert by_state["SETTLED"].contact == 1
    assert by_state["SETTLED"].phase is None


def test_phase_excludes_settled(tmp_path: Path):
    ep = _write_fake_episode(tmp_path)
    samples = load_episode_samples(ep, interpolate_tips=False)
    phase = phase_samples(samples)
    assert {s.state for s in phase} == {"TOUCHING", "FLOWING"}
