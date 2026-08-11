from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOFTWARE_WIN = Path(__file__).resolve().parents[4]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.vision.dataset import interpolate_tips_along_stems
from high_level_jobs.epoxy_app.vision.preprocess import crop_around_tip, gaussian_heatmap
from high_level_jobs.epoxy_app.vision.tip_model import TipNet


def test_interpolate_tips():
    stems = [f"{i:06d}" for i in range(5)]
    marked = {"000000": (0.0, 0.0), "000004": (40.0, 20.0)}
    out = interpolate_tips_along_stems(stems, marked)
    assert out["000002"] == (20.0, 10.0)
    assert out["000000"] == (0.0, 0.0)


def test_crop_around_tip_shape():
    img = np.zeros((100, 200, 1), dtype=np.uint8)
    crop = crop_around_tip(img, x=10, y=10, crop_size=64)
    assert crop.shape == (64, 64, 1)


def test_gaussian_heatmap_peak():
    h = gaussian_heatmap(32, 32, 10.0, 15.0, sigma=1.5)
    assert h.shape == (32, 32)
    assert abs(float(h.max()) - 1.0) < 1e-5
    iy, ix = np.unravel_index(int(h.argmax()), h.shape)
    assert ix == 10 and iy == 15


def test_predict_tip_prior_prefers_local_peak():
    """Global blob vs weaker local tip: prior should pick local."""
    import torch

    class FakeTip(TipNet):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.heatmap_size = 64
            # Build a fixed heatmap: strong peak at (50,50), weaker at (20,20)
            heat = torch.full((1, 1, 64, 64), 0.05)
            heat[0, 0, 50, 50] = 0.95  # apply blob
            heat[0, 0, 20, 20] = 0.55  # tip
            self._heat = heat

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Return logits so sigmoid ≈ heat (logit of p)
            p = self._heat.clamp(1e-4, 1 - 1e-4)
            return torch.log(p / (1 - p)).expand(x.size(0), -1, -1, -1)

    model = FakeTip().eval()
    dummy = torch.zeros(1, 1, 224, 224)
    # Global: blob wins
    gx, gy, _ = model.predict_tip(dummy, frame_w=640, frame_h=480)
    assert float(gx[0]) > 400  # ~50/64 * 640
    # Prior near tip: local wins
    lx, ly, _ = model.predict_tip(
        dummy,
        frame_w=640,
        frame_h=480,
        prior_xy=(200.0, 150.0),
        track_radius_px=120,
        track_sigma_px=40,
    )
    assert float(lx[0]) < 280
    assert float(ly[0]) < 220
