"""Train proposal detector from labeled bboxes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.grab_sensor.paths import DATASET_DIR, MODEL_DIR  # noqa: E402
from high_level_jobs.grab_sensor.vision.dataset_io import iter_labels, load_label  # noqa: E402
from high_level_jobs.grab_sensor.vision.models import FULL_SIZE, ProposalDetector  # noqa: E402


def _letterbox(rgb: np.ndarray, size: int = FULL_SIZE) -> tuple[np.ndarray, float, float, float]:
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = img
    return canvas, scale, float(pad_x), float(pad_y)


class ProposalDataset(Dataset):
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.samples: list[tuple[Path, dict]] = []
        for lp in iter_labels(data_dir):
            data = load_label(lp)
            if data.get("objects"):
                self.samples.append((lp, data))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        _lp, data = self.samples[idx]
        img_rel = (data.get("files") or {}).get("image")
        rgb = np.asarray(Image.open(self.data_dir / img_rel).convert("RGB"))
        canvas, scale, pad_x, pad_y = _letterbox(rgb, FULL_SIZE)
        heat = np.zeros((FULL_SIZE // 8, FULL_SIZE // 8), dtype=np.float32)
        size_map = np.zeros((2, FULL_SIZE // 8, FULL_SIZE // 8), dtype=np.float32)
        for obj in data["objects"]:
            x0, y0, x1, y1 = obj["bbox_xyxy"]
            cx = (x0 + x1) * 0.5 * scale + pad_x
            cy = (y0 + y1) * 0.5 * scale + pad_y
            bw = max(1.0, (x1 - x0) * scale)
            bh = max(1.0, (y1 - y0) * scale)
            gx = int(np.clip(cx / 8.0, 0, heat.shape[1] - 1))
            gy = int(np.clip(cy / 8.0, 0, heat.shape[0] - 1))
            heat[gy, gx] = 1.0
            size_map[0, gy, gx] = np.log(bw / FULL_SIZE)
            size_map[1, gy, gx] = np.log(bh / FULL_SIZE)
        x = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1)
        return x, torch.from_numpy(heat)[None], torch.from_numpy(size_map)


def train_proposal(
    data_dir: Path,
    out_path: Path,
    *,
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 1e-3,
) -> Path:
    ds = ProposalDataset(data_dir)
    if len(ds) == 0:
        raise RuntimeError(f"No labeled samples in {data_dir}")
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProposalDetector().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, heat, sizes in loader:
            x, heat, sizes = x.to(device), heat.to(device), sizes.to(device)
            pred_h, pred_s = model(x)
            # resize pred to label grid if needed
            if pred_h.shape[-2:] != heat.shape[-2:]:
                pred_h = F.interpolate(pred_h, size=heat.shape[-2:], mode="bilinear", align_corners=False)
                pred_s = F.interpolate(pred_s, size=heat.shape[-2:], mode="bilinear", align_corners=False)
            mask = heat > 0.5
            loss_h = F.binary_cross_entropy_with_logits(pred_h, heat)
            if mask.any():
                loss_s = F.l1_loss(pred_s[mask.expand_as(pred_s)], sizes[mask.expand_as(sizes)])
            else:
                loss_s = pred_s.sum() * 0.0
            loss = loss_h + loss_s
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"[proposal] epoch {epoch + 1}/{epochs} loss={total / max(1, len(loader)):.4f}")

    torch.save({"state_dict": model.state_dict(), "kind": "proposal"}, out_path)
    print(f"Saved {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATASET_DIR)
    p.add_argument("--out", type=Path, default=MODEL_DIR / "proposal.pt")
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args(argv)
    train_proposal(args.data, args.out, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
