"""Shared tip-detect → crop → contact/phase inference helpers."""



from __future__ import annotations



from typing import Any



import numpy as np

import torch



from high_level_jobs.epoxy_app.vision.config_util import resolve_repo_path

from high_level_jobs.epoxy_app.vision.model import PHASE_NAMES

from high_level_jobs.epoxy_app.vision.preprocess import (

    build_temporal_mono,

    crop_around_tip,

    to_model_tensor,

)

from high_level_jobs.epoxy_app.vision.tip_model import TipNet, build_tip_model, load_tip_checkpoint

from high_level_jobs.epoxy_app.vision.tip_track import TipTracker





def maybe_load_tip_model(

    vcfg: dict[str, Any],

    device: torch.device,

    cascade_ckpt: dict[str, Any] | None = None,

) -> TipNet | None:

    tip_cfg = vcfg.get("tip", {})

    if not tip_cfg.get("enabled", False):

        return None

    if cascade_ckpt and "tip_state" in cascade_ckpt:

        model = build_tip_model(vcfg)

        model.load_state_dict(cascade_ckpt["tip_state"])

        model.to(device)

        model.eval()

        return model

    tip_path = tip_cfg.get("checkpoint")

    if tip_path:

        path = resolve_repo_path(tip_path)

        if path.exists():

            model, _ = load_tip_checkpoint(path, device=device, cfg=vcfg)

            return model

    # fallback tip_best.pt weights-only

    ckpt_dir = resolve_repo_path(vcfg["paths"]["checkpoints"])

    weights = ckpt_dir / "tip_best.pt"

    if weights.exists():

        model = build_tip_model(vcfg)

        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))

        model.to(device)

        model.eval()

        return model

    return None





@torch.inference_mode()

def infer_frame_cascade(

    *,

    model,

    tip_model: TipNet | None,

    gray: np.ndarray,

    prev_gray: np.ndarray | None,

    device: torch.device,

    image_size: int,

    contact_thr: float,

    tip_cfg: dict[str, Any],

    tip_tracker: TipTracker | None = None,

) -> dict[str, Any]:

    """Run tip (optional) → crop → contact/phase. gray is HxWx1.



    Pass a persistent ``tip_tracker`` across frames so the tip does not snap

    to the stationary apply blob near contact.

    """

    h, w = gray.shape[:2]

    tip_xy = None

    tip_conf = 0.0

    used_crop = False

    crop_size = int(tip_cfg.get("crop_size", 256))

    conf_thr = float(tip_cfg.get("conf_threshold", 0.15))



    if tip_model is not None and tip_cfg.get("enabled", False):

        full_t = to_model_tensor(gray, image_size).unsqueeze(0).to(device)

        if tip_tracker is not None and tip_tracker.cfg.enabled:

            xf, yf, tip_conf = tip_tracker.step(tip_model, full_t, frame_w=w, frame_h=h)

            tip_xy = (xf, yf)

        else:

            x_px, y_px, conf = tip_model.predict_tip(full_t, frame_w=w, frame_h=h)

            tip_conf = float(conf[0])

            tip_xy = (float(x_px[0]), float(y_px[0]))

        if tip_conf >= conf_thr:

            used_crop = True

        else:

            tip_xy = None



    if used_crop and tip_xy is not None:

        g = crop_around_tip(gray, tip_xy[0], tip_xy[1], crop_size)

        p = (

            crop_around_tip(prev_gray, tip_xy[0], tip_xy[1], crop_size)

            if prev_gray is not None

            else None

        )

    else:

        g, p = gray, prev_gray



    temporal = build_temporal_mono(g, p)

    gray_t = to_model_tensor(g, image_size).unsqueeze(0).to(device)

    contact_p = float(model.contact.predict_proba(gray_t)[0].item())

    phase_probs = [1.0, 0.0]

    phase_name = "-"

    if contact_p >= contact_thr:

        tmp_t = to_model_tensor(temporal, image_size).unsqueeze(0).to(device)

        pp = model.phase.predict_proba(tmp_t)[0].cpu().tolist()

        phase_probs = pp

        phase_name = PHASE_NAMES[int(max(range(len(pp)), key=lambda j: pp[j]))]



    return {

        "contact_probability": contact_p,

        "phase_probs": phase_probs,

        "phase_name": phase_name,

        "tip_xy": tip_xy,

        "tip_conf": tip_conf,

        "used_crop": used_crop,

        "crop_size": crop_size,

    }


