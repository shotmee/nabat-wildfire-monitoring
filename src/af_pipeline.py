"""AF module (active-fire detection) inference pipeline: feature engineering,
LightGBM ensemble, secondary CNN, max-blending, thresholds, postprocessing.

This is the SAME logic as embedded in notebooks/01_af_binary_fire_detection.ipynb
(via build_notebooks.py) - kept here too as a plain importable module so
`inference.py` can run outside Colab/Jupyter, without Kaggle/Google Drive, using
just a local directory of weights. If you change feature engineering or model
architecture here, mirror the change in build_notebooks.py (and vice versa) -
the two are not auto-synced.

Only inference-time code lives here (feature extraction, forward pass, blending,
thresholding, postprocessing, weight load/save). Training-only code (building the
hard-negative-mined pixel table, the LightGBM/CNN training loops, threshold
search) stays in the notebook, since training only ever happens on Colab.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.ndimage import uniform_filter

AF_FEATURE_NAMES = [
    "I1", "I2", "I3", "I4", "I5", "diff_I4_I5",
    "z_I4_w9", "z_diff_w9",
    "z_I4_w21", "z_diff_w21", "bg_I4", "bg_diff",
    "z_I4_w41", "z_diff_w41", "z_I4_scale_ratio",
    "solar_zenith", "sensor_zenith", "is_night",
    "I1_I2_ratio", "I2_I3_ratio", "mean_refl_day",
    "landcover", "dem", "t2m", "rh2m", "wind_speed",
    "z_I4_x_cropland", "z_diff_x_urban",
]
AF_LANDCOVER_IDX = AF_FEATURE_NAMES.index("landcover")
AF_LANDCOVER_CROPLAND_CODE = 40  # ESA WorldCover: cropland
AF_LANDCOVER_URBAN_CODE = 50  # ESA WorldCover: built-up


def _fill_nan(arr, fallback=280.0):
    if np.isnan(arr).any():
        fill = np.nanmean(arr) if not np.isnan(arr).all() else fallback
        arr = np.where(np.isnan(arr), fill, arr)
    return arr


def _local_mean_std(arr, win=21):
    mean = uniform_filter(arr, size=win)
    sq_mean = uniform_filter(arr ** 2, size=win)
    std = np.sqrt(np.clip(sq_mean - mean ** 2, 0, None)) + 1e-3
    return mean, std


def extract_af_features(chip, win=21):
    """chip = source.load_af(...). Returns (C, H, W) float32 in AF_FEATURE_NAMES order."""
    v, aux = chip["viirs"], chip["aux"]
    I1, I2, I3 = v["I1"], v["I2"], v["I3"]
    I4, I5 = _fill_nan(v["I4"]), _fill_nan(v["I5"])
    diff = I4 - I5

    bg_I4_9, std_I4_9 = _local_mean_std(I4, 9)
    bg_diff_9, std_diff_9 = _local_mean_std(diff, 9)
    z_I4_w9 = (I4 - bg_I4_9) / std_I4_9
    z_diff_w9 = (diff - bg_diff_9) / std_diff_9

    bg_I4, std_I4 = _local_mean_std(I4, win)
    bg_diff, std_diff = _local_mean_std(diff, win)
    z_I4_w21 = (I4 - bg_I4) / std_I4
    z_diff_w21 = (diff - bg_diff) / std_diff

    bg_I4_41, std_I4_41 = _local_mean_std(I4, 41)
    bg_diff_41, std_diff_41 = _local_mean_std(diff, 41)
    z_I4_w41 = (I4 - bg_I4_41) / std_I4_41
    z_diff_w41 = (diff - bg_diff_41) / std_diff_41
    z_I4_scale_ratio = z_I4_w9 / (np.abs(z_I4_w41) + 1e-3)

    solar_zenith = v["solar_zenith"]
    is_night = (solar_zenith > 90).astype(np.float32)
    I1f = np.nan_to_num(I1, nan=0.0)
    I2f = np.nan_to_num(I2, nan=0.0)
    I3f = np.nan_to_num(I3, nan=0.0)
    I1_I2_ratio = I1f / (I2f + 1e-3)
    I2_I3_ratio = I2f / (I3f + 1e-3)
    mean_refl_day = (I1f + I2f + I3f) / 3.0 * (1 - is_night)

    landcover = aux["landcover"]
    is_cropland = (landcover == AF_LANDCOVER_CROPLAND_CODE).astype(np.float32)
    is_urban = (landcover == AF_LANDCOVER_URBAN_CODE).astype(np.float32)

    feats = np.stack([
        I1, I2, I3, I4, I5, diff,
        z_I4_w9, z_diff_w9,
        z_I4_w21, z_diff_w21, bg_I4, bg_diff,
        z_I4_w41, z_diff_w41, z_I4_scale_ratio,
        solar_zenith, v["sensor_zenith"], is_night,
        I1_I2_ratio, I2_I3_ratio, mean_refl_day,
        landcover, aux["dem"], aux["t2m"], aux["rh2m"], aux["wind_speed"],
        z_I4_w21 * is_cropland, z_diff_w21 * is_urban,
    ], axis=0).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def predict_af_proba(chip, model, win=21):
    """Full-image inference (all pixels, no sampling). Works on test chips (no mask).
    A model trained with objective='none' (focal loss) predicts a raw logit -
    converted to a probability via the `_af_focal` flag set on that Booster."""
    feats = extract_af_features(chip, win=win)
    flat = feats.reshape(feats.shape[0], -1).T
    raw = model.predict(flat, num_iteration=model.best_iteration)
    if getattr(model, "_af_focal", False):
        raw = 1.0 / (1.0 + np.exp(-raw))
    return raw.reshape(feats.shape[1:])


def predict_af_proba_ensemble(chip, models, win=21):
    """Equal-weight average across the LightGBM ensemble (xView2 1st place pattern)."""
    probs = [predict_af_proba(chip, m, win=win) for m in models]
    return np.mean(probs, axis=0)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AFCNN(nn.Module):
    """Compact U-Net for binary fire segmentation - a second, spatial opinion on
    top of the per-pixel LightGBM ensemble."""

    def __init__(self, in_channels, base=16):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(in_channels, affine=False)
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x = self.input_norm(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1).squeeze(1)  # (N, H, W) logits


def predict_af_proba_blend(chip, models, cnn_model, win=21, device="cpu"):
    """max() of the LightGBM ensemble and the CNN, not their average - the
    DrivenData STAC Overflow 1st-place pattern: max of two independent models
    catches pixels either one misses on its own. cnn_model=None -> LightGBM only
    (this is what training picks when the CNN turns out weaker on validation -
    see notebooks/01_af_binary_fire_detection.ipynb section 10)."""
    p_lgb = predict_af_proba_ensemble(chip, models, win=win)
    if cnn_model is None:
        return p_lgb
    feats = extract_af_features(chip, win=win)
    with torch.no_grad():
        p_cnn = torch.sigmoid(
            cnn_model(torch.from_numpy(feats).unsqueeze(0).to(device))
        ).squeeze(0).cpu().numpy()
    return np.maximum(p_lgb, p_cnn)


def apply_af_thresholds(proba, landcover, global_threshold, per_landcover_thresholds=None):
    if not per_landcover_thresholds:
        return proba >= global_threshold
    thr_map = np.full(landcover.shape, global_threshold, dtype=np.float32)
    for code, t in per_landcover_thresholds.items():
        thr_map[landcover == int(code)] = t
    return proba >= thr_map


def confidence_tiered_filter(proba, pred_mask, high_conf=0.9, min_blob_size=3):
    """Drop a small connected component only if it's ALSO low-confidence; a
    single high-confidence pixel is kept untouched (median positive AF chip is
    only ~21 pixels - a flat area filter would erase real small fires)."""
    labeled, n = ndi.label(pred_mask)
    keep = np.zeros_like(pred_mask, dtype=bool)
    for i in range(1, n + 1):
        blob = labeled == i
        if blob.sum() >= min_blob_size or proba[blob].max() >= high_conf:
            keep |= blob
    return keep


def predict_af_mask(chip_id, models, cnn_model, global_threshold, per_landcover_thresholds,
                     source, device="cpu"):
    """Full AF pipeline for one chip -> boolean mask. RLE-encoding is the caller's
    job (`submission_row(chip_id, 1, mask)` from src/rle.py), kept out of this
    module so it has no dependency on rle.py's import context."""
    chip = source.load_af(chip_id)
    proba = predict_af_proba_blend(chip, models, cnn_model, device=device)
    pred_mask = apply_af_thresholds(proba, chip["aux"]["landcover"], global_threshold,
                                     per_landcover_thresholds)
    return confidence_tiered_filter(proba, pred_mask)


def save_af_artifacts(weights_dir, models, cnn_model, global_threshold, per_landcover_thresholds,
                       use_cnn_in_blend, best_iterations, focal_model_idx, seed):
    """Mirrors exactly what the training notebook writes to CHECKPOINT_DIR - a
    weights folder saved this way can be copied straight into `weights/af/` for
    `inference.py` (or vice versa: files downloaded from Colab's CHECKPOINT_DIR
    already have the right names, drop them in unchanged)."""
    os.makedirs(weights_dir, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(os.path.join(weights_dir, f"af_lightgbm_{i}.txt"))
    if cnn_model is not None:
        torch.save(cnn_model.state_dict(), os.path.join(weights_dir, "af_cnn_best.pt"))
    meta = {
        "feature_names": AF_FEATURE_NAMES,
        "categorical_feature_idx": [AF_LANDCOVER_IDX],
        "n_seeds": len(models),
        "best_iterations": best_iterations,
        "focal_model_idx": focal_model_idx,
        "seed": seed,
        "global_threshold": float(global_threshold),
        "per_landcover_thresholds": per_landcover_thresholds,
        "use_cnn_in_blend": use_cnn_in_blend,
    }
    with open(os.path.join(weights_dir, "af_lightgbm_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_booster(lgb, path):
    """Loads a LightGBM model WITHOUT handing the path to the C library.

    `lgb.Booster(model_file=...)` opens the file inside LightGBM's native code,
    which on Windows uses the system codepage and fails outright on any path
    containing non-ASCII characters - a checkout under "C:\\Users\\...\\Рабочий
    стол\\..." raises "Could not open <path>" even though the file is there and
    os.path.isfile() says True. Reading the text in Python and passing
    `model_str` sidesteps the native file handling entirely.
    """
    with open(path, encoding="utf-8") as f:
        return lgb.Booster(model_str=f.read())


def load_af_artifacts(weights_dir, device="cpu"):
    """Returns a dict: models (list of lgb.Booster), cnn_model (AFCNN or None),
    global_threshold (float), per_landcover_thresholds (dict or None)."""
    import lightgbm as lgb  # local import: only needed at inference time

    meta_path = os.path.join(weights_dir, "af_lightgbm_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    models = []
    for i in range(meta["n_seeds"]):
        m = _load_booster(lgb, os.path.join(weights_dir, f"af_lightgbm_{i}.txt"))
        # a Booster reloaded from disk has no .best_iteration of its own - restore it
        m.best_iteration = meta["best_iterations"][i]
        if i == meta.get("focal_model_idx"):
            m._af_focal = True
        models.append(m)

    cnn_model = None
    cnn_path = os.path.join(weights_dir, "af_cnn_best.pt")
    if meta.get("use_cnn_in_blend") and os.path.exists(cnn_path):
        cnn_model = AFCNN(in_channels=len(AF_FEATURE_NAMES)).to(device)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn_model.eval()

    return {
        "models": models,
        "cnn_model": cnn_model,
        "global_threshold": meta.get("global_threshold", 0.5),
        "per_landcover_thresholds": meta.get("per_landcover_thresholds"),
    }
