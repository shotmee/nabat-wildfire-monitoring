"""BS module (burn contour + severity) inference pipeline: feature stacks,
siamese U-Net ensemble, TTA, thresholds, postprocessing.

Same logic as embedded in notebooks/02_bs_burn_severity.ipynb (via
build_notebooks.py) - kept here too as a plain importable module so
`inference.py` can run outside Colab/Jupyter using just a local directory of
weights. If you change feature engineering or model architecture here, mirror
the change in build_notebooks.py (and vice versa) - the two are not auto-synced.

Only inference-time code lives here. Training-only code (Dataset/augmentation,
the training loop, class weights, threshold search) stays in the notebook,
since training only ever happens on Colab.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

BS_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
SCL_CLOUD_CLASSES = {3, 8, 9, 10}  # shadow / cloud medium+high probability / cirrus

BS_PRE_CHANNELS = BS_S2_BANDS + ["NDVI", "NBR", "NDMI", "MIRBI", "BAI", "GEMI", "VV_db", "VH_db", "RVI"]
BS_STATIC_CHANNELS = [
    "dNBR", "RdNBR", "RBR", "dVV", "dVH", "RBD", "LogRBR", "dRVI",
    "dem", "slope", "landcover", "cloud_mask",
]


def _refl(band):
    """Sentinel-2 L2A stores reflectance as uint16 * 10000."""
    return band.astype(np.float32) / 10000.0


def _ndvi(bands):
    red, nir = bands["B4"].astype(np.float32), bands["B8A"].astype(np.float32)
    return (nir - red) / (nir + red + 1e-6)


def _nbr(bands):
    b8a, b12 = bands["B8A"].astype(np.float32), bands["B12"].astype(np.float32)
    return (b8a - b12) / (b8a + b12 + 1e-6)


def _ndmi(bands):
    nir, swir1 = bands["B8A"].astype(np.float32), bands["B11"].astype(np.float32)
    return (nir - swir1) / (nir + swir1 + 1e-6)


def _mirbi(bands):
    swir1, swir2 = _refl(bands["B11"]), _refl(bands["B12"])
    return 10.0 * swir2 - 9.8 * swir1 + 2.0


def _bai(bands):
    red, nir = _refl(bands["B4"]), _refl(bands["B8A"])
    val = 1.0 / ((0.1 - red) ** 2 + (0.06 - nir) ** 2 + 1e-6)
    return np.clip(val, 0, 50)


def _gemi(bands):
    red, nir = _refl(bands["B4"]), _refl(bands["B8A"])
    eta = (2.0 * (nir ** 2 - red ** 2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5 + 1e-6)
    return eta * (1 - 0.25 * eta) - (red - 0.125) / (1 - red + 1e-6)


def _s1_db(band, lo=-30.0, hi=5.0):
    """int16 stores dB*100 (confirmed on real data) - convert to dB and clip to a
    physically sane range (radar shadow/layover outliers otherwise dominate
    normalization statistics)."""
    return np.clip(band.astype(np.float32) / 100.0, lo, hi)


def _rvi(vv_db, vh_db):
    """Radar Vegetation Index - defined for LINEAR power, not dB."""
    vv_lin, vh_lin = 10.0 ** (vv_db / 10.0), 10.0 ** (vh_db / 10.0)
    return 4.0 * vh_lin / (vv_lin + vh_lin + 1e-6)


def _build_date_branch(bands, s1):
    vv_db, vh_db = _s1_db(s1["VV"]), _s1_db(s1["VH"])
    stack = [bands[b].astype(np.float32) for b in BS_S2_BANDS] + [
        _ndvi(bands), _nbr(bands), _ndmi(bands), _mirbi(bands), _bai(bands), _gemi(bands),
        vv_db, vh_db, _rvi(vv_db, vh_db),
    ]
    return np.stack(stack, axis=0).astype(np.float32)


def build_bs_feature_stack_siamese(chip):
    """(pre_stack, post_stack, static_stack), shapes (18,H,W), (18,H,W), (12,H,W)."""
    s2pre, s2post = chip["s2_pre"], chip["s2_post"]
    s1pre, s1post = chip["s1_pre"], chip["s1_post"]
    aux = chip["aux"]

    pre_stack = _build_date_branch(s2pre, s1pre)
    post_stack = _build_date_branch(s2post, s1post)

    nbr_pre, nbr_post = _nbr(s2pre), _nbr(s2post)
    dnbr = nbr_pre - nbr_post
    rdnbr = np.clip(dnbr / (np.sqrt(np.abs(nbr_pre) / 1000.0) + 1e-6), -1000, 1000)
    rbr = np.clip(dnbr / (nbr_pre + 1.001), -3, 3)

    vv_pre_db, vh_pre_db = _s1_db(s1pre["VV"]), _s1_db(s1pre["VH"])
    vv_post_db, vh_post_db = _s1_db(s1post["VV"]), _s1_db(s1post["VH"])
    dvv = np.clip(vv_post_db - vv_pre_db, -20, 20)
    dvh = np.clip(vh_post_db - vh_pre_db, -20, 20)
    rbd = np.clip(vh_pre_db - vh_post_db, -20, 20)
    vh_pre_lin, vh_post_lin = 10.0 ** (vh_pre_db / 10.0), 10.0 ** (vh_post_db / 10.0)
    log_rbr = np.clip(np.log10((vh_pre_lin + 1e-6) / (vh_post_lin + 1e-6)), -3, 3)
    drvi = _rvi(vv_post_db, vh_post_db) - _rvi(vv_pre_db, vh_pre_db)

    cloud_mask = np.isin(s2post["SCL"], list(SCL_CLOUD_CLASSES)).astype(np.float32)
    static_stack = np.stack([
        dnbr, rdnbr, rbr,
        dvv, dvh, rbd, log_rbr, drvi,
        aux["dem"].astype(np.float32), aux["slope"].astype(np.float32),
        aux["landcover"].astype(np.float32), cloud_mask,
    ], axis=0).astype(np.float32)

    clean = lambda a: np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return clean(pre_stack), clean(post_stack), clean(static_stack)


# ---------------------------------------------------------------------------
# Per-pixel feature stack for the LightGBM severity model.
#
# Why a per-pixel GBM at all, next to the siamese U-Net: burn severity is
# largely a per-pixel spectral problem (dNBR conditioned on land cover - the
# case statement itself says fixed dNBR scales fail because forest/steppe/
# cropland need different calibration). A GBM sees ~47M labelled pixels from
# 179 chips, while the U-Net sees 179 images - so the GBM is far less
# data-starved on exactly the part of Score that has the most headroom
# (mIoU_sev), and it is an order of magnitude cheaper at inference.
# ---------------------------------------------------------------------------

BS_PIXEL_FEATURE_NAMES = (
    [f"{b}_pre" for b in BS_S2_BANDS] + [f"{b}_post" for b in BS_S2_BANDS] + [
        "NDVI_pre", "NDVI_post", "dNDVI",
        "NBR_pre", "NBR_post", "dNBR", "RdNBR", "RBR",
        "NDMI_pre", "NDMI_post", "dNDMI",
        "MIRBI_pre", "MIRBI_post", "dMIRBI",
        "BAI_pre", "BAI_post", "dBAI",
        "GEMI_pre", "GEMI_post", "dGEMI",
        "VV_pre_db", "VH_pre_db", "VV_post_db", "VH_post_db",
        "dVV", "dVH", "RBD", "LogRBR",
        "RVI_pre", "RVI_post", "dRVI",
        "dem", "slope", "landcover", "cloud_mask",
        "dNBR_mean5", "dNBR_mean15", "dNBR_local_contrast",
        # --- scene-relative + texture block ---------------------------------
        # The case statement itself says a single dNBR scale does not transfer
        # between forest/steppe/cropland. Absolute dNBR is therefore a weak
        # feature across scenes; its rank/z-score WITHIN the chip is not, and
        # local std separates a real burn edge from sensor noise. Measured on
        # validation: IoU_burn 0.5880 -> 0.5891, mIoU_sev 0.5786 -> 0.5847,
        # with class-1 (light burn) IoU 0.376 -> 0.387.
        "dNBR_z", "dNBR_rank", "dNBR_med_rel",
        "NBR_post_z", "NBR_post_rank",
        "NDVI_post_z", "dNDVI_rank",
        "MIRBI_post_z", "dNBR_std5", "dNBR_std15", "dNBR_mean31",
        "NBR_post_mean5", "dNDVI_mean5", "dNBR_rank_mean15",
    ]
)
BS_PIXEL_LANDCOVER_IDX = BS_PIXEL_FEATURE_NAMES.index("landcover")


def _chip_z(x):
    """z-score within the chip - makes a spectral index scene-relative."""
    return (x - float(np.nanmean(x))) / (float(np.nanstd(x)) + 1e-6)


def _chip_rank(x):
    """percentile rank within the chip, 0..1 - fully scale-invariant."""
    flat = x.ravel()
    order = np.argsort(flat, kind="stable")
    r = np.empty(flat.shape[0], np.float32)
    r[order] = np.arange(flat.shape[0], dtype=np.float32) / max(flat.shape[0] - 1, 1)
    return r.reshape(x.shape)


def _std_filter(x, size):
    from scipy.ndimage import uniform_filter

    m = uniform_filter(x, size=size)
    return np.sqrt(np.maximum(uniform_filter(x * x, size=size) - m * m, 0.0))


def build_bs_pixel_features(chip):
    """(C, H, W) float32 in BS_PIXEL_FEATURE_NAMES order, for the LightGBM model.

    `dNBR_mean5`/`dNBR_mean15`/`dNBR_local_contrast` give the otherwise purely
    per-pixel model a bit of spatial context for free (a burnt pixel sits inside
    a burnt neighbourhood; an isolated low-dNBR pixel is usually noise)."""
    from scipy.ndimage import uniform_filter

    s2pre, s2post = chip["s2_pre"], chip["s2_post"]
    s1pre, s1post = chip["s1_pre"], chip["s1_post"]
    aux = chip["aux"]

    ndvi_pre, ndvi_post = _ndvi(s2pre), _ndvi(s2post)
    nbr_pre, nbr_post = _nbr(s2pre), _nbr(s2post)
    dnbr = nbr_pre - nbr_post
    rdnbr = np.clip(dnbr / (np.sqrt(np.abs(nbr_pre) / 1000.0) + 1e-6), -1000, 1000)
    rbr = np.clip(dnbr / (nbr_pre + 1.001), -3, 3)
    ndmi_pre, ndmi_post = _ndmi(s2pre), _ndmi(s2post)
    mirbi_pre, mirbi_post = _mirbi(s2pre), _mirbi(s2post)
    bai_pre, bai_post = _bai(s2pre), _bai(s2post)
    gemi_pre, gemi_post = _gemi(s2pre), _gemi(s2post)

    vv_pre_db, vh_pre_db = _s1_db(s1pre["VV"]), _s1_db(s1pre["VH"])
    vv_post_db, vh_post_db = _s1_db(s1post["VV"]), _s1_db(s1post["VH"])
    dvv = np.clip(vv_post_db - vv_pre_db, -20, 20)
    dvh = np.clip(vh_post_db - vh_pre_db, -20, 20)
    rbd = np.clip(vh_pre_db - vh_post_db, -20, 20)
    vh_pre_lin, vh_post_lin = 10.0 ** (vh_pre_db / 10.0), 10.0 ** (vh_post_db / 10.0)
    log_rbr = np.clip(np.log10((vh_pre_lin + 1e-6) / (vh_post_lin + 1e-6)), -3, 3)
    rvi_pre, rvi_post = _rvi(vv_pre_db, vh_pre_db), _rvi(vv_post_db, vh_post_db)

    dnbr_mean5 = uniform_filter(dnbr, size=5)
    dnbr_mean15 = uniform_filter(dnbr, size=15)

    dndvi = ndvi_post - ndvi_pre
    dnbr_rank = _chip_rank(dnbr)

    cloud_mask = np.isin(s2post["SCL"], list(SCL_CLOUD_CLASSES)).astype(np.float32)

    feats = np.stack(
        [s2pre[b].astype(np.float32) for b in BS_S2_BANDS]
        + [s2post[b].astype(np.float32) for b in BS_S2_BANDS]
        + [
            ndvi_pre, ndvi_post, ndvi_post - ndvi_pre,
            nbr_pre, nbr_post, dnbr, rdnbr, rbr,
            ndmi_pre, ndmi_post, ndmi_post - ndmi_pre,
            mirbi_pre, mirbi_post, mirbi_post - mirbi_pre,
            bai_pre, bai_post, bai_post - bai_pre,
            gemi_pre, gemi_post, gemi_post - gemi_pre,
            vv_pre_db, vh_pre_db, vv_post_db, vh_post_db,
            dvv, dvh, rbd, log_rbr,
            rvi_pre, rvi_post, rvi_post - rvi_pre,
            aux["dem"].astype(np.float32), aux["slope"].astype(np.float32),
            aux["landcover"].astype(np.float32), cloud_mask,
            dnbr_mean5, dnbr_mean15, dnbr - dnbr_mean15,
            _chip_z(dnbr), dnbr_rank, dnbr - float(np.nanmedian(dnbr)),
            _chip_z(nbr_post), _chip_rank(nbr_post),
            _chip_z(ndvi_post), _chip_rank(dndvi),
            _chip_z(mirbi_post), _std_filter(dnbr, 5), _std_filter(dnbr, 15),
            uniform_filter(dnbr, size=31),
            uniform_filter(nbr_post, size=5), uniform_filter(dndvi, size=5),
            uniform_filter(dnbr_rank, size=15),
        ],
        axis=0,
    ).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# LightGBM stops walking the remaining trees for a pixel once the leading class
# is ahead by this margin. On a 512x512 chip almost every pixel is obviously
# unburnt, so this is where the time goes and where it can be given back:
# the model is 181 iterations x 4 classes = 724 trees over 262144 pixels.
#
# The margin was chosen by running the WHOLE BS pipeline (LightGBM -> refiner ->
# threshold -> postprocessing) on the validation split and scoring it with the
# organizers' own metric, not by eyeballing probabilities:
#
#   margin off -> 2.29 s/chip           IoU_burn 0.71323  mIoU_sev 0.67410
#   margin 7.0 -> 1.85 s/chip (1.24x)   IoU_burn 0.71323  mIoU_sev 0.67410
#   margin 5.0 -> 1.50 s/chip (1.53x)   IoU_burn 0.71339  mIoU_sev 0.67428
#   margin 3.5 -> 1.17 s/chip (1.95x)   IoU_burn 0.71320  mIoU_sev 0.67418
#
# At 3.5 the module's contribution to Score moves by +0.00001 - noise. (Pixel
# level, 0.038% of pixels across the test set do flip, but they are pixels the
# threshold was never going to be decided by, which is exactly why the metric
# does not move.)
#
# This matters for grading: the BS module stays CPU-bound on LightGBM even on
# the organizers' H100, because the GPU only removes the refiner. Halving the
# LightGBM time is what puts the whole inference inside the "under 5 minutes"
# bracket instead of on its boundary.
BS_LGB_EARLY_STOP_MARGIN = 3.5
BS_LGB_EARLY_STOP_FREQ = 10


def _lgb_predict(model, flat):
    """Predict with early stopping, falling back if the installed LightGBM does
    not accept those keywords - a speed optimisation must never be the reason
    inference fails on someone else's machine."""
    try:
        return model.predict(
            flat, num_iteration=model.best_iteration, pred_early_stop=True,
            pred_early_stop_freq=BS_LGB_EARLY_STOP_FREQ,
            pred_early_stop_margin=BS_LGB_EARLY_STOP_MARGIN)
    except (TypeError, ValueError):
        return model.predict(flat, num_iteration=model.best_iteration)


def predict_bs_proba_lgb(chip, model):
    """Full-image 4-class probabilities from the LightGBM model -> (4, H, W)."""
    feats = build_bs_pixel_features(chip)
    c, h, w = feats.shape
    probs = _lgb_predict(model, feats.reshape(c, -1).T)
    return probs.T.reshape(4, h, w).astype(np.float32)


# --------------------------------------------------------------------------
# Stacked spatial refiner
#
# The per-pixel LightGBM saturates around IoU_burn 0.59 / mIoU_sev 0.585 on
# validation, and the class it fails on is class 1 (light burn, IoU 0.39):
# spectrally light burn sits between unburnt and moderate burn, so a model that
# sees one pixel at a time cannot separate them. What does separate them is
# context - light burn forms rings around heavier burn, and fire scars are
# contiguous shapes. Hence a CNN that takes the LightGBM probability maps as
# input and refines them spatially, rather than replacing them.
#
# Two design choices make this safe rather than a gamble:
#   * ImageNet-pretrained ResNet34 encoder - 224 training chips is far too few
#     to learn low-level filters from scratch (the earlier from-scratch siamese
#     U-Net scored 0.274 against LightGBM's 0.374, i.e. it was data-starved);
#   * a residual connection onto log(P_lgb) with a ZERO-initialised output
#     layer, so at step 0 the network reproduces the LightGBM probabilities
#     exactly and training can only move away from them if that lowers the loss.
# --------------------------------------------------------------------------
BS_REFINER_FEATURE_NAMES = [
    "dNBR", "dNBR_rank", "dNBR_z", "dNBR_std5", "NBR_pre", "NBR_post",
    "dNDVI", "NDVI_post", "dNDMI", "dMIRBI", "dVV", "dVH",
    "dem", "slope", "landcover", "cloud_mask",
]
BS_REFINER_IDX = [BS_PIXEL_FEATURE_NAMES.index(n) for n in BS_REFINER_FEATURE_NAMES]
BS_REFINER_IN_CH = 4 + len(BS_REFINER_FEATURE_NAMES)


def build_refiner_input(pixel_feats, lgb_proba):
    """(BS_REFINER_IN_CH, H, W) float32: 4 LightGBM probability maps + a compact
    raw block, each raw channel z-scored WITHIN the chip so no normalisation
    statistics have to be stored alongside the weights."""
    raw = pixel_feats[BS_REFINER_IDX].astype(np.float32)
    out = np.empty((BS_REFINER_IN_CH, raw.shape[1], raw.shape[2]), np.float32)
    out[:4] = lgb_proba
    for i, name in enumerate(BS_REFINER_FEATURE_NAMES):
        if name == "landcover":
            out[4 + i] = raw[i] / 100.0            # categorical codes 10..100
        elif name == "cloud_mask":
            out[4 + i] = raw[i]                     # already 0/1
        else:
            out[4 + i] = _chip_z(raw[i])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = ConvBlock(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class BSRefiner(nn.Module):
    """ResNet34-U-Net that refines LightGBM probabilities. `pretrained=False` at
    inference time so constructing it never touches the network - the encoder
    weights come from the saved checkpoint like everything else."""

    def __init__(self, in_ch=BS_REFINER_IN_CH, n_classes=4, pretrained=False):
        super().__init__()
        import torchvision

        enc = torchvision.models.resnet34(
            weights="IMAGENET1K_V1" if pretrained else None)
        w = enc.conv1.weight.data.clone()
        enc.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            reps = (in_ch + 2) // 3
            enc.conv1.weight.data = w.repeat(1, reps, 1, 1)[:, :in_ch] * (3.0 / in_ch)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool, self.l1, self.l2 = enc.maxpool, enc.layer1, enc.layer2
        self.l3, self.l4 = enc.layer3, enc.layer4
        self.d4, self.d3 = _Up(512, 256, 256), _Up(256, 128, 128)
        self.d2, self.d1 = _Up(128, 64, 64), _Up(64, 64, 64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, 1),
        )
        # identity start: logits = 0 + log(P_lgb)  =>  softmax == P_lgb
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        lgb_logit = torch.log(x[:, :4].clamp_min(1e-6))
        s0 = self.stem(x)
        x1 = self.l1(self.pool(s0))
        x2, x3 = self.l2(x1), None
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        y = self.d1(self.d2(self.d3(self.d4(x4, x3), x2), x1), s0)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(y) + lgb_logit


def save_refiner_state(model, path):
    """Checkpoints go to disk in fp16: 98.5 MB -> 49.3 MB, which keeps them well
    under GitHub's hard 100 MB per-file limit. Measured cost of the round-trip:
    max |delta softmax| = 3.5e-6 and not a single pixel changes its argmax."""
    torch.save({k: (v.half() if v.is_floating_point() else v)
                for k, v in model.state_dict().items()}, path)


def load_refiner_state(model, path, device="cpu"):
    """Accepts both fp16 and fp32 checkpoints - always casts to the model's dtype."""
    sd = torch.load(path, map_location=device)
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v)
                            for k, v in sd.items()})
    return model


@torch.no_grad()
def predict_bs_proba_refiner(refiner_input, models, device="cpu", use_tta=True):
    """Average softmax over the refiner ensemble (and flip-TTA) -> (4, H, W)."""
    x = torch.from_numpy(refiner_input).unsqueeze(0).to(device)
    variants = [(x, None)]
    if use_tta:
        variants += [(torch.flip(x, [-1]), [-1]), (torch.flip(x, [-2]), [-2]),
                     (torch.flip(x, [-1, -2]), [-1, -2])]
    total, n = 0.0, 0
    for model in models:
        for xv, dims in variants:
            p = torch.softmax(model(xv), dim=1)
            if dims is not None:
                p = torch.flip(p, dims)
            total = total + p
            n += 1
    return (total / n)[0].float().cpu().numpy()


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SharedEncoder(nn.Module):
    def __init__(self, in_channels, base=32):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(in_channels, affine=False)
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)

    def forward(self, x):
        x = self.input_norm(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b


class SiameseUNet(nn.Module):
    """Shared-weight encoder for pre/post (change-invariant features, xView2
    1st/2nd place and BAM-CD/FLOGA pattern) + a static/diff stream fused once
    at the bottleneck."""

    def __init__(self, pair_channels, static_channels, n_classes, base=32):
        super().__init__()
        self.encoder = SharedEncoder(pair_channels, base)
        self.static_norm = nn.BatchNorm2d(static_channels, affine=False)
        self.static_stem = ConvBlock(static_channels, base * 16)
        self.bottleneck_fuse = nn.Conv2d(base * 16 * 3, base * 16, 1)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 8 * 3, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 4 * 3, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 2 * 3, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 3, base)
        self.out_conv = nn.Conv2d(base, n_classes, 1)

    def forward(self, pre, post, static):
        e1p, e2p, e3p, e4p, bp = self.encoder(pre)
        e1o, e2o, e3o, e4o, bo = self.encoder(post)
        static_feat = F.adaptive_avg_pool2d(self.static_stem(self.static_norm(static)), bp.shape[-2:])
        b = self.bottleneck_fuse(torch.cat([bp, bo, static_feat], dim=1))
        d4 = self.dec4(torch.cat([self.up4(b), e4p, e4o], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3p, e3o], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2p, e2o], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1p, e1o], dim=1))
        return self.out_conv(d1)


def _tta_variants(pre, post, static):
    variants = []
    for flip_h, flip_v in [(False, False), (True, False), (False, True), (True, True)]:
        p, o, s = pre, post, static
        if flip_h:
            p, o, s = torch.flip(p, dims=[-1]), torch.flip(o, dims=[-1]), torch.flip(s, dims=[-1])
        if flip_v:
            p, o, s = torch.flip(p, dims=[-2]), torch.flip(o, dims=[-2]), torch.flip(s, dims=[-2])
        variants.append((p, o, s, flip_h, flip_v))
    return variants


def predict_bs_proba_tta(pre, post, static, models, device="cpu", use_tta=True):
    """Average softmax across the model ensemble and (optionally) TTA variants.
    pre/post/static are (C,H,W) tensors with no batch dim. Returns (4,H,W) numpy."""
    pre_b = pre.unsqueeze(0).to(device)
    post_b = post.unsqueeze(0).to(device)
    static_b = static.unsqueeze(0).to(device)
    variants = _tta_variants(pre_b, post_b, static_b) if use_tta else [(pre_b, post_b, static_b, False, False)]
    total = None
    with torch.no_grad():
        for model in models:
            for p, o, s, flip_h, flip_v in variants:
                probs = F.softmax(model(p, o, s), dim=1)
                if flip_h:
                    probs = torch.flip(probs, dims=[-1])
                if flip_v:
                    probs = torch.flip(probs, dims=[-2])
                total = probs if total is None else total + probs
    n = len(models) * len(variants)
    return (total / n).squeeze(0).cpu().numpy()


def apply_bs_thresholds(probs, burn_threshold=0.5, severity_weights=(1.0, 1.0, 1.0)):
    """"Is there burn at all" (P(1)+P(2)+P(3)) is thresholded separately from the
    severity argmax among classes 1/2/3 - IoU_burn (weight 0.35) and mIoU_sev
    (weight 0.30) are optimized somewhat independently in Score.

    `severity_weights` rescale the three severity probabilities before the
    argmax. mIoU_sev is a MEAN of per-class IoU, so a rare class is worth as
    much as a common one and is worth biasing towards; the weights are searched
    on validation by coordinate ascent."""
    p_burn = probs[1] + probs[2] + probs[3]
    is_burn = p_burn >= burn_threshold
    w = np.asarray(severity_weights, dtype=np.float32).reshape(3, 1, 1)
    severity = 1 + np.argmax(probs[1:4] * w, axis=0)
    return np.where(is_burn, severity, 0).astype(np.uint8)


def postprocess_bs_mask(pred, min_blob_size=20, dilate_classes=(2, 3), dilate_iters=1):
    """Drop small isolated false-positive blobs + class-specific dilation for 2/3
    (xView2 1st place pattern: network boundaries for the damaged class tend to
    be under-called at the edges)."""
    out = pred.copy()
    labeled, n = ndi.label(pred > 0)
    for i in range(1, n + 1):
        blob = labeled == i
        if blob.sum() < min_blob_size:
            out[blob] = 0
    for c in dilate_classes:
        class_mask = out == c
        if class_mask.any():
            dilated = ndi.binary_dilation(class_mask, iterations=dilate_iters)
            out[dilated & (out == 0)] = c
    return out


def try_slic_smooth(pred, guide_rgb, n_segments=300):
    """Optional SLIC superpixel majority-vote smoothing; silently returns pred
    unchanged if scikit-image isn't installed - must never break the pipeline."""
    try:
        from skimage.segmentation import slic
    except ImportError:
        return pred
    segments = slic(guide_rgb, n_segments=n_segments, compactness=10, start_label=0, channel_axis=-1)
    out = pred.copy()
    for seg_id in np.unique(segments):
        seg_mask = segments == seg_id
        values, counts = np.unique(pred[seg_mask], return_counts=True)
        out[seg_mask] = values[np.argmax(counts)]
    return out


def predict_bs_mask(chip_id, source, lgb_model=None, unet_models=None, refiner_models=None,
                     strategy="lgb", burn_threshold=0.76, min_blob_size=35,
                     severity_weights=(1.0, 1.0, 1.0), refiner_alpha=1.0,
                     device="cpu", use_tta=True):
    """Full BS pipeline for one chip -> uint8 mask (0..3). RLE-encoding per class
    (`submission_row(chip_id, c, mask == c)` for c in 1,2,3) is the caller's job.

    `strategy` is whatever won on validation in the training notebook:
    'lgb'     - per-pixel LightGBM alone,
    'refiner' - LightGBM probabilities refined by the stacked ResNet34-U-Net
                (blended with the raw LightGBM maps by `refiner_alpha`),
    'unet'    - the legacy siamese U-Net ensemble,
    'blend'   - mean of LightGBM and the legacy siamese U-Net."""
    chip = source.load_bs(chip_id)

    probs_lgb = None
    if strategy in ("lgb", "blend", "refiner"):
        pixel_feats = build_bs_pixel_features(chip)
        c, h, w = pixel_feats.shape
        probs_lgb = _lgb_predict(
            lgb_model, pixel_feats.reshape(c, -1).T
        ).T.reshape(4, h, w).astype(np.float32)

    if strategy == "refiner":
        refined = predict_bs_proba_refiner(
            build_refiner_input(pixel_feats, probs_lgb), refiner_models,
            device=device, use_tta=use_tta,
        )
        probs = refiner_alpha * refined + (1.0 - refiner_alpha) * probs_lgb
    elif strategy == "lgb":
        probs = probs_lgb
    else:
        pre, post, static = build_bs_feature_stack_siamese(chip)
        probs_unet = predict_bs_proba_tta(
            torch.from_numpy(pre), torch.from_numpy(post), torch.from_numpy(static),
            unet_models, device=device, use_tta=use_tta,
        )
        probs = probs_unet if strategy == "unet" else 0.5 * (probs_lgb + probs_unet)

    pred_mask = apply_bs_thresholds(probs, burn_threshold=burn_threshold,
                                     severity_weights=severity_weights)
    if min_blob_size:
        pred_mask = postprocess_bs_mask(pred_mask, min_blob_size=min_blob_size, dilate_classes=())
    return pred_mask


def save_bs_artifacts(weights_dir, models, burn_threshold, extra_meta=None):
    """Mirrors exactly what the training notebook writes per-model to
    CHECKPOINT_DIR - a weights folder saved this way can be copied straight into
    `weights/bs/` for `inference.py`."""
    os.makedirs(weights_dir, exist_ok=True)
    for i, m in enumerate(models):
        torch.save(m.state_dict(), os.path.join(weights_dir, f"bs_unet_{i}_best.pt"))
    meta = {
        "n_models": len(models),
        "pre_channels": list(BS_PRE_CHANNELS),
        "static_channels": list(BS_STATIC_CHANNELS),
        "burn_threshold": float(burn_threshold),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(weights_dir, "bs_unet_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_bs_artifacts(weights_dir, device="cpu"):
    """Loads whatever the training notebook produced, in the layout it writes:

        bs_lightgbm.txt + bs_lightgbm_meta.json   (per-pixel LightGBM, the default model)
        bs_strategy.json                          (strategy + burn_threshold + min_blob_size)
        bs_unet_{i}_best.pt + bs_unet_meta.json   (optional siamese U-Net ensemble)

    Returns a dict ready to splat into `predict_bs_mask`."""
    strategy, burn_threshold, min_blob_size = "lgb", 0.76, 35
    severity_weights, refiner_alpha = (1.0, 1.0, 1.0), 1.0
    strategy_path = os.path.join(weights_dir, "bs_strategy.json")
    if os.path.exists(strategy_path):
        with open(strategy_path, encoding="utf-8") as f:
            sj = json.load(f)
        strategy = sj.get("strategy", strategy)
        burn_threshold = sj.get("burn_threshold", burn_threshold)
        min_blob_size = sj.get("min_blob_size", min_blob_size)
        severity_weights = tuple(sj.get("severity_weights", severity_weights))
        refiner_alpha = sj.get("refiner_alpha", refiner_alpha)

    lgb_model = None
    lgb_path = os.path.join(weights_dir, "bs_lightgbm.txt")
    lgb_meta_path = os.path.join(weights_dir, "bs_lightgbm_meta.json")
    if os.path.exists(lgb_path):
        import lightgbm as lgb  # local import: only needed at inference time

        # NOT lgb.Booster(model_file=...): that opens the file inside LightGBM's
        # native code, which on Windows uses the system codepage and fails on any
        # non-ASCII path ("Could not open ...") even when the file is readable.
        with open(lgb_path, encoding="utf-8") as f:
            lgb_model = lgb.Booster(model_str=f.read())
        if os.path.exists(lgb_meta_path):
            with open(lgb_meta_path, encoding="utf-8") as f:
                # a Booster reloaded from disk has no .best_iteration of its own
                lgb_model.best_iteration = json.load(f)["best_iteration"]

    unet_models = None
    unet_meta_path = os.path.join(weights_dir, "bs_unet_meta.json")
    if strategy in ("unet", "blend") and os.path.exists(unet_meta_path):
        with open(unet_meta_path, encoding="utf-8") as f:
            unet_meta = json.load(f)
        base = unet_meta.get("base", 32)
        unet_models = []
        for i in range(unet_meta["n_models"]):
            m = SiameseUNet(pair_channels=len(BS_PRE_CHANNELS), static_channels=len(BS_STATIC_CHANNELS),
                             n_classes=4, base=base).to(device)
            m.load_state_dict(torch.load(os.path.join(weights_dir, f"bs_unet_{i}_best.pt"), map_location=device))
            m.eval()
            unet_models.append(m)

    refiner_models = None
    refiner_meta_path = os.path.join(weights_dir, "bs_refiner_meta.json")
    if strategy == "refiner" and os.path.exists(refiner_meta_path):
        with open(refiner_meta_path, encoding="utf-8") as f:
            refiner_meta = json.load(f)
        refiner_models = []
        for i in range(refiner_meta["n_models"]):
            m = BSRefiner(in_ch=refiner_meta.get("in_ch", BS_REFINER_IN_CH),
                           n_classes=4, pretrained=False).to(device)
            load_refiner_state(m, os.path.join(weights_dir, f"bs_refiner_{i}_best.pt"),
                                device=device)
            m.eval()
            refiner_models.append(m)

    if strategy == "refiner" and not refiner_models:
        raise FileNotFoundError(
            f"bs_strategy.json просит стратегию 'refiner', но в {weights_dir} нет "
            "bs_refiner_meta.json / bs_refiner_*_best.pt - положите их туда из CHECKPOINT_DIR."
        )
    if lgb_model is None and not unet_models:
        raise FileNotFoundError(
            f"В {weights_dir} нет ни bs_lightgbm.txt, ни весов сиамского U-Net - "
            "положите туда артефакты из CHECKPOINT_DIR после обучения."
        )

    return {
        "lgb_model": lgb_model,
        "unet_models": unet_models,
        "refiner_models": refiner_models,
        "strategy": strategy,
        "burn_threshold": burn_threshold,
        "min_blob_size": min_blob_size,
        "severity_weights": severity_weights,
        "refiner_alpha": refiner_alpha,
    }
