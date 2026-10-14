#!/usr/bin/env python
"""Standalone inference entry point - no Colab, no Kaggle, no Google Drive.

Usage:
    python inference.py --data-dir /path/to/test --weights-dir ./weights --output submission.csv

`--data-dir` must contain the same layout as the organizers' test archive:
    <data-dir>/sample_submission.csv
    <data-dir>/af/viirs/..., <data-dir>/af/aux/...
    <data-dir>/bs/sentinel2_pre/..., .../sentinel2_post/..., .../sentinel1_pre/...,
        .../sentinel1_post/..., .../aux/...
(this is exactly the `test/` folder inside fire-test-renamed, or the raw
fire-test-renamed.tar itself - both are auto-detected, see
`resolve_data_source()` below.)

`--weights-dir` must contain the trained artifacts, produced by
notebooks/01_af_binary_fire_detection.ipynb and 02_bs_burn_severity.ipynb (see
`save_af_artifacts`/`save_bs_artifacts` in src/af_pipeline.py / src/bs_pipeline.py):
    weights/af/af_lightgbm_0.txt, af_lightgbm_1.txt, ... , af_lightgbm_meta.json,
             af_cnn_best.pt (optional, only if the CNN blend won on validation)
    weights/bs/bs_unet_0_best.pt, bs_unet_1_best.pt, ... , bs_unet_meta.json

No network access, no GPU required (falls back to CPU automatically) - CUDA is
used only if available, which speeds up the BS module (siamese U-Net + TTA)
substantially but is not required for correctness.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from dataio import DirChipSource, TarChipSource
from rle import submission_row, validate_submission
import af_pipeline as afp
import bs_pipeline as bsp


def resolve_data_source(data_dir: str):
    """`data_dir` is either an already-extracted test/ folder (has af/, bs/,
    sample_submission.csv directly inside) or the raw fire-test-renamed.tar -
    both are accepted so this script works whether the organizers hand over a
    folder or an archive."""
    if os.path.isfile(data_dir) and data_dir.endswith(".tar"):
        return TarChipSource(data_dir, root="test")
    if os.path.isdir(data_dir):
        # tar-vs-folder auto-detect, same convention as resolve_test_source()
        # in the training notebooks (Kaggle sometimes shows an archive "expanded"
        # in its file browser without it actually being extracted on disk).
        tar_path = os.path.join(data_dir, "fire-test-renamed.tar")
        if os.path.isfile(tar_path):
            return TarChipSource(tar_path, root="test")
        for candidate in (os.path.join(data_dir, "fire-test-renamed", "test"), data_dir):
            if os.path.isfile(os.path.join(candidate, "sample_submission.csv")):
                return DirChipSource(candidate)
        raise FileNotFoundError(
            f"Не нашли sample_submission.csv ни в {data_dir}, ни в его подпапках - "
            "проверьте структуру --data-dir."
        )
    raise FileNotFoundError(f"--data-dir не найден: {data_dir}")


def run_af(source, sample_submission: pd.DataFrame, weights_dir: str, device: str) -> pd.DataFrame:
    af_weights_dir = os.path.join(weights_dir, "af")
    artifacts = afp.load_af_artifacts(af_weights_dir, device=device)
    af_ids = sample_submission.loc[sample_submission["chip_id"].str.startswith("AF_"), "chip_id"].unique()

    rows = []
    for chip_id in tqdm(af_ids, desc="AF inference"):
        mask = afp.predict_af_mask(
            chip_id, artifacts["models"], artifacts["cnn_model"],
            artifacts["global_threshold"], artifacts["per_landcover_thresholds"],
            source, device=device,
        )
        rows.append(submission_row(chip_id, 1, mask))
    return pd.DataFrame(rows)


def run_bs(source, sample_submission: pd.DataFrame, weights_dir: str, device: str, use_tta: bool) -> pd.DataFrame:
    bs_weights_dir = os.path.join(weights_dir, "bs")
    artifacts = bsp.load_bs_artifacts(bs_weights_dir, device=device)
    print(f"BS strategy: {artifacts['strategy']}, burn_threshold={artifacts['burn_threshold']}, "
          f"min_blob_size={artifacts['min_blob_size']}, "
          f"severity_weights={artifacts['severity_weights']}, alpha={artifacts['refiner_alpha']}")
    bs_ids = sample_submission.loc[sample_submission["chip_id"].str.startswith("BS_"), "chip_id"].unique()

    rows = []
    for chip_id in tqdm(bs_ids, desc="BS inference"):
        mask = bsp.predict_bs_mask(
            chip_id, source,
            lgb_model=artifacts["lgb_model"], unet_models=artifacts["unet_models"],
            refiner_models=artifacts["refiner_models"],
            strategy=artifacts["strategy"], burn_threshold=artifacts["burn_threshold"],
            min_blob_size=artifacts["min_blob_size"],
            severity_weights=artifacts["severity_weights"],
            refiner_alpha=artifacts["refiner_alpha"],
            device=device, use_tta=use_tta,
        )
        rows.extend(submission_row(chip_id, c, mask == c) for c in (1, 2, 3))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="test/ folder or fire-test-renamed.tar")
    parser.add_argument("--weights-dir", default="weights", help="folder with af/ and bs/ subfolders (default: ./weights)")
    parser.add_argument("--output", default="submission.csv", help="where to write the final submission.csv")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available, else cpu)")
    parser.add_argument("--no-tta", action="store_true",
                         help="skip BS test-time augmentation (4x fewer forward passes, faster, slightly lower IoU_burn/mIoU_sev)")
    parser.add_argument("--skip-af", action="store_true", help="debug only: skip the AF module")
    parser.add_argument("--skip-bs", action="store_true", help="debug only: skip the BS module")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    t0 = time.time()
    source = resolve_data_source(args.data_dir)
    sample_submission = source.read_csv("sample_submission.csv")

    parts = []
    if not args.skip_af:
        t_af = time.time()
        parts.append(run_af(source, sample_submission, args.weights_dir, device))
        print(f"AF done in {time.time() - t_af:.1f}s")
    if not args.skip_bs:
        t_bs = time.time()
        parts.append(run_bs(source, sample_submission, args.weights_dir, device, use_tta=not args.no_tta))
        print(f"BS done in {time.time() - t_bs:.1f}s")
    source.close()

    submission = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    submission.to_csv(args.output, index=False)

    problems = validate_submission(submission, sample_submission if len(parts) == 2 else
                                    sample_submission[sample_submission["chip_id"].isin(submission["chip_id"])])
    elapsed = time.time() - t0
    print(f"Сохранено: {args.output} {submission.shape}")
    print(f"Проблемы валидации (пусто = ок): {problems}")
    print(f"Полное время (загрузка весов + инференс): {elapsed:.1f}s")

    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
