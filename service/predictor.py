"""Turns a chip into map-ready vector features, running the trained models.

The service deliberately shows what OUR solution predicts, not the reference
masks - a map drawn from the answer key would demonstrate nothing. Ground truth
is still reachable through `source=truth`, which is what makes the service
useful for inspecting errors side by side, and every response says which of the
two produced it.

Vectorising a chip costs ~1-2 s (feature stack + LightGBM + polygonisation), so
results are cached on disk keyed by (source, chip_id). `precompute.py` warms the
whole archive once; after that queries are file reads.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import geo
from .config import CACHE_DIR, DEVICE, SOURCE, WEIGHTS_DIR


class Predictor:
    def __init__(self, catalog, weights_dir=WEIGHTS_DIR, source=SOURCE,
                  cache_dir=CACHE_DIR, device=DEVICE):
        self.catalog = catalog
        self.weights_dir = str(weights_dir)
        self.cache_dir = str(cache_dir)
        self.requested_source = source
        self._device = device
        self._af_artifacts = None
        self._bs_artifacts = None
        self.source, self.fallback_reason = self._resolve_source(source)
        os.makedirs(os.path.join(self.cache_dir, self.source), exist_ok=True)

    # ------------------------------------------------------------------ setup
    def _resolve_source(self, source):
        if source == "truth":
            return "truth", None
        missing = [p for p in ("af/af_lightgbm_meta.json", "bs/bs_lightgbm.txt")
                   if not os.path.isfile(os.path.join(self.weights_dir, p))]
        if missing:
            return "truth", (
                f"в {self.weights_dir} нет {', '.join(missing)} - сервис работает "
                "на эталонных масках. Положите веса из CHECKPOINT_DIR/weights, "
                "чтобы увидеть предсказания модели."
            )
        return "model", None

    def _device_name(self):
        if self._device:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def af_artifacts(self):
        if self._af_artifacts is None:
            import af_pipeline as afp
            self._af_artifacts = afp.load_af_artifacts(
                os.path.join(self.weights_dir, "af"), device=self._device_name())
        return self._af_artifacts

    @property
    def bs_artifacts(self):
        if self._bs_artifacts is None:
            import bs_pipeline as bsp
            self._bs_artifacts = bsp.load_bs_artifacts(
                os.path.join(self.weights_dir, "bs"), device=self._device_name())
        return self._bs_artifacts

    def info(self):
        return {
            "source": self.source,
            "requested_source": self.requested_source,
            "weights_dir": self.weights_dir,
            "device": self._device_name(),
            "fallback_reason": self.fallback_reason,
        }

    # ------------------------------------------------------------------ masks
    def _af_mask(self, chip_id):
        if self.source == "truth":
            return (self.catalog.source.load_af(chip_id)["mask"] > 0).astype(np.uint8)
        import af_pipeline as afp
        art = self.af_artifacts
        return afp.predict_af_mask(
            chip_id, art["models"], art["cnn_model"], art["global_threshold"],
            art["per_landcover_thresholds"], self.catalog.source,
            device=self._device_name()).astype(np.uint8)

    def _bs_mask(self, chip_id):
        if self.source == "truth":
            return self.catalog.source.load_bs(chip_id)["mask"].astype(np.uint8)
        import bs_pipeline as bsp
        art = self.bs_artifacts
        return bsp.predict_bs_mask(
            chip_id, self.catalog.source,
            lgb_model=art["lgb_model"], unet_models=art["unet_models"],
            refiner_models=art["refiner_models"], strategy=art["strategy"],
            burn_threshold=art["burn_threshold"], min_blob_size=art["min_blob_size"],
            severity_weights=art["severity_weights"], refiner_alpha=art["refiner_alpha"],
            device=self._device_name(), use_tta=True).astype(np.uint8)

    # --------------------------------------------------------------- features
    def _cache_path(self, chip_id):
        return os.path.join(self.cache_dir, self.source, f"{chip_id}.json")

    def features(self, chip_id, use_cache=True):
        """GeoJSON features for one chip: points for AF, polygons for BS."""
        path = self._cache_path(chip_id)
        if use_cache and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        rec = self.catalog.records[chip_id]
        if rec.kind == "af":
            feats = geo.hotspots_to_features(
                self._af_mask(chip_id), rec.epsg, rec.x_min, rec.y_max, rec.gsd,
                chip_id, rec.date_start.isoformat(), rec.satellite)
        else:
            feats = geo.polygonize_severity(
                self._bs_mask(chip_id), rec.epsg, rec.x_min, rec.y_max, rec.gsd, chip_id)
            for f in feats:
                f["properties"]["date_pre"] = rec.date_start.isoformat()
                f["properties"]["date_post"] = rec.date_end.isoformat()
                f["properties"]["fire_event_id"] = rec.fire_event_id

        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(feats, f, ensure_ascii=False)
        os.replace(tmp, path)   # атомарно: параллельный запрос не прочитает half-written файл
        return feats

    def cached_count(self):
        directory = os.path.join(self.cache_dir, self.source)
        if not os.path.isdir(directory):
            return 0
        return sum(1 for n in os.listdir(directory) if n.endswith(".json"))
