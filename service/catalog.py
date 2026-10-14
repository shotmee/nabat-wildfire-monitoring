"""Spatio-temporal index over the chips.

The service answers "what burned inside this box between these dates", so every
chip needs a lon/lat footprint and a time interval. Both come straight from the
organizers' meta.csv - nothing is reconstructed or guessed.

Note on the two modules' notion of time:
  * AF chips carry `acq_datetime` - the instant of the VIIRS overpass, so a
    hotspot either falls inside the requested interval or it does not;
  * BS chips carry `date_pre` and `date_post` - the burn happened somewhere
    between the two Sentinel-2 acquisitions, so a BS chip matches when that
    interval OVERLAPS the request. Requiring containment would silently drop
    fires whose "before" image predates the requested window.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import pandas as pd

from . import geo
from .config import DATA_DIR


@dataclass(frozen=True)
class ChipRecord:
    chip_id: str
    kind: str                 # 'af' | 'bs'
    epsg: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    gsd: float
    width: int
    height: int
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    date_start: dt.date
    date_end: dt.date
    satellite: str | None
    fire_event_id: str | None
    cloud_frac: float | None
    valid_frac: float | None

    @property
    def bounds_wgs84(self):
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)


def _as_date(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts.date()


def resolve_chip_source(data_dir=None, root="train"):
    """Accepts the tar, an extracted folder, or the parent of either - the same
    tar-vs-folder ambiguity the training notebook has to deal with, because
    Kaggle sometimes shows an archive 'expanded' without extracting it."""
    from dataio import DirChipSource, TarChipSource

    data_dir = str(data_dir or DATA_DIR)
    tar_path = os.path.join(data_dir, f"fire-{root}-renamed.tar")
    if os.path.isfile(data_dir) and data_dir.endswith(".tar"):
        return TarChipSource(data_dir, root=root)
    if os.path.isfile(tar_path):
        return TarChipSource(tar_path, root=root)
    for candidate in (os.path.join(data_dir, f"fire-{root}-renamed", root),
                      os.path.join(data_dir, root), data_dir):
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "af")):
            return DirChipSource(candidate)
    raise FileNotFoundError(
        f"Не нашли данные '{root}' в {data_dir}. Укажите каталог с "
        f"fire-{root}-renamed.tar через переменную окружения FIRE_DATA_DIR."
    )


class Catalog:
    """In-memory index. 644 chips - a dataframe scan is microseconds, an R-tree
    would be premature."""

    def __init__(self, source):
        self.source = source
        self.records: dict[str, ChipRecord] = {}
        for kind in ("af", "bs"):
            for rec in self._build_records(source.meta(kind), kind):
                self.records[rec.chip_id] = rec
        self.af_ids = [r.chip_id for r in self.records.values() if r.kind == "af"]
        self.bs_ids = [r.chip_id for r in self.records.values() if r.kind == "bs"]

    @staticmethod
    def _build_records(meta: pd.DataFrame, kind: str):
        for row in meta.itertuples(index=False):
            epsg = int(row.epsg)
            bounds = (float(row.x_min), float(row.y_min), float(row.x_max), float(row.y_max))
            lon_min, lat_min, lon_max, lat_max = geo.bounds_to_wgs84(bounds, epsg)
            if kind == "af":
                start = end = _as_date(getattr(row, "acq_datetime", None))
            else:
                start = _as_date(getattr(row, "date_pre", None))
                end = _as_date(getattr(row, "date_post", None))
            if start is None or end is None:
                continue
            yield ChipRecord(
                chip_id=row.chip_id, kind=kind, epsg=epsg,
                x_min=bounds[0], y_min=bounds[1], x_max=bounds[2], y_max=bounds[3],
                gsd=float(row.gsd), width=int(row.width), height=int(row.height),
                lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max,
                date_start=start, date_end=end,
                satellite=_clean(getattr(row, "satellite", None)),
                fire_event_id=_clean(getattr(row, "fire_event_id", None)),
                cloud_frac=_clean_float(getattr(row, "cloud_frac", None)),
                valid_frac=_clean_float(getattr(row, "valid_frac", None)),
            )

    def extent(self):
        recs = list(self.records.values())
        return {
            "bbox": [min(r.lon_min for r in recs), min(r.lat_min for r in recs),
                      max(r.lon_max for r in recs), max(r.lat_max for r in recs)],
            "date_from": min(r.date_start for r in recs).isoformat(),
            "date_to": max(r.date_end for r in recs).isoformat(),
            "n_af_chips": len(self.af_ids),
            "n_bs_chips": len(self.bs_ids),
        }

    def select(self, bbox, date_from: dt.date, date_to: dt.date, kind: str):
        """Chips of `kind` whose footprint intersects `bbox` and whose time
        interval overlaps [date_from, date_to]."""
        out = []
        for rec in self.records.values():
            if rec.kind != kind:
                continue
            if not geo.bbox_overlaps(rec.bounds_wgs84, bbox):
                continue
            if rec.date_end < date_from or rec.date_start > date_to:
                continue
            out.append(rec)
        out.sort(key=lambda r: (r.date_start, r.chip_id))
        return out


def _clean(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def _clean_float(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
