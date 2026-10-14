"""Chip readers for the КосмоХакатон fire-monitoring dataset.

Disk on the dev machine is essentially full, so `TarChipSource` reads every
GeoTIFF straight out of `fire-train-renamed.tar` into memory (via
rasterio.io.MemoryFile) and never extracts anything to disk. `DirChipSource`
is the same interface for a plain extracted folder, for when the organizers'
test set turns out to ship as loose files instead of a tar - swap the source,
keep every notebook/inference cell unchanged.

Band names are read from each GeoTIFF's own band descriptions (they're
embedded in the files), not hardcoded from the case statement PDF, so this
keeps working even if a channel gets reordered.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile


class _ChipSourceBase:
    def _read_bytes(self, relpath: str) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        """No-op by default; TarChipSource overrides this to close its tarfile handle."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read_named(self, relpath: str) -> dict:
        """Multi-band GeoTIFF -> dict keyed by band description, plus '_bands' (C,H,W),
        '_names' (tuple) and '_profile' (rasterio profile, has crs/transform/nodata)."""
        raw = self._read_bytes(relpath)
        with MemoryFile(raw) as mem, mem.open() as src:
            names = src.descriptions
            arr = src.read()
            profile = src.profile
        out = {"_bands": arr, "_names": tuple(names), "_profile": profile}
        for i, name in enumerate(names):
            if name:
                out[name] = arr[i]
        return out

    def read_csv(self, relpath: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self._read_bytes(relpath)))

    def meta(self, kind: str) -> pd.DataFrame:
        return self.read_csv(f"{kind}/meta.csv")

    def chip_ids(self, kind: str) -> list[str]:
        return self.meta(kind)["chip_id"].tolist()

    def _try_load_mask(self, relpath: str) -> tuple[np.ndarray | None, float | None]:
        """Test chips ship with no mask (that's what we predict) - returns (None, None)
        instead of raising, so load_af/load_bs work unchanged on train, val AND test."""
        try:
            mask_raw = self.read_named(relpath)
        except FileNotFoundError:
            return None, None
        return mask_raw["_bands"][0], mask_raw["_profile"].get("nodata")

    def load_af(self, chip_id: str) -> dict:
        viirs = self.read_named(f"af/viirs/{chip_id}_VIIRS_I1-I5.tif")
        aux = self.read_named(f"af/aux/{chip_id}_AUX.tif")
        mask, mask_nodata = self._try_load_mask(f"af/masks/{chip_id}_MASK.tif")
        return {
            "chip_id": chip_id,
            "viirs": viirs,
            "aux": aux,
            "mask": mask,
            "mask_nodata": mask_nodata,
        }

    def load_bs(self, chip_id: str) -> dict:
        s2_pre = self.read_named(f"bs/sentinel2_pre/{chip_id}_Sentinel-2_pre.tif")
        s2_post = self.read_named(f"bs/sentinel2_post/{chip_id}_Sentinel-2_post.tif")
        s1_pre = self.read_named(f"bs/sentinel1_pre/{chip_id}_Sentinel-1_pre.tif")
        s1_post = self.read_named(f"bs/sentinel1_post/{chip_id}_Sentinel-1_post.tif")
        aux = self.read_named(f"bs/aux/{chip_id}_AUX.tif")
        mask, mask_nodata = self._try_load_mask(f"bs/masks/{chip_id}_MASK.tif")
        return {
            "chip_id": chip_id,
            "s2_pre": s2_pre,
            "s2_post": s2_post,
            "s1_pre": s1_pre,
            "s1_post": s1_post,
            "aux": aux,
            "mask": mask,
            "mask_nodata": mask_nodata,
        }


class TarChipSource(_ChipSourceBase):
    """Reads chips directly out of fire-train-renamed.tar, no extraction to disk."""

    def __init__(self, tar_path: str | Path, root: str = "train"):
        self.tar_path = str(tar_path)
        self.root = root
        self._tar = tarfile.open(self.tar_path)

    def _read_bytes(self, relpath: str) -> bytes:
        member = f"{self.root}/{relpath}"
        try:
            buf = self._tar.extractfile(member)
        except KeyError:
            # tarfile raises KeyError (not FileNotFoundError) for a missing member -
            # normalize it so callers (e.g. _try_load_mask) can catch one exception
            # type regardless of which ChipSource they're using.
            raise FileNotFoundError(member) from None
        if buf is None:
            raise FileNotFoundError(member)
        return buf.read()

    def close(self) -> None:
        self._tar.close()


class DirChipSource(_ChipSourceBase):
    """Same interface as TarChipSource, for an already-extracted directory
    (e.g. the organizers' --data-dir at inference time)."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def _read_bytes(self, relpath: str) -> bytes:
        path = self.root_dir / relpath
        return path.read_bytes()


def nodata_to_ignore(mask: np.ndarray, nodata: float | None) -> np.ndarray:
    """Boolean valid-pixel mask (True = usable) from a mask array + its nodata value."""
    if nodata is None:
        return np.ones_like(mask, dtype=bool)
    return mask != nodata


def iter_chips(source: _ChipSourceBase, kind: str, chip_ids: Iterable[str] | None = None):
    """Yield load_af()/load_bs() dicts for every chip_id (or all chips if None)."""
    loader = source.load_af if kind == "af" else source.load_bs
    ids = list(chip_ids) if chip_ids is not None else source.chip_ids(kind)
    for chip_id in ids:
        yield loader(chip_id)
