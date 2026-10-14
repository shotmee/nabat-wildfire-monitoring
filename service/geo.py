"""Geometry helpers for the analytical service.

Everything here runs on rasterio alone, which already bundles GDAL and PROJ.
That is deliberate: shapely/pyproj/geopandas are the usual reason a geo stack
fails to install on someone else's machine, and the service has to come up on
the reviewers' machine from `pip install -r requirements.txt` without help.

Coordinate handling, in one place so it cannot drift:
  * chips are stored in per-chip UTM zones (EPSG:326xx) - metric, so areas are
    computed there and nowhere else;
  * GeoJSON is emitted in EPSG:4326, as RFC 7946 requires;
  * the map UI therefore gets lon/lat and never has to know about UTM.
"""
from __future__ import annotations

import numpy as np
import rasterio.features
import rasterio.transform
import rasterio.warp
from rasterio.crs import CRS

WGS84 = CRS.from_epsg(4326)

# Class ids of the severity mask, shared with the model side of the project.
SEVERITY_LABELS = {
    1: "слабое",
    2: "умеренное",
    3: "сильное",
}


def chip_transform(x_min: float, y_max: float, gsd: float):
    """Affine transform of a chip: origin is the TOP-LEFT corner, and y grows
    downwards in raster space, hence y_max rather than y_min."""
    return rasterio.transform.from_origin(x_min, y_max, gsd, gsd)


def pixel_area_ha(gsd: float) -> float:
    """Area of one pixel in hectares.

    Verified against the organizers' own `burn_area_ha` column: for
    BS_tr_000001 the mask holds 11360+10+0 = 11370 burnt pixels at gsd=20 m,
    11370 * 400 / 10000 = 454.8 ha, and meta.csv says exactly 454.8.
    """
    return (gsd * gsd) / 10000.0


def bounds_to_wgs84(bounds, epsg: int):
    """(x_min, y_min, x_max, y_max) in the chip CRS -> the same in lon/lat."""
    return rasterio.warp.transform_bounds(CRS.from_epsg(int(epsg)), WGS84, *bounds)


def bbox_overlaps(a, b) -> bool:
    """Interval overlap of two (min_x, min_y, max_x, max_y) boxes."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def point_in_ring(x: float, y: float, ring) -> bool:
    """Ray casting. Used only to refine a polygon query - chip selection itself
    goes through bounding boxes, so this runs on few points."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        if (y1 > y) != (y2 > y):
            x_at_y = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
            if x < x_at_y:
                inside = not inside
    return inside


def point_in_polygon(x: float, y: float, geometry: dict) -> bool:
    """GeoJSON Polygon / MultiPolygon containment, holes included."""
    polys = ([geometry["coordinates"]] if geometry["type"] == "Polygon"
             else geometry["coordinates"])
    for rings in polys:
        if not rings:
            continue
        if point_in_ring(x, y, rings[0]) and not any(
                point_in_ring(x, y, hole) for hole in rings[1:]):
            return True
    return False


def geometry_bounds(geometry: dict):
    """Bounding box of a GeoJSON Polygon/MultiPolygon."""
    polys = ([geometry["coordinates"]] if geometry["type"] == "Polygon"
             else geometry["coordinates"])
    xs, ys = [], []
    for rings in polys:
        for ring in rings:
            for pt in ring:
                xs.append(pt[0])
                ys.append(pt[1])
    if not xs:
        raise ValueError("пустая геометрия в запросе")
    return min(xs), min(ys), max(xs), max(ys)


def polygonize_severity(mask: np.ndarray, epsg: int, x_min: float, y_max: float,
                         gsd: float, chip_id: str):
    """Severity mask (0..3) -> list of GeoJSON features in EPSG:4326.

    One feature per connected region of one severity class. Area is counted in
    PIXELS and converted with `pixel_area_ha`, not measured on the reprojected
    lon/lat polygon: after reprojection to a geographic CRS a planar area is
    meaningless, and pixel counting is what reproduces the organizers' numbers
    exactly.
    """
    from scipy import ndimage as ndi

    src_crs = CRS.from_epsg(int(epsg))
    transform = chip_transform(x_min, y_max, gsd)
    px_ha = pixel_area_ha(gsd)
    features = []

    for severity in (1, 2, 3):
        binary = mask == severity
        if not binary.any():
            continue
        # Label first, then polygonize the LABEL raster: rasterio returns one
        # geometry per connected region, and np.bincount gives every region's
        # exact pixel count in a single pass. Rasterizing each polygon back to
        # count its pixels would be quadratic in the number of contours.
        labels, n_labels = ndi.label(binary, structure=np.ones((3, 3), dtype=int))
        if n_labels == 0:
            continue
        counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
        for geom, value in rasterio.features.shapes(
                labels.astype(np.int32), mask=binary, transform=transform, connectivity=8):
            label_id = int(value)
            if label_id == 0:
                continue
            n_px = int(counts[label_id])
            geom_wgs = rasterio.warp.transform_geom(src_crs, WGS84, geom, precision=6)
            lon, lat = _representative_point(geom_wgs)
            features.append({
                "type": "Feature",
                "geometry": geom_wgs,
                "properties": {
                    "contour_id": f"{chip_id}_s{severity}_{label_id:04d}",
                    "chip_id": chip_id,
                    "severity_class": severity,
                    "severity_label": SEVERITY_LABELS[severity],
                    "area_ha": round(n_px * px_ha, 4),
                    "pixels": n_px,
                    "gsd_m": float(gsd),
                    "source_epsg": int(epsg),
                    "lon": lon,
                    "lat": lat,
                },
            })
    return features


def hotspots_to_features(mask: np.ndarray, epsg: int, x_min: float, y_max: float,
                          gsd: float, chip_id: str, acq_datetime, satellite):
    """Active-fire mask -> one point Feature per burning pixel, at the pixel
    centre. VIIRS pixels are 375 m, so a point per pixel is the honest
    representation - the same way FIRMS publishes thermal anomalies."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return []
    src_crs = CRS.from_epsg(int(epsg))
    # pixel centres in the chip CRS
    east = x_min + (xs + 0.5) * gsd
    north = y_max - (ys + 0.5) * gsd
    lons, lats = rasterio.warp.transform(src_crs, WGS84, east.tolist(), north.tolist())
    px_ha = pixel_area_ha(gsd)
    return [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
        "properties": {
            "hotspot_id": f"{chip_id}_{i:05d}",
            "chip_id": chip_id,
            "acq_datetime": acq_datetime,
            "satellite": satellite,
            "pixel_area_ha": round(px_ha, 4),
            "gsd_m": float(gsd),
        },
    } for i, (lon, lat) in enumerate(zip(lons, lats))]


def _representative_point(geom_wgs: dict):
    """Centroid of the outer ring's vertices - good enough to refine a polygon
    query and to place a marker; we never use it for measurement."""
    rings = (geom_wgs["coordinates"] if geom_wgs["type"] == "Polygon"
             else geom_wgs["coordinates"][0])
    ring = rings[0]
    return (round(float(np.mean([p[0] for p in ring])), 6),
            round(float(np.mean([p[1] for p in ring])), 6))
