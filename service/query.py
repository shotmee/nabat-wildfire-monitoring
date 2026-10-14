"""The one place a spatio-temporal request turns into an answer.

Both the REST API and the UI go through `run_query`, so the JSON a reviewer gets
from curl and the numbers drawn on the map cannot diverge.
"""
from __future__ import annotations

import datetime as dt

from . import geo
from .config import MAX_CHIPS_PER_QUERY, MAX_HOTSPOTS_IN_RESPONSE
from .geo import SEVERITY_LABELS


def parse_date(value, fallback: dt.date) -> dt.date:
    if value in (None, ""):
        return fallback
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def normalise_area(bbox=None, geometry=None):
    """A request may carry either a bbox or a GeoJSON polygon.

    Chip selection always runs on bounding boxes (cheap, and chips are big).
    A polygon additionally filters the resulting features by their
    representative point, which is why the response reports both the polygon
    and the box actually scanned - so nobody has to guess what was clipped.
    """
    if geometry:
        if geometry.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("geometry должна быть Polygon или MultiPolygon")
        return geo.geometry_bounds(geometry), geometry
    if bbox:
        if len(bbox) != 4:
            raise ValueError("bbox должен быть [min_lon, min_lat, max_lon, max_lat]")
        lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
        if lon_min > lon_max or lat_min > lat_max:
            raise ValueError("в bbox перепутаны углы: min больше max")
        return (lon_min, lat_min, lon_max, lat_max), None
    raise ValueError("нужен bbox или geometry")


def _representative(feature):
    """Точка, по которой объект относится к запросу: для термоточки это она
    сама, для контура - центр его внешнего кольца (посчитан при векторизации)."""
    props = feature["properties"]
    if "lon" in props:
        return props["lon"], props["lat"]
    return feature["geometry"]["coordinates"]


def _keep(feature, area_bbox, polygon):
    """Объект попадает в выборку, если его представительная точка внутри области.

    Фильтровать обязательно и для bbox тоже: чип AF - это 256 пикселей по 375 м,
    то есть 96 км, и запрос на 10 км пересекает такой чип целиком. Без этой
    проверки в справку уходили бы все термоточки чипа, а площадь гари считалась
    бы по контурам за сотни километров от запрошенной территории.

    Площадь контура берётся ЦЕЛИКОМ, без обрезки по границе запроса: так сумма
    в справке всегда равна сумме `area_ha` выгруженных контуров, а один и тот же
    контур не делится между соседними запросами.
    """
    lon, lat = _representative(feature)
    if not (area_bbox[0] <= lon <= area_bbox[2] and area_bbox[1] <= lat <= area_bbox[3]):
        return False
    return polygon is None or geo.point_in_polygon(lon, lat, polygon)


def run_query(catalog, predictor, bbox=None, geometry=None, date_from=None,
               date_to=None, include_hotspots=True, include_contours=True,
               limit_hotspots=MAX_HOTSPOTS_IN_RESPONSE):
    extent = catalog.extent()
    area_bbox, polygon = normalise_area(bbox, geometry)
    d_from = parse_date(date_from, dt.date.fromisoformat(extent["date_from"]))
    d_to = parse_date(date_to, dt.date.fromisoformat(extent["date_to"]))
    if d_from > d_to:
        raise ValueError("date_from позже date_to")

    af_chips = catalog.select(area_bbox, d_from, d_to, "af") if include_hotspots else []
    bs_chips = catalog.select(area_bbox, d_from, d_to, "bs") if include_contours else []
    truncated_chips = False
    if len(af_chips) + len(bs_chips) > MAX_CHIPS_PER_QUERY:
        truncated_chips = True
        budget = max(MAX_CHIPS_PER_QUERY // 2, 1)
        af_chips, bs_chips = af_chips[:budget], bs_chips[:budget]

    hotspots, contours = [], []
    for rec in af_chips:
        hotspots.extend(f for f in predictor.features(rec.chip_id)
                         if _keep(f, area_bbox, polygon))
    for rec in bs_chips:
        contours.extend(f for f in predictor.features(rec.chip_id)
                         if _keep(f, area_bbox, polygon))

    summary = build_summary(contours, hotspots, af_chips, bs_chips,
                             area_bbox, polygon, d_from, d_to, predictor)
    summary["chips_truncated"] = truncated_chips

    hotspots_out = hotspots[:limit_hotspots] if limit_hotspots else hotspots
    return {
        "summary": summary,
        "hotspots": _fc(hotspots_out),
        "hotspots_truncated": len(hotspots_out) < len(hotspots),
        "burn_contours": _fc(contours),
    }


def build_summary(contours, hotspots, af_chips, bs_chips, area_bbox, polygon,
                   d_from, d_to, predictor):
    """The analytical brief: total burnt hectares plus the split across the three
    severity classes.

    Area is summed from per-contour pixel counts, each converted with
    `gsd^2 / 10000`. That reproduces the organizers' own `burn_area_ha` column
    exactly (checked chip by chip), which is the point - the numbers in the brief
    have to be re-derivable from the supplied data.
    """
    by_class = {}
    for c in (1, 2, 3):
        subset = [f for f in contours if f["properties"]["severity_class"] == c]
        area = sum(f["properties"]["area_ha"] for f in subset)
        by_class[c] = {
            "severity_class": c,
            "severity_label": SEVERITY_LABELS[c],
            "area_ha": round(area, 2),
            "contours": len(subset),
        }
    total = sum(v["area_ha"] for v in by_class.values())
    for v in by_class.values():
        v["share_pct"] = round(100.0 * v["area_ha"] / total, 2) if total else 0.0

    gsds = sorted({f["properties"]["gsd_m"] for f in contours})
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "query": {
            "bbox": [round(v, 6) for v in area_bbox],
            "geometry_type": (polygon or {}).get("type"),
            "date_from": d_from.isoformat(),
            "date_to": d_to.isoformat(),
        },
        "source": predictor.source,
        "source_note": predictor.fallback_reason,
        "total_burn_area_ha": round(total, 2),
        "total_contours": len(contours),
        "by_severity": [by_class[c] for c in (1, 2, 3)],
        "hotspots_count": len(hotspots),
        "hotspot_area_ha": round(sum(f["properties"]["pixel_area_ha"] for f in hotspots), 2),
        "chips": {
            "af": [r.chip_id for r in af_chips],
            "bs": [r.chip_id for r in bs_chips],
            "n_af": len(af_chips),
            "n_bs": len(bs_chips),
        },
        "method": {
            "area_formula": "площадь = число пикселей × gsd² / 10000 (га)",
            "gsd_m": gsds,
            "measured_in": "исходная проекция чипа (UTM, EPSG:326xx) — метрическая",
            "output_crs": "EPSG:4326 (WGS 84), как требует RFC 7946 для GeoJSON",
            "validation": "совпадает с колонкой burn_area_ha из meta.csv организаторов",
            "selection_rule": ("объект относится к запросу, если его представительная "
                                "точка внутри области; площадь контура берётся целиком, "
                                "без обрезки по границе запроса"),
        },
    }


def summary_rows(summary):
    """Flat rows for the CSV version of the brief."""
    q = summary["query"]
    rows = [
        ("запрос: bbox", " ".join(str(v) for v in q["bbox"])),
        ("запрос: период", f"{q['date_from']} .. {q['date_to']}"),
        ("источник контуров", summary["source"]),
        ("чипов AF / BS", f"{summary['chips']['n_af']} / {summary['chips']['n_bs']}"),
        ("термоточек", summary["hotspots_count"]),
        ("площадь термоточек, га", summary["hotspot_area_ha"]),
        ("суммарная площадь гари, га", summary["total_burn_area_ha"]),
        ("контуров всего", summary["total_contours"]),
    ]
    for item in summary["by_severity"]:
        rows.append((f"степень {item['severity_class']} ({item['severity_label']}), га",
                      item["area_ha"]))
        rows.append((f"степень {item['severity_class']} ({item['severity_label']}), доля %",
                      item["share_pct"]))
        rows.append((f"степень {item['severity_class']} ({item['severity_label']}), контуров",
                      item["contours"]))
    rows.append(("формула площади", summary["method"]["area_formula"]))
    rows.append(("проекция измерения", summary["method"]["measured_in"]))
    rows.append(("проекция выгрузки", summary["method"]["output_crs"]))
    return rows


def _fc(features):
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features}
