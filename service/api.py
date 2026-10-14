"""FastAPI приложение информационно-аналитического сервиса.

Запуск:
    python -m service            (или: uvicorn service.api:app --port 8000)

Каждый ответ можно получить и без интерфейса - у всех эндпоинтов есть GET-форма
с query-параметрами, чтобы проверяющий мог воспроизвести любую цифру с карты
одной командой curl.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import query as q
from .catalog import Catalog, resolve_chip_source
from .config import DATA_DIR, SERVICE_DIR, WEIGHTS_DIR
from .predictor import Predictor

app = FastAPI(
    title="Мониторинг природных пожаров — аналитический сервис",
    description=(
        "Пространственно-временные запросы по данным VIIRS / Sentinel-2 / Sentinel-1: "
        "карта термоточек, векторные контуры гари со степенью поражения и "
        "аналитическая справка по площадям."
    ),
    version="1.0",
)

_state: dict = {}


def get_state():
    """Ленивая инициализация: процесс должен подниматься быстро, а архив
    открывается и индексируется при первом запросе."""
    if not _state:
        t0 = time.time()
        source = resolve_chip_source()
        catalog = Catalog(source)
        _state["source"] = source
        _state["catalog"] = catalog
        _state["predictor"] = Predictor(catalog)
        _state["startup_seconds"] = round(time.time() - t0, 2)
    return _state


# --------------------------------------------------------------------- модели
class QueryRequest(BaseModel):
    bbox: list[float] | None = Field(
        None, description="[min_lon, min_lat, max_lon, max_lat] в EPSG:4326")
    geometry: dict | None = Field(
        None, description="GeoJSON Polygon/MultiPolygon в EPSG:4326 (альтернатива bbox)")
    date_from: str | None = Field(None, description="YYYY-MM-DD включительно")
    date_to: str | None = Field(None, description="YYYY-MM-DD включительно")
    include_hotspots: bool = True
    include_contours: bool = True


# ------------------------------------------------------------------ эндпоинты
@app.get("/api/health", tags=["служебные"])
def health():
    state = get_state()
    return {
        "status": "ok",
        "data_dir": str(DATA_DIR),
        "weights_dir": str(WEIGHTS_DIR),
        "startup_seconds": state["startup_seconds"],
        "predictor": state["predictor"].info(),
        "cached_chips": state["predictor"].cached_count(),
    }


@app.get("/api/catalog", tags=["каталог"])
def catalog_info():
    """Границы и период доступных данных - интерфейс использует это, чтобы
    выставить карту и ползунки дат без хардкода."""
    state = get_state()
    extent = state["catalog"].extent()
    extent["source"] = state["predictor"].source
    extent["source_note"] = state["predictor"].fallback_reason
    return extent


@app.get("/api/chips", tags=["каталог"])
def chips(kind: str = Query("bs", pattern="^(af|bs)$")):
    """Футпринты чипов — слой «покрытие данными» на карте."""
    catalog = get_state()["catalog"]
    feats = []
    for rec in catalog.records.values():
        if rec.kind != kind:
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [rec.lon_min, rec.lat_min], [rec.lon_max, rec.lat_min],
                [rec.lon_max, rec.lat_max], [rec.lon_min, rec.lat_max],
                [rec.lon_min, rec.lat_min]]]},
            "properties": {
                "chip_id": rec.chip_id, "kind": rec.kind,
                "date_start": rec.date_start.isoformat(),
                "date_end": rec.date_end.isoformat(),
                "fire_event_id": rec.fire_event_id,
                "cloud_frac": rec.cloud_frac, "gsd_m": rec.gsd,
            },
        })
    return q._fc(feats)


def _run(bbox, geometry, date_from, date_to, include_hotspots=True,
          include_contours=True, limit_hotspots=None):
    state = get_state()
    try:
        kwargs = {}
        if limit_hotspots is not None:
            kwargs["limit_hotspots"] = limit_hotspots
        return q.run_query(
            state["catalog"], state["predictor"], bbox=bbox, geometry=geometry,
            date_from=date_from, date_to=date_to,
            include_hotspots=include_hotspots, include_contours=include_contours,
            **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _parse_bbox(bbox: str | None):
    if not bbox:
        return None
    parts = [p for p in bbox.replace(";", ",").split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise HTTPException(status_code=422,
                             detail="bbox должен быть четырьмя числами через запятую") from exc


@app.post("/api/query", tags=["запросы"])
def query_post(request: QueryRequest):
    """Основной запрос: полигон или bbox + интервал дат -> термоточки, контуры,
    справка. Тот же результат доступен через GET /api/query."""
    return _run(request.bbox, request.geometry, request.date_from, request.date_to,
                 request.include_hotspots, request.include_contours)


@app.get("/api/query", tags=["запросы"])
def query_get(
    bbox: str = Query(..., description="min_lon,min_lat,max_lon,max_lat"),
    date_from: str | None = None,
    date_to: str | None = None,
    include_hotspots: bool = True,
    include_contours: bool = True,
):
    return _run(_parse_bbox(bbox), None, date_from, date_to,
                 include_hotspots, include_contours)


@app.get("/api/export/contours.geojson", tags=["выгрузка"])
def export_contours(bbox: str = Query(...), date_from: str | None = None,
                     date_to: str | None = None):
    """Контуры гари в GeoJSON (EPSG:4326). Атрибуты каждого контура:
    contour_id, chip_id, severity_class, severity_label, area_ha, pixels,
    gsd_m, source_epsg, date_pre, date_post, fire_event_id."""
    result = _run(_parse_bbox(bbox), None, date_from, date_to,
                   include_hotspots=False, include_contours=True)
    return _download(result["burn_contours"], "burn_contours.geojson")


@app.get("/api/export/hotspots.geojson", tags=["выгрузка"])
def export_hotspots(bbox: str = Query(...), date_from: str | None = None,
                     date_to: str | None = None):
    """Термоточки в GeoJSON (EPSG:4326), по точке на пиксель активного горения."""
    result = _run(_parse_bbox(bbox), None, date_from, date_to,
                   include_hotspots=True, include_contours=False, limit_hotspots=0)
    return _download(result["hotspots"], "hotspots.geojson")


@app.get("/api/export/report.json", tags=["выгрузка"])
def export_report_json(bbox: str = Query(...), date_from: str | None = None,
                        date_to: str | None = None):
    """Аналитическая справка в машиночитаемом виде."""
    result = _run(_parse_bbox(bbox), None, date_from, date_to)
    return _download(result["summary"], "report.json")


@app.get("/api/export/report.csv", tags=["выгрузка"])
def export_report_csv(bbox: str = Query(...), date_from: str | None = None,
                       date_to: str | None = None):
    result = _run(_parse_bbox(bbox), None, date_from, date_to)
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";", lineterminator="\n")
    writer.writerow(["показатель", "значение"])
    writer.writerows(q.summary_rows(result["summary"]))
    # BOM: Excel иначе открывает кириллический CSV в кодировке системы
    data = ("﻿" + buf.getvalue()).encode("utf-8")
    return Response(content=data, media_type="text/csv; charset=utf-8",
                     headers={"Content-Disposition": 'attachment; filename="report.csv"'})


def _download(payload, filename):
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(content=body, media_type="application/geo+json",
                     headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ----------------------------------------------------------------- интерфейс
STATIC_DIR = SERVICE_DIR / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        return JSONResponse({"detail": "интерфейс не найден, REST API доступен на /api"},
                             status_code=404)
    return FileResponse(str(page))
