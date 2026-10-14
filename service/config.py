"""Service settings. Everything is an environment variable with a repo-relative
default, so the service starts with no arguments on a fresh clone and nothing
here contains a path from a developer's machine.

    FIRE_DATA_DIR     каталог с fire-train-renamed.tar (или распакованной train/)
    FIRE_WEIGHTS_DIR  каталог с af/ и bs/ (по умолчанию solution/weights)
    FIRE_CACHE_DIR    куда складывать посчитанные контуры
    FIRE_SOURCE       model | truth - чем рисовать карту (по умолчанию model,
                      с автоматическим откатом на truth, если весов нет)
"""
from __future__ import annotations

import os
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = SERVICE_DIR.parent
REPO_DIR = SOLUTION_DIR.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


# Кандидаты, где искать данные, если FIRE_DATA_DIR не задан. Порядок - от
# самого явного к самому общему; берётся первый существующий.
_DATA_CANDIDATES = [
    SOLUTION_DIR / "data",
    REPO_DIR / "Мониторинг DATA" / "Мониторинг DATA",
    REPO_DIR / "data",
    REPO_DIR,
]


def default_data_dir() -> Path:
    for candidate in _DATA_CANDIDATES:
        if (candidate / "fire-train-renamed.tar").is_file():
            return candidate
        if (candidate / "fire-train-renamed" / "train").is_dir():
            return candidate
        if (candidate / "train").is_dir():
            return candidate
    return _DATA_CANDIDATES[0]


DATA_DIR = _env_path("FIRE_DATA_DIR", default_data_dir())
WEIGHTS_DIR = _env_path("FIRE_WEIGHTS_DIR", SOLUTION_DIR / "weights")
CACHE_DIR = _env_path("FIRE_CACHE_DIR", SERVICE_DIR / "cache")
SOURCE = os.environ.get("FIRE_SOURCE", "model").strip().lower()
DEVICE = os.environ.get("FIRE_DEVICE", "").strip().lower() or None

# Запрос, охватывающий весь архив, вернул бы сотни тысяч термоточек и повесил бы
# браузер. Ограничиваем то, что уходит в UI; выгрузка в GeoJSON идёт без лимита.
MAX_HOTSPOTS_IN_RESPONSE = int(os.environ.get("FIRE_MAX_HOTSPOTS", "20000"))
MAX_CHIPS_PER_QUERY = int(os.environ.get("FIRE_MAX_CHIPS", "400"))
