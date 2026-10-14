"""`python -m service` — точка входа сервиса."""
from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("FIRE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FIRE_PORT", "8000")))
    parser.add_argument("--data-dir", default=None,
                         help="каталог с fire-train-renamed.tar (иначе FIRE_DATA_DIR/автопоиск)")
    parser.add_argument("--weights-dir", default=None,
                         help="каталог с af/ и bs/ (иначе FIRE_WEIGHTS_DIR или solution/weights)")
    parser.add_argument("--source", choices=("model", "truth"), default=None,
                         help="чем рисовать карту: предсказаниями модели или эталонными масками")
    parser.add_argument("--reload", action="store_true", help="автоперезапуск при правках кода")
    args = parser.parse_args()

    # Настройки читаются модулем config при импорте, поэтому проставляем их ДО
    # того, как uvicorn импортирует приложение.
    if args.data_dir:
        os.environ["FIRE_DATA_DIR"] = args.data_dir
    if args.weights_dir:
        os.environ["FIRE_WEIGHTS_DIR"] = args.weights_dir
    if args.source:
        os.environ["FIRE_SOURCE"] = args.source

    import uvicorn
    print(f"Сервис: http://{args.host}:{args.port}")
    print(f"REST API (Swagger): http://{args.host}:{args.port}/docs")
    uvicorn.run("service.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
