"""Прогревает кэш сервиса: прогоняет модель по всем чипам и складывает готовые
векторные слои на диск.

Запускать не обязательно — сервис посчитает чип при первом обращении к нему.
Но на защите это единственный способ гарантировать, что любой запрос жюри
отработает мгновенно, а не за десятки секунд.

    python -m service.precompute                  # все чипы
    python -m service.precompute --kind bs        # только контуры гари
    python -m service.precompute --source truth   # эталон для сравнения
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("af", "bs", "all"), default="all")
    parser.add_argument("--source", choices=("model", "truth"), default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="обработать только первые N чипов")
    parser.add_argument("--force", action="store_true", help="пересчитать даже то, что уже в кэше")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["FIRE_DATA_DIR"] = args.data_dir
    if args.weights_dir:
        os.environ["FIRE_WEIGHTS_DIR"] = args.weights_dir
    if args.source:
        os.environ["FIRE_SOURCE"] = args.source

    from service.catalog import Catalog, resolve_chip_source
    from service.predictor import Predictor

    source = resolve_chip_source()
    catalog = Catalog(source)
    predictor = Predictor(catalog)
    print(f"источник: {predictor.source}, устройство: {predictor.info()['device']}")
    if predictor.fallback_reason:
        print("ВНИМАНИЕ:", predictor.fallback_reason)

    kinds = ("af", "bs") if args.kind == "all" else (args.kind,)
    chip_ids = [cid for cid, rec in catalog.records.items() if rec.kind in kinds]
    if args.limit:
        chip_ids = chip_ids[: args.limit]

    t0 = time.time()
    done = skipped = 0
    for i, chip_id in enumerate(chip_ids, 1):
        cached = os.path.isfile(predictor._cache_path(chip_id))
        if cached and not args.force:
            skipped += 1
            continue
        n = len(predictor.features(chip_id, use_cache=False))
        done += 1
        elapsed = time.time() - t0
        rate = elapsed / max(done, 1)
        left = (len(chip_ids) - i) * rate
        print(f"[{i}/{len(chip_ids)}] {chip_id}: {n} объектов "
              f"| {elapsed/60:.1f} мин прошло, ~{left/60:.1f} мин осталось", flush=True)

    source.close()
    print(f"\nготово: посчитано {done}, из кэша {skipped}, "
          f"время {(time.time()-t0)/60:.1f} мин")
    print("кэш:", os.path.join(predictor.cache_dir, predictor.source))


if __name__ == "__main__":
    main()
