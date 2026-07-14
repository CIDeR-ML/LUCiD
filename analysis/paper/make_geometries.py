#!/usr/bin/env python3
"""Generate SK_like detector geometry JSONs at varying PMT counts (the 'geom' study axis).

Copies ``config/SK_like_geom_config.json`` and overrides ``geometry_definitions.n_sensors``
for each count in 2000..20000 step 1000, writing them next to this script under
``analysis/paper/geometries/SK_like_<N>_geom_config.json``.

The physics config is independent of PMT count, so every generated geometry reuses
``config/SK_like_physics_config.json`` (referenced from the study configs, not copied here).

    python analysis/paper/make_geometries.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_GEOM = REPO_ROOT / 'config' / 'SK_like_geom_config.json'
OUT_DIR = Path(__file__).resolve().parent / 'geometries'

SENSOR_COUNTS = list(range(2000, 20001, 1000))     # 2k..20k step 1k (19 geometries)


def main():
    base = json.loads(BASE_GEOM.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in SENSOR_COUNTS:
        cfg = json.loads(json.dumps(base))          # deep copy
        cfg['geometry_definitions']['n_sensors'] = n
        out = OUT_DIR / f'SK_like_{n}_geom_config.json'
        out.write_text(json.dumps(cfg, indent=2))
        print(f"wrote {out.relative_to(REPO_ROOT)}  (n_sensors={n})")
    print(f"\n{len(SENSOR_COUNTS)} geometry configs in {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == '__main__':
    main()
