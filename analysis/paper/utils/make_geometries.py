#!/usr/bin/env python3
"""Generate SK_like detector geometry JSONs at varying PMT counts (the 'geom' study axis).

Copies ``config/SK_like_geom_config.json`` and overrides ``geometry_definitions.n_sensors``
for each count in 2000..20000 step 1000, writing them next to this script under
``analysis/paper/geometries/SK_like_<N>_geom_config.json``.

The physics config is independent of PMT count, so every generated geometry reuses
``config/SK_like_physics_config.json`` (referenced from the study configs, not copied here).

    python analysis/paper/utils/make_geometries.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_GEOM = REPO_ROOT / 'config' / 'SK_like_geom_config.json'
# analysis/paper/geometries/ — the path studies.geom_configs() writes into its configs.
OUT_DIR = Path(__file__).resolve().parents[1] / 'geometries'

SENSOR_COUNTS = list(range(2000, 20001, 1000))     # 2k..20k step 1k (19 geometries)


def geom_path(n):
    """Path to the SK_like geometry config at ``n`` sensors (may not exist yet)."""
    return OUT_DIR / f'SK_like_{n}_geom_config.json'


def ensure(sensors=None, verbose=True):
    """Write any MISSING geometry configs for ``sensors``; return the paths.

    Idempotent, so ``fig_geometry_scan.py`` can call it unconditionally — the geom
    study then works from a fresh clone with no manual prerequisite step.
    """
    sensors = SENSOR_COUNTS if sensors is None else sensors
    base = json.loads(BASE_GEOM.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made, paths = [], []
    for n in sensors:
        out = geom_path(n)
        paths.append(out)
        if out.exists():
            continue
        cfg = json.loads(json.dumps(base))          # deep copy
        cfg['geometry_definitions']['n_sensors'] = n
        out.write_text(json.dumps(cfg, indent=2))
        made.append(n)
    if verbose and made:
        print(f"[geometries] wrote {len(made)} missing geometry config(s) "
              f"({', '.join(str(n) for n in made)}) in {OUT_DIR.relative_to(REPO_ROOT)}")
    return paths


def main():
    for n in SENSOR_COUNTS:                          # force a full regeneration
        geom_path(n).unlink(missing_ok=True)
    ensure(SENSOR_COUNTS, verbose=False)
    print(f"{len(SENSOR_COUNTS)} geometry configs in {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == '__main__':
    main()
