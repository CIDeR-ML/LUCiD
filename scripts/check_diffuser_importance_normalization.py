#!/usr/bin/env python
"""Audit analytic versus self-normalized diffuser importance weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from lucid.juno_laser.config import JunoLaserConfig
from lucid.juno_laser.diagnostics import diffuser_importance_audit
from lucid.juno_laser.source import build_juno_laser_source


def json_value(value):
    """Recursively convert NumPy values into JSON-compatible objects."""
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    """Run the paired source-estimator audit and write transparent JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default="config/JUNO_laser_config.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--photon-counts", type=int, nargs="+",
        default=(256, 1024, 4096),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=tuple(range(16))
    )
    parser.add_argument("--n-angular-bins", type=int, default=8)
    args = parser.parse_args()
    config = JunoLaserConfig.from_file(args.config)
    source = build_juno_laser_source(
        config.data["source"],
        (0.0, 0.0, 0.0),
        intensity=config.data["source"]["default_intensity"],
        wavelength_nm=config.wavelength_nm,
    )
    result = diffuser_importance_audit(
        source,
        args.photon_counts,
        args.seeds,
        n_angular_bins=args.n_angular_bins,
    )
    result["config"] = str(config.path)
    result["baseline_mode"] = config.data["source"].get(
        "importance_normalization", "self_normalized"
    )
    serializable = json_value(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serializable, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(serializable, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
