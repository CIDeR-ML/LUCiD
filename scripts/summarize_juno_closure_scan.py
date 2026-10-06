#!/usr/bin/env python
"""Summarize a completed photon-by-ensemble-by-seed closure scan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from lucid.juno_laser.diagnostics import summarize_closure_scan


def json_value(value):
    """Convert nested NumPy values to strict JSON-compatible objects."""
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
    """Load closure results, enforce grid completeness and write summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photons", type=int, nargs="+", required=True)
    parser.add_argument("--ensembles", type=int, nargs="+", required=True)
    parser.add_argument(
        "--fit-seed-offsets", type=int, nargs="+", required=True
    )
    args = parser.parse_args()
    records = []
    inputs = []
    missing = []
    for photons in args.photons:
        for ensemble in args.ensembles:
            for seed_offset in args.fit_seed_offsets:
                path = (
                    args.root
                    / f"photons_{photons}"
                    / f"ensemble_{ensemble}"
                    / f"seed_offset_{seed_offset}"
                    / "closure_result.json"
                )
                if not path.exists():
                    missing.append(str(path))
                    continue
                record = json.loads(path.read_text())
                expected = (photons, ensemble, seed_offset)
                actual = (
                    record.get("n_photons"),
                    record.get("fit_ensemble_size"),
                    record.get("fit_seed_offset"),
                )
                if actual != expected:
                    raise ValueError(
                        f"scan metadata mismatch in {path}: "
                        f"expected {expected}, got {actual}"
                    )
                records.append(record)
                inputs.append(str(path.resolve()))
    if missing:
        raise FileNotFoundError(
            "closure scan is incomplete; missing:\n" + "\n".join(missing)
        )
    result = summarize_closure_scan(records)
    result["inputs"] = inputs
    result["grid"] = {
        "photons": args.photons,
        "ensembles": args.ensembles,
        "fit_seed_offsets": args.fit_seed_offsets,
    }
    serializable = json_value(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(serializable, indent=2, allow_nan=False)
    args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
