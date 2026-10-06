#!/usr/bin/env python
"""Summarize campaign IRF FWHM by JUNO 20-inch PMT technology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PMT_NAMES = {1: "Hamamatsu", 2: "NNVT"}
FWHM_TO_SIGMA = 2.354820045


def summarize(irf_path, pmt_map_path):
    """Return highest-load IRF width statistics for matched CD PMTs."""
    with h5py.File(irf_path, "r") as source:
        rows = source["hist"][:]
    best = {}
    for row in rows:
        serial = row["sn"].decode()
        if serial not in best or row["load"] > best[serial]["load"]:
            best[serial] = row
    table = pd.read_parquet(
        pmt_map_path, columns=["SN", "copy_number", "pmt_type"]
    )
    table = table[
        (table["copy_number"] >= 0) & (table["copy_number"] < 17612)
    ]
    type_by_serial = {
        str(serial): int(pmt_type)
        for serial, pmt_type in zip(table["SN"], table["pmt_type"])
        if pd.notna(serial)
    }
    values = {code: [] for code in PMT_NAMES}
    for serial, row in best.items():
        pmt_type = type_by_serial.get(serial)
        if pmt_type in values:
            values[pmt_type].append(float(row["fwhm"]))
    groups = {}
    for code, name in PMT_NAMES.items():
        fwhm = np.asarray(values[code], dtype=np.float64)
        sigma = fwhm / FWHM_TO_SIGMA
        groups[name] = {
            "count": int(fwhm.size),
            "fwhm_ns": {
                "mean": float(np.mean(fwhm)),
                "median": float(np.median(fwhm)),
                "std": float(np.std(fwhm)),
                "q05": float(np.quantile(fwhm, 0.05)),
                "q95": float(np.quantile(fwhm, 0.95)),
                "min": float(np.min(fwhm)),
                "max": float(np.max(fwhm)),
            },
            "gaussian_equivalent_sigma_ns": {
                "mean": float(np.mean(sigma)),
                "median": float(np.median(sigma)),
            },
        }
    return {
        "schema": "juno_irf_type_width_summary_v1",
        "irf_file": str(Path(irf_path).resolve()),
        "pmt_map": str(Path(pmt_map_path).resolve()),
        "selection": "highest load per serial; CD copy_number < 17612",
        "fwhm_to_sigma": FWHM_TO_SIGMA,
        "hist_rows": int(len(rows)),
        "unique_serials": int(len(best)),
        "matched_pmts": int(sum(len(item) for item in values.values())),
        "groups": groups,
    }


def main():
    """Parse paths and print or write the reproducible JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--irf",
        type=Path,
        default=Path(
            "/junofs/PmtCharacterization/IRF/"
            "jpwaptool_ser_baseline.h5"
        ),
    )
    parser.add_argument(
        "--pmt-map",
        type=Path,
        default=Path("/junofs/C/junoid_J25.7.3.pq"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(args.irf, args.pmt_map)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
