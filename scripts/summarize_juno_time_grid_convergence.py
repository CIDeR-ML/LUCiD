#!/usr/bin/env python
"""Compare completed JUNO closure results across numerical time grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COMPARABILITY_FIELDS = (
    "config",
    "source_positions_m",
    "n_events",
    "n_photons",
    "k",
    "intensity",
    "event_exposure",
    "event_time_offset_ns",
    "parameter_names",
)


def parse_result_spec(spec):
    """Parse ``BIN_WIDTH_NS=RESULT_JSON`` without reading the file."""
    try:
        width_text, path_text = spec.split("=", 1)
        width = float(width_text)
    except ValueError as exc:
        raise ValueError(
            f"result must be BIN_WIDTH_NS=JSON_PATH, got {spec!r}"
        ) from exc
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("time-bin width must be finite and positive")
    return width, Path(path_text).expanduser().resolve()


def _relative_vector_difference(values, reference):
    """Return component and norm differences relative to a reference."""
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = np.maximum(np.abs(reference), 1.0e-12)
    return {
        "component_relative": ((values - reference) / scale).tolist(),
        "relative_l2": float(
            np.linalg.norm(values - reference)
            / max(np.linalg.norm(reference), 1.0e-12)
        ),
    }


def summarize_results(
    records,
    *,
    parameter_relative_tolerance,
    loss_absolute_tolerance,
    gradient_relative_tolerance,
):
    """Build a convergence summary using the finest grid as reference."""
    if len(records) < 2:
        raise ValueError("at least two time-grid results are required")
    ordered = sorted(records, key=lambda item: item[0], reverse=True)
    widths = [item[0] for item in ordered]
    if len(set(widths)) != len(widths):
        raise ValueError("time-bin widths must be unique")
    baseline = ordered[0][1]
    for width, result in ordered:
        if not np.isclose(float(result["time_bin_ns"]), width):
            raise ValueError(
                f"declared width {width} disagrees with result metadata"
            )
        for field in COMPARABILITY_FIELDS:
            if result.get(field) != baseline.get(field):
                raise ValueError(
                    f"grid results differ in non-grid field {field!r}"
                )
    reference_width, reference = ordered[-1]
    names = tuple(reference["parameter_names"])
    reference_fit = [reference["fit"][name] for name in names]
    reference_gradient = [
        reference["final_gradient"][name] for name in names
    ]
    comparisons = []
    for width, result in ordered:
        fit = [result["fit"][name] for name in names]
        gradient = [result["final_gradient"][name] for name in names]
        fit_difference = _relative_vector_difference(fit, reference_fit)
        gradient_difference = _relative_vector_difference(
            gradient, reference_gradient
        )
        comparisons.append(
            {
                "time_bin_ns": width,
                "fit": dict(zip(names, fit)),
                "fit_relative_to_finest": dict(
                    zip(names, fit_difference["component_relative"])
                ),
                "fit_relative_l2": fit_difference["relative_l2"],
                "final_loss_per_pe": result["final_loss_per_pe"],
                "loss_minus_finest": (
                    result["final_loss_per_pe"]
                    - reference["final_loss_per_pe"]
                ),
                "gradient_relative_l2": gradient_difference["relative_l2"],
            }
        )
    finest_pair = comparisons[-2:]
    coarser, finest = finest_pair
    max_parameter_change = max(
        abs(value)
        for value in coarser["fit_relative_to_finest"].values()
    )
    loss_change = abs(coarser["loss_minus_finest"])
    gradient_change = coarser["gradient_relative_l2"]
    converged = bool(
        max_parameter_change <= parameter_relative_tolerance
        and loss_change <= loss_absolute_tolerance
        and gradient_change <= gradient_relative_tolerance
    )
    return {
        "schema": "lucid_time_grid_convergence_v1",
        "reference_time_bin_ns": reference_width,
        "tested_time_bins_ns": widths,
        "comparisons": comparisons,
        "finest_pair": {
            "coarser_time_bin_ns": coarser["time_bin_ns"],
            "finest_time_bin_ns": finest["time_bin_ns"],
            "max_parameter_relative_change": max_parameter_change,
            "loss_absolute_change": loss_change,
            "gradient_relative_l2_change": gradient_change,
        },
        "tolerances": {
            "parameter_relative": parameter_relative_tolerance,
            "loss_absolute_per_pe": loss_absolute_tolerance,
            "gradient_relative_l2": gradient_relative_tolerance,
        },
        "converged": converged,
    }


def main():
    """Read closure JSON files and write one convergence summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="BIN_WIDTH_NS=JSON_PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parameter-relative-tolerance", type=float, default=0.01
    )
    parser.add_argument(
        "--loss-absolute-tolerance", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--gradient-relative-tolerance", type=float, default=0.1
    )
    parser.add_argument("--require-converged", action="store_true")
    args = parser.parse_args()
    tolerances = (
        args.parameter_relative_tolerance,
        args.loss_absolute_tolerance,
        args.gradient_relative_tolerance,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in tolerances):
        parser.error("convergence tolerances must be finite and non-negative")
    records = []
    for spec in args.result:
        width, path = parse_result_spec(spec)
        with path.open() as stream:
            records.append((width, json.load(stream)))
    summary = summarize_results(
        records,
        parameter_relative_tolerance=args.parameter_relative_tolerance,
        loss_absolute_tolerance=args.loss_absolute_tolerance,
        gradient_relative_tolerance=args.gradient_relative_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    if args.require_converged and not summary["converged"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
