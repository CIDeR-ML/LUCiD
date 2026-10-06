#!/usr/bin/env python
"""Compute per-source numerical-basis Jacobians and five-parameter Fisher."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import numpy as np

from lucid.detector_params import load_physics_config
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.juno_laser.config import JunoLaserConfig, PARAMETER_NAMES
from lucid.juno_laser.diagnostics import (
    poisson_basis_fisher,
    response_jacobian,
    summarize_fisher_matrix,
)
from lucid.juno_laser.forward import build_per_pe_forward
from lucid.juno_laser.irf import (
    load_type_average_irf,
    make_irf_response_forward,
    resample_irf_bank,
)
from lucid.juno_laser.optimizer import make_ensemble_forward
from lucid.juno_laser.parameters import (
    encode_calibration_values,
    fixed_params_from_geometry,
    truth_from_geometry,
)
from lucid.juno_laser.source import (
    LSPHASE_SHAPE_NAMES,
    build_juno_laser_source,
)
from lucid.juno_laser.time_grid import TimeGrid


COORDINATE_NAMES = (
    "log_scat_ls",
    "log_abs_ls",
    *LSPHASE_SHAPE_NAMES,
)


def parse_position(text):
    """Parse one comma-separated source position in metres."""
    values = tuple(float(value) for value in text.split(","))
    if len(values) != 3:
        raise ValueError(f"source position requires x,y,z: {text!r}")
    return values


def json_value(value):
    """Convert nested NumPy values into strict JSON objects."""
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def git_metadata():
    """Return repository identity without modifying Git state."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def run(args):
    """Build the full baseline and calculate per-source Fisher matrices."""
    started = time.perf_counter()
    config = JunoLaserConfig.from_file(args.config)
    geometry = DetectorGeometry.from_config(
        str(config.geometry_path),
        temperature=None,
        max_candidates_per_ray=4,
        detector_type="nested_sphere",
    )
    physics, _, _ = load_physics_config(
        str(config.physics_path),
        num_sensors=geometry.num_sensors,
        scalar_ref_wavelength=config.wavelength_nm,
    )
    truth = truth_from_geometry(config, geometry)
    coordinates = encode_calibration_values(truth)
    fixed = fixed_params_from_geometry(config, geometry)
    positions = tuple(parse_position(item) for item in args.source_position)
    intensity = float(args.intensity)
    sources = tuple(
        build_juno_laser_source(
            config.data["source"],
            position,
            intensity=intensity,
            wavelength_nm=config.wavelength_nm,
        )
        for position in positions
    )
    time_grid = TimeGrid(
        args.time_min_ns,
        args.time_max_ns,
        args.time_bin_ns,
    )
    hist_shape = (geometry.num_sensors, time_grid.n_bins)
    forward = build_per_pe_forward(
        config,
        geometry,
        physics,
        n_photons=args.n_photons,
        k=args.k,
        intensity=intensity,
        time_grid=time_grid,
    )
    pmt_types = geometry.detector.pmt_type_codes
    if pmt_types is None:
        raise ValueError("measured IRF response requires JUNO PMT types")
    native_irf, irf_metadata = load_type_average_irf(
        config.irf_path, config.irf_pmt_map_path
    )
    irf_bank = resample_irf_bank(native_irf, time_grid.bin_width_ns)
    keys = jax.random.split(
        jax.random.PRNGKey(args.seed), args.ensemble_size
    )
    background_mass = (
        args.background_rate_hz * 1.0e-9 * time_grid.bin_width_ns
    )
    event_exposures = np.full(args.n_events, args.event_exposure)
    source_results = []
    total_fisher = np.zeros((len(PARAMETER_NAMES),) * 2)
    for index, (position, source) in enumerate(zip(positions, sources)):
        optical = make_ensemble_forward(
            forward,
            (source,),
            hist_shape,
            fixed_params=fixed,
            source_indices=(index,),
        )
        response = make_irf_response_forward(
            optical, pmt_types, irf_bank
        )

        def source_forward(theta, selected_keys):
            """Return this source's PMT-by-time numerical masses."""
            return response(theta, selected_keys)[0]

        source_started = time.perf_counter()
        masses, jacobian = response_jacobian(
            source_forward, coordinates, keys
        )
        fisher = poisson_basis_fisher(
            masses,
            jacobian,
            event_exposures=event_exposures,
            background_mass=background_mass,
        )
        total_fisher += fisher["fisher"]
        source_results.append({
            "source_index": index,
            "position_m": list(position),
            "response_shape": list(masses.shape),
            "response_mass": float(np.sum(masses)),
            "seconds": time.perf_counter() - source_started,
            **json_value(fisher),
        })
    total = summarize_fisher_matrix(total_fisher)
    result = {
        "schema": "lucid_per_source_numerical_basis_fisher_v1",
        "definition": "expected Poisson Fisher on IRF-convolved time bins",
        "not_yet": "continuous-time unbinned PPP Fisher",
        "git": git_metadata(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "config": str(config.path),
        "parameter_names": list(PARAMETER_NAMES),
        "optimizer_coordinate_names": list(COORDINATE_NAMES),
        "truth": {name: float(truth[name]) for name in PARAMETER_NAMES},
        "n_photons": args.n_photons,
        "k": args.k,
        "ensemble_size": args.ensemble_size,
        "seed": args.seed,
        "n_events": args.n_events,
        "event_exposure": args.event_exposure,
        "background_rate_hz_per_pmt": args.background_rate_hz,
        "time_window_ns": [args.time_min_ns, args.time_max_ns],
        "time_bin_ns": args.time_bin_ns,
        "irf": irf_metadata,
        "sources": source_results,
        "total": json_value(total),
        "total_is_sum_of_sources": bool(np.allclose(
            total_fisher,
            np.sum(
                [np.asarray(item["fisher"]) for item in source_results],
                axis=0,
            ),
        )),
        "seconds": time.perf_counter() - started,
    }
    serializable = json_value(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serializable, indent=2, allow_nan=False) + "\n"
    )
    return serializable


def main():
    """Parse arguments, run the Fisher diagnostic and write JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default="config/JUNO_laser_config.json"
    )
    parser.add_argument(
        "--source-position", action="append", required=True,
        metavar="X,Y,Z",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-photons", type=int, default=512)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--intensity", type=float, default=100000.0)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-events", type=int, default=1)
    parser.add_argument("--event-exposure", type=float, default=1.0)
    parser.add_argument("--background-rate-hz", type=float, default=30000.0)
    parser.add_argument("--time-min-ns", type=float, default=0.0)
    parser.add_argument("--time-max-ns", type=float, default=300.0)
    parser.add_argument("--time-bin-ns", type=float, default=50.0)
    args = parser.parse_args()
    positive = (
        args.n_photons,
        args.k,
        args.intensity,
        args.ensemble_size,
        args.n_events,
        args.event_exposure,
        args.time_bin_ns,
    )
    if any(value <= 0 for value in positive):
        parser.error("transport, exposure and time-bin values must be > 0")
    if args.background_rate_hz < 0.0:
        parser.error("background rate must be non-negative")
    span = (args.time_max_ns - args.time_min_ns) / args.time_bin_ns
    if args.time_max_ns <= args.time_min_ns or not np.isclose(
        span, round(span)
    ):
        parser.error("time window must be increasing with integer bin count")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
