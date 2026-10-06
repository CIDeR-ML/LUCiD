#!/usr/bin/env python
"""Audit full-baseline JUNO calibration gradients across random seeds.

This diagnostic uses the same measured-IRF unbinned PPP objective as the
calibration entry point.  It does not optimize parameters or submit jobs.
"""

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
from lucid.juno_laser.data import load_eventwise_pe_observation
from lucid.juno_laser.eventwise import make_gaussian_eventwise_objective
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
from lucid.juno_laser.source import build_juno_laser_source
from lucid.juno_laser.time_grid import TimeGrid


def parse_source(spec):
    """Parse ``X,Y,Z:PARQUET_OR_DIRECTORY`` into a position and path."""
    position_text, data_text = spec.split(":", 1)
    position = tuple(float(value) for value in position_text.split(","))
    if len(position) != 3:
        raise ValueError(f"source requires three coordinates: {spec!r}")
    return position, Path(data_text).expanduser().resolve()


def git_metadata():
    """Return the current commit and dirty flag without changing Git state."""
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


def run_audit(args):
    """Evaluate and summarize the full-baseline objective gradient."""
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
    fixed_params = fixed_params_from_geometry(config, geometry)
    source_defs = tuple(parse_source(spec) for spec in args.source)
    if len(args.source_exposure) != len(source_defs):
        raise ValueError(
            "provide exactly one --source-exposure for every --source"
        )
    if len(args.source_time_offset_ns) != len(source_defs):
        raise ValueError(
            "provide one --source-time-offset-ns for every --source"
        )
    time_grid = TimeGrid(
        args.time_min_ns,
        args.time_max_ns,
        args.time_bin_ns,
    )
    observations = tuple(
        load_eventwise_pe_observation(
            path,
            n_sensors=geometry.num_sensors,
            time_grid=time_grid,
            posterior_step=args.fsmp_step,
            event_exposure=exposure,
            event_time_offset_ns=time_offset,
            max_events=args.max_events,
        )
        for (_, path), exposure, time_offset in zip(
            source_defs,
            args.source_exposure,
            args.source_time_offset_ns,
        )
    )
    intensity = float(config.data["source"]["default_intensity"])
    sources = tuple(
        build_juno_laser_source(
            config.data["source"],
            position,
            intensity=intensity,
            wavelength_nm=config.wavelength_nm,
        )
        for position, _ in source_defs
    )
    forward = build_per_pe_forward(
        config,
        geometry,
        physics,
        n_photons=args.n_photons,
        k=args.k,
        intensity=intensity,
        time_grid=time_grid,
    )
    optical_ensemble = make_ensemble_forward(
        forward,
        sources,
        (geometry.num_sensors, time_grid.n_bins),
        fixed_params=fixed_params,
    )
    pmt_types = geometry.detector.pmt_type_codes
    if pmt_types is None:
        raise ValueError("measured IRF response requires JUNO PMT types")
    native_irf, irf_metadata = load_type_average_irf(
        config.irf_path, config.irf_pmt_map_path
    )
    irf_bank = resample_irf_bank(native_irf, time_grid.bin_width_ns)
    ensemble = make_irf_response_forward(
        optical_ensemble, pmt_types, irf_bank
    )
    objective = make_gaussian_eventwise_objective(
        ensemble,
        observations,
        sigma_ns=config.interpolation_sigma_ns,
        truncate_sigma=args.truncate_sigma,
        background_rate_per_ns=args.background_rate_hz * 1.0e-9,
    )
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    coordinates = encode_calibration_values(truth)
    records = []
    for seed in args.seeds:
        keys = jax.random.split(
            jax.random.PRNGKey(seed), args.ensemble_size
        )
        call_start = time.perf_counter()
        value, gradient = value_and_grad(coordinates, keys)
        value.block_until_ready()
        gradient_array = np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(float(value)) or not np.all(
            np.isfinite(gradient_array)
        ):
            raise FloatingPointError(
                f"non-finite objective or gradient for seed {seed}: "
                f"loss={float(value)}, gradient={gradient_array.tolist()}"
            )
        records.append({
            "seed": seed,
            "loss": float(value),
            "gradient": dict(zip(PARAMETER_NAMES,
                                 gradient_array.tolist())),
            "gradient_norm": float(np.linalg.norm(gradient_array)),
            "seconds": time.perf_counter() - call_start,
        })
    gradients = np.asarray([
        [record["gradient"][name] for name in PARAMETER_NAMES]
        for record in records
    ])
    result = {
        "git": git_metadata(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "config": str(config.path),
        "sources": [
            {
                "position": list(position),
                "data": str(path),
                "fixed_event_exposure": exposure,
                "fixed_event_time_offset_ns": time_offset,
            }
            for (position, path), exposure, time_offset in zip(
                source_defs,
                args.source_exposure,
                args.source_time_offset_ns,
            )
        ],
        "exposure_mode": "explicit_fixed_per_source",
        "n_photons": args.n_photons,
        "k": args.k,
        "ensemble_size": args.ensemble_size,
        "max_events": args.max_events,
        "time_window_ns": [args.time_min_ns, args.time_max_ns],
        "time_bin_ns": args.time_bin_ns,
        "fsmp_posterior_step": args.fsmp_step,
        "irf": {
            **irf_metadata,
            "resampled_bin_width_ns": irf_bank.bin_width_ns,
            "interpolation_sigma_ns": config.interpolation_sigma_ns,
        },
        "truncate_sigma": args.truncate_sigma,
        "background_rate_hz_per_pmt": args.background_rate_hz,
        "parameter_names": list(PARAMETER_NAMES),
        "truth": {name: truth[name] for name in PARAMETER_NAMES},
        "records": records,
        "gradient_mean": dict(zip(
            PARAMETER_NAMES, np.mean(gradients, axis=0).tolist()
        )),
        "gradient_std": dict(zip(
            PARAMETER_NAMES, np.std(gradients, axis=0, ddof=0).tolist()
        )),
        "total_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(result, stream, indent=2)
    return result


def main():
    """Parse arguments and write the gradient-audit JSON output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/JUNO_laser_config.json"
    )
    parser.add_argument(
        "--source", action="append", required=True,
        metavar="X,Y,Z:DATA",
    )
    parser.add_argument(
        "--source-exposure",
        action="append",
        type=float,
        required=True,
        metavar="VALUE",
    )
    parser.add_argument(
        "--source-time-offset-ns",
        action="append",
        type=float,
        required=True,
        metavar="NS",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-photons", type=int, default=512)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--time-min-ns", type=float, default=0.0)
    parser.add_argument("--time-max-ns", type=float, default=300.0)
    parser.add_argument("--time-bin-ns", type=float, default=10.0)
    parser.add_argument("--fsmp-step", type=int, default=0)
    parser.add_argument("--truncate-sigma", type=float, default=5.0)
    parser.add_argument("--background-rate-hz", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.n_photons <= 0 or args.k <= 0 or args.ensemble_size <= 0:
        parser.error("photon, K and ensemble settings must be positive")
    if args.fsmp_step < 0 or args.background_rate_hz < 0.0:
        parser.error("FSMP step and background rate must be non-negative")
    if args.truncate_sigma <= 0.0:
        parser.error("Gaussian truncation must be positive")
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in args.source_exposure
    ):
        parser.error("source exposures must be finite and positive")
    if any(
        not np.isfinite(value) for value in args.source_time_offset_ns
    ):
        parser.error("source time offsets must be finite")
    result = run_audit(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
