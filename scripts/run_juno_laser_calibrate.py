#!/usr/bin/env python
"""Run the current unbinned JUNO 532 nm PE-time calibration prototype.

This CLI is the repository-facing calibration entry point.  It consumes
one prescribed FSMP posterior step per physical event, applies measured
PMT-family IRFs, builds a continuous Poisson point-process objective, and
runs the Adam fit:

1. LS scattering and absorption lengths,
2. three identifiable diffuser LSPhase shape coordinates,
3. all five parameters jointly.

Example::

    python scripts/run_juno_laser_calibrate.py \
        --source 0,0,0:/junofs/C/liukainan/LUCiD/outputs/juno_laser_center \
        --outdir /junofs/C/liukainan/LUCiD/outputs/juno_laser_calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make direct ``python scripts/run_juno_laser_calibrate.py`` execution resolve
# the repository package without a caller-specific PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import numpy as np

from lucid.detector_params import load_physics_config
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.juno_laser.config import JunoLaserConfig, PARAMETER_NAMES
from lucid.juno_laser.data import load_eventwise_pe_observation
from lucid.juno_laser.eventwise import (
    gaussian_eventwise_poisson_terms,
    gaussian_window_diagnostics,
    make_gaussian_eventwise_objective,
    require_positive_pe_support,
)
from lucid.juno_laser.forward import build_per_pe_forward
from lucid.juno_laser.irf import (
    load_type_average_irf,
    make_irf_response_forward,
    resample_irf_bank,
)
from lucid.juno_laser.optimizer import (
    adam_stage,
    make_ensemble_forward,
    select_stage_coordinates,
)
from lucid.juno_laser.parameters import (
    decode_calibration_array,
    encode_calibration_values,
    fixed_params_from_geometry,
    truth_from_geometry,
)
from lucid.juno_laser.source import (
    LSPHASE_SHAPE_NAMES,
    build_juno_laser_source,
    lsphase_polynomial_coefficients,
    set_diffuser_angular_shape,
)
from lucid.juno_laser.time_grid import TimeGrid


LOG = logging.getLogger("lucid.juno_laser.calibrate")
DEFAULT_OUTPUT_DIR = "/junofs/C/liukainan/LUCiD/outputs/juno_laser_calibration"


def _parse_source(spec: str, config: JunoLaserConfig):
    try:
        position_text, data_text = spec.split(":", 1)
        position = tuple(float(value) for value in position_text.split(","))
    except ValueError as exc:
        raise ValueError(
            f"source must be X,Y,Z:DATA_DIR, got {spec!r}"
        ) from exc
    if len(position) != 3:
        raise ValueError(
            "source position must have three coordinates, "
            f"got {position}"
        )
    data_dir = Path(data_text).expanduser()
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()
    return position, data_dir


def _parameter_start(
    truth, seed: int, start_values: str | None, fraction: float
):
    if start_values:
        values = tuple(float(value) for value in start_values.split(","))
        if len(values) != len(PARAMETER_NAMES):
            raise ValueError(
                "--start-values must contain " + ",".join(PARAMETER_NAMES)
            )
        return encode_calibration_values(dict(zip(PARAMETER_NAMES, values)))
    rng = np.random.default_rng(seed)
    coordinates = np.asarray(
        encode_calibration_values(truth), dtype=np.float32
    )
    perturbation = rng.uniform(
        -fraction, fraction, size=coordinates.shape
    )
    return coordinates + perturbation.astype(np.float32)


def calibrate(args):
    config = JunoLaserConfig.from_file(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(outdir / "calibration.log", mode="w"),
        ],
    )

    geom = DetectorGeometry.from_config(
        str(config.geometry_path), temperature=None,
        max_candidates_per_ray=4, detector_type="nested_sphere")
    physics, _, _ = load_physics_config(
        str(config.physics_path), num_sensors=geom.num_sensors,
        scalar_ref_wavelength=config.wavelength_nm)
    truth = truth_from_geometry(config, geom)
    fixed_params = fixed_params_from_geometry(config, geom)
    source_specs = args.source or []
    if not source_specs:
        raise ValueError("at least one --source X,Y,Z:DATA_DIR is required")
    source_defs = [_parse_source(spec, config) for spec in source_specs]
    source_exposures = args.source_exposure or []
    if len(source_exposures) != len(source_defs):
        raise ValueError(
            "provide exactly one --source-exposure for every --source"
        )
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in source_exposures
    ):
        raise ValueError("source exposures must be finite and positive")
    source_time_offsets = args.source_time_offset_ns or []
    if len(source_time_offsets) != len(source_defs):
        raise ValueError(
            "provide exactly one --source-time-offset-ns for every --source"
        )
    if any(not np.isfinite(value) for value in source_time_offsets):
        raise ValueError("source time offsets must be finite")
    calibration = config.calibration
    t_min, t_max = calibration["time_window_ns"]
    t_bin = float(args.time_bin_ns or calibration["time_bin_ns"])
    time_grid = TimeGrid(
        float(args.time_min_ns if args.time_min_ns is not None else t_min),
        float(args.time_max_ns if args.time_max_ns is not None else t_max),
        t_bin,
    )
    observations = tuple(
        load_eventwise_pe_observation(
            data_dir, n_sensors=geom.num_sensors,
            time_grid=time_grid, posterior_step=args.fsmp_step,
            event_exposure=exposure,
            event_time_offset_ns=time_offset,
            max_events=args.max_events)
        for (_, data_dir), exposure, time_offset in zip(
            source_defs, source_exposures, source_time_offsets
        ))
    if any(item.n_events <= 0 for item in observations):
        raise ValueError(
            "all source observations must contain at least one event"
        )
    intensity = float(
        args.intensity or config.data["source"]["default_intensity"]
    )
    sources = tuple(
        build_juno_laser_source(
            config.data["source"],
            position,
            intensity=intensity,
            wavelength_nm=config.wavelength_nm,
        )
        for position, _ in source_defs
    )
    n_photons = int(args.n_photons or calibration["n_photons"])
    n_steps = int(args.k or config.data["transport"]["max_scattering_steps"])
    forward = build_per_pe_forward(
        config, geom, physics, n_photons=n_photons, k=n_steps,
        intensity=intensity, time_grid=time_grid)
    optical_ensemble = make_ensemble_forward(
        forward,
        sources,
        (geom.num_sensors, time_grid.n_bins),
        fixed_params=fixed_params,
    )
    pmt_types = geom.detector.pmt_type_codes
    if pmt_types is None:
        raise ValueError("measured IRF response requires JUNO PMT types")
    native_irf, irf_metadata = load_type_average_irf(
        config.irf_path, config.irf_pmt_map_path
    )
    irf_bank = resample_irf_bank(native_irf, time_grid.bin_width_ns)
    ensemble = make_irf_response_forward(
        optical_ensemble, pmt_types, irf_bank
    )
    interpolation_sigma_ns = config.interpolation_sigma_ns
    train_keys = jax.random.split(
        jax.random.PRNGKey(args.seed),
        int(args.nb_h or calibration["train_ensemble_size"]))
    mc_validation_keys = jax.random.split(
        jax.random.PRNGKey(args.seed + args.mc_validation_seed_offset),
        int(
            args.mc_validation_nb_h
            or calibration["mc_validation_ensemble_size"]
        ),
    )
    background_rate_per_ns = args.background_rate_hz * 1.0e-9
    full_objective = make_gaussian_eventwise_objective(
        ensemble,
        observations,
        sigma_ns=interpolation_sigma_ns,
        truncate_sigma=args.truncate_sigma,
        background_rate_per_ns=background_rate_per_ns,
    )
    bounds = config.parameter_bounds
    theta = _parameter_start(
        truth, args.seed, args.start_values, args.start_fraction
    )
    LOG.info(
        "sources=%d, events=%s, n_photons=%d, K=%d, bins=%d x %.3g ns",
        len(sources),
        [item.n_events for item in observations],
        n_photons,
        n_steps,
        time_grid.n_bins,
        t_bin,
    )
    LOG.info("truth=%s", truth)

    initial_predictions = ensemble(theta, train_keys)
    support = []
    window_stability = []
    for index, (observation, prediction) in enumerate(
        zip(observations, initial_predictions)
    ):
        terms = gaussian_eventwise_poisson_terms(
            observation,
            prediction,
            sigma_ns=interpolation_sigma_ns,
            truncate_sigma=args.truncate_sigma,
            background_rate_per_ns=background_rate_per_ns,
        )
        terms.pe_rate.block_until_ready()
        support.append(
            require_positive_pe_support(
                terms.pe_rate,
                context=f"real-data initial source {index}",
                low_rate_threshold=args.low_rate_threshold,
            )
        )
        window_stability.append(
            gaussian_window_diagnostics(
                prediction,
                time_grid=time_grid,
                sigma_ns=interpolation_sigma_ns,
                expansion_ns=args.window_stability_expansion_ns,
                observation=observation,
            )
        )
    LOG.info("initial PE-support diagnostics=%s", support)

    stage_steps = calibration.get("stage_steps", {})
    stage1_steps = args.stage1_steps or int(
        stage_steps.get("ls_lengths", 64)
    )
    stage2_steps = args.stage2_steps or int(
        stage_steps.get("diffuser_angular_shape", 64)
    )
    stage3_steps = args.stage3_steps or int(
        stage_steps.get("joint", 64)
    )

    stages = (
        ("ls_lengths", stage1_steps,
         ("scat_ls", "abs_ls"), full_objective),
        ("diffuser_angular_shape", stage2_steps,
         LSPHASE_SHAPE_NAMES, full_objective),
        ("joint", stage3_steps, PARAMETER_NAMES, full_objective),
    )
    stage_results = []
    for name, steps, active, objective in stages:
        LOG.info("starting stage %s: active=%s", name, active)
        result = adam_stage(
            objective, theta, PARAMETER_NAMES, active, bounds=bounds,
            train_keys=train_keys, steps=steps,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            mc_validation_objective=full_objective,
            mc_validation_keys=mc_validation_keys,
            validation_every=args.validation_every,
            early_stopping_patience=args.early_stopping_patience,
            min_improvement=args.min_improvement,
        )
        stage_results.append(
            {
                "name": name,
                **{
                    key: (
                        value.tolist()
                        if isinstance(value, np.ndarray)
                        else value
                    )
                    for key, value in result.items()
                    if not key.endswith("_validation_coordinates")
                },
            }
        )
        theta = select_stage_coordinates(result)

    fit_values = np.asarray(
        decode_calibration_array(theta), dtype=np.float64
    )
    rows = []
    for index, name in enumerate(PARAMETER_NAMES):
        difference = float(fit_values[index] - truth[name])
        relative = (
            difference / truth[name] if truth[name] != 0.0 else None
        )
        rows.append({"param": name, "truth": truth[name],
                     "fit": float(fit_values[index]),
                     "difference": difference,
                     "relative_error": relative})
        LOG.info(
            "%-12s truth=%12.6g fit=%12.6g err=%+8.2f%%",
            name,
            truth[name],
            fit_values[index],
            (relative * 100.0 if relative is not None else difference),
        )
    truth_source = set_diffuser_angular_shape(
        sources[0], np.zeros(3, dtype=np.float32)
    )
    fit_source = set_diffuser_angular_shape(
        sources[0], fit_values[2:5]
    )
    coefficient_names = ("c0", "c1", "c2", "c3")
    truth_coefficients = np.asarray(
        lsphase_polynomial_coefficients(truth_source.angular_logits)
    )
    fit_coefficients = np.asarray(
        lsphase_polynomial_coefficients(fit_source.angular_logits)
    )
    result_data = {
        "config": str(config.path),
        "sources": [
            {
                "position": list(position),
                "data_dir": str(data_dir),
                "fixed_event_exposure": exposure,
                "fixed_event_time_offset_ns": time_offset,
            }
            for (position, data_dir), exposure, time_offset in zip(
                source_defs, source_exposures, source_time_offsets
            )
        ],
        "n_photons": n_photons,
        "k": n_steps,
        "likelihood": "unbinned_measured_irf_ppp",
        "fsmp_posterior_step": args.fsmp_step,
        "irf": {
            **irf_metadata,
            "resampled_bin_width_ns": irf_bank.bin_width_ns,
            "resampled_n_bins": int(irf_bank.kernels.shape[1]),
            "interpolation_sigma_ns": interpolation_sigma_ns,
            "future_mode": "per_channel_irf_when_available",
        },
        "truncate_sigma": args.truncate_sigma,
        "background_rate_hz_per_pmt": args.background_rate_hz,
        "exposure": {
            "mode": "explicit_fixed_per_source",
            "monitor_measurement_used": False,
            "fitted_nuisance": False,
            "nu_lc_used": False,
        },
        "support_diagnostics": support,
        "window_stability_diagnostics": window_stability,
        "validation": {
            "mc_validation_same_data": True,
            "mc_validation_independent_transport_keys": True,
            "heldout_validation": False,
            "checkpoint_priority": "mc_validation_then_train",
            "early_stopping_unit": "validation_evaluation",
        },
        "params": rows,
        "lsphase_polynomial_coefficients": {
            "gauge": "fixed nominal integral",
            "truth": dict(zip(
                coefficient_names, truth_coefficients.tolist()
            )),
            "fit": dict(zip(coefficient_names, fit_coefficients.tolist())),
        },
        "stages": stage_results,
    }
    with (outdir / "calibration_result.json").open("w") as stream:
        json.dump(result_data, stream, indent=2)


def build_parser():
    """Return the CLI parser used by execution and contract tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/JUNO_laser_config.json")
    parser.add_argument("--source", action="append", metavar="X,Y,Z:DATA_DIR")
    parser.add_argument(
        "--source-exposure",
        action="append",
        type=float,
        metavar="VALUE",
        help=(
            "fixed event exposure for the corresponding --source; "
            "repeat once per source"
        ),
    )
    parser.add_argument(
        "--source-time-offset-ns",
        action="append",
        type=float,
        metavar="NS",
        help=(
            "fixed time offset for the corresponding --source; "
            "repeat once per source"
        ),
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--intensity", type=float, default=0.0)
    parser.add_argument("--n-photons", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--time-min-ns", type=float, default=None)
    parser.add_argument("--time-max-ns", type=float, default=None)
    parser.add_argument("--time-bin-ns", type=float, default=0.0)
    parser.add_argument("--nb-h", type=int, default=0)
    parser.add_argument("--mc-validation-nb-h", type=int, default=0)
    parser.add_argument(
        "--mc-validation-seed-offset", type=int, default=5_000_000
    )
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=10_000.0)
    parser.add_argument("--stage1-steps", type=int, default=0)
    parser.add_argument("--stage2-steps", type=int, default=0)
    parser.add_argument("--stage3-steps", type=int, default=0)
    parser.add_argument("--fsmp-step", type=int, default=0)
    parser.add_argument("--truncate-sigma", type=float, default=5.0)
    parser.add_argument(
        "--window-stability-expansion-ns", type=float, default=50.0
    )
    parser.add_argument("--background-rate-hz", type=float, default=0.0)
    parser.add_argument(
        "--low-rate-threshold", type=float, default=1.0e-12
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument(
        "--start-values",
        default=None,
        help=("comma-separated physical values in order: "
              + ",".join(PARAMETER_NAMES)),
    )
    parser.add_argument("--start-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    """Parse arguments and run the unbinned real-data calibration."""
    parser = build_parser()
    args = parser.parse_args()
    if args.time_bin_ns < 0 or args.start_fraction < 0:
        parser.error("--time-bin-ns and --start-fraction must be non-negative")
    if args.fsmp_step < 0:
        parser.error("--fsmp-step must be non-negative")
    if args.truncate_sigma <= 0.0:
        parser.error("Gaussian truncation must be positive")
    if args.background_rate_hz < 0.0:
        parser.error("--background-rate-hz must be non-negative")
    if args.low_rate_threshold <= 0.0:
        parser.error("--low-rate-threshold must be positive")
    if args.window_stability_expansion_ns <= 0.0:
        parser.error("--window-stability-expansion-ns must be positive")
    if args.validation_every <= 0 or args.early_stopping_patience < 0:
        parser.error("validation interval must be positive and patience >= 0")
    calibrate(args)


if __name__ == "__main__":
    main()
