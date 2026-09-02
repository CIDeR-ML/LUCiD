#!/usr/bin/env python
"""Calibrate JUNO 532 nm optical parameters from per-PE laser data.

This CLI is the repository-facing calibration entry point.  It consumes
the event NPZ files written by :mod:`run_juno_laser_sim`, builds a binned
Poisson objective, and runs the Adam fit:

1. prompt-time refractive indices (``n_ls``, ``n_water``),
2. four scattering/absorption lengths,
3. all six parameters jointly.

Example::

    python scripts/run_juno_laser_calibrate.py \
        --source 0,0,0:outputs/juno_laser_center \
        --outdir outputs/juno_laser_calibration
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jax
import numpy as np

from lucid.detector_params import load_physics_config
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.sources import isotropic_source
from lucid.juno_laser.config import JunoLaserConfig, PARAMETER_NAMES
from lucid.juno_laser.data import load_per_pe_histogram
from lucid.juno_laser.forward import build_per_pe_forward
from lucid.juno_laser.optimizer import (
    adam_stage,
    make_ensemble_forward,
    make_per_pe_objective,
)
from lucid.juno_laser.parameters import truth_from_geometry


LOG = logging.getLogger("lucid.juno_laser.calibrate")


def _parse_source(spec: str, config: JunoLaserConfig):
    try:
        position_text, data_text = spec.split(":", 1)
        position = tuple(float(value) for value in position_text.split(","))
    except ValueError as exc:
        raise ValueError(f"source must be X,Y,Z:DATA_DIR, got {spec!r}") from exc
    if len(position) != 3:
        raise ValueError(f"source position must have three coordinates, got {position}")
    data_dir = Path(data_text).expanduser()
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()
    return position, data_dir


def _parameter_start(truth, seed: int, start_values: str | None, fraction: float):
    if start_values:
        values = tuple(float(value) for value in start_values.split(","))
        if len(values) != len(PARAMETER_NAMES) or any(value <= 0 for value in values):
            raise ValueError("--start-values must contain six positive values")
        return np.log(np.asarray(values, dtype=np.float32))
    rng = np.random.default_rng(seed)
    values = np.asarray([truth[name] for name in PARAMETER_NAMES], dtype=np.float64)
    perturbation = rng.uniform(-fraction, fraction, size=values.shape)
    return np.log(values) + perturbation.astype(np.float32)


def _make_prompt_masks(observations, prompt_max_ns: float):
    masks = []
    for observation in observations:
        centers = observation.time_min_ns + observation.bin_width_ns * (
            np.arange(observation.n_bins) + 0.5)
        prompt = (centers < prompt_max_ns).astype(np.float32)
        masks.append(np.broadcast_to(prompt[None, :], observation.counts.shape).copy())
    return tuple(masks)


def calibrate(args):
    config = JunoLaserConfig.from_file(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(outdir / "calibration.log", mode="w")])

    geom = DetectorGeometry.from_config(
        str(config.geometry_path), temperature=None,
        max_candidates_per_ray=4, detector_type="nested_sphere")
    physics, _, _ = load_physics_config(
        str(config.physics_path), num_sensors=geom.num_sensors,
        scalar_ref_wavelength=config.wavelength_nm)
    truth = truth_from_geometry(config, geom)
    source_specs = args.source or []
    if not source_specs:
        raise ValueError("at least one --source X,Y,Z:DATA_DIR is required")
    source_defs = [_parse_source(spec, config) for spec in source_specs]
    calibration = config.calibration
    t_min, t_max = calibration["time_window_ns"]
    t_bin = float(args.time_bin_ns or calibration["time_bin_ns"])
    observations = tuple(
        load_per_pe_histogram(
            data_dir, n_sensors=geom.num_sensors,
            time_min_ns=float(args.time_min_ns if args.time_min_ns is not None else t_min),
            time_max_ns=float(args.time_max_ns if args.time_max_ns is not None else t_max),
            bin_width_ns=t_bin, max_events=args.max_events)
        for _, data_dir in source_defs)
    if any(item.n_events <= 0 for item in observations):
        raise ValueError("all source observations must contain at least one event")
    intensity = float(args.intensity or config.data["source"]["default_intensity"])
    sources = tuple(isotropic_source(position, intensity=intensity,
                                     wavelength=config.wavelength_nm)
                    for position, _ in source_defs)
    n_photons = int(args.n_photons or calibration["n_photons"])
    n_steps = int(args.k or config.data["transport"]["max_scattering_steps"])
    forward = build_per_pe_forward(
        config, geom, physics, n_photons=n_photons, k=n_steps,
        intensity=intensity, time_window_ns=(observations[0].time_min_ns,
                                             observations[0].time_max_ns),
        time_bin_ns=t_bin)
    ensemble = make_ensemble_forward(forward, sources, observations[0].counts.shape)
    train_keys = jax.random.split(
        jax.random.PRNGKey(args.seed),
        int(args.nb_h or calibration["train_ensemble_size"]))
    validation_keys = jax.random.split(
        jax.random.PRNGKey(args.seed + args.validation_seed_offset),
        int(args.validation_nb_h or calibration["validation_ensemble_size"]))
    full_masks = tuple(np.ones_like(item.counts, dtype=np.float32) for item in observations)
    prompt_masks = _make_prompt_masks(observations, args.prompt_max_ns)
    full_objective = make_per_pe_objective(
        ensemble, observations, time_masks=full_masks,
        charge_weight=args.charge_weight)
    prompt_objective = make_per_pe_objective(
        ensemble, observations, time_masks=prompt_masks,
        charge_weight=args.stage1_charge_weight)
    bounds = config.parameter_bounds
    theta = _parameter_start(truth, args.seed, args.start_values, args.start_fraction)
    LOG.info("sources=%d, events=%s, n_photons=%d, K=%d, bins=%d x %.3g ns",
             len(sources), [item.n_events for item in observations], n_photons,
             n_steps, observations[0].n_bins, t_bin)
    LOG.info("truth=%s", truth)

    stages = (
        ("prompt_refractive_index", args.stage1_steps, ("n_ls", "n_water"), prompt_objective),
        ("lengths", args.stage2_steps,
         ("scat_ls", "abs_ls", "scat_water", "abs_water"), full_objective),
        ("joint", args.stage3_steps, PARAMETER_NAMES, full_objective),
    )
    stage_results = []
    for name, steps, active, objective in stages:
        LOG.info("starting stage %s: active=%s", name, active)
        result = adam_stage(
            objective, theta, PARAMETER_NAMES, active, bounds=bounds,
            keys=train_keys, steps=steps, learning_rate=args.learning_rate,
            grad_clip=args.grad_clip, validation_objective=full_objective,
            validation_keys=validation_keys, validation_every=args.validation_every,
            early_stopping_patience=args.early_stopping_patience,
            min_improvement=args.min_improvement)
        stage_results.append({"name": name, **{
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in result.items() if key != "best_validation_log_theta"}})
        theta = (result["best_validation_log_theta"]
                 if result["best_validation_log_theta"] is not None
                 else result["log_theta"])

    fit_values = np.exp(np.asarray(theta, dtype=np.float64))
    rows = []
    for index, name in enumerate(PARAMETER_NAMES):
        rows.append({"param": name, "truth": truth[name],
                     "fit": float(fit_values[index]),
                     "err": float(fit_values[index] / truth[name] - 1.0)})
        LOG.info("%-12s truth=%12.6g fit=%12.6g err=%+8.2f%%",
                 name, truth[name], fit_values[index], rows[-1]["err"] * 100)
    result_data = {
        "config": str(config.path),
        "sources": [{"position": list(position), "data_dir": str(data_dir)}
                    for position, data_dir in source_defs],
        "n_photons": n_photons,
        "k": n_steps,
        "params": rows,
        "stages": stage_results,
    }
    with (outdir / "calibration_result.json").open("w") as stream:
        json.dump(result_data, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/JUNO_laser_config.json")
    parser.add_argument("--source", action="append", metavar="X,Y,Z:DATA_DIR")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--intensity", type=float, default=0.0)
    parser.add_argument("--n-photons", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--time-min-ns", type=float, default=None)
    parser.add_argument("--time-max-ns", type=float, default=None)
    parser.add_argument("--time-bin-ns", type=float, default=0.0)
    parser.add_argument("--nb-h", type=int, default=0)
    parser.add_argument("--validation-nb-h", type=int, default=0)
    parser.add_argument("--validation-seed-offset", type=int, default=5_000_000)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=10_000.0)
    parser.add_argument("--stage1-steps", type=int, default=100)
    parser.add_argument("--stage2-steps", type=int, default=200)
    parser.add_argument("--stage3-steps", type=int, default=200)
    parser.add_argument("--prompt-max-ns", type=float, default=250.0)
    parser.add_argument("--stage1-charge-weight", type=float, default=1.0)
    parser.add_argument("--charge-weight", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--start-values", default=None,
                        help="six comma-separated linear-space parameter values")
    parser.add_argument("--start-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.time_bin_ns < 0 or args.start_fraction < 0:
        parser.error("--time-bin-ns and --start-fraction must be non-negative")
    calibrate(args)


if __name__ == "__main__":
    main()
