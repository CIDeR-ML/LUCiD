#!/usr/bin/env python
"""Run a minimal event-wise synthetic closure with full JUNO geometry.

The script deliberately omits Monitor-PMT corrections, exposure variation
and channel offsets.  An optional explicit uniform PMT background is sampled
and fitted with the same PPP model; it is never a numerical rate floor.
Forward PMT-by-time-cell masses are convolved with measured PMT-family IRFs
before constructing a continuous intensity.  Synthetic PEs retain event ID,
PMT ID and exact timestamps.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import jax
import numpy as np

from lucid.detector_params import load_physics_config
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.juno_laser.config import JunoLaserConfig, PARAMETER_NAMES
from lucid.juno_laser.diagnostics import (
    objective_ad_fd_comparison,
    split_source_indices,
)
from lucid.juno_laser.eventwise import (
    EventwisePeObservation,
    gaussian_eventwise_poisson_terms,
    gaussian_window_diagnostics,
    make_chunked_gaussian_eventwise_objective,
    make_host_accumulated_gaussian_eventwise_objective,
    require_positive_pe_support,
    sample_gaussian_mixture_events,
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
from lucid.sources.diffuser_ball import IMPORTANCE_NORMALIZATION_MODES


LOG = logging.getLogger("lucid.juno_laser.full_geometry_closure")
OPTIMIZER_COORDINATE_NAMES = (
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


def git_metadata():
    """Return traceable repository state without changing the worktree."""
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


def json_value(value):
    """Convert optimizer values into JSON-compatible Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def checkpoint_metadata(result):
    """Describe the checkpoint selected by the current automatic policy."""
    if result.get("best_heldout_validation_coordinates") is not None:
        return {
            "kind": "heldout-best",
            "step": int(result["best_heldout_validation_step"]),
            "loss_per_pe": float(
                result["best_heldout_validation_loss"]
            ),
        }
    if result.get("best_mc_validation_coordinates") is not None:
        return {
            "kind": "mc-validation-best",
            "step": int(result["best_mc_validation_step"]),
            "loss_per_pe": float(result["best_mc_validation_loss"]),
        }
    return {
        "kind": "train-best",
        "step": int(result["best_step"]),
        "loss_per_pe": float(result["best_loss"]),
    }


def lsphase_curve(coefficients, n_points=181):
    """Evaluate one JUNOSW-convention LSPhase polynomial curve."""
    theta = np.linspace(0.0, 180.0, n_points)
    cos_theta = np.cos(np.deg2rad(theta))
    values = np.polynomial.polynomial.polyval(cos_theta, coefficients)
    return theta, values


def write_observations(
    path,
    observations,
    predictions,
    positions,
    source_indices,
    args,
    irf_bank,
):
    """Write event-wise synthetic data and truth predictions to HDF5."""
    with h5py.File(path, "w") as output:
        output.attrs["schema"] = "lucid_eventwise_closure_v1"
        output.attrs["likelihood"] = "measured_irf_mixture"
        output.attrs["time_bin_ns"] = args.time_bin_ns
        output.attrs["truncate_sigma"] = args.truncate_sigma
        output.attrs["event_exposure"] = args.event_exposure
        output.attrs["event_time_offset_ns"] = args.event_time_offset_ns
        output.attrs["seed"] = args.seed
        output.attrs["data_n_photons"] = args.data_n_photons
        output.attrs["data_ensemble_size"] = args.data_ensemble_size
        output.attrs["data_importance_normalization"] = (
            args.data_importance_normalization
        )
        output.attrs["fit_n_photons"] = args.fit_n_photons
        output.attrs["k"] = args.k
        output.attrs["intensity"] = args.intensity
        output.attrs["optimization_plan"] = args.optimization_plan
        output.attrs["background_rate_hz_per_pmt"] = (
            args.background_rate_hz
        )
        output.attrs["time_min_ns"] = args.time_min_ns
        output.attrs["time_max_ns"] = args.time_max_ns
        output.attrs["source_count"] = len(positions)
        irf_group = output.create_group("irf")
        irf_group.attrs["granularity"] = "pmt_type_average"
        irf_group.attrs["bin_width_ns"] = irf_bank.bin_width_ns
        irf_group.create_dataset(
            "family_kernel_mass", data=np.asarray(irf_bank.kernels)
        )
        items = zip(
            observations, predictions, positions, source_indices
        )
        for index, (observation, prediction, position, source_index) in (
            enumerate(items)
        ):
            group = output.create_group(f"source_{index:02d}")
            group.attrs["position_m"] = position
            group.attrs["absolute_source_index"] = source_index
            group.attrs["n_events"] = observation.n_events
            group.create_dataset(
                "event_id", data=observation.event_ids, compression="gzip"
            )
            group.create_dataset(
                "pmt_id", data=observation.pmt_ids, compression="gzip"
            )
            group.create_dataset(
                "pe_time_ns",
                data=observation.pe_times_ns,
                compression="gzip",
            )
            group.create_dataset(
                "event_exposure",
                data=observation.event_exposures,
                compression="gzip",
            )
            group.create_dataset(
                "event_time_offset_ns",
                data=observation.event_time_offsets_ns,
                compression="gzip",
            )
            group.create_dataset(
                "truth_bin_mass",
                data=np.asarray(prediction, dtype=np.float32),
                compression="gzip",
            )


def _require_matching_attr(handle, name, expected):
    """Require one HDF5 attribute to match the active closure contract."""
    actual = handle.attrs[name]
    if isinstance(expected, float):
        matches = np.isclose(float(actual), expected, rtol=0.0, atol=1e-9)
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(
            f"observation attribute {name!r} is {actual!r}, "
            f"expected {expected!r}"
        )


def read_observation_fragments(
    paths,
    positions,
    source_indices,
    time_grid,
    n_sensors,
    args,
    irf_bank,
):
    """Read and validate independently generated source observations."""
    records = {}
    for path in paths:
        with h5py.File(path, "r") as source_file:
            _require_matching_attr(
                source_file, "schema", "lucid_eventwise_closure_v1"
            )
            expected_attrs = {
                "time_bin_ns": time_grid.bin_width_ns,
                "time_min_ns": time_grid.time_min_ns,
                "time_max_ns": time_grid.time_max_ns,
                "event_exposure": args.event_exposure,
                "event_time_offset_ns": args.event_time_offset_ns,
                "seed": args.seed,
                "data_n_photons": args.data_n_photons,
                "data_ensemble_size": args.data_ensemble_size,
                "data_importance_normalization": (
                    args.data_importance_normalization
                ),
                "k": args.k,
                "intensity": args.intensity,
                "truncate_sigma": args.truncate_sigma,
                "background_rate_hz_per_pmt": args.background_rate_hz,
            }
            for name, expected in expected_attrs.items():
                _require_matching_attr(source_file, name, expected)
            saved_kernel = np.asarray(
                source_file["irf/family_kernel_mass"]
            )
            np.testing.assert_allclose(
                saved_kernel,
                np.asarray(irf_bank.kernels),
                rtol=1e-6,
                atol=1e-7,
            )
            groups = sorted(
                name for name in source_file if name.startswith("source_")
            )
            for name in groups:
                group = source_file[name]
                saved_n_events = int(group.attrs["n_events"])
                if saved_n_events != args.n_events:
                    raise ValueError(
                        "observation event count is "
                        f"{saved_n_events}, expected {args.n_events}"
                    )
                source_index = int(group.attrs["absolute_source_index"])
                if source_index in records:
                    raise ValueError(
                        f"duplicate observation source index {source_index}"
                    )
                prediction = np.asarray(
                    group["truth_bin_mass"], dtype=np.float32
                )
                time_grid.validate_mass_shape(
                    prediction, name="truth bin mass"
                )
                if prediction.shape[0] != n_sensors:
                    raise ValueError(
                        "truth bin mass uses the wrong PMT dimension"
                    )
                observation = EventwisePeObservation(
                    event_ids=group["event_id"][...],
                    pmt_ids=group["pmt_id"][...],
                    pe_times_ns=group["pe_time_ns"][...],
                    n_events=int(group.attrs["n_events"]),
                    n_sensors=n_sensors,
                    time_grid=time_grid,
                    event_exposures=group["event_exposure"][...],
                    event_time_offsets_ns=(
                        group["event_time_offset_ns"][...]
                    ),
                )
                if observation.n_events != args.n_events:
                    raise ValueError(
                        f"source {source_index} contains "
                        f"{observation.n_events} events, expected "
                        f"{args.n_events}"
                    )
                records[source_index] = (
                    tuple(float(value) for value in group.attrs["position_m"]),
                    observation,
                    prediction,
                )
    expected = set(source_indices)
    if set(records) != expected:
        raise ValueError(
            "observation source indices do not match requested sources: "
            f"found={sorted(records)}, expected={sorted(expected)}"
        )
    observations = []
    predictions = []
    for source_index, position in zip(source_indices, positions):
        saved_position, observation, prediction = records[source_index]
        if not np.allclose(saved_position, position, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"source {source_index} position is {saved_position}, "
                f"expected {position}"
            )
        observations.append(observation)
        predictions.append(prediction)
    return tuple(observations), tuple(predictions)


def parameter_start(truth, fraction, seed):
    """Return a reproducibly perturbed start in optimizer coordinates."""
    start = np.asarray(encode_calibration_values(truth), dtype=np.float32)
    rng = np.random.default_rng(seed)
    shift = rng.uniform(-fraction, fraction, size=start.shape)
    return start + shift.astype(np.float32)


def source_config_with_normalization(config, selected):
    """Return a source config with an optional importance-mode override."""
    source_config = dict(config.data["source"])
    if selected != "config":
        source_config["importance_normalization"] = selected
    return source_config


def partition_source_indices(indices, chunk_size):
    """Partition absolute source indices without changing their identity."""
    indices = tuple(int(index) for index in indices)
    if not indices:
        return ()
    size = len(indices) if chunk_size <= 0 else int(chunk_size)
    if size <= 0:
        raise ValueError("source chunk size must be non-negative")
    return tuple(
        indices[start:start + size]
        for start in range(0, len(indices), size)
    )


def build_response_chunks(
    forward,
    sources,
    source_index_chunks,
    absolute_source_indices,
    hist_shape,
    fixed,
    pmt_types,
    irf_bank,
):
    """Build one IRF-convolved ensemble forward per source chunk."""
    responses = []
    for indices in source_index_chunks:
        absolute_indices = tuple(
            absolute_source_indices[index] for index in indices
        )
        optical = make_ensemble_forward(
            forward,
            tuple(sources[index] for index in indices),
            hist_shape,
            fixed_params=fixed,
            source_indices=absolute_indices,
        )
        responses.append(
            make_irf_response_forward(optical, pmt_types, irf_bank)
        )
    return tuple(responses)


def evaluate_response_chunks(responses, theta, keys):
    """Evaluate source chunks sequentially and return host predictions."""
    predictions = []
    for response in responses:
        chunk = response(theta, keys)
        for prediction in chunk:
            prediction.block_until_ready()
            predictions.append(np.asarray(prediction, dtype=np.float32))
    return tuple(predictions)


def observation_chunks(observations, source_index_chunks):
    """Select observations using the same absolute-index chunk layout."""
    return tuple(
        tuple(observations[index] for index in indices)
        for indices in source_index_chunks
    )


def build_joint_objective(
    execution,
    responses,
    chunks,
    *,
    sigma_ns,
    truncate_sigma,
    background_rate_per_ns,
):
    """Build either a fused or host-accumulated joint PPP objective."""
    factory = (
        make_host_accumulated_gaussian_eventwise_objective
        if execution == "per-source"
        else make_chunked_gaussian_eventwise_objective
    )
    return factory(
        responses,
        chunks,
        sigma_ns=sigma_ns,
        truncate_sigma=truncate_sigma,
        background_rate_per_ns=background_rate_per_ns,
    )


def evaluate_value_and_gradient(objective, theta, keys):
    """Evaluate a fused or host-accumulated objective and gradient."""
    if hasattr(objective, "value_and_grad"):
        return objective.value_and_grad(theta, keys)
    value, gradient = jax.value_and_grad(objective)(theta, keys)
    value.block_until_ready()
    return value, gradient


def evaluate_reference_objective_losses(
    objective,
    truth_coordinates,
    fit_keys,
    mc_validation_keys,
    *,
    heldout_objective=None,
    heldout_keys=None,
):
    """Evaluate truth coordinates under each configured objective."""
    losses = {
        "truth_train_loss_per_pe": float(
            objective(truth_coordinates, fit_keys)
        ),
        "truth_mc_validation_loss_per_pe": float(
            objective(truth_coordinates, mc_validation_keys)
        ),
        "truth_heldout_loss_per_pe": None,
    }
    if heldout_objective is not None:
        losses["truth_heldout_loss_per_pe"] = float(
            heldout_objective(truth_coordinates, heldout_keys)
        )
    return losses


def optimization_stages(args):
    """Return the requested staged or all-parameter optimization plan."""
    if args.optimization_plan == "joint-only":
        return (("joint", args.joint_steps, PARAMETER_NAMES),)
    return (
        ("ls_lengths", args.stage1_steps, ("scat_ls", "abs_ls")),
        (
            "diffuser_shape",
            args.stage2_steps,
            LSPHASE_SHAPE_NAMES,
        ),
        ("joint", args.stage3_steps, PARAMETER_NAMES),
    )


def checkpoint_writer(outdir, stage_name):
    """Return a callback that atomically replaces one recovery snapshot."""
    destination = outdir / f"checkpoint_{stage_name}.npz"

    def write(state):
        temporary = destination.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            stage=np.asarray(stage_name),
            **state,
        )
        temporary.replace(destination)

    return write


def support_diagnostics(
    observations,
    predictions,
    args,
    sigma_ns,
    background_rate_per_ns,
    *,
    label,
):
    """Check that every observed PE has strictly positive model support."""
    summaries = []
    for index, (observation, prediction) in enumerate(
        zip(observations, predictions)
    ):
        terms = gaussian_eventwise_poisson_terms(
            observation,
            prediction,
            sigma_ns=sigma_ns,
            truncate_sigma=args.truncate_sigma,
            background_rate_per_ns=background_rate_per_ns,
        )
        terms.pe_rate.block_until_ready()
        summaries.append(
            require_positive_pe_support(
                terms.pe_rate,
                context=f"{label} source {index}",
                low_rate_threshold=args.low_rate_threshold,
            )
        )
    return summaries


def run_closure(args):
    """Generate event-wise synthetic PEs and run the requested fit plan."""
    started = time.perf_counter()
    timings = {}
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.outdir / "closure.log", mode="w"),
        ],
    )
    config = JunoLaserConfig.from_file(args.config)
    calibration = config.data["calibration"]
    legacy_photons = args.n_photons
    args.data_n_photons = int(
        args.data_n_photons
        or legacy_photons
        or calibration["n_photons"]
    )
    args.fit_n_photons = int(
        args.fit_n_photons
        or legacy_photons
        or calibration["n_photons"]
    )
    args.mc_validation_ensemble_size = int(
        args.mc_validation_ensemble_size
        or calibration["mc_validation_ensemble_size"]
    )
    args.joint_steps = int(
        args.joint_steps or calibration["stage_steps"]["joint"]
    )
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
    positions = tuple(parse_position(text) for text in args.source_position)
    if args.heldout_source_index:
        train_indices, heldout_indices = split_source_indices(
            len(positions), args.heldout_source_index
        )
    else:
        train_indices = tuple(range(len(positions)))
        heldout_indices = ()
    data_source_config = source_config_with_normalization(
        config, args.data_importance_normalization
    )
    fit_source_config = source_config_with_normalization(
        config, args.fit_importance_normalization
    )
    data_sources = tuple(
        build_juno_laser_source(
            data_source_config,
            position,
            intensity=args.intensity,
            wavelength_nm=config.wavelength_nm,
        )
        for position in positions
    )
    fit_sources = tuple(
        build_juno_laser_source(
            fit_source_config,
            position,
            intensity=args.intensity,
            wavelength_nm=config.wavelength_nm,
        )
        for position in positions
    )
    truth = truth_from_geometry(config, geometry)
    fixed = fixed_params_from_geometry(config, geometry)
    time_grid = TimeGrid(
        args.time_min_ns,
        args.time_max_ns,
        args.time_bin_ns,
    )
    hist_shape = (geometry.num_sensors, time_grid.n_bins)
    pmt_types = geometry.detector.pmt_type_codes
    if pmt_types is None:
        raise ValueError("measured IRF response requires JUNO PMT types")
    native_irf, irf_metadata = load_type_average_irf(
        config.irf_path, config.irf_pmt_map_path
    )
    irf_bank = resample_irf_bank(native_irf, time_grid.bin_width_ns)
    source_indices = tuple(args.source_index) or tuple(
        range(len(positions))
    )
    all_indices = tuple(range(len(positions)))
    effective_chunk_size = (
        1
        if args.objective_execution == "per-source"
        else args.source_chunk_size
    )
    train_index_chunks = partition_source_indices(
        train_indices, effective_chunk_size
    )
    heldout_index_chunks = partition_source_indices(
        heldout_indices, effective_chunk_size
    )
    sigma_ns = config.interpolation_sigma_ns
    background_rate_per_ns = args.background_rate_hz * 1.0e-9
    truth_coordinates = encode_calibration_values(truth)
    if args.phase == "fit":
        loading_started = time.perf_counter()
        observations, truth_predictions = read_observation_fragments(
            args.observation_input,
            positions,
            source_indices,
            time_grid,
            geometry.num_sensors,
            args,
            irf_bank,
        )
        timings["observation_loading_seconds"] = (
            time.perf_counter() - loading_started
        )
        LOG.info(
            "loaded %d independently generated source fragment(s)",
            len(observations),
        )
    else:
        data_forward_build_started = time.perf_counter()
        data_forward = build_per_pe_forward(
            config,
            geometry,
            physics,
            n_photons=args.data_n_photons,
            k=args.k,
            intensity=args.intensity,
            time_grid=time_grid,
        )
        timings["data_forward_build_seconds"] = (
            time.perf_counter() - data_forward_build_started
        )
        data_index_chunks = partition_source_indices(
            all_indices, effective_chunk_size
        )
        data_responses = build_response_chunks(
            data_forward,
            data_sources,
            data_index_chunks,
            source_indices,
            hist_shape,
            fixed,
            pmt_types,
            irf_bank,
        )
        data_keys = jax.random.split(
            jax.random.PRNGKey(args.seed), args.data_ensemble_size
        )
        LOG.info(
            "evaluating data forward: %d source(s), %d photons, K=%d",
            len(positions),
            args.data_n_photons,
            args.k,
        )
        data_generation_started = time.perf_counter()
        truth_predictions = evaluate_response_chunks(
            data_responses, truth_coordinates, data_keys
        )
        observations = tuple(
            sample_gaussian_mixture_events(
                prediction,
                n_events=args.n_events,
                time_grid=time_grid,
                seed=args.seed + 1000 + source_index,
                sigma_ns=sigma_ns,
                truncate_sigma=args.truncate_sigma,
                event_exposure=args.event_exposure,
                event_time_offset_ns=args.event_time_offset_ns,
                background_rate_per_ns=background_rate_per_ns,
            )
            for source_index, prediction in zip(
                source_indices, truth_predictions
            )
        )
        timings["data_generation_seconds"] = (
            time.perf_counter() - data_generation_started
        )
    empty_predictions = [
        index for index, prediction in enumerate(truth_predictions)
        if float(np.sum(prediction)) <= 0.0
    ]
    if empty_predictions:
        raise RuntimeError(
            "zero predicted PE mass for source indices "
            f"{empty_predictions}; increase K and/or n_photons"
        )
    empty_observations = [
        index for index, observation in enumerate(observations)
        if observation.n_pe == 0
    ]
    if empty_observations:
        raise RuntimeError(
            "zero sampled PEs for source indices "
            f"{empty_observations}; increase intensity or n_events"
        )
    truth_support = support_diagnostics(
        observations,
        truth_predictions,
        args,
        sigma_ns,
        background_rate_per_ns,
        label="synthetic truth",
    )
    write_observations(
        args.outdir / "closure_observations.h5",
        observations,
        truth_predictions,
        positions,
        source_indices,
        args,
        irf_bank,
    )
    if args.phase == "generate":
        result = {
            "schema": "lucid_closure_generation_v1",
            "completed": True,
            "phase": "generate",
            "source_positions_m": [list(item) for item in positions],
            "source_indices": list(source_indices),
            "n_pe": [item.n_pe for item in observations],
            "support_diagnostics": truth_support,
            "timings": timings,
            "seconds": time.perf_counter() - started,
        }
        path = args.outdir / "generation_result.json"
        with path.open("w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        LOG.info("source generation completed: %s", args.outdir)
        return result
    if args.phase == "all":
        LOG.info("releasing data-forward executable before fit build")
        del data_responses, data_forward
        jax.clear_caches()
        gc.collect()

    LOG.info(
        "building fit forward: %d source(s), %d photons, K=%d",
        len(train_indices),
        args.fit_n_photons,
        args.k,
    )
    fit_forward_build_started = time.perf_counter()
    fit_forward = build_per_pe_forward(
        config,
        geometry,
        physics,
        n_photons=args.fit_n_photons,
        k=args.k,
        intensity=args.intensity,
        time_grid=time_grid,
    )
    train_responses = build_response_chunks(
        fit_forward,
        fit_sources,
        train_index_chunks,
        source_indices,
        hist_shape,
        fixed,
        pmt_types,
        irf_bank,
    )
    heldout_responses = build_response_chunks(
        fit_forward,
        fit_sources,
        heldout_index_chunks,
        source_indices,
        hist_shape,
        fixed,
        pmt_types,
        irf_bank,
    )
    timings["fit_forward_build_seconds"] = (
        time.perf_counter() - fit_forward_build_started
    )
    train_observation_chunks = observation_chunks(
        observations, train_index_chunks
    )
    heldout_observation_chunks = observation_chunks(
        observations, heldout_index_chunks
    )
    objective = build_joint_objective(
        args.objective_execution,
        train_responses,
        train_observation_chunks,
        sigma_ns=sigma_ns,
        truncate_sigma=args.truncate_sigma,
        background_rate_per_ns=background_rate_per_ns,
    )
    heldout_objective = None
    if heldout_responses:
        heldout_objective = build_joint_objective(
            args.objective_execution,
            heldout_responses,
            heldout_observation_chunks,
            sigma_ns=sigma_ns,
            truncate_sigma=args.truncate_sigma,
            background_rate_per_ns=background_rate_per_ns,
        )
    fit_keys = jax.random.split(
        jax.random.PRNGKey(args.seed + args.fit_seed_offset),
        args.fit_ensemble_size,
    )
    mc_validation_keys = jax.random.split(
        jax.random.PRNGKey(
            args.seed
            + args.fit_seed_offset
            + args.mc_validation_seed_offset
        ),
        args.mc_validation_ensemble_size,
    )
    heldout_keys = None
    if heldout_objective is not None:
        heldout_keys = jax.random.split(
            jax.random.PRNGKey(
                args.seed
                + args.fit_seed_offset
                + args.heldout_seed_offset
            ),
            args.fit_ensemble_size,
        )
    reference_losses = evaluate_reference_objective_losses(
        objective,
        truth_coordinates,
        fit_keys,
        mc_validation_keys,
        heldout_objective=heldout_objective,
        heldout_keys=heldout_keys,
    )
    theta = parameter_start(truth, args.start_fraction, args.seed + 2)
    if args.restart_checkpoint is not None:
        if args.optimization_plan != "joint-only":
            raise ValueError(
                "checkpoint restart currently requires joint-only"
            )
        with np.load(args.restart_checkpoint) as checkpoint:
            theta = np.asarray(checkpoint["coordinates"], dtype=np.float32)
        if theta.shape != (len(PARAMETER_NAMES),):
            raise ValueError("restart checkpoint has invalid coordinates")
    initial_theta = np.asarray(theta, dtype=np.float32)
    LOG.info("evaluating initial fit and MC-validation forwards")
    initial_predictions = evaluate_response_chunks(
        train_responses, theta, fit_keys
    )
    train_observations = tuple(
        observations[index] for index in train_indices
    )
    initial_support = support_diagnostics(
        train_observations,
        initial_predictions,
        args,
        sigma_ns,
        background_rate_per_ns,
        label="initial fit",
    )
    heldout_support = None
    mc_validation_predictions = evaluate_response_chunks(
        train_responses, theta, mc_validation_keys
    )
    mc_validation_support = support_diagnostics(
        train_observations,
        mc_validation_predictions,
        args,
        sigma_ns,
        background_rate_per_ns,
        label="initial MC-validation fit",
    )
    if heldout_responses:
        heldout_predictions = evaluate_response_chunks(
            heldout_responses, theta, heldout_keys
        )
        heldout_observations = tuple(
            observations[index] for index in heldout_indices
        )
        heldout_support = support_diagnostics(
            heldout_observations,
            heldout_predictions,
            args,
            sigma_ns,
            background_rate_per_ns,
            label="initial held-out fit",
        )
    window_diagnostics = [
        gaussian_window_diagnostics(
            prediction,
            time_grid=time_grid,
            sigma_ns=sigma_ns,
            expansion_ns=args.window_stability_expansion_ns,
            observation=observation,
        )
        for observation, prediction in zip(
            observations, truth_predictions
        )
    ]
    initial_gradient_started = time.perf_counter()
    initial_loss, initial_gradient = evaluate_value_and_gradient(
        objective, theta, fit_keys
    )
    timings["initial_value_and_gradient_seconds"] = (
        time.perf_counter() - initial_gradient_started
    )
    initial_gradient = np.asarray(initial_gradient, dtype=np.float64)
    if not np.isfinite(float(initial_loss)) or not np.all(
        np.isfinite(initial_gradient)
    ):
        raise FloatingPointError("initial closure loss or gradient is invalid")
    zero_gradient = [
        name for name, value in zip(PARAMETER_NAMES, initial_gradient)
        if value == 0.0
    ]
    if zero_gradient:
        raise FloatingPointError(
            "zero initial closure gradient for "
            f"{zero_gradient}; increase the transport budget or use "
            "common random numbers"
        )
    ad_fd = None
    if args.ad_fd_relative_step > 0.0:
        audit_background_rate_hz = (
            args.background_rate_hz
            if args.ad_fd_background_rate_hz is None
            else args.ad_fd_background_rate_hz
        )
        audit_objective = build_joint_objective(
            args.objective_execution,
            train_responses,
            train_observation_chunks,
            sigma_ns=sigma_ns,
            truncate_sigma=args.truncate_sigma,
            background_rate_per_ns=(
                audit_background_rate_hz * 1.0e-9
            ),
        )
        audit = objective_ad_fd_comparison(
            audit_objective,
            theta,
            fit_keys,
            relative_step=args.ad_fd_relative_step,
            minimum_step=args.ad_fd_minimum_step,
            value_and_grad_fn=getattr(
                audit_objective, "value_and_grad", None
            ),
        )
        vector_fields = (
            "steps",
            "ad_gradient",
            "finite_difference_gradient",
            "absolute_difference",
            "relative_difference",
            "plus_objective",
            "minus_objective",
        )
        ad_fd = {
            key: (
                dict(zip(OPTIMIZER_COORDINATE_NAMES, audit[key].tolist()))
                if key in vector_fields
                else value
            )
            for key, value in audit.items()
        }
        ad_fd["diagnostic_background_rate_hz_per_pmt"] = (
            audit_background_rate_hz
        )
    stages = optimization_stages(args)
    stage_results = []
    for name, steps, active in stages:
        result = adam_stage(
            objective,
            theta,
            PARAMETER_NAMES,
            active,
            bounds=config.parameter_bounds,
            train_keys=fit_keys,
            steps=steps,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            mc_validation_objective=objective,
            mc_validation_keys=mc_validation_keys,
            heldout_validation_objective=heldout_objective,
            heldout_validation_keys=heldout_keys,
            validation_every=(
                args.validation_every
            ),
            early_stopping_patience=args.early_stopping_patience,
            min_improvement=args.min_improvement,
            checkpoint_callback=checkpoint_writer(args.outdir, name),
            checkpoint_every=args.checkpoint_every,
            value_and_grad_fn=getattr(
                objective, "value_and_grad", None
            ),
        )
        stage_results.append(
            {
                "name": name,
                "selected_checkpoint": checkpoint_metadata(result),
                **{
                    key: json_value(value)
                    for key, value in result.items()
                    if not key.endswith("_validation_coordinates")
                },
            }
        )
        theta = select_stage_coordinates(result)
    final_loss, final_gradient = evaluate_value_and_gradient(
        objective, theta, fit_keys
    )
    final_gradient = np.asarray(final_gradient, dtype=np.float64)
    completed = bool(
        np.isfinite(float(final_loss))
        and np.all(np.isfinite(final_gradient))
    )
    if not completed:
        raise FloatingPointError("final closure loss or gradient is invalid")
    heldout_initial_loss = None
    heldout_final_loss = None
    mc_validation_initial_loss = float(
        objective(initial_theta, mc_validation_keys)
    )
    mc_validation_final_loss = float(
        objective(theta, mc_validation_keys)
    )
    if heldout_objective is not None:
        heldout_initial_loss = float(
            heldout_objective(initial_theta, heldout_keys)
        )
        heldout_final_loss = float(heldout_objective(theta, heldout_keys))
    fit_values = np.asarray(decode_calibration_array(theta), dtype=np.float64)
    initial_values = np.asarray(
        decode_calibration_array(initial_theta), dtype=np.float64
    )
    truth_source = set_diffuser_angular_shape(
        fit_sources[0], np.zeros(3, dtype=np.float32)
    )
    initial_source = set_diffuser_angular_shape(
        fit_sources[0], initial_values[2:5]
    )
    fit_source = set_diffuser_angular_shape(
        fit_sources[0], fit_values[2:5]
    )
    coefficient_names = ("c0", "c1", "c2", "c3")
    truth_coefficients = np.asarray(
        lsphase_polynomial_coefficients(truth_source.angular_logits)
    )
    initial_coefficients = np.asarray(
        lsphase_polynomial_coefficients(initial_source.angular_logits)
    )
    fit_coefficients = np.asarray(
        lsphase_polynomial_coefficients(fit_source.angular_logits)
    )
    theta_degrees, truth_curve = lsphase_curve(truth_coefficients)
    _, initial_curve = lsphase_curve(initial_coefficients)
    _, fit_curve = lsphase_curve(fit_coefficients)
    result = {
        "schema": "lucid_full_geometry_eventwise_closure_v1",
        "completed": completed,
        "scope": {
            "eventwise_pe": True,
            "continuous_time_unbinned": True,
            "time_model": "measured_pmt_family_irf",
            "interpolation_sigma_ns": sigma_ns,
            "interpolation_sigma_is_physical_tts_irf": False,
            "synthetic_event_exposure_fixed": True,
            "synthetic_event_time_offset_fixed": True,
            "monitor_exposure_measurement_used": False,
            "exposure_nuisance_fitted": False,
            "tts_irf_convolution": True,
            "irf_granularity": "pmt_type_average",
            "channel_time_offsets": False,
            "dark_noise_background": args.background_rate_hz > 0.0,
            "mc_validation": True,
            "heldout_validation": bool(heldout_indices),
        },
        "git": git_metadata(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "config": str(config.path),
        "execution_phase": args.phase,
        "source_positions_m": [list(item) for item in positions],
        "source_indices": list(source_indices),
        "train_source_indices": list(train_indices),
        "heldout_source_indices": list(heldout_indices),
        "n_events": args.n_events,
        "n_pe": [item.n_pe for item in observations],
        "n_photons": args.fit_n_photons,
        "data_n_photons": args.data_n_photons,
        "fit_n_photons": args.fit_n_photons,
        "k": args.k,
        "intensity": args.intensity,
        "event_exposure": args.event_exposure,
        "event_time_offset_ns": args.event_time_offset_ns,
        "background_rate_hz_per_pmt": args.background_rate_hz,
        "time_window_ns": [args.time_min_ns, args.time_max_ns],
        "time_bin_ns": args.time_bin_ns,
        "truncate_sigma": args.truncate_sigma,
        "support_diagnostics": {
            "truth": truth_support,
            "initial_fit": initial_support,
            "initial_mc_validation_fit": mc_validation_support,
            "initial_heldout_fit": heldout_support,
        },
        "window_stability_diagnostics": window_diagnostics,
        "data_ensemble_size": args.data_ensemble_size,
        "fit_ensemble_size": args.fit_ensemble_size,
        "mc_validation_ensemble_size": (
            args.mc_validation_ensemble_size
        ),
        "seed": args.seed,
        "fit_seed_offset": args.fit_seed_offset,
        "mc_validation_seed_offset": args.mc_validation_seed_offset,
        "heldout_seed_offset": args.heldout_seed_offset,
        "data_importance_normalization": (
            IMPORTANCE_NORMALIZATION_MODES[
                int(data_sources[0].importance_normalization)
            ]
        ),
        "fit_importance_normalization": (
            IMPORTANCE_NORMALIZATION_MODES[
                int(fit_sources[0].importance_normalization)
            ]
        ),
        "parameter_names": list(PARAMETER_NAMES),
        "optimizer_coordinate_names": list(OPTIMIZER_COORDINATE_NAMES),
        "truth": {name: float(truth[name]) for name in PARAMETER_NAMES},
        "start": dict(zip(PARAMETER_NAMES, initial_values.tolist())),
        "fit": dict(zip(PARAMETER_NAMES, fit_values.tolist())),
        "lsphase_polynomial_coefficients": {
            "gauge": "fixed nominal integral",
            "truth": dict(zip(
                coefficient_names, truth_coefficients.tolist()
            )),
            "start": dict(zip(
                coefficient_names, initial_coefficients.tolist()
            )),
            "fit": dict(zip(coefficient_names, fit_coefficients.tolist())),
        },
        "lsphase_theta_curve": {
            "definition": "JUNOSW pol3 evaluated at cos(theta)",
            "theta_degrees": theta_degrees.tolist(),
            "truth": truth_curve.tolist(),
            "start": initial_curve.tolist(),
            "fit": fit_curve.tolist(),
        },
        "irf": {
            **irf_metadata,
            "resampled_bin_width_ns": irf_bank.bin_width_ns,
            "resampled_n_bins": int(irf_bank.kernels.shape[1]),
        },
        "initial_loss_per_pe": float(initial_loss),
        "final_loss_per_pe": float(final_loss),
        **reference_losses,
        "mc_validation_initial_loss_per_pe": (
            mc_validation_initial_loss
        ),
        "mc_validation_final_loss_per_pe": mc_validation_final_loss,
        "heldout_initial_loss_per_pe": heldout_initial_loss,
        "heldout_final_loss_per_pe": heldout_final_loss,
        "initial_gradient": dict(
            zip(PARAMETER_NAMES, initial_gradient.tolist())
        ),
        "final_gradient": dict(
            zip(PARAMETER_NAMES, final_gradient.tolist())
        ),
        "ad_fd_audit_at_initial_coordinates": ad_fd,
        "optimization_plan": args.optimization_plan,
        "optimizer": {
            "learning_rate": args.learning_rate,
            "grad_clip": args.grad_clip,
            "start_fraction": args.start_fraction,
            "joint_steps": args.joint_steps,
            "stage1_steps": args.stage1_steps,
            "stage2_steps": args.stage2_steps,
            "stage3_steps": args.stage3_steps,
            "validation_every": args.validation_every,
            "early_stopping_patience": args.early_stopping_patience,
            "min_improvement": args.min_improvement,
        },
        "source_chunk_size": args.source_chunk_size,
        "effective_source_chunk_size": effective_chunk_size,
        "objective_execution": args.objective_execution,
        "checkpoint_every": args.checkpoint_every,
        "restart_checkpoint": (
            None
            if args.restart_checkpoint is None
            else str(args.restart_checkpoint.resolve())
        ),
        "restart_resets_adam_moments": (
            args.restart_checkpoint is not None
        ),
        "timings": timings,
        "stages": stage_results,
        "seconds": time.perf_counter() - started,
    }
    with (args.outdir / "closure_result.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    LOG.info("closure completed: %s", args.outdir)
    return result


def main():
    """Parse command-line arguments and execute synthetic closure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "generate", "fit"),
        default="all",
        help=(
            "all is the local single-process regression path; GPU jobs "
            "use independent generate and fit processes"
        ),
    )
    parser.add_argument(
        "--config", type=Path, default="config/JUNO_laser_config.json"
    )
    parser.add_argument(
        "--source-position", action="append", required=True,
        metavar="X,Y,Z",
    )
    parser.add_argument(
        "--source-index",
        action="append",
        type=int,
        default=[],
        help="stable absolute index paired with each source position",
    )
    parser.add_argument(
        "--observation-input",
        action="append",
        type=Path,
        default=[],
        help="HDF5 source fragment read by the fit phase",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-events", type=int, default=2)
    parser.add_argument(
        "--n-photons",
        type=int,
        default=None,
        help="deprecated compatibility alias setting both photon budgets",
    )
    parser.add_argument("--data-n-photons", type=int, default=None)
    parser.add_argument("--fit-n-photons", type=int, default=None)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--intensity", type=float, default=1000.0)
    parser.add_argument("--event-exposure", type=float, default=1.0)
    parser.add_argument("--event-time-offset-ns", type=float, default=0.0)
    parser.add_argument(
        "--background-rate-hz",
        type=float,
        default=0.0,
        help=(
            "explicit uniform background per PMT, sampled and fitted in "
            "both PPP terms"
        ),
    )
    parser.add_argument("--time-min-ns", type=float, default=0.0)
    parser.add_argument("--time-max-ns", type=float, default=300.0)
    parser.add_argument("--time-bin-ns", type=float, default=10.0)
    parser.add_argument("--truncate-sigma", type=float, default=5.0)
    parser.add_argument(
        "--ad-fd-relative-step", type=float, default=0.0,
        help="positive value enables a five-parameter centered FD audit",
    )
    parser.add_argument(
        "--ad-fd-minimum-step", type=float, default=1.0e-4
    )
    parser.add_argument(
        "--ad-fd-background-rate-hz", type=float, default=None,
        help=(
            "optional AD-FD-only background override; by default use "
            "--background-rate-hz"
        ),
    )
    parser.add_argument(
        "--window-stability-expansion-ns", type=float, default=50.0
    )
    parser.add_argument("--data-ensemble-size", type=int, default=1)
    parser.add_argument("--fit-ensemble-size", type=int, default=1)
    parser.add_argument(
        "--mc-validation-ensemble-size", type=int, default=None
    )
    parser.add_argument(
        "--data-importance-normalization",
        choices=("config", "analytic", "self_normalized"),
        default="config",
    )
    parser.add_argument(
        "--fit-importance-normalization",
        choices=("config", "analytic", "self_normalized"),
        default="config",
    )
    parser.add_argument(
        "--fit-seed-offset",
        type=int,
        default=0,
        help="0 uses common transport random numbers for minimal closure",
    )
    parser.add_argument(
        "--mc-validation-seed-offset", type=int, default=2000003
    )
    parser.add_argument(
        "--heldout-source-index",
        action="append",
        type=int,
        default=[],
        help="zero-based source index excluded from training and validation",
    )
    parser.add_argument("--heldout-seed-offset", type=int, default=0)
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--min-improvement", type=float, default=1.0e-4)
    parser.add_argument(
        "--optimization-plan",
        choices=("joint-only", "staged"),
        default="joint-only",
    )
    parser.add_argument("--joint-steps", type=int, default=None)
    parser.add_argument("--stage1-steps", type=int, default=64)
    parser.add_argument("--stage2-steps", type=int, default=64)
    parser.add_argument("--stage3-steps", type=int, default=64)
    parser.add_argument(
        "--source-chunk-size",
        type=int,
        default=0,
        help="0 places all selected sources in one response chunk",
    )
    parser.add_argument(
        "--objective-execution",
        choices=("fused", "per-source"),
        default="fused",
        help=(
            "per-source evaluates one-source XLA modules and sums loss/"
            "gradient on the host"
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--restart-checkpoint",
        type=Path,
        default=None,
        help=(
            "restart joint-only Adam from saved coordinates with fresh "
            "moments"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0e4)
    parser.add_argument(
        "--low-rate-threshold", type=float, default=1.0e-12
    )
    parser.add_argument("--start-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.source_index and (
        len(args.source_index) != len(args.source_position)
    ):
        parser.error("source indices must match source positions")
    if len(set(args.source_index)) != len(args.source_index):
        parser.error("source indices must be unique")
    if any(index < 0 for index in args.source_index):
        parser.error("source indices must be non-negative")
    if args.phase == "generate" and len(args.source_position) != 1:
        parser.error("generate phase requires exactly one source")
    if args.phase == "fit" and not args.observation_input:
        parser.error("fit phase requires --observation-input")
    if args.phase != "fit" and args.observation_input:
        parser.error("observation inputs are only valid for fit phase")
    missing_inputs = [
        path for path in args.observation_input if not path.is_file()
    ]
    if missing_inputs:
        parser.error(f"missing observation input: {missing_inputs[0]}")
    configured_photons = tuple(
        value for value in (
            args.n_photons,
            args.data_n_photons,
            args.fit_n_photons,
        )
        if value is not None
    )
    positive = (
        args.n_events,
        args.k,
        args.data_ensemble_size,
        args.fit_ensemble_size,
        *configured_photons,
    )
    if any(value <= 0 for value in positive):
        parser.error("event, photon, K and ensemble counts must be > 0")
    if args.mc_validation_ensemble_size is not None and (
        args.mc_validation_ensemble_size <= 0
    ):
        parser.error("MC-validation ensemble must be positive")
    selected_steps = (
        (args.joint_steps,)
        if args.optimization_plan == "joint-only"
        and args.joint_steps is not None
        else (
            args.stage1_steps,
            args.stage2_steps,
            args.stage3_steps,
        )
        if args.optimization_plan == "staged"
        else ()
    )
    if any(value <= 0 for value in selected_steps):
        parser.error("selected optimizer stage steps must be positive")
    if args.heldout_source_index and args.validation_every <= 0:
        parser.error("held-out validation requires --validation-every > 0")
    if args.early_stopping_patience < 0 or args.min_improvement < 0.0:
        parser.error("early-stopping controls must be non-negative")
    if args.validation_every <= 0:
        parser.error("MC validation requires --validation-every > 0")
    if args.source_chunk_size < 0 or args.checkpoint_every < 0:
        parser.error("chunk and checkpoint intervals must be non-negative")
    if (
        args.restart_checkpoint is not None
        and not args.restart_checkpoint.is_file()
    ):
        parser.error("--restart-checkpoint must name an existing file")
    if args.time_max_ns <= args.time_min_ns or args.time_bin_ns <= 0.0:
        parser.error("invalid time window or bin width")
    if args.truncate_sigma <= 0.0:
        parser.error("Gaussian truncation must be positive")
    if args.ad_fd_relative_step < 0.0 or args.ad_fd_minimum_step <= 0.0:
        parser.error("AD-FD relative step must be >=0 and minimum step >0")
    if args.background_rate_hz < 0.0:
        parser.error("background rate must be non-negative")
    if (
        args.ad_fd_background_rate_hz is not None
        and args.ad_fd_background_rate_hz < 0.0
    ):
        parser.error("AD-FD background rate must be non-negative")
    if args.low_rate_threshold <= 0.0:
        parser.error("--low-rate-threshold must be positive")
    if (
        not np.isfinite(args.event_exposure)
        or args.event_exposure <= 0.0
        or not np.isfinite(args.event_time_offset_ns)
        or args.window_stability_expansion_ns <= 0.0
    ):
        parser.error("exposure and window expansion must be positive")
    span = (args.time_max_ns - args.time_min_ns) / args.time_bin_ns
    if not np.isclose(span, round(span)):
        parser.error("the time window must contain an integer number of bins")
    result = run_closure(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
