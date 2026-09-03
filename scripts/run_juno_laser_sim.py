#!/usr/bin/env python
"""Simulate JUNO 532 nm isotropic laser flashes.

This is the repository-facing JUNO adapter for the laser data format.
Each event uses the Fibonacci isotropic source with an independent uniform
rotation and samples Poisson photo-electrons from the differentiable expected
hit weights.  All accepted PE records are written to one FSMP-style Parquet
file per simulation run.  The table uses one row per ``(eid, ch)`` pair and
stores the PE times in a ``list<float64>`` ``PEt`` column, including empty
lists for zero-PE channels.  Diagnostic photon trajectories are optional and
are written to a separate HDF5 file when explicitly enabled.

Example::

    python scripts/run_juno_laser_sim.py \
        --outdir /junofs/C/liukainan/LUCiD/outputs/juno_laser_center \
        --n-events 10

The default configuration is ``config/JUNO_laser_config.json``.  The script
does not alter the public simulator API; it only supplies the nested-sphere
transport state required by the two-medium laser adapter.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import sys
import time
from pathlib import Path

# Make direct ``python scripts/run_juno_laser_sim.py`` execution resolve the
# repository package without requiring a caller-specific PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from lucid.detector_params import load_physics_config
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.simulation.photon_step_factory import make_photon_step
from lucid.simulation.reflection import get_reflection_model
from lucid.simulation.sensor_response import make_hits_likelihood
from lucid.sources import isotropic_source
from lucid.juno_laser.data import SIMULATION_SCHEMA, simulation_table


LOG = logging.getLogger("lucid.juno_laser")
INVALID_LOG_WEIGHT = -1e10
DEFAULT_OUTPUT_DIR = "/junofs/C/liukainan/LUCiD/outputs/juno_laser"


def _simulation_metadata(
    args,
    config,
    n_events,
    n_photons,
    n_steps,
    n_channels,
    record_trajectories,
):
    """Return file metadata documenting the simulation provenance."""
    values = {
        "format_version": "lucid-juno-laser-pe-parquet",
        "config": str(Path(args.config).resolve()),
        "source": args.source,
        "n_events": str(n_events),
        "n_channels": str(n_channels),
        "n_photons": str(n_photons),
        "intensity": str(
            float(args.intensity or config["source"]["default_intensity"])
        ),
        "wavelength_nm": str(config["wavelength_nm"]),
        "max_scattering_steps": str(n_steps),
        "seed": str(args.seed),
        "record_trajectories": str(bool(record_trajectories)),
        "pe_time_units": "ns",
        "offset_units": "ns",
    }
    return {key.encode(): value.encode() for key, value in values.items()}


def _rotation_matrix(key, dtype):
    q = jax.random.normal(key, (4,), dtype=dtype)
    q = q / (jnp.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=dtype)


def _rotate_directions(directions, key):
    return directions @ _rotation_matrix(key, directions.dtype).T


def _lengths_at(material_dir: Path, wavelength_nm: float):
    """Evaluate scalar lengths at ``wavelength_nm``."""
    values = {}
    wl = float(wavelength_nm)
    for name in ("inner", "outer"):
        filename = (
            "juno_laser_labls.json"
            if name == "inner"
            else "juno_laser_water.json"
        )
        with (material_dir / filename).open() as stream:
            data = json.load(stream)
        symmetric = data["scattering"]["symmetric"]
        alpha_rayleigh = symmetric["P4"] / wl**4 * (
            1 + symmetric["P5"] / wl**2
        )
        asymmetric = data["scattering"]["asymmetric"]
        alpha_mie = asymmetric["P6"] * (
            1 + asymmetric["P7"] / wl**4 * (wl - asymmetric["P8"]) ** 2
        )
        absorption = data["absorption"]
        table_wl = np.asarray(
            absorption["pope_fry_wavelengths_nm"], dtype=float
        )
        table_alpha = np.asarray(
            absorption["pope_fry_absorption_m_inv"], dtype=float
        )
        if wl <= 464:
            alpha_continuum = absorption["P0"] * absorption["P2"] * (
                wl / 500.0
            ) ** absorption["P3"]
        else:
            alpha_continuum = float(np.interp(wl, table_wl, table_alpha))
        alpha_abs = (
            absorption["P0"] * absorption["P1"] / wl**4 + alpha_continuum
        )
        values[name] = (
            1 / alpha_rayleigh,
            1 / max(alpha_mie, 1e-30),
            1 / max(alpha_abs, 1e-30),
            float(data["refractive_index"]),
        )
    return values


def _make_forward(
    det_geom, dp, *, n_photons, n_steps, source, intensity, wavelength_nm,
    record_trajectories=False,
):
    """Create a JIT nested-sphere forward returning raw hit arrays."""
    detector = det_geom.detector
    propagate = det_geom.propagator
    n_sensors = det_geom.num_sensors
    n_inner = jnp.asarray(
        det_geom.medium.refractive_index, dtype=jnp.float32
    )
    n_outer = jnp.asarray(
        det_geom.medium_outer.refractive_index, dtype=jnp.float32
    )
    c_vacuum = float(det_geom.speed_of_light) * float(
        det_geom.medium.refractive_index
    )
    refl_params = get_reflection_model("scalar")[1](dp)
    photon_step = make_photon_step("sample", has_interface=True)
    source = isotropic_source(
        source, intensity=intensity, wavelength=wavelength_nm
    )
    qc = dp.per_pmt.qe_corrections
    qe = float(dp.response.qe)
    sl_in, ml_in, al_in, _ = _lengths_at(
        Path(__file__).resolve().parents[1] / "config" / "materials",
        wavelength_nm,
    )["inner"]
    sl_out, ml_out, al_out, _ = _lengths_at(
        Path(__file__).resolve().parents[1] / "config" / "materials",
        wavelength_nm,
    )["outer"]

    @jax.jit
    def forward(key, source_key):
        directions, positions, intensities = source(n_photons, source_key)
        directions = _rotate_directions(directions, source_key)
        medium_id = detector.region_of(positions)

        if record_trajectories:
            trajectory_positions = jnp.zeros(
                (n_steps + 1, n_photons, 3), dtype=positions.dtype
            )
            trajectory_times = jnp.zeros(
                (n_steps + 1, n_photons), dtype=positions.dtype
            )
            trajectory_alive = jnp.zeros(
                (n_steps + 1, n_photons), dtype=positions.dtype
            )
            trajectory_positions = trajectory_positions.at[0].set(positions)
            trajectory_alive = trajectory_alive.at[0].set(1.0)

        def step(carry, step_index):
            if record_trajectories:
                (
                    positions,
                    directions,
                    times,
                    survival,
                    medium_id,
                    key,
                    trajectory_positions,
                    trajectory_times,
                    trajectory_alive,
                ) = carry
            else:
                positions, directions, times, survival, medium_id, key = carry
            hit = propagate(positions, directions)
            distances = jnp.sqrt(
                jnp.sum((hit["positions"] - positions) ** 2, axis=1)
                + 1e-12
            ) - 1e-6
            hit_sensor = jnp.max(hit["inside_sensor"], axis=0)
            key, subkey = jax.random.split(key)
            ray_keys = jax.random.split(subkey, n_photons)
            in_inner = medium_id == 0
            scatter = jnp.where(in_inner, sl_in, sl_out)
            mie = jnp.where(in_inner, ml_in, ml_out)
            absorb = jnp.where(in_inner, al_in, al_out)
            speed = jnp.where(
                in_inner, c_vacuum / n_inner, c_vacuum / n_outer
            )
            result = jax.vmap(
                photon_step,
                in_axes=(0, 0, 0, 0, 0,
                         0, 0, None, None, 0,
                         0, None, 0, 0,
                         0, 0, None, None),
            )(
                positions,
                directions,
                times,
                distances,
                hit["normals"],
                scatter,
                mie,
                dp.scattering.g,
                refl_params,
                absorb,
                hit_sensor,
                400.0,
                ray_keys,
                speed,
                hit["hit_interface"],
                medium_id,
                n_inner,
                n_outer,
            )
            (
                new_pos,
                new_dir,
                new_time,
                detect,
                reflect,
                cont,
                _,
                new_medium,
            ) = result
            inside = detector.bounds_check(new_pos)
            new_survival = survival * jnp.where(inside, cont, 0.0)
            weights = hit["sensor_weights"] * intensities[None, :] * survival
            weights = weights * detect[None, :] * reflect[None, :]
            arrival = hit["times"] / speed[None, :, None] + times[:, None]
            if record_trajectories:
                trajectory_positions = trajectory_positions.at[
                    step_index + 1
                ].set(new_pos)
                trajectory_times = trajectory_times.at[step_index + 1].set(
                    new_time
                )
                trajectory_alive = trajectory_alive.at[step_index + 1].set(
                    (new_survival > 0).astype(positions.dtype)
                )
                new_carry = (
                    new_pos,
                    new_dir,
                    new_time,
                    new_survival,
                    new_medium,
                    key,
                    trajectory_positions,
                    trajectory_times,
                    trajectory_alive,
                )
            else:
                new_carry = (
                    new_pos,
                    new_dir,
                    new_time,
                    new_survival,
                    new_medium,
                    key,
                )
            return new_carry, (
                weights,
                hit["sensor_indices"],
                arrival.squeeze(-1),
            )

        if record_trajectories:
            initial = (
                positions,
                directions,
                jnp.zeros(n_photons),
                jnp.ones(n_photons),
                medium_id,
                key,
                trajectory_positions,
                trajectory_times,
                trajectory_alive,
            )
        else:
            initial = (
                positions,
                directions,
                jnp.zeros(n_photons),
                jnp.ones(n_photons),
                medium_id,
                key,
            )
        final_carry, (weights, indices, times) = jax.lax.scan(
            step, initial, jnp.arange(n_steps))
        log_w, safe_times, indices, total_charge = make_hits_likelihood(
            weights.reshape(-1),
            indices.reshape(-1),
            times.reshape(-1),
            n_sensors,
            qe=qe,
            qe_corrections=qc,
        )
        if record_trajectories:
            return (
                log_w,
                safe_times,
                indices,
                total_charge,
                final_carry[6],
                final_carry[7],
                final_carry[8],
            )
        return log_w, safe_times, indices, total_charge

    return forward


def simulate(args):
    """Run the configured laser simulation and write its PE table."""
    root = Path(args.config).resolve().parent
    with Path(args.config).open() as stream:
        config = json.load(stream)
    geom_path = root / config["geometry_config"]
    simulation = config["simulation"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    t0 = time.perf_counter()
    LOG.info("building nested-sphere detector geometry from %s", geom_path)
    geom = DetectorGeometry.from_config(
        str(geom_path),
        temperature=None,
        max_candidates_per_ray=4,
        detector_type="nested_sphere",
    )
    n_sensors = geom.num_sensors
    LOG.info("detector geometry ready: %d PMTs (%.2fs)", n_sensors,
             time.perf_counter() - t0)

    t0 = time.perf_counter()
    LOG.info(
        "loading physics parameters from %s",
        root / config["physics_config"],
    )
    physics_path = root / config["physics_config"]
    dp, _, _ = load_physics_config(
        str(physics_path),
        num_sensors=n_sensors,
        scalar_ref_wavelength=config["wavelength_nm"],
    )
    LOG.info("physics parameters ready (%.2fs)", time.perf_counter() - t0)
    n_photons = int(args.n_photons or simulation["n_photons"])
    intensity = float(
        args.intensity or config["source"]["default_intensity"]
    )
    source_position = np.asarray(
        [float(value) for value in args.source.split(",")]
    )
    if source_position.shape != (3,):
        raise ValueError(
            "--source must contain three comma-separated coordinates"
        )
    n_steps = int(args.k or simulation["max_scattering_steps"])
    record_trajectories = bool(
        args.record_trajectories
        or simulation.get("record_trajectories", False)
        or config.get("transport", {}).get("record_trajectories", False)
    )
    LOG.info("constructing forward function: events=%d, photons=%d, steps=%d",
             int(args.n_events or simulation["n_events"]), n_photons, n_steps)
    forward = _make_forward(
        geom,
        dp,
        n_photons=n_photons,
        n_steps=n_steps,
        source=source_position,
        intensity=intensity,
        wavelength_nm=float(config["wavelength_nm"]),
        record_trajectories=record_trajectories,
    )
    LOG.info(
        "forward function constructed; first event will trigger JIT "
        "compilation"
    )
    base_key = jax.random.PRNGKey(args.seed)
    event_stats = []
    n_events = int(args.n_events or simulation["n_events"])
    output = Path(args.output)
    if not output.is_absolute():
        output = outdir / output
    if output.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError(
            "simulation output must use a .parquet or .pq suffix"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectory_output = Path(args.trajectory_output)
    if not trajectory_output.is_absolute():
        trajectory_output = outdir / trajectory_output
    if record_trajectories:
        trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    schema = SIMULATION_SCHEMA.with_metadata(
        _simulation_metadata(
            args,
            config,
            n_events,
            n_photons,
            n_steps,
            n_sensors,
            record_trajectories,
        )
    )
    trajectory_context = (
        h5py.File(trajectory_output, "w")
        if record_trajectories
        else nullcontext()
    )
    LOG.info("opening Parquet output: %s", output)
    if record_trajectories:
        LOG.info("opening trajectory output: %s", trajectory_output)
    with trajectory_context as trajectory_file, pq.ParquetWriter(
        output, schema=schema, compression="zstd"
    ) as writer:
        LOG.info("output files opened; entering event loop")
        if trajectory_file is not None:
            trajectory_file.attrs["format"] = (
                "lucid-juno-laser-trajectories"
            )
            trajectory_file.attrs["n_events"] = n_events
            trajectory_file.attrs["n_photons"] = n_photons
            trajectory_file.attrs["n_steps"] = n_steps
        for event in range(n_events):
            seed = args.seed + event
            t0 = time.perf_counter()
            if event == 0:
                LOG.info("event 0: starting first forward call and JIT compilation")
            else:
                LOG.info("event %d: starting forward call", event)
            result = forward(
                jax.random.fold_in(base_key, event),
                jax.random.fold_in(
                    jax.random.PRNGKey(args.seed + 1_000_000), event
                ),
            )
            jax.block_until_ready(result)
            elapsed = time.perf_counter() - t0
            LOG.info("event %d: forward result ready (%.2fs)", event, elapsed)
            if record_trajectories:
                (
                    log_w,
                    times,
                    indices,
                    total_charge,
                    trajectory_positions,
                    trajectory_times,
                    trajectory_alive,
                ) = result
                event_group = trajectory_file.create_group(
                    f"event_{event:06d}"
                )
                event_group.create_dataset(
                    "positions",
                    data=np.asarray(trajectory_positions, dtype=np.float32),
                    compression="gzip",
                )
                event_group.create_dataset(
                    "times",
                    data=np.asarray(trajectory_times, dtype=np.float32),
                    compression="gzip",
                )
                event_group.create_dataset(
                    "alive",
                    data=np.asarray(trajectory_alive, dtype=np.uint8),
                    compression="gzip",
                )
                event_group.attrs["event"] = event
                event_group.attrs["seed"] = seed
            else:
                log_w, times, indices, total_charge = result
            log_w, times, indices = (
                np.asarray(value) for value in (log_w, times, indices)
            )
            valid = (
                (log_w > INVALID_LOG_WEIGHT / 2)
                & (times > 0)
                & np.isfinite(times)
                & (indices >= 0)
                & (indices < n_sensors)
            )
            expected = np.exp(log_w[valid])
            hit_indices = indices[valid].astype(np.int32)
            hit_times = times[valid].astype(np.float64)
            rng = np.random.default_rng(seed + 100_000)
            multiplicity = rng.poisson(expected)
            keep = multiplicity > 0
            pe_pmt = np.repeat(
                hit_indices[keep], multiplicity[keep]
            ).astype(np.int32)
            pe_time = np.repeat(hit_times[keep], multiplicity[keep])
            nu_lc = -np.asarray(total_charge, dtype=np.float64)
            if np.any(keep):
                log_intensity = np.asarray(log_w[valid], dtype=np.float64)
                np.add.at(
                    nu_lc,
                    hit_indices[keep],
                    multiplicity[keep].astype(np.float64)
                    * log_intensity[keep],
                )
            writer.write_table(
                simulation_table(
                    event,
                    pe_pmt,
                    pe_time,
                    n_channels=n_sensors,
                    nu_lc=nu_lc,
                )
            )
            LOG.info("event %d: Parquet rows written", event)
            stats = {
                "event": event,
                "seed": seed,
                "n_pe": int(pe_pmt.size),
                "n_hit_pmts": int(np.unique(pe_pmt).size),
                "mean_pe_per_pmt": float(pe_pmt.size / n_sensors),
                "elapsed_s": float(elapsed),
            }
            event_stats.append(stats)
            LOG.info(
                "event %d: PE=%d, PMTs=%d, %.2fs",
                event,
                stats["n_pe"],
                stats["n_hit_pmts"],
                elapsed,
            )
    LOG.info("wrote %s", output)
    with (outdir / "results_summary.json").open("w") as stream:
        json.dump(
            {
                "config": str(Path(args.config)),
                "output": str(output),
                "format_version": "lucid-juno-laser-pe-parquet",
                "trajectory_output": (
                    str(trajectory_output)
                    if record_trajectories
                    else None
                ),
                "events": event_stats,
            },
            stream,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/JUNO_laser_config.json")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        default="simulation.parquet",
        help="single FSMP-style Parquet file name or path",
    )
    parser.add_argument("--source", default="0,0,0", metavar="X,Y,Z")
    parser.add_argument(
        "--n-events",
        type=int,
        default=0,
        help="事件数 (0=读取配置中的 simulation.n_events)",
    )
    parser.add_argument("--n-photons", type=int, default=0)
    parser.add_argument("--intensity", type=float, default=0.0)
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="最大散射步数 (0=读取配置中的 simulation.max_scattering_steps)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--record-trajectories",
        "--record",
        dest="record_trajectories",
        action="store_true",
        help="write debug photon trajectories to an HDF5 file",
    )
    parser.add_argument(
        "--trajectory-output",
        default="trajectories.h5",
        help="debug trajectory HDF5 path (used with --record-trajectories)",
    )
    args = parser.parse_args()
    if args.n_events < 0 or args.n_photons < 0 or args.k < 0:
        parser.error("--n-events, --n-photons and --k must be non-negative")
    simulate(args)


if __name__ == "__main__":
    main()
