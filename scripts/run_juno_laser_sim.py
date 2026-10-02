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
does not alter the public simulator API; it supplies the nested-sphere
transport state required by the two- or three-medium laser adapter.
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
from lucid.geometry.juno_structures import (
    BOWL_ARC_AXIAL_CENTRE_M,
    BOWL_ARC_LENGTH_M,
    BOWL_ARC_RADIAL_CENTRE_M,
    BOWL_ARC_RADIUS_M,
    BOWL_CANDIDATES_PER_CELL,
    BOWL_NEWTON_MAX_STEP_M,
    BOWL_NEWTON_RESIDUAL_M,
    BOWL_NEWTON_STEPS,
)
from lucid.simulation.photon_step_factory import make_photon_step
from lucid.simulation.engineering_surfaces import (
    SURFACE_NAMES,
    engineering_surface_parameters,
)
from lucid.simulation.reflection import REFLECTION_MODELS, get_reflection_model
from lucid.simulation.sensor_response import make_hits_likelihood
from lucid.sources import sample_source
from lucid.juno_laser.source import (
    SOURCE_MODELS,
    build_juno_laser_source,
    rotate_source_directions,
)
from lucid.juno_laser.data import SIMULATION_SCHEMA, simulation_table


LOG = logging.getLogger("lucid.juno_laser")
INVALID_LOG_WEIGHT = -1e10
DEFAULT_OUTPUT_DIR = "/junofs/C/liukainan/LUCiD/outputs/juno_laser"


def _pmt_geometry_override(request):
    """Map the CLI ablation mode to the detector-loader override."""
    if request == "config":
        return None
    return request == "on"


def _support_geometry_override(request):
    """Map the support ablation mode to the detector-loader override."""
    if request == "config":
        return None
    return request == "on"


def _component_geometry_override(request):
    """Map an independent component-ablation request to an override."""
    if request == "config":
        return None
    return request == "on"


def _simulation_metadata(
    args,
    config,
    ls_provenance,
    n_events,
    n_photons,
    n_steps,
    n_channels,
    record_trajectories,
    pmt_geometry_request,
    pmt_geometry,
    support_geometry_request,
    support_geometry,
    support_count,
    support_reflectance,
    support_fspec,
    chimney_geometry_request,
    chimney_geometry,
    bottom_chimney_geometry_request,
    bottom_chimney_geometry,
    acrylic_bowl_geometry_request,
    acrylic_bowl_geometry,
    acrylic_bowl_count,
    diffuser_hardware_request,
    diffuser_hardware_geometry,
    diffuser_hardware,
    surface_reflectances,
    surface_transmittances,
    surface_fspecs,
    acrylic_absorption_length_m,
    reflection_model_request,
    reflection_model,
):
    """Return file metadata documenting the simulation provenance."""
    values = {
        "format_version": "lucid-juno-laser-pe-parquet",
        "config": str(Path(args.config).resolve()),
        "source": args.source,
        "source_model_request": args.source_model,
        "source_model": str(
            args.source_model
            if args.source_model != "config"
            else config["source"]["model"]
        ),
        "diffuser_radius_m": str(
            config["source"].get("radius_m", 0.010)
        ),
        "diffuser_axis": json.dumps(
            config["source"].get("axis", [0.0, 0.0, 1.0])
        ),
        "diffuser_time_sigma_ns": str(
            config["source"].get("time_sigma_ns", 0.23)
        ),
        "diffuser_angular_weights": json.dumps(
            config["source"].get("angular_weights", [])
        ),
        "n_events": str(n_events),
        "n_channels": str(n_channels),
        "n_photons": str(n_photons),
        "intensity": str(
            float(args.intensity or config["source"]["default_intensity"])
        ),
        "wavelength_nm": str(config["wavelength_nm"]),
        "max_scattering_steps": str(n_steps),
        "reflection_model_request": reflection_model_request,
        "reflection_model": reflection_model,
        "seed": str(args.seed),
        "record_trajectories": str(bool(record_trajectories)),
        "pmt_geometry_request": pmt_geometry_request,
        "pmt_geometry": pmt_geometry,
        "support_geometry_request": support_geometry_request,
        "support_geometry": support_geometry,
        "support_rod_count": str(support_count),
        "support_reflectance": str(support_reflectance),
        "support_fspec": str(support_fspec),
        "chimney_geometry_request": chimney_geometry_request,
        "chimney_geometry": chimney_geometry,
        "bottom_chimney_geometry_request": (
            bottom_chimney_geometry_request
        ),
        "bottom_chimney_geometry": bottom_chimney_geometry,
        "acrylic_bowl_geometry_request": acrylic_bowl_geometry_request,
        "acrylic_bowl_geometry": acrylic_bowl_geometry,
        "acrylic_bowl_count": str(acrylic_bowl_count),
        "diffuser_hardware_request": diffuser_hardware_request,
        "diffuser_hardware_geometry": diffuser_hardware_geometry,
        "acrylic_bowl_arc_radius_m": str(BOWL_ARC_RADIUS_M),
        "acrylic_bowl_arc_radial_centre_m": str(
            BOWL_ARC_RADIAL_CENTRE_M
        ),
        "acrylic_bowl_arc_axial_centre_m": str(
            BOWL_ARC_AXIAL_CENTRE_M
        ),
        "acrylic_bowl_arc_length_m": str(BOWL_ARC_LENGTH_M),
        "acrylic_bowl_newton_steps": str(BOWL_NEWTON_STEPS),
        "acrylic_bowl_newton_max_step_m": str(
            BOWL_NEWTON_MAX_STEP_M
        ),
        "acrylic_bowl_newton_residual_m": str(
            BOWL_NEWTON_RESIDUAL_M
        ),
        "acrylic_bowl_candidates_per_cell": str(
            BOWL_CANDIDATES_PER_CELL
        ),
        "acrylic_bowl_max_candidates_per_ray": str(
            2 * BOWL_CANDIDATES_PER_CELL
        ),
        "engineering_surface_reflectances": json.dumps(
            list(surface_reflectances)
        ),
        "engineering_surface_transmittances": json.dumps(
            list(surface_transmittances)
        ),
        "engineering_surface_fspecs": json.dumps(list(surface_fspecs)),
        "engineering_acrylic_transport": "finite_volume_snell_fresnel",
        "engineering_acrylic_absorption_length_m": str(
            acrylic_absorption_length_m
        ),
        "pe_time_units": "ns",
        "offset_units": "ns",
    }
    values.update(ls_provenance)
    if diffuser_hardware is not None:
        values.update({
            "diffuser_ball_radius_m": str(
                diffuser_hardware.ball_radius_m
            ),
            "diffuser_peek_radius_m": str(
                diffuser_hardware.peek_radius_m
            ),
            "diffuser_peek_length_m": str(
                2.0 * diffuser_hardware.peek_half_length_m
            ),
            "diffuser_fiber_radius_m": str(
                diffuser_hardware.fiber_radius_m
            ),
            "diffuser_fiber_length_m": str(
                2.0 * diffuser_hardware.fiber_half_length_m
            ),
        })
    return {
        key.encode(): str(value).encode() for key, value in values.items()
    }


def _absorption_length_at(absorption, wavelength_nm):
    """Evaluate one material's absorption length in metres.

    JUNO's laser-specific LS material is deliberately monochromatic.  Other
    materials retain the historical wavelength-table parameterisation.
    """
    if absorption.get("model") == "monochromatic_reference":
        reference = float(absorption["reference_wavelength_nm"])
        if not np.isclose(wavelength_nm, reference, rtol=0.0, atol=1e-9):
            raise ValueError(
                "monochromatic absorption material is defined only at "
                f"{reference:g} nm, not {wavelength_nm:g} nm"
            )
        return float(absorption["reference_absorption_length_m"])

    table_wl = np.asarray(
        absorption["pope_fry_wavelengths_nm"], dtype=float
    )
    table_alpha = np.asarray(
        absorption["pope_fry_absorption_m_inv"], dtype=float
    )
    if wavelength_nm <= 464:
        alpha_continuum = absorption["P0"] * absorption["P2"] * (
            wavelength_nm / 500.0
        ) ** absorption["P3"]
    else:
        alpha_continuum = float(
            np.interp(wavelength_nm, table_wl, table_alpha)
        )
    alpha_abs = (
        absorption["P0"] * absorption["P1"] / wavelength_nm**4
        + alpha_continuum
    )
    return 1 / max(alpha_abs, 1e-30)


def _ls_optical_provenance(physics_config, absorption_length_m):
    """Validate and return the LS optical-model provenance metadata."""
    declared = float(physics_config["absorption_length"])
    if not np.isclose(
        declared, absorption_length_m, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "LS absorption mismatch between physics config "
            f"({declared} m) and transported material "
            f"({absorption_length_m} m)"
        )
    return {
        "ls_optical_model": physics_config["ls_optical_model"],
        "ls_absorption_table": physics_config["ls_absorption_table"],
        "ls_absorption_scale": physics_config["ls_absorption_scale"],
        "ls_absorption_reference": physics_config[
            "ls_absorption_reference"
        ],
        "ls_absorption_length_532_m": float(absorption_length_m),
    }


def _lengths_at(material_dir: Path, wavelength_nm: float,
                filenames=None):
    """Evaluate scalar optical lengths at ``wavelength_nm``.

    ``filenames`` is ordered from the innermost to outermost transport
    medium.  The default preserves the historical LS/Water configuration.
    """
    values = {}
    wl = float(wavelength_nm)
    if filenames is None:
        filenames = ("juno_laser_labls.json", "juno_laser_water.json")
    for index, filename in enumerate(filenames):
        name = str(index)
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
        absorption_length = _absorption_length_at(
            data["absorption"], wl
        )
        values[name] = (
            1 / alpha_rayleigh,
            1 / max(alpha_mie, 1e-30),
            absorption_length,
            float(data["refractive_index"]),
        )
    return tuple(values[str(index)] for index in range(len(filenames)))


def _make_forward(
    det_geom, dp, *, n_photons, n_steps, source_model, wavelength_nm,
    reflection_model="scalar", record_trajectories=False,
    surface_reflectances=(0.535, 0.0, 0.4, 0.0),
    surface_transmittances=(0.0, 1.0, 0.0, 1.0),
    surface_fspecs=(0.5, 1.0, 0.5, 1.0),
):
    """Create a JIT nested-sphere forward returning raw hit arrays.

    The number of transport media is inferred from ``det_geom``.  A geometry
    without ``medium_acrylic`` follows the legacy LS/Water path; a geometry
    with it uses the LS/Acrylic/Water state machine.
    """
    detector = det_geom.detector
    propagate = det_geom.propagator
    n_sensors = det_geom.num_sensors
    n_inner = jnp.asarray(
        det_geom.medium.refractive_index, dtype=jnp.float32
    )
    n_outer = jnp.asarray(
        det_geom.medium_outer.refractive_index, dtype=jnp.float32
    )
    has_acrylic = det_geom.medium_acrylic is not None
    has_engineering = any((
        detector.support_rods is not None,
        detector.chimney is not None,
        detector.bottom_chimney is not None,
        detector.acrylic_bowls is not None,
        detector.diffuser_hardware is not None,
    ))
    has_diffuser_hardware = detector.diffuser_hardware is not None
    diffuser_center = (
        source_model.center
        if hasattr(source_model, "center")
        else source_model.position
    )
    diffuser_axis = (
        source_model.axis
        if hasattr(source_model, "axis")
        else jnp.asarray([0.0, 0.0, 1.0], dtype=diffuser_center.dtype)
    )
    media = (det_geom.medium, det_geom.medium_acrylic,
             det_geom.medium_outer) if has_acrylic else (
                 det_geom.medium, det_geom.medium_outer)
    refractive_indices = jnp.asarray(
        [medium.refractive_index for medium in media], dtype=jnp.float32)
    speed_values = jnp.asarray(
        [medium.speed_of_light for medium in media], dtype=jnp.float32)
    reflection_fn, build_reflection_params = get_reflection_model(
        reflection_model
    )
    refl_params = build_reflection_params(dp)
    photon_step = make_photon_step(
        "sample", has_interface=True, reflection_fn=reflection_fn,
        multi_interface=has_acrylic,
        has_engineering_surfaces=has_engineering,
    )
    surface_reflectances = jnp.asarray(surface_reflectances)
    surface_transmittances = jnp.asarray(surface_transmittances)
    surface_fspecs = jnp.asarray(surface_fspecs)
    source = source_model
    qc = dp.per_pmt.qe_corrections
    qe = float(dp.response.qe)
    material_dir = Path(__file__).resolve().parents[1] / "config" / "materials"
    filenames = tuple(f"{medium.material}.json" for medium in media)
    lengths = _lengths_at(material_dir, wavelength_nm, filenames)
    sl_values = [item[0] for item in lengths]
    ml_values = [item[1] for item in lengths]
    al_values = [item[2] for item in lengths]
    scatter_lengths = jnp.asarray(sl_values, dtype=jnp.float32)
    mie_scatter_lengths = jnp.asarray(ml_values, dtype=jnp.float32)
    absorption_lengths = jnp.asarray(al_values, dtype=jnp.float32)

    @jax.jit
    def forward(key, source_key):
        source_batch = sample_source(source, n_photons, source_key)
        directions = rotate_source_directions(
            source, source_batch.directions, source_key
        )
        positions = source_batch.positions
        intensities = source_batch.weights
        emission_times = source_batch.times
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
            trajectory_times = trajectory_times.at[0].set(emission_times)
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
            if has_diffuser_hardware:
                hit = propagate(
                    positions,
                    directions,
                    diffuser_center,
                    diffuser_axis,
                )
            else:
                hit = propagate(positions, directions)
            # Keep scan carry precision stable when measured PMT coordinates
            # or geometry constants arrive as float64 under X64 mode.
            hit_positions = jnp.asarray(hit["positions"], dtype=positions.dtype)
            hit_normals = jnp.asarray(hit["normals"], dtype=positions.dtype)
            hit_times = jnp.asarray(hit["times"], dtype=positions.dtype)
            distances = jnp.sqrt(
                jnp.sum((hit_positions - positions) ** 2, axis=1)
                + 1e-12
            ) - 1e-6
            hit_sensor = jnp.max(hit["inside_sensor"], axis=0)
            key, subkey = jax.random.split(key)
            ray_keys = jax.random.split(subkey, n_photons)
            scatter = scatter_lengths[medium_id]
            mie = mie_scatter_lengths[medium_id]
            absorb = absorption_lengths[medium_id]
            speed = speed_values[medium_id]
            interface_index = hit.get(
                "interface_index", jnp.zeros_like(medium_id))
            result = jax.vmap(
                photon_step,
                in_axes=(0, 0, 0, 0, 0,
                         0, 0, None, None, 0,
                         0, None, 0, 0,
                         0, 0, None, None, 0, None),
            ) if not has_engineering else jax.vmap(
                photon_step,
                in_axes=(0, 0, 0, 0, 0,
                         0, 0, None, None, 0,
                         0, None, 0, 0,
                         0, 0, None, None, 0, None,
                         0, 0, 0, 0, 0, 0, 0),
            )
            result = result(
                positions,
                directions,
                times,
                distances,
                hit_normals,
                scatter,
                mie,
                dp.scattering.g,
                refl_params,
                absorb,
                hit_sensor,
                wavelength_nm,
                ray_keys,
                speed,
                hit["hit_interface"],
                medium_id,
                n_inner,
                n_outer,
                interface_index,
                refractive_indices,
                *(
                    (
                        hit["engineering_surface_code"] >= 0,
                        surface_reflectances[jnp.maximum(
                            hit["engineering_surface_code"], 0
                        )],
                        surface_transmittances[jnp.maximum(
                            hit["engineering_surface_code"], 0
                        )],
                        surface_fspecs[jnp.maximum(
                            hit["engineering_surface_code"], 0
                        )],
                        hit["interface_target_medium_id"],
                        hit["engineering_entering"],
                        jnp.where(
                            hit["engineering_acrylic_inside"],
                            hit["engineering_surface_code"],
                            -1,
                        ),
                    )
                    if has_engineering else ()
                ),
            )
            base_result = result[:8]
            (
                new_pos,
                new_dir,
                new_time,
                detect,
                reflect,
                cont,
                _,
                new_medium,
            ) = base_result
            new_pos = jnp.asarray(new_pos, dtype=positions.dtype)
            new_dir = jnp.asarray(new_dir, dtype=directions.dtype)
            new_time = jnp.asarray(new_time, dtype=times.dtype)
            inside = detector.bounds_check(new_pos)
            new_survival = survival * jnp.where(inside, cont, 0.0)
            new_survival = jnp.asarray(new_survival, dtype=survival.dtype)
            weights = hit["sensor_weights"] * intensities[None, :] * survival
            weights = weights * detect[None, :] * reflect[None, :]
            arrival = hit_times / speed[None, :, None] + times[:, None]
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
            outputs = (
                weights,
                hit["sensor_indices"],
                arrival.squeeze(-1),
            )
            if has_engineering:
                surface_events = result[8]
                alive = (survival > 0).astype(surface_events.dtype)
                codes = hit["engineering_surface_code"]
                one_hot = jax.nn.one_hot(
                    jnp.maximum(codes, 0),
                    len(SURFACE_NAMES),
                    dtype=surface_events.dtype,
                )
                one_hot *= (codes >= 0)[:, None]
                surface_counts = jnp.sum(
                    one_hot[:, :, None]
                    * surface_events[:, None, :]
                    * alive[:, None, None],
                    axis=0,
                )
                outputs = outputs + (surface_counts,)
            return new_carry, outputs

        if record_trajectories:
            initial = (
                positions,
                directions,
                emission_times,
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
                emission_times,
                jnp.ones(n_photons),
                medium_id,
                key,
            )
        final_carry, scan_outputs = jax.lax.scan(
            step, initial, jnp.arange(n_steps))
        weights, indices, times = scan_outputs[:3]
        log_w, safe_times, indices, total_charge = make_hits_likelihood(
            weights.reshape(-1),
            indices.reshape(-1),
            times.reshape(-1),
            n_sensors,
            qe=qe,
            qe_corrections=qc,
        )
        if record_trajectories:
            outputs = (
                log_w,
                safe_times,
                indices,
                total_charge,
                final_carry[6],
                final_carry[7],
                final_carry[8],
            )
        else:
            outputs = (log_w, safe_times, indices, total_charge)
        if has_engineering:
            outputs = outputs + (jnp.sum(scan_outputs[3], axis=0),)
        return outputs

    return forward


def simulate(args):
    """Run the configured laser simulation and write its PE table."""
    root = Path(args.config).resolve().parent
    with Path(args.config).open() as stream:
        config = json.load(stream)
    geom_path = root / config["geometry_config"]
    simulation = config["simulation"]
    pmt_geometry_request = getattr(args, "pmt_geometry", "config")
    support_geometry_request = getattr(
        args, "support_geometry", "config"
    )
    chimney_geometry_request = getattr(
        args, "chimney_geometry", "config"
    )
    bottom_chimney_geometry_request = getattr(
        args, "bottom_chimney_geometry", "config"
    )
    acrylic_bowl_geometry_request = getattr(
        args, "acrylic_bowl_geometry", "config"
    )
    diffuser_hardware_request = getattr(
        args, "diffuser_hardware", "config"
    )
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
        pmt_geometry_enabled=_pmt_geometry_override(
            pmt_geometry_request
        ),
        support_geometry_enabled=_support_geometry_override(
            support_geometry_request
        ),
        chimney_geometry_enabled=_component_geometry_override(
            chimney_geometry_request
        ),
        bottom_chimney_geometry_enabled=_component_geometry_override(
            bottom_chimney_geometry_request
        ),
        acrylic_bowl_geometry_enabled=_component_geometry_override(
            acrylic_bowl_geometry_request
        ),
        diffuser_hardware_enabled=_component_geometry_override(
            diffuser_hardware_request
        ),
    )
    n_sensors = geom.num_sensors
    pmt_geometry = geom.detector.pmt_geometry or "legacy_sphere"
    support_geometry = (
        geom.detector.support_geometry or "disabled"
    )
    support_count = (
        0 if geom.detector.support_rods is None
        else len(geom.detector.support_rods.centers_m)
    )
    chimney_geometry = geom.detector.chimney_geometry or "disabled"
    bottom_chimney_geometry = (
        geom.detector.bottom_chimney_geometry or "disabled"
    )
    acrylic_bowl_geometry = (
        geom.detector.acrylic_bowl_geometry or "disabled"
    )
    acrylic_bowl_count = (
        0 if geom.detector.acrylic_bowls is None
        else len(geom.detector.acrylic_bowls.copy_numbers)
    )
    diffuser_hardware_geometry = (
        geom.detector.diffuser_hardware_geometry or "disabled"
    )
    has_engineering = any((
        support_count,
        geom.detector.chimney is not None,
        geom.detector.bottom_chimney is not None,
        acrylic_bowl_count,
        geom.detector.diffuser_hardware is not None,
    ))
    LOG.info("detector geometry ready: %d PMTs (%.2fs)", n_sensors,
             time.perf_counter() - t0)
    LOG.info(
        "PMT geometry: request=%s, effective=%s",
        pmt_geometry_request,
        pmt_geometry,
    )
    LOG.info(
        "support geometry: request=%s, effective=%s, rods=%d",
        support_geometry_request,
        support_geometry,
        support_count,
    )
    LOG.info(
        "chimney geometry: request=%s, effective=%s",
        chimney_geometry_request,
        chimney_geometry,
    )
    LOG.info(
        "bottom-chimney geometry: request=%s, effective=%s",
        bottom_chimney_geometry_request,
        bottom_chimney_geometry,
    )
    LOG.info(
        "Acrylic-bowl geometry: request=%s, effective=%s, bowls=%d",
        acrylic_bowl_geometry_request,
        acrylic_bowl_geometry,
        acrylic_bowl_count,
    )
    LOG.info(
        "diffuser hardware: request=%s, effective=%s",
        diffuser_hardware_request,
        diffuser_hardware_geometry,
    )

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
    with physics_path.open() as stream:
        physics_config = json.load(stream)
    inner_material = Path(config["materials"]["inner"]).name
    ls_absorption_length_m = _lengths_at(
        PROJECT_ROOT / "config" / "materials",
        float(config["wavelength_nm"]),
        (inner_material,),
    )[0][2]
    ls_provenance = _ls_optical_provenance(
        physics_config, ls_absorption_length_m
    )
    LOG.info(
        "LS absorption baseline: model=%s, table=%s, L_abs=%.6f m",
        ls_provenance["ls_optical_model"],
        ls_provenance["ls_absorption_table"],
        ls_absorption_length_m,
    )
    support_reflectance = float(
        physics_config.get("support_reflection_rate", 0.535)
    )
    support_fspec = float(physics_config.get("support_fspec", 0.5))
    required_surface_codes = []
    if support_count:
        required_surface_codes.append(0)
    if geom.detector.chimney is not None:
        required_surface_codes.extend((1, 2))
    if geom.detector.bottom_chimney is not None:
        required_surface_codes.extend((7, 8))
    if acrylic_bowl_count:
        required_surface_codes.append(3)
    if geom.detector.diffuser_hardware is not None:
        required_surface_codes.extend((4, 5, 6))
    surface_parameters = engineering_surface_parameters(
        physics_config, required_surface_codes
    )
    surface_reflectances, surface_transmittances, surface_fspecs = (
        surface_parameters
    )
    acrylic_material = Path(config["materials"]["acrylic"]).name
    acrylic_absorption_length_m = _lengths_at(
        PROJECT_ROOT / "config" / "materials",
        float(config["wavelength_nm"]),
        (acrylic_material,),
    )[0][2]
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
    source_model_name = (
        config["source"]["model"]
        if args.source_model == "config"
        else args.source_model
    )
    source_model = build_juno_laser_source(
        config["source"],
        source_position,
        intensity=intensity,
        wavelength_nm=float(config["wavelength_nm"]),
        model=source_model_name,
    )
    LOG.info(
        "laser source: request=%s, effective=%s",
        args.source_model,
        source_model_name,
    )
    n_steps = int(args.k or simulation["max_scattering_steps"])
    reflection_model_request = args.reflection_model
    reflection_model = reflection_model_request
    if reflection_model_request == "config":
        reflection_model = config.get("transport", {}).get(
            "reflection_model", "scalar"
        )
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
        source_model=source_model,
        wavelength_nm=float(config["wavelength_nm"]),
        reflection_model=reflection_model,
        record_trajectories=record_trajectories,
        surface_reflectances=surface_reflectances,
        surface_transmittances=surface_transmittances,
        surface_fspecs=surface_fspecs,
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
            ls_provenance,
            n_events,
            n_photons,
            n_steps,
            n_sensors,
            record_trajectories,
            pmt_geometry_request,
            pmt_geometry,
            support_geometry_request,
            support_geometry,
            support_count,
            support_reflectance,
            support_fspec,
            chimney_geometry_request,
            chimney_geometry,
            bottom_chimney_geometry_request,
            bottom_chimney_geometry,
            acrylic_bowl_geometry_request,
            acrylic_bowl_geometry,
            acrylic_bowl_count,
            diffuser_hardware_request,
            diffuser_hardware_geometry,
            geom.detector.diffuser_hardware,
            surface_reflectances,
            surface_transmittances,
            surface_fspecs,
            acrylic_absorption_length_m,
            reflection_model_request,
            reflection_model,
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
            source_event_key = jax.random.fold_in(
                jax.random.PRNGKey(args.seed + 1_000_000), event
            )
            result = forward(
                jax.random.fold_in(base_key, event),
                source_event_key,
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
                ) = result[:7]
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
                source_batch = sample_source(
                    source_model, n_photons, source_event_key
                )
                source_directions = rotate_source_directions(
                    source_model,
                    source_batch.directions,
                    source_event_key,
                )
                event_group.create_dataset(
                    "initial_directions",
                    data=np.asarray(source_directions, dtype=np.float32),
                    compression="gzip",
                )
                event_group.create_dataset(
                    "source_weights",
                    data=np.asarray(source_batch.weights, dtype=np.float32),
                    compression="gzip",
                )
                event_group.attrs["event"] = event
                event_group.attrs["seed"] = seed
            else:
                log_w, times, indices, total_charge = result[:4]
            surface_counts = (
                np.asarray(result[-1], dtype=np.float64)
                if has_engineering else None
            )
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
            if surface_counts is not None:
                names = (
                    "hits",
                    "surface_absorbed",
                    "reflected",
                    "specular",
                    "diffuse",
                    "transmitted",
                    "entered",
                    "exited",
                    "bulk_absorbed",
                    "acrylic_path_m",
                )
                component_counts = {
                    "support": surface_counts[0],
                    "chimney": surface_counts[1] + surface_counts[2],
                    "chimney_acrylic": surface_counts[1],
                    "chimney_steel": surface_counts[2],
                    "acrylic_bowl": surface_counts[3],
                    "diffuser_hardware": (
                        surface_counts[4]
                        + surface_counts[5]
                        + surface_counts[6]
                    ),
                    "bottom_chimney": (
                        surface_counts[7] + surface_counts[8]
                    ),
                    "bottom_chimney_acrylic": surface_counts[7],
                    "bottom_chimney_steel": surface_counts[8],
                }
                component_counts.update({
                    "diffuser_ptfe": surface_counts[4],
                    "diffuser_peek": surface_counts[5],
                    "diffuser_fiber": surface_counts[6],
                })
                for component, counts in component_counts.items():
                    for name, value in zip(names, counts):
                        key = f"{component}_{name}"
                        stats[key] = (
                            float(value)
                            if name == "acrylic_path_m"
                            else int(value)
                        )
                    stats[f"{component}_absorbed"] = (
                        stats[f"{component}_surface_absorbed"]
                        + stats[f"{component}_bulk_absorbed"]
                    )
                stats["support_blocked"] = stats["support_hits"]
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
                "source_model_request": args.source_model,
                "source_model": source_model_name,
                "source_position_m": source_position.tolist(),
                "source_intensity": intensity,
                **ls_provenance,
                "diffuser_radius_m": config["source"].get(
                    "radius_m", 0.010
                ),
                "diffuser_axis": config["source"].get(
                    "axis", [0.0, 0.0, 1.0]
                ),
                "diffuser_time_sigma_ns": config["source"].get(
                    "time_sigma_ns", 0.23
                ),
                "diffuser_angular_weights": config["source"].get(
                    "angular_weights", []
                ),
                "pmt_geometry_request": pmt_geometry_request,
                "pmt_geometry": pmt_geometry,
                "support_geometry_request": support_geometry_request,
                "support_geometry": support_geometry,
                "support_rod_count": support_count,
                "support_reflection_rate": support_reflectance,
                "support_fspec": support_fspec,
                "chimney_geometry_request": chimney_geometry_request,
                "chimney_geometry": chimney_geometry,
                "bottom_chimney_geometry_request": (
                    bottom_chimney_geometry_request
                ),
                "bottom_chimney_geometry": bottom_chimney_geometry,
                "acrylic_bowl_geometry_request": (
                    acrylic_bowl_geometry_request
                ),
                "acrylic_bowl_geometry": acrylic_bowl_geometry,
                "acrylic_bowl_count": acrylic_bowl_count,
                "diffuser_hardware_request": diffuser_hardware_request,
                "diffuser_hardware_geometry": (
                    diffuser_hardware_geometry
                ),
                "diffuser_hardware_dimensions_m": (
                    None
                    if geom.detector.diffuser_hardware is None
                    else {
                        "ball_radius": (
                            geom.detector.diffuser_hardware.ball_radius_m
                        ),
                        "peek_radius": (
                            geom.detector.diffuser_hardware.peek_radius_m
                        ),
                        "peek_length": (
                            2.0
                            * geom.detector.diffuser_hardware
                            .peek_half_length_m
                        ),
                        "fiber_radius": (
                            geom.detector.diffuser_hardware.fiber_radius_m
                        ),
                        "fiber_length": (
                            2.0
                            * geom.detector.diffuser_hardware
                            .fiber_half_length_m
                        ),
                    }
                ),
                "acrylic_bowl_arc_radius_m": BOWL_ARC_RADIUS_M,
                "acrylic_bowl_arc_radial_centre_m": (
                    BOWL_ARC_RADIAL_CENTRE_M
                ),
                "acrylic_bowl_arc_axial_centre_m": (
                    BOWL_ARC_AXIAL_CENTRE_M
                ),
                "acrylic_bowl_arc_length_m": BOWL_ARC_LENGTH_M,
                "acrylic_bowl_newton_steps": BOWL_NEWTON_STEPS,
                "acrylic_bowl_newton_max_step_m": (
                    BOWL_NEWTON_MAX_STEP_M
                ),
                "acrylic_bowl_newton_residual_m": (
                    BOWL_NEWTON_RESIDUAL_M
                ),
                "acrylic_bowl_candidates_per_cell": (
                    BOWL_CANDIDATES_PER_CELL
                ),
                "acrylic_bowl_max_candidates_per_ray": (
                    2 * BOWL_CANDIDATES_PER_CELL
                ),
                "engineering_surface_reflectances": (
                    surface_reflectances.tolist()
                ),
                "engineering_surface_transmittances": (
                    surface_transmittances.tolist()
                ),
                "engineering_surface_fspecs": surface_fspecs.tolist(),
                "engineering_acrylic_transport": (
                    "finite_volume_snell_fresnel"
                ),
                "engineering_acrylic_absorption_length_m": (
                    acrylic_absorption_length_m
                ),
                "reflection_model_request": reflection_model_request,
                "reflection_model": reflection_model,
                "max_scattering_steps": n_steps,
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
    parser.add_argument(
        "--support-geometry",
        choices=("config", "on", "off"),
        default="config",
        help=(
            "支撑杆几何消融：config=遵循几何配置，on=有效圆柱，"
            "off=关闭支撑杆"
        ),
    )
    parser.add_argument(
        "--chimney-geometry",
        choices=("config", "on", "off"),
        default="config",
        help=(
            "烟囱几何消融：config=遵循几何配置，on=有效下烟囱，"
            "off=关闭烟囱"
        ),
    )
    parser.add_argument(
        "--bottom-chimney-geometry",
        choices=("config", "on", "off"),
        default="config",
        help=(
            "南极底部烟囱消融：config=遵循几何配置，"
            "on=启用 Acrylic/steel 有效几何，off=关闭"
        ),
    )
    parser.add_argument(
        "--acrylic-bowl-geometry",
        choices=("config", "on", "off"),
        default="config",
        help=(
            "亚克力碗消融：config=遵循几何配置，on=有效圆弧碗，"
            "off=关闭亚克力碗"
        ),
    )
    parser.add_argument(
        "--diffuser-hardware",
        choices=("config", "on", "off"),
        default="config",
        help=(
            "漫射球硬件消融：config=遵循几何配置，on=PTFE球/PEEK管/"
            "PE光纤，off=只保留发射模型"
        ),
    )
    parser.add_argument("--source", default="0,0,0", metavar="X,Y,Z")
    parser.add_argument(
        "--source-model",
        choices=("config", *SOURCE_MODELS),
        default="config",
        help=(
            "光源消融：config=遵循主配置；isotropic=旧数学点源；"
            "juno_diffuser_ball=J26.5.1有效漫射球发射"
        ),
    )
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
        "--pmt-geometry",
        choices=("config", "on", "off"),
        default="config",
        help=(
            "PMT 几何消融：config=遵循几何配置，on=显式前端，"
            "off=旧球形接收面"
        ),
    )
    parser.add_argument(
        "--reflection-model",
        choices=("config", *sorted(REFLECTION_MODELS)),
        default="config",
        help=(
            "反射模型消融：config=遵循主配置；pmt_surface=整体 PMT；"
            "angular=旧多层模型；scalar=旧标量模型"
        ),
    )
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
