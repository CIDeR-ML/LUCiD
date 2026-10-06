"""Differentiable nested-sphere forward model for JUNO laser calibration."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from lucid.detector_params import DetectorParams
from lucid.simulation.photon_step_factory import make_photon_step
from lucid.simulation.engineering_surfaces import (
    engineering_surface_parameters,
)
from lucid.simulation.reflection import get_reflection_model
from lucid.sources import sample_source

from .config import JunoLaserConfig
from .histogram import weighted_time_histogram
from .parameters import CalibParams, medium_lengths_at
from .source import rotate_source_directions
from .time_grid import TimeGrid


def build_per_pe_forward(
    config: JunoLaserConfig,
    det_geom,
    physics_params: DetectorParams,
    *,
    n_photons: int,
    k: int,
    intensity: float | None = None,
    fit_amp: bool = False,
    time_grid: TimeGrid | None = None,
):
    """Build ``forward(source, params, key) -> (histogram, charge)``.

    The returned function is JIT compiled.  ``params`` is a :class:`CalibParams`
    pytree; six optical fields are traced through speed, Snell refraction and
    DiCE transport scores.  The fixed detector response comes from the standard
    physics config and is not duplicated in this module.
    """
    if n_photons <= 0 or k <= 0:
        raise ValueError("n_photons and k must be positive")
    detector = det_geom.detector
    propagate = det_geom.propagator
    n_sensors = det_geom.num_sensors
    n_in_reference = float(det_geom.medium.refractive_index)
    c_vacuum = float(det_geom.speed_of_light) * n_in_reference
    n_outer_reference = float(det_geom.medium_outer.refractive_index)
    del n_outer_reference  # kept in geometry for validation/documentation

    reflection_name = config.data["transport"].get(
        "reflection_model", "scalar"
    )
    reflection_fn, build_reflection_params = get_reflection_model(
        reflection_name
    )
    reflection_params = build_reflection_params(physics_params)
    has_engineering = any((
        detector.support_rods is not None,
        detector.chimney is not None,
        detector.bottom_chimney is not None,
        detector.acrylic_bowls is not None,
        detector.diffuser_hardware is not None,
    ))
    has_diffuser_hardware = detector.diffuser_hardware is not None
    diffuser_axis_default = jnp.asarray(
        config.data["source"].get("axis", [0.0, 0.0, 1.0]),
        dtype=jnp.float32,
    )
    photon_step = make_photon_step(
        "update_factors",
        has_interface=True,
        reflection_fn=reflection_fn,
        multi_interface=det_geom.medium_acrylic is not None,
        has_engineering_surfaces=has_engineering,
    )
    acrylic_path = config.material_paths.get("acrylic")
    fixed_lengths = medium_lengths_at(
        {} if acrylic_path is None else {"acrylic": acrylic_path},
        config.wavelength_nm,
    )
    acrylic = fixed_lengths.get("acrylic")
    with config.physics_path.open() as stream:
        fixed_physics = json.load(stream)
    required_surface_codes = []
    if detector.support_rods is not None:
        required_surface_codes.append(0)
    if detector.chimney is not None:
        required_surface_codes.extend((1, 2))
    if detector.bottom_chimney is not None:
        required_surface_codes.extend((7, 8))
    if detector.acrylic_bowls is not None:
        required_surface_codes.append(3)
    if has_diffuser_hardware:
        required_surface_codes.extend((4, 5, 6))
    surface_values = engineering_surface_parameters(
        fixed_physics, required_surface_codes
    )
    transport_dtype = jnp.float32
    surface_reflectances = jnp.asarray(
        surface_values[0], dtype=transport_dtype
    )
    surface_transmittances = jnp.asarray(
        surface_values[1], dtype=transport_dtype
    )
    surface_fspecs = jnp.asarray(
        surface_values[2], dtype=transport_dtype
    )
    g = jnp.asarray(physics_params.scattering.g, dtype=transport_dtype)
    qe = jnp.asarray(physics_params.response.qe, dtype=transport_dtype)
    wavelength = jnp.asarray(
        config.wavelength_nm, dtype=transport_dtype
    )
    qe_corrections = jnp.asarray(
        physics_params.per_pmt.qe_corrections,
        dtype=transport_dtype,
    )
    if qe_corrections.ndim == 0:
        qe_corrections = jnp.ones(n_sensors, dtype=jnp.float32) * qe_corrections

    time_grid = config.time_grid if time_grid is None else time_grid
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid")
    n_bins = time_grid.n_bins
    source_intensity = (float(config.data["source"]["default_intensity"])
                        if intensity is None else float(intensity))

    @jax.jit
    def forward(source, params: CalibParams, key):
        key, source_key = jax.random.split(key)
        source_batch = sample_source(source, n_photons, source_key)
        directions = rotate_source_directions(
            source, source_batch.directions, source_key
        )
        positions = source_batch.positions
        intensities = source_batch.weights
        emission_times = source_batch.times
        medium_id0 = detector.region_of(positions)

        dtype = positions.dtype
        vacuum_speed = jnp.asarray(c_vacuum, dtype=dtype)
        c_in = vacuum_speed / params.n_ls
        c_out = vacuum_speed / params.n_water
        if acrylic is None:
            refractive_indices = jnp.stack(
                (params.n_ls, params.n_water)
            ).astype(dtype)
            scatter_values = jnp.stack(
                (params.scat_ls, params.scat_water)
            ).astype(dtype)
            mie_values = jnp.asarray((1.0e9, 1.0e9), dtype=dtype)
            absorb_values = jnp.stack(
                (params.abs_ls, params.abs_water)
            ).astype(dtype)
            speed_values = jnp.stack((c_in, c_out)).astype(dtype)
        else:
            acrylic_n = jnp.asarray(acrylic["n"], dtype=dtype)
            acrylic_scat = jnp.asarray(acrylic["scat"], dtype=dtype)
            acrylic_mie = jnp.asarray(acrylic["mie"], dtype=dtype)
            acrylic_abs = jnp.asarray(acrylic["abs"], dtype=dtype)
            refractive_indices = jnp.stack(
                (params.n_ls, acrylic_n, params.n_water)
            ).astype(dtype)
            scatter_values = jnp.stack(
                (params.scat_ls, acrylic_scat, params.scat_water)
            ).astype(dtype)
            mie_values = jnp.stack(
                (jnp.asarray(1.0e9, dtype=dtype),
                 acrylic_mie,
                 jnp.asarray(1.0e9, dtype=dtype))
            )
            absorb_values = jnp.stack(
                (params.abs_ls, acrylic_abs, params.abs_water)
            ).astype(dtype)
            speed_values = vacuum_speed / refractive_indices

        def step(carry, index):
            (
                positions,
                directions,
                times,
                survival,
                medium_id,
                key,
                log_p,
                hist,
            ) = carry
            if has_diffuser_hardware:
                diffuser_center = (
                    source.center
                    if hasattr(source, "center")
                    else source.position
                )
                diffuser_axis = (
                    source.axis
                    if hasattr(source, "axis")
                    else diffuser_axis_default
                )
                prop = propagate(
                    positions,
                    directions,
                    diffuser_center,
                    diffuser_axis,
                )
            else:
                prop = propagate(positions, directions)
            depositions = jnp.asarray(
                prop["sensor_weights"], dtype=positions.dtype
            )
            sensor_indices = prop["sensor_indices"]
            hit_positions = jnp.asarray(
                prop["positions"], dtype=positions.dtype
            )
            hit_times_meters = jnp.asarray(
                prop["times"], dtype=times.dtype
            )
            normals = jax.lax.stop_gradient(jnp.asarray(
                prop["normals"], dtype=directions.dtype
            ))
            hit_interface = prop["hit_interface"]
            surface_distance = (
                jnp.sqrt(
                    jnp.sum(
                        (hit_positions - positions) ** 2, axis=1
                    ) + 1e-12
                )
                - 1e-6
            )
            hit_sensor = jnp.max(prop["inside_sensor"], axis=0)

            key, subkey = jax.random.split(key)
            ray_keys = jax.random.split(subkey, n_photons)
            scatter = scatter_values[medium_id]
            mie = mie_values[medium_id]
            absorb = absorb_values[medium_id]
            speed = speed_values[medium_id]
            interface_index = prop.get(
                "interface_index", jnp.zeros_like(medium_id)
            )
            vectorized_step = jax.vmap(
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
            result = vectorized_step(
              positions, directions, times, surface_distance, normals,
              scatter, mie, g, reflection_params, absorb, hit_sensor,
              wavelength,
              ray_keys, speed, hit_interface, medium_id,
              params.n_ls, params.n_water,
              interface_index, refractive_indices,
              *(
                  (
                      prop["engineering_surface_code"] >= 0,
                      surface_reflectances[jnp.maximum(
                          prop["engineering_surface_code"], 0
                      )],
                      surface_transmittances[jnp.maximum(
                          prop["engineering_surface_code"], 0
                      )],
                      surface_fspecs[jnp.maximum(
                          prop["engineering_surface_code"], 0
                      )],
                      prop["interface_target_medium_id"],
                      prop["engineering_entering"],
                      jnp.where(
                          prop["engineering_acrylic_inside"],
                          prop["engineering_surface_code"],
                          -1,
                      ),
                  )
                  if has_engineering else ()
              ))
            (new_positions, new_directions, new_times,
             detect_probs, reflection_attenuations,
             continuing_factors, logp_increment, new_medium_id) = result
            new_positions = new_positions.astype(positions.dtype)
            new_directions = new_directions.astype(directions.dtype)
            new_times = new_times.astype(times.dtype)
            detect_probs = detect_probs.astype(survival.dtype)
            reflection_attenuations = reflection_attenuations.astype(
                survival.dtype
            )
            continuing_factors = continuing_factors.astype(survival.dtype)
            logp_increment = logp_increment.astype(log_p.dtype)
            inside = detector.bounds_check(new_positions)
            new_survival = survival * jnp.where(inside, continuing_factors, 0.0)

            physical_intensities = intensities * survival
            dice_deposit = jnp.exp(log_p - jax.lax.stop_gradient(log_p))
            updated_weights = (depositions * physical_intensities[None, :]
                               * detect_probs[None, :]
                               * reflection_attenuations[None, :]
                               * dice_deposit[None, :])
            total_times = (
                hit_times_meters / speed[None, :, None]
                + times[:, None]
            )
            step_hist = weighted_time_histogram(
                updated_weights.reshape(-1), sensor_indices.reshape(-1),
                total_times.squeeze(-1).reshape(-1), n_sensors=n_sensors,
                time_grid=time_grid,
                qe=qe, qe_corrections=qe_corrections)
            if fit_amp:
                step_hist = step_hist * params.amp
            next_carry = (
                new_positions,
                new_directions,
                new_times,
                new_survival,
                new_medium_id,
                key,
                log_p + logp_increment,
                hist + step_hist,
            )
            return next_carry, index

        histogram0 = jnp.zeros((n_sensors, n_bins), dtype=jnp.float32)
        carry0 = (positions, directions, emission_times,
                  jnp.ones(n_photons, dtype=jnp.float32), medium_id0, key,
                  jnp.zeros(n_photons, dtype=jnp.float32), histogram0)
        final_carry, _ = jax.lax.scan(jax.remat(step), carry0, jnp.arange(k))
        histogram = final_carry[-1]
        return histogram, histogram.sum(axis=1)

    return forward
