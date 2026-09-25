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
from .parameters import CalibParams
from .source import rotate_source_directions


def build_per_pe_forward(
    config: JunoLaserConfig,
    det_geom,
    physics_params: DetectorParams,
    *,
    n_photons: int,
    k: int,
    intensity: float | None = None,
    fit_amp: bool = False,
    time_window_ns=None,
    time_bin_ns: float | None = None,
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

    reflection_fn, build_reflection_params = get_reflection_model("scalar")
    reflection_params = build_reflection_params(physics_params)
    has_engineering = any((
        detector.support_rods is not None,
        detector.chimney is not None,
        detector.acrylic_bowls is not None,
    ))
    photon_step = make_photon_step(
        "update_factors",
        has_interface=True,
        reflection_fn=reflection_fn,
        has_engineering_surfaces=has_engineering,
    )
    with config.physics_path.open() as stream:
        fixed_physics = json.load(stream)
    required_surface_codes = []
    if detector.support_rods is not None:
        required_surface_codes.append(0)
    if detector.chimney is not None:
        required_surface_codes.extend((1, 2))
    if detector.acrylic_bowls is not None:
        required_surface_codes.append(3)
    surface_values = engineering_surface_parameters(
        fixed_physics, required_surface_codes
    )
    surface_reflectances = jnp.asarray(surface_values[0])
    surface_transmittances = jnp.asarray(surface_values[1])
    surface_fspecs = jnp.asarray(surface_values[2])
    g = physics_params.scattering.g
    qe = physics_params.response.qe
    qe_corrections = physics_params.per_pmt.qe_corrections
    if qe_corrections.ndim == 0:
        qe_corrections = jnp.ones(n_sensors, dtype=jnp.float32) * qe_corrections

    calibration = config.calibration
    window = (calibration["time_window_ns"] if time_window_ns is None
              else time_window_ns)
    t_min, t_max = (float(window[0]), float(window[1]))
    t_bin = float(calibration["time_bin_ns"] if time_bin_ns is None else time_bin_ns)
    n_bins = int((t_max - t_min + t_bin - 1e-12) // t_bin)
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

        c_in = jnp.asarray(c_vacuum / params.n_ls, dtype=jnp.float32)
        c_out = jnp.asarray(c_vacuum / params.n_water, dtype=jnp.float32)
        speed_in = jnp.full(n_photons, c_in)
        speed_out = jnp.full(n_photons, c_out)
        scatter_in = jnp.full(n_photons, params.scat_ls)
        scatter_out = jnp.full(n_photons, params.scat_water)
        absorb_in = jnp.full(n_photons, params.abs_ls)
        absorb_out = jnp.full(n_photons, params.abs_water)
        # The JUNO laser materials have no Mie component at 532 nm.
        mie_in = jnp.full(n_photons, 1.0e9)
        mie_out = jnp.full(n_photons, 1.0e9)

        def step(carry, index):
            positions, directions, times, survival, medium_id, key, log_p, hist = carry
            prop = propagate(positions, directions)
            depositions = prop["sensor_weights"]
            sensor_indices = prop["sensor_indices"]
            hit_positions = prop["positions"]
            hit_times_meters = prop["times"]
            normals = jax.lax.stop_gradient(prop["normals"])
            hit_interface = prop["hit_interface"]
            surface_distance = jnp.sqrt(
                jnp.sum((hit_positions - positions) ** 2, axis=1) + 1e-12) - 1e-6
            hit_sensor = jnp.max(prop["inside_sensor"], axis=0)

            key, subkey = jax.random.split(key)
            ray_keys = jax.random.split(subkey, n_photons)
            in_inner = medium_id == 0
            scatter = jnp.where(in_inner, scatter_in, scatter_out)
            mie = jnp.where(in_inner, mie_in, mie_out)
            absorb = jnp.where(in_inner, absorb_in, absorb_out)
            speed = jnp.where(in_inner, speed_in, speed_out)
            vectorized_step = jax.vmap(
                photon_step,
                in_axes=(0, 0, 0, 0, 0,
                         0, 0, None, None, 0,
                         0, None, 0, 0,
                         0, 0, None, None),
            ) if not has_engineering else jax.vmap(
                photon_step,
                in_axes=(0, 0, 0, 0, 0,
                         0, 0, None, None, 0,
                         0, None, 0, 0,
                         0, 0, None, None, None, None,
                         0, 0, 0, 0),
            )
            result = vectorized_step(
              positions, directions, times, surface_distance, normals,
              scatter, mie, g, reflection_params, absorb, hit_sensor, 400.0,
              ray_keys, speed, hit_interface, medium_id,
              params.n_ls, params.n_water,
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
                  )
                  if has_engineering else ()
              ))
            (new_positions, new_directions, new_times,
             detect_probs, reflection_attenuations,
             continuing_factors, logp_increment, new_medium_id) = result
            inside = detector.bounds_check(new_positions)
            new_survival = survival * jnp.where(inside, continuing_factors, 0.0)

            physical_intensities = intensities * survival
            dice_deposit = jnp.exp(log_p - jax.lax.stop_gradient(log_p))
            updated_weights = (depositions * physical_intensities[None, :]
                               * detect_probs[None, :]
                               * reflection_attenuations[None, :]
                               * dice_deposit[None, :])
            total_times = hit_times_meters / speed[None, :, None] + times[:, None]
            step_hist = weighted_time_histogram(
                updated_weights.reshape(-1), sensor_indices.reshape(-1),
                total_times.squeeze(-1).reshape(-1), n_sensors=n_sensors,
                time_min_ns=t_min, time_max_ns=t_max, bin_width_ns=t_bin,
                qe=qe, qe_corrections=qe_corrections)
            if fit_amp:
                step_hist = step_hist * params.amp
            next_carry = (new_positions, new_directions, new_times, new_survival,
                          new_medium_id, key, log_p + logp_increment, hist + step_hist)
            return next_carry, index

        histogram0 = jnp.zeros((n_sensors, n_bins), dtype=jnp.float32)
        carry0 = (positions, directions, emission_times,
                  jnp.ones(n_photons, dtype=jnp.float32), medium_id0, key,
                  jnp.zeros(n_photons, dtype=jnp.float32), histogram0)
        final_carry, _ = jax.lax.scan(jax.remat(step), carry0, jnp.arange(k))
        histogram = final_carry[-1]
        return histogram, histogram.sum(axis=1)

    return forward
