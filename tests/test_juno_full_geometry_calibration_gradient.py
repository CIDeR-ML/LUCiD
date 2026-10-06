"""Full JUNO-geometry regression for the restricted calibration gradient."""

from pathlib import Path

import jax
import numpy as np

from lucid.detector_params import load_physics_config
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.juno_laser.config import JunoLaserConfig
from lucid.juno_laser.eventwise import (
    gaussian_eventwise_poisson_terms,
    make_gaussian_eventwise_objective,
    require_positive_pe_support,
    sample_gaussian_mixture_events,
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
from lucid.juno_laser.source import build_juno_laser_source
from lucid.juno_laser.time_grid import TimeGrid


def test_full_geometry_eventwise_gradient_is_finite_and_nonzero():
    """All deployed surfaces must support a non-vacuous event-wise AD pass."""
    assert jax.config.x64_enabled
    root = Path(__file__).resolve().parents[1]
    config = JunoLaserConfig.from_file(
        root / "config" / "JUNO_laser_config.json"
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
    truth = truth_from_geometry(config, geometry)
    fixed = fixed_params_from_geometry(config, geometry)
    source = build_juno_laser_source(
        config.data["source"],
        (0.0, 0.0, 17.1),
        intensity=1.0e5,
        wavelength_nm=config.wavelength_nm,
    )
    time_grid = TimeGrid(0.0, 300.0, 50.0)
    forward = build_per_pe_forward(
        config,
        geometry,
        physics,
        n_photons=128,
        k=4,
        intensity=1.0e5,
        time_grid=time_grid,
    )
    optical_ensemble = make_ensemble_forward(
        forward,
        (source,),
        (geometry.num_sensors, 6),
        fixed_params=fixed,
    )
    native_irf, _ = load_type_average_irf(
        config.irf_path, config.irf_pmt_map_path
    )
    irf_bank = resample_irf_bank(native_irf, time_grid.bin_width_ns)
    ensemble = make_irf_response_forward(
        optical_ensemble,
        geometry.detector.pmt_type_codes,
        irf_bank,
    )
    keys = jax.random.split(jax.random.PRNGKey(42), 1)
    truth_coordinates = encode_calibration_values(truth)
    baseline = ensemble(truth_coordinates, keys)[0]
    sigma_ns = config.interpolation_sigma_ns
    observation = sample_gaussian_mixture_events(
        baseline,
        n_events=1,
        time_grid=time_grid,
        sigma_ns=sigma_ns,
        seed=1042,
    )
    assert observation.n_pe > 0
    objective = make_gaussian_eventwise_objective(
        ensemble,
        (observation,),
        sigma_ns=sigma_ns,
    )
    terms = gaussian_eventwise_poisson_terms(
        observation,
        baseline,
        sigma_ns=sigma_ns,
    )
    require_positive_pe_support(terms.pe_rate, context="full-geometry truth")
    coordinates = truth_coordinates
    value, gradient = jax.value_and_grad(objective)(coordinates, keys)
    value.block_until_ready()
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.asarray(gradient).shape == (5,)
    assert np.all(np.abs(np.asarray(gradient)) > 0.0)
