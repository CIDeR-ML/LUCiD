"""PMT SIREN artifact, coordinate, response, and simulator integration tests."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.detector_params import DetectorParams
from lucid.simulation.pmt_siren import (
    PmtSirenResponse,
    build_pmt_frames,
    load_pmt_siren_response,
    pmt_siren_inputs,
    save_pmt_siren_response,
)
from lucid.simulation.sensor_response import make_hits_data, make_hits_simulation
from lucid.simulation.simulator import setup_event_simulator
from lucid.siren.core import SIREN
from lucid.sources import laser_source


def _constant_response(value=0.5):
    model = SIREN(
        hidden_features=4,
        hidden_layers=1,
        out_features=1,
        outermost_linear=True,
        first_omega_0=30.0,
        hidden_omega_0=15.0,
        square_output=False,
    )
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))["params"]
    params = jax.tree.map(jnp.zeros_like, params)
    raw_bias = jnp.log(jnp.expm1(jnp.asarray(value)))
    params = dict(params)
    params["Dense_0"] = dict(params["Dense_0"])
    params["Dense_0"]["bias"] = jnp.asarray([raw_bias])
    return PmtSirenResponse(
        model=model,
        params=params,
        input_mean=jnp.zeros(3),
        input_scale=jnp.ones(3),
    )


def test_portable_artifact_roundtrip(tmp_path):
    response = _constant_response(0.4)
    path = tmp_path / "pmt_response.npz"
    save_pmt_siren_response(
        path,
        response.params,
        input_mean=response.input_mean,
        input_scale=response.input_scale,
    )
    loaded = load_pmt_siren_response(path)
    got = loaded.efficiency(jnp.asarray([[0.1, -0.2, 0.8]]))
    np.testing.assert_allclose(got, 0.4, rtol=1e-6)
    assert loaded.model.hidden_layers == 1


def test_legacy_artifact_requires_training_normalization(tmp_path):
    path = tmp_path / "siren_weights.npy"
    np.save(path, _constant_response().params, allow_pickle=True)
    with pytest.raises(FileNotFoundError, match="input_mean and input_scale"):
        load_pmt_siren_response(path)


def test_cylinder_local_frame_matches_prototype_convention():
    centers = jnp.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    directions = jnp.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    frames = build_pmt_frames(centers, sensor_directions=directions)

    # Hit each bulb at its front pole and send the photon outward into it.
    hit_positions = jnp.asarray([[[1.75, 0.0, 0.0], [0.0, 0.0, 2.75]]])
    photon_directions = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    sensor_indices = jnp.asarray([[0, 1]])
    inputs = pmt_siren_inputs(
        hit_positions, photon_directions, sensor_indices, centers, frames)

    np.testing.assert_allclose(inputs[..., 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(inputs[..., 2], 1.0, atol=1e-6)
    # Top-cap basis is x=+X, y=-Y, z=-Z, as in Titouan's prototype.
    np.testing.assert_allclose(frames[1], np.diag([1.0, -1.0, -1.0]), atol=1e-6)


def test_expected_and_sampled_response_have_physical_efficiency_semantics():
    weights = jnp.asarray([1.0, 2.0])
    indices = jnp.asarray([0, 0])
    times = jnp.asarray([1.0, 2.0])
    corrections = jnp.ones(1)

    charge, _ = make_hits_simulation(
        weights, indices, times, 1, qe=1.0, qe_corrections=corrections,
        pmt_response_factor=jnp.asarray([0.25, 0.25]))
    np.testing.assert_allclose(charge, [0.75], rtol=1e-6)

    sampled_charge, sampled_time = make_hits_data(
        weights, indices, times, 1, qe=1.0, qe_corrections=corrections,
        rng_key=jax.random.PRNGKey(0), pmt_response_factor=jnp.zeros(2))
    np.testing.assert_array_equal(sampled_charge, [0.0])
    np.testing.assert_array_equal(sampled_time, [0.0])


@pytest.mark.slow
def test_event_simulator_applies_constant_pmt_siren():
    root = Path(__file__).resolve().parents[1]
    geometry = root / "config" / "WCTE_like_geom_config.json"
    physics = root / "config" / "WCTE_like_physics_config.json"
    params = DetectorParams.from_flat(
        num_sensors=1,
        scatter_length=1e12,
        absorption_length=1e12,
        qe=1.0,
        qe_corrections=jnp.asarray(1.0),
    )
    common = dict(
        n_photons=128,
        K=1,
        is_calibration=True,
        use_expected_value=True,
        wavelength_mode=False,
        hit_mode="aggregated",
        temperature=None,
        physics_config=str(physics),
        default_detector_params=params,
    )
    baseline = setup_event_simulator(str(geometry), **common)
    corrected = setup_event_simulator(
        str(geometry), pmt_response_model=_constant_response(0.5), **common)
    source = laser_source(
        [0.0, 0.0, 0.0], intensity=128.0, direction=[1.0, 0.0, 0.0],
        fiber_NA=0.22, wavelength=400.0)
    key = jax.random.PRNGKey(7)
    charge_base, time_base = baseline(source, key)
    charge_corr, time_corr = corrected(source, key)

    assert float(jnp.sum(charge_base)) > 0.0
    np.testing.assert_allclose(charge_corr, charge_base * 0.5, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(time_corr, time_base, rtol=1e-6, atol=1e-6)
