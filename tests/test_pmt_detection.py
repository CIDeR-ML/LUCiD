import json

import jax
import jax.numpy as jnp
import numpy as np

from lucid.simulation.pmt_detection import (
    load_pmt_detection_response,
    make_tabulated_pmt_detection_response,
)
from lucid.simulation.sensor_response import make_hits_simulation
from lucid.propagation.sk_pmt_sensor import (
    compute_sk20inch_sensor_intersections,
)


def test_tabulated_response_interpolates_and_allows_enhancement():
    response = make_tabulated_pmt_detection_response(
        [0.0, 0.5, 1.0], [0.2, 1.4, 1.0], wavelength_nm=405.5)

    factors = response.factor(jnp.array([-1.0, 0.25, 0.5, 2.0]))
    np.testing.assert_allclose(factors, [0.2, 0.8, 1.4, 1.0])


def test_tabulated_response_supports_forward_and_reverse_autodiff():
    response = make_tabulated_pmt_detection_response(
        [0.0, 0.5, 1.0], [0.2, 1.4, 1.0], wavelength_nm=405.5)

    primal, tangent = jax.jvp(response.factor, (jnp.asarray(0.25),),
                              (jnp.asarray(1.0),))
    reverse = jax.grad(response.factor)(jnp.asarray(0.25))
    np.testing.assert_allclose(primal, 0.8)
    np.testing.assert_allclose(tangent, 2.4, rtol=1e-6)
    np.testing.assert_allclose(reverse, 2.4, rtol=1e-6)


def test_response_artifact_round_trip(tmp_path):
    path = tmp_path / "pmt_response.npz"
    np.savez_compressed(
        path,
        incidence_cosine=np.array([0.0, 0.5, 1.0]),
        relative_efficiency=np.array([0.2, 1.4, 1.0]),
        wavelength_nm=np.asarray(405.5),
        metadata_json=np.asarray(json.dumps({"source": "unit test"})),
    )

    response = load_pmt_detection_response(path)
    assert response.wavelength_nm == 405.5
    assert response.metadata == {"source": "unit test"}
    np.testing.assert_allclose(response.factor(0.25), 0.8)


def test_expected_hit_charge_scales_with_response_and_is_differentiable():
    weights = jnp.array([2.0, 3.0])
    indices = jnp.array([0, 0])
    times = jnp.array([1.0, 2.0])
    qe_corrections = jnp.ones(1)

    def total_charge(response_factor):
        charge, _ = make_hits_simulation(
            weights, indices, times, 1,
            qe=0.25,
            qe_corrections=qe_corrections,
            pmt_response_factor=response_factor,
        )
        return charge.sum()

    response_factor = jnp.array([1.0, 1.5])
    np.testing.assert_allclose(total_charge(response_factor), 1.625)
    np.testing.assert_allclose(
        jax.grad(total_charge)(response_factor), [0.5, 0.75])


def test_curved_surface_response_supports_forward_and_reverse_autodiff():
    """Differentiate through the bulb root, local normal, and response table."""
    response = make_tabulated_pmt_detection_response(
        [0.0, 0.5, 1.0], [0.2, 1.4, 1.0], wavelength_nm=405.5)
    direction = jnp.array([[0.0, 0.0, -1.0]])

    def detected_weight(offset):
        weights, _, _, normals, _, _ = compute_sk20inch_sensor_intersections(
            sensor_indices=jnp.array([0]),
            sensor_positions=jnp.zeros((1, 3)),
            sensor_directions=jnp.array([[0.0, 0.0, 1.0]]),
            sensor_is_barrel=jnp.array([False]),
            ray_origins=jnp.array([[offset, 0.0, 1.0]]),
            ray_directions=direction,
        )
        mu_local = jnp.clip(
            jnp.sum(normals * direction[None, :, :], axis=-1), 0.0, 1.0)
        return jnp.squeeze(weights * response.factor(mu_local))

    offset = 0.15
    reverse = jax.grad(detected_weight)(offset)
    forward = jax.jvp(detected_weight, (offset,), (1.0,))[1]
    step = 1e-4
    finite_difference = (
        detected_weight(offset + step) - detected_weight(offset - step)
    ) / (2.0 * step)

    assert np.isfinite(float(reverse))
    assert abs(float(reverse)) > 0.1
    np.testing.assert_allclose(reverse, forward, rtol=2e-6)
    np.testing.assert_allclose(reverse, finite_difference, rtol=2e-3)
