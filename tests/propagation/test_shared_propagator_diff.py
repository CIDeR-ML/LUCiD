"""Differentiability tests for the shared propagator.

Verify that gradients flow correctly through:
1. Shared propagator (weights, positions, normals w.r.t. ray inputs)
2. Full simulation chain: propagator → photon_step → sensor_response
3. Gradient w.r.t. DetectorParams through the pipeline
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from lucid.propagation.shared import create_propagator
from lucid.geometry import generate_detector
from lucid.simulation.photon_step import photon_iteration_update_factors_safe
from lucid.simulation.sensor_response import make_hits_simulation

pytestmark = pytest.mark.slow


class TestSharedPropagatorGradients:
    """Verify gradients flow through the shared propagator."""

    @pytest.fixture(scope="class")
    def cylinder_prop(self):
        det = generate_detector("config/WCTE_like_geom_config.json")
        prop = create_propagator(
            det, jnp.array(det.all_points), det.S_radius,
            n_cap=150, n_angular=250, n_height=150)
        return prop

    def test_grad_weights_wrt_origin(self, cylinder_prop):
        """Sensor weights must be differentiable w.r.t. ray origin."""
        def loss(origin):
            result = cylinder_prop(origin[None, :], jnp.array([[1., 0., 0.]]))
            return jnp.sum(result['sensor_weights'])
        grad = jax.grad(loss)(jnp.array([0., 0., 0.]))
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_weights_wrt_direction(self, cylinder_prop):
        """Sensor weights must be differentiable w.r.t. ray direction."""
        def loss(direction):
            result = cylinder_prop(jnp.zeros((1, 3)), direction[None, :])
            return jnp.sum(result['sensor_weights'])
        grad = jax.grad(loss)(jnp.array([1., 0.1, 0.]))
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_positions_wrt_origin(self, cylinder_prop):
        """Hit positions must be differentiable w.r.t. ray origin."""
        def loss(origin):
            result = cylinder_prop(origin[None, :], jnp.array([[1., 0., 0.]]))
            return jnp.sum(result['positions'])
        grad = jax.grad(loss)(jnp.array([0., 0., 0.]))
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_normals_wrt_origin(self, cylinder_prop):
        """Normals must be differentiable w.r.t. ray origin."""
        def loss(origin):
            result = cylinder_prop(origin[None, :], jnp.array([[1., 0., 0.]]))
            return jnp.sum(result['normals'])
        grad = jax.grad(loss)(jnp.array([0., 0., 0.]))
        assert jnp.all(jnp.isfinite(grad))


class TestFullPipelineGradients:
    """Test gradient flow: propagator → photon_step → sensor_response.

    This simulates what happens inside _common_propagation.
    """

    @pytest.fixture(scope="class")
    def pipeline_setup(self):
        det = generate_detector("config/WCTE_like_geom_config.json")
        prop = create_propagator(
            det, jnp.array(det.all_points), det.S_radius,
            n_cap=150, n_angular=250, n_height=150)
        return det, prop

    def test_grad_charge_wrt_scatter_length(self, pipeline_setup):
        """Gradient of total charge w.r.t. scatter_length through full chain."""
        det, prop = pipeline_setup
        key = jax.random.PRNGKey(42)

        def loss(scatter_length):
            # 1. Propagator
            origins = jnp.zeros((5, 3))
            dirs = jax.random.normal(key, (5, 3))
            dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
            result = prop(origins, dirs)

            # 2. Photon step (one iteration, vmapped)
            normals = result['normals']
            surface_distances = jnp.linalg.norm(
                result['positions'] - origins, axis=1) - 1e-6
            hit_sensor = jnp.max(result['inside_sensor'], axis=0)

            keys = jax.random.split(key, 5)
            _, _, _, detect_probs, refl_attens, _ = jax.vmap(
                photon_iteration_update_factors_safe,
                in_axes=(0, 0, 0, 0, 0, None, None, None, None, 0, 0, None)
            )(origins, dirs, jnp.zeros(5), surface_distances, normals,
              scatter_length, 0.5, 0.3, 100.0,
              hit_sensor, keys, 0.2253)

            # 3. Sensor response
            depositions = result['sensor_weights']
            weights = depositions * detect_probs[None, :] * refl_attens[None, :]
            flat_w = weights.reshape(-1)
            flat_i = result['sensor_indices'].reshape(-1)
            flat_t = (result['sensor_distances'] / 0.2253).reshape(-1)
            qe_corr = jnp.ones(len(det.all_points))
            q, _ = make_hits_simulation(flat_w, flat_i, flat_t,
                                         len(det.all_points), qe=0.2,
                                         qe_corrections=qe_corr)
            return jnp.sum(q)

        grad = jax.grad(loss)(50.0)
        assert jnp.isfinite(grad)
        # Longer scatter length → more photons reach sensors → more charge
        assert float(grad) > 0

    def test_grad_charge_wrt_qe(self, pipeline_setup):
        """Gradient of charge w.r.t. QE through sensor response."""
        det, prop = pipeline_setup

        def loss(qe):
            origins = jnp.zeros((3, 3))
            dirs = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            result = prop(origins, dirs)
            w = result['sensor_weights'].reshape(-1)
            i = result['sensor_indices'].reshape(-1)
            t = (result['sensor_distances'] / 0.2253).reshape(-1)
            qe_corr = jnp.ones(len(det.all_points))
            q, _ = make_hits_simulation(w, i, t, len(det.all_points),
                                         qe=qe, qe_corrections=qe_corr)
            return jnp.sum(q)

        grad = jax.grad(loss)(0.2)
        assert jnp.isfinite(grad)
        assert float(grad) > 0  # more QE → more charge


class TestSphereBoxPropagatorGradients:
    """Verify gradients work for sphere and box too."""

    def test_sphere_grad(self):
        det = generate_detector("config/JUNO_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius,
                                  n_divisions=100)

        def loss(origin):
            return jnp.sum(prop(origin[None, :], jnp.array([[1., 0., 0.]]))['sensor_weights'])

        grad = jax.grad(loss)(jnp.array([0., 0., 0.]))
        assert jnp.all(jnp.isfinite(grad))

    def test_box_grad(self):
        det = generate_detector("config/nuSCOPE_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius,
                                  n_x=50, n_y=50, n_z=50)

        def loss(origin):
            return jnp.sum(prop(origin[None, :], jnp.array([[1., 0., 0.]]))['sensor_weights'])

        grad = jax.grad(loss)(jnp.array([0., 0., 0.]))
        assert jnp.all(jnp.isfinite(grad))
