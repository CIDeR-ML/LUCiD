"""Tests for pipeline NamedTuples (Phase 7).

Verify that all NamedTuples work correctly as JAX pytrees
and can be used with jax.lax.scan, jax.tree.map, etc.
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation.types import (
    PhotonRays, PropagationResult, PhotonStepResult, PhotonState,
)


class TestPhotonRays:
    def test_construction(self):
        rays = PhotonRays(
            directions=jnp.ones((10, 3)),
            origins=jnp.zeros((10, 3)),
            weights=jnp.ones(10),
        )
        assert rays.wavelengths is None
        assert rays.directions.shape == (10, 3)

    def test_with_wavelengths(self):
        rays = PhotonRays(
            directions=jnp.ones((10, 3)),
            origins=jnp.zeros((10, 3)),
            weights=jnp.ones(10),
            wavelengths=jnp.full(10, 400.0),
        )
        assert rays.wavelengths is not None
        assert rays.wavelengths.shape == (10,)

    def test_is_jax_pytree(self):
        rays = PhotonRays(
            directions=jnp.ones((5, 3)),
            origins=jnp.zeros((5, 3)),
            weights=jnp.ones(5),
        )
        leaves = jax.tree.leaves(rays)
        assert len(leaves) == 3  # wavelengths=None is not a leaf


class TestPhotonState:
    def test_construction(self):
        state = PhotonState(
            positions=jnp.zeros((10, 3)),
            directions=jnp.ones((10, 3)),
            times=jnp.zeros(10),
            survival=jnp.ones(10),
            key=jax.random.PRNGKey(0),
        )
        assert state.positions.shape == (10, 3)

    def test_works_with_lax_scan(self):
        """PhotonState as carry in jax.lax.scan must work correctly."""
        def step(state, i):
            new_state = PhotonState(
                positions=state.positions + 0.1,
                directions=state.directions,
                times=state.times + 1.0,
                survival=state.survival * 0.9,
                key=state.key,
            )
            return new_state, state.times

        init = PhotonState(
            positions=jnp.zeros((5, 3)),
            directions=jnp.ones((5, 3)),
            times=jnp.zeros(5),
            survival=jnp.ones(5),
            key=jax.random.PRNGKey(42),
        )
        final_state, all_times = jax.lax.scan(step, init, jnp.arange(10))
        # After 10 steps, positions should be 10 * 0.1 = 1.0
        npt.assert_allclose(final_state.positions[0, 0], 1.0, atol=1e-5)
        # survival should be 0.9^10
        npt.assert_allclose(final_state.survival[0], 0.9**10, atol=1e-5)
        # times should be 10.0
        npt.assert_allclose(final_state.times[0], 10.0, atol=1e-5)

    def test_jit_compatible(self):
        """PhotonState should work inside jit."""
        @jax.jit
        def update(state):
            return PhotonState(
                positions=state.positions * 2,
                directions=state.directions,
                times=state.times,
                survival=state.survival,
                key=state.key,
            )
        state = PhotonState(
            positions=jnp.ones((3, 3)),
            directions=jnp.zeros((3, 3)),
            times=jnp.zeros(3),
            survival=jnp.ones(3),
            key=jax.random.PRNGKey(0),
        )
        result = update(state)
        npt.assert_allclose(result.positions, 2.0 * jnp.ones((3, 3)))


class TestPhotonStepResult:
    def test_construction(self):
        result = PhotonStepResult(
            position=jnp.array([1.0, 2.0, 3.0]),
            direction=jnp.array([0.0, 0.0, 1.0]),
            time=5.0,
            detect_prob=0.7,
            reflection_attenuation=0.98,
            continuing_factor=0.3,
        )
        assert result.detect_prob == 0.7

    def test_tree_map(self):
        """jax.tree.map should work on PhotonStepResult."""
        r = PhotonStepResult(
            position=jnp.array([1.0, 2.0, 3.0]),
            direction=jnp.array([0.0, 0.0, 1.0]),
            time=jnp.float32(5.0),
            detect_prob=jnp.float32(0.7),
            reflection_attenuation=jnp.float32(0.98),
            continuing_factor=jnp.float32(0.3),
        )
        doubled = jax.tree.map(lambda x: x * 2, r)
        npt.assert_allclose(doubled.time, 10.0)
        npt.assert_allclose(doubled.detect_prob, 1.4)


class TestPropagationResult:
    def test_construction(self):
        n_rays, max_sensors = 10, 4
        result = PropagationResult(
            sensor_weights=jnp.zeros((max_sensors, n_rays)),
            sensor_indices=jnp.zeros((max_sensors, n_rays), dtype=jnp.int32),
            times=jnp.zeros((max_sensors, n_rays)),
            positions=jnp.zeros((n_rays, 3)),
            normals=jnp.zeros((n_rays, 3)),
            inside_sensor=jnp.zeros((max_sensors, n_rays), dtype=jnp.bool_),
        )
        assert result.sensor_weights.shape == (4, 10)


class TestJITCompatibility:
    """Verify all NamedTuples work correctly inside jax.jit — no retracing
    issues, no static field problems."""

    def test_photon_state_jit_scan(self):
        """Full jax.lax.scan with PhotonState carry inside jit."""
        @jax.jit
        def run_scan(init_state):
            def step(state, i):
                new_state = PhotonState(
                    positions=state.positions + state.directions * 0.1,
                    directions=state.directions,
                    times=state.times + 0.1 / 0.3,
                    survival=state.survival * 0.95,
                    key=state.key,
                )
                return new_state, state.survival
            return jax.lax.scan(step, init_state, jnp.arange(5))

        init = PhotonState(
            positions=jnp.zeros((3, 3)),
            directions=jnp.ones((3, 3)) / jnp.sqrt(3.0),
            times=jnp.zeros(3),
            survival=jnp.ones(3),
            key=jax.random.PRNGKey(0),
        )
        final, all_survival = run_scan(init)
        assert final.positions.shape == (3, 3)
        assert all_survival.shape == (5, 3)
        # After 5 steps: survival = 0.95^5
        npt.assert_allclose(final.survival[0], 0.95**5, atol=1e-5)

    def test_photon_step_result_jit_vmap(self):
        """PhotonStepResult as return from vmap inside jit."""
        @jax.jit
        def make_results(positions):
            def single(pos):
                return PhotonStepResult(
                    position=pos,
                    direction=jnp.array([0., 0., 1.]),
                    time=jnp.linalg.norm(pos),
                    detect_prob=0.5,
                    reflection_attenuation=0.99,
                    continuing_factor=0.49,
                )
            return jax.vmap(single)(positions)

        positions = jnp.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        result = make_results(positions)
        assert result.position.shape == (3, 3)
        npt.assert_allclose(result.time, [1.0, 2.0, 3.0], atol=1e-5)

    def test_photon_rays_optional_wavelengths_jit(self):
        """PhotonRays with wavelengths=None must work in jit."""
        @jax.jit
        def process(rays):
            return jnp.sum(rays.weights)

        rays = PhotonRays(
            directions=jnp.ones((5, 3)),
            origins=jnp.zeros((5, 3)),
            weights=jnp.ones(5) * 0.5,
        )
        result = process(rays)
        npt.assert_allclose(result, 2.5)

    def test_grad_through_photon_state(self):
        """Gradients should flow through PhotonState fields."""
        @jax.jit
        def loss_fn(positions):
            state = PhotonState(
                positions=positions,
                directions=jnp.ones_like(positions),
                times=jnp.zeros(positions.shape[0]),
                survival=jnp.ones(positions.shape[0]),
                key=jax.random.PRNGKey(0),
            )
            return jnp.sum(state.positions ** 2)

        pos = jnp.array([[1.0, 2.0, 3.0]])
        grad = jax.grad(loss_fn)(pos)
        npt.assert_allclose(grad, [[2.0, 4.0, 6.0]])
