"""Tests for make_hits_* functions (sensor response aggregation)."""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation import (
    make_hits_simulation, make_hits_data, make_hits_likelihood,
)


class TestMakeHitsSimulation:
    def test_reference(self, fixed_flat_hits):
        h = fixed_flat_hits
        q, t = make_hits_simulation(
            h["flat_weights"], h["flat_indices"], h["flat_times"],
            h["num_detectors"], qe=0.2,
            qe_corrections=h["qe_corrections"], temperature=0.01,
        )
        # sensor 0: weight 0.5 * 0.2 = 0.1
        npt.assert_allclose(q[0], 0.1, atol=1e-6)
        npt.assert_allclose(t[0], 10.0, atol=1e-3)
        # sensor 5: two photons (0.8, 0.1) * 0.2 → charge 0.18
        npt.assert_allclose(q[5], 0.18, atol=1e-5)
        # sensor 5 time: soft-min of [12.0, 20.0] with temp=0.01 → ~12.0
        npt.assert_allclose(t[5], 12.0, atol=0.1)
        # empty sensor
        assert q[3] == 0.0
        assert t[3] == 0.0

    def test_output_shape(self, fixed_flat_hits):
        h = fixed_flat_hits
        q, t = make_hits_simulation(
            h["flat_weights"], h["flat_indices"], h["flat_times"],
            h["num_detectors"], qe=0.2, qe_corrections=h["qe_corrections"],
        )
        assert q.shape == (20,)
        assert t.shape == (20,)


class TestMakeHitsLikelihood:
    def test_reference(self, fixed_flat_hits):
        h = fixed_flat_hits
        log_w, safe_t, fi, total_q = make_hits_likelihood(
            h["flat_weights"], h["flat_indices"], h["flat_times"],
            h["num_detectors"], qe=0.2, qe_corrections=h["qe_corrections"],
        )
        # log_w[0] = log(0.5 * 0.2 * 1.0) = log(0.1)
        npt.assert_allclose(log_w[0], jnp.log(0.1), atol=1e-5)
        # total_q same as make_hits_simulation charge
        npt.assert_allclose(total_q[0], 0.1, atol=1e-6)
        npt.assert_allclose(total_q[5], 0.18, atol=1e-5)

    def test_output_shapes(self, fixed_flat_hits):
        h = fixed_flat_hits
        log_w, safe_t, fi, total_q = make_hits_likelihood(
            h["flat_weights"], h["flat_indices"], h["flat_times"],
            h["num_detectors"], qe=0.2, qe_corrections=h["qe_corrections"],
        )
        assert log_w.shape == (5,)
        assert safe_t.shape == (5,)
        assert fi.shape == (5,)
        assert total_q.shape == (20,)


class TestMakeHitsData:
    def test_output_shape(self, fixed_flat_hits):
        h = fixed_flat_hits
        key = jax.random.PRNGKey(42)
        q, t_true, t_reco = make_hits_data(
            h["flat_weights"], h["flat_indices"], h["flat_times"],
            h["num_detectors"], qe=0.2, rng_key=key,
        )
        assert q.shape == (20,)
        assert t_true.shape == (20,)
        assert t_reco.shape == (20,)

    def test_no_smearing_true_equals_reco(self, fixed_flat_hits):
        h = fixed_flat_hits
        key = jax.random.PRNGKey(42)
        q, t_true, t_reco = make_hits_data(
            h["flat_weights"], h["flat_indices"], h["flat_times"],
            h["num_detectors"], qe=0.2, rng_key=key, apply_smearing=False,
        )
        npt.assert_array_equal(t_true, t_reco)

    def test_no_negative_charge(self, fixed_flat_hits):
        h = fixed_flat_hits
        key = jax.random.PRNGKey(42)
        q, _, _ = make_hits_data(
            h["flat_weights"], h["flat_indices"], h["flat_times"],
            h["num_detectors"], qe=0.2, rng_key=key,
        )
        assert jnp.all(q >= 0)
