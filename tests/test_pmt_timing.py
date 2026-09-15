"""Tests for the optional SK-IV per-photoelectron timing response."""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.simulation.pmt_timing import (
    SK4_COMPONENT_MEANS_NS,
    SK4_COMPONENT_PROBABILITIES,
    apply_sk4_pmt_timing,
    get_pmt_timing_model,
    sample_sk4_time_offsets,
)


def test_none_is_exact_identity_and_does_not_advance_key():
    times = jnp.asarray([1.0, 2.0, jnp.inf, 0.0, -1.0])
    key = jax.random.PRNGKey(17)
    output, output_key = get_pmt_timing_model(None)(times, key)
    npt.assert_array_equal(np.asarray(output), np.asarray(times))
    npt.assert_array_equal(np.asarray(output_key), np.asarray(key))


def test_unknown_model_fails_at_setup_resolution():
    with pytest.raises(ValueError, match="Unknown PMT timing model"):
        get_pmt_timing_model("sk5")


def test_sk4_sampling_matches_mixture_mean_and_unshifted_fraction():
    n = 500_000
    offsets = sample_sk4_time_offsets(jax.random.PRNGKey(20260915), (n,))
    offsets = np.asarray(offsets)
    expected_mean = float(np.sum(
        np.asarray(SK4_COMPONENT_PROBABILITIES)
        * np.asarray(SK4_COMPONENT_MEANS_NS)
    ))
    # The broad mixture has an offset RMS around 25 ns, giving a mean SEM
    # below 0.04 ns for this sample.
    assert abs(float(offsets.mean()) - expected_mean) < 0.15
    # Sigma is exactly zero only for the 93.9% unshifted component, so those
    # samples are identifiable by an exact zero offset.
    assert abs(float(np.mean(offsets == 0.0)) - 0.939) < 0.002


def test_sk4_response_does_not_advance_downstream_rng():
    times = jnp.asarray([100.0, 101.0, 102.0])
    key = jax.random.PRNGKey(23)
    _, output_key = apply_sk4_pmt_timing(times, key)
    npt.assert_array_equal(np.asarray(output_key), np.asarray(key))


def test_invalid_times_are_unchanged():
    times = jnp.asarray([100.0, 0.0, -3.0, jnp.inf, jnp.nan])
    shifted, _ = apply_sk4_pmt_timing(times, jax.random.PRNGKey(4))
    shifted = np.asarray(shifted)
    assert np.isfinite(shifted[0])
    npt.assert_array_equal(shifted[1:4], np.asarray(times)[1:4])
    assert np.isnan(shifted[4])


def test_time_gradient_is_preserved():
    key = jax.random.PRNGKey(9)

    def total(times):
        shifted, _ = apply_sk4_pmt_timing(times, key)
        return shifted.sum()

    grad = jax.grad(total)(jnp.asarray([10.0, 20.0, 30.0]))
    npt.assert_array_equal(np.asarray(grad), np.ones(3))
