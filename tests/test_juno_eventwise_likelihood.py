"""Deterministic tests for the minimal event-wise PE likelihood."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.juno_laser.eventwise import (
    EventwisePeObservation,
    gaussian_eventwise_poisson_terms,
    gaussian_mixture_rate,
    gaussian_window_diagnostics,
    gaussian_window_probability,
    make_chunked_gaussian_eventwise_objective,
    make_gaussian_eventwise_objective,
    make_host_accumulated_gaussian_eventwise_objective,
    pe_support_summary,
    require_positive_pe_support,
    sample_gaussian_mixture_events,
)
from lucid.juno_laser.time_grid import TimeGrid


TIME_GRID = TimeGrid(0.0, 4.0, 2.0)


def _observation():
    """Return two retained PEs in two unit-exposure laser events."""
    return EventwisePeObservation(
        event_ids=np.asarray((0, 1), dtype=np.int32),
        pmt_ids=np.asarray((0, 1), dtype=np.int32),
        pe_times_ns=np.asarray((1.0, 3.0), dtype=np.float32),
        n_events=2,
        n_sensors=2,
        time_grid=TIME_GRID,
    )


def test_zero_predicted_rate_is_not_replaced_by_a_floor():
    """An observed PE outside model support gives an infinite strict NLL."""
    observation = EventwisePeObservation(
        event_ids=np.asarray((0,), dtype=np.int32),
        pmt_ids=np.asarray((0,), dtype=np.int32),
        pe_times_ns=np.asarray((1.0,), dtype=np.float32),
        n_events=1,
        n_sensors=1,
        time_grid=TIME_GRID,
    )
    terms = gaussian_eventwise_poisson_terms(
        observation,
        jnp.asarray(((0.0, 1.0),)),
        sigma_ns=0.3,
    )
    assert np.isinf(float(terms.nll))
    summary = pe_support_summary(terms.pe_rate)
    assert summary["n_nonpositive_pe_rate"] == 1
    with pytest.raises(ValueError, match="PPP support failure"):
        require_positive_pe_support(terms.pe_rate, context="unit test")


def test_observation_rejects_times_outside_the_window():
    """Invalid event-wise inputs fail before entering a JIT trace."""
    with pytest.raises(ValueError, match="outside the observation window"):
        EventwisePeObservation(
            event_ids=np.asarray((0,), dtype=np.int32),
            pmt_ids=np.asarray((0,), dtype=np.int32),
            pe_times_ns=np.asarray((4.0,), dtype=np.float32),
            n_events=1,
            n_sensors=1,
            time_grid=TIME_GRID,
        )


def test_gaussian_rate_uses_the_continuous_pe_time():
    """PEs in one transport bin retain different exact-time likelihoods."""
    rates = gaussian_mixture_rate(
        jnp.asarray(((2.0, 0.0),)),
        jnp.asarray((0, 0)),
        jnp.asarray((1.0, 1.8)),
        time_grid=TIME_GRID,
        sigma_ns=0.4,
    )
    assert float(rates[0]) > float(rates[1])


def test_gaussian_point_process_has_expected_scale_gradient():
    """Kernel normalization preserves the analytic Poisson scale score."""
    base = jnp.asarray(((2.0, 4.0), (1.0, 3.0)))

    def ensemble(theta, _keys):
        return (jnp.exp(theta[0]) * base,)

    objective = make_gaussian_eventwise_objective(
        ensemble, (_observation(),), sigma_ns=0.4, normalize=False
    )
    value, gradient = jax.value_and_grad(objective)(
        jnp.zeros(1), jnp.zeros((1, 2), dtype=jnp.uint32)
    )
    terms = gaussian_eventwise_poisson_terms(
        _observation(), base, sigma_ns=0.4
    )
    assert np.isfinite(float(value))
    assert float(gradient[0]) == pytest.approx(
        float(terms.integral) - _observation().n_pe,
        rel=1e-5,
    )


def test_gaussian_density_is_not_renormalized_at_window_edge():
    """The observed-event term uses the full-axis Gaussian density."""
    rate = gaussian_mixture_rate(
        jnp.asarray(((2.0,),)),
        jnp.asarray((0,)),
        jnp.asarray((1.0,)),
        time_grid=TimeGrid(0.0, 2.0, 2.0),
        sigma_ns=1.0,
    )
    assert float(rate[0]) == pytest.approx(
        2.0 / np.sqrt(2.0 * np.pi), rel=1e-6
    )


def test_gaussian_integral_uses_fixed_unit_exposure_per_event():
    """The PPP integral is event count times accepted signal mass."""
    grid = TimeGrid(0.0, 2.0, 2.0)
    observation = EventwisePeObservation(
        event_ids=np.empty(0, dtype=np.int32),
        pmt_ids=np.empty(0, dtype=np.int32),
        pe_times_ns=np.empty(0, dtype=np.float32),
        n_events=2,
        n_sensors=1,
        time_grid=grid,
    )
    probability = gaussian_window_probability(
        time_grid=grid, sigma_ns=1.0, n_sensors=1
    )
    terms = gaussian_eventwise_poisson_terms(
        observation, jnp.asarray(((3.0,),)), sigma_ns=1.0
    )
    expected = 2.0 * 3.0 * float(probability[0, 0])
    assert float(terms.integral) == pytest.approx(expected, rel=1e-6)


def test_window_diagnostics_recovers_mass_when_window_expands():
    """The stability check exposes rather than hides edge leakage."""
    summary = gaussian_window_diagnostics(
        jnp.asarray(((2.0,),)),
        time_grid=TimeGrid(0.0, 2.0, 2.0),
        sigma_ns=1.0,
        expansion_ns=10.0,
    )
    assert summary["current_window_fraction"] < 1.0
    assert summary["expanded_window_fraction"] > 0.999


def test_multi_source_objective_uses_global_pe_normalization():
    """Sources form one joint PPP rather than equal source means."""
    first = EventwisePeObservation(
        np.asarray((0,), dtype=np.int32),
        np.asarray((0,), dtype=np.int32),
        np.asarray((1.0,), dtype=np.float32),
        1,
        1,
        TIME_GRID,
    )
    second = EventwisePeObservation(
        np.asarray((0, 0, 0), dtype=np.int32),
        np.asarray((0, 0, 0), dtype=np.int32),
        np.asarray((1.0, 1.1, 1.2), dtype=np.float32),
        1,
        1,
        TIME_GRID,
    )
    predictions = (
        jnp.asarray(((2.0, 0.5),)),
        jnp.asarray(((4.0, 1.0),)),
    )

    def ensemble(_theta, _keys):
        return predictions

    objective = make_gaussian_eventwise_objective(
        ensemble, (first, second), sigma_ns=0.4
    )
    expected = sum(
        float(
            gaussian_eventwise_poisson_terms(
                observation, prediction, sigma_ns=0.4
            ).nll
        )
        for observation, prediction in zip((first, second), predictions)
    ) / 4.0
    value = objective(jnp.zeros(1), jnp.zeros((1, 2), dtype=jnp.uint32))
    assert float(value) == pytest.approx(expected, rel=1e-6)


def test_chunked_objective_matches_multi_source_objective():
    """Source chunking preserves the joint globally normalized PPP."""
    predictions = (
        jnp.asarray(((2.0, 0.5), (0.4, 1.0))),
        jnp.asarray(((1.5, 0.2), (0.8, 1.3))),
    )

    def together(_theta, _keys):
        return predictions

    def first_only(_theta, _keys):
        return predictions[:1]

    def second_only(_theta, _keys):
        return predictions[1:]

    unchunked = make_gaussian_eventwise_objective(
        together, (_observation(), _observation()), sigma_ns=0.4
    )
    chunked = make_chunked_gaussian_eventwise_objective(
        (first_only, second_only),
        ((_observation(),), (_observation(),)),
        sigma_ns=0.4,
    )
    theta = jnp.zeros(1)
    keys = jnp.zeros((1, 2), dtype=jnp.uint32)
    np.testing.assert_allclose(
        chunked(theta, keys), unchunked(theta, keys), rtol=1e-6
    )


def test_host_accumulated_objective_matches_fused_gradient():
    """Per-source host accumulation preserves the joint loss and gradient."""
    first_base = jnp.asarray(((2.0, 0.5), (0.4, 1.0)))
    second_base = jnp.asarray(((1.5, 0.2), (0.8, 1.3)))

    def together(theta, _keys):
        scale = jnp.exp(theta[0])
        return scale * first_base, scale * second_base

    def first_only(theta, _keys):
        return (jnp.exp(theta[0]) * first_base,)

    def second_only(theta, _keys):
        return (jnp.exp(theta[0]) * second_base,)

    observations = (_observation(), _observation())
    fused = make_gaussian_eventwise_objective(
        together, observations, sigma_ns=0.4
    )
    split = make_host_accumulated_gaussian_eventwise_objective(
        (first_only, second_only),
        ((observations[0],), (observations[1],)),
        sigma_ns=0.4,
    )
    theta = jnp.asarray((0.2,))
    keys = jnp.zeros((1, 2), dtype=jnp.uint32)
    fused_value, fused_gradient = jax.value_and_grad(fused)(theta, keys)
    split_value, split_gradient = split.value_and_grad(theta, keys)
    np.testing.assert_allclose(split_value, fused_value, rtol=1e-6)
    np.testing.assert_allclose(
        split_gradient, fused_gradient, rtol=1e-6
    )


def test_gaussian_sampler_is_reproducible_and_continuous():
    """Fixed-seed timestamps are reproducible continuous samples."""
    masses = np.asarray(((4.0, 2.0), (1.0, 3.0)))
    kwargs = {
        "n_events": 4,
        "time_grid": TIME_GRID,
        "sigma_ns": 0.3,
        "seed": 9,
    }
    first = sample_gaussian_mixture_events(masses, **kwargs)
    second = sample_gaussian_mixture_events(masses, **kwargs)
    np.testing.assert_array_equal(first.event_ids, second.event_ids)
    np.testing.assert_array_equal(first.pmt_ids, second.pmt_ids)
    np.testing.assert_array_equal(first.pe_times_ns, second.pe_times_ns)
    coordinates = first.pe_times_ns / 2.0 - 0.5
    assert np.any(np.abs(coordinates - np.round(coordinates)) > 1.0e-4)


def test_gaussian_rate_accepts_one_sigma_per_pmt():
    """A wider numerical kernel lowers its central peak density."""
    rates = gaussian_mixture_rate(
        jnp.asarray(((2.0,), (2.0,))),
        jnp.asarray((0, 1)),
        jnp.asarray((1.0, 1.0)),
        time_grid=TimeGrid(0.0, 2.0, 2.0),
        sigma_ns=np.asarray((0.2, 0.8)),
    )
    assert float(rates[0]) > float(rates[1])
