"""Deterministic tests for the minimal event-wise PE likelihood."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr

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
    """Return two retained PEs in two laser events."""
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
    assert summary["min_pe_rate_per_ns"] == 0.0
    assert summary["n_nonpositive_pe_rate"] == 1
    with pytest.raises(ValueError, match="PPP support failure"):
        require_positive_pe_support(terms.pe_rate, context="unit test")


def test_background_enters_both_ppp_terms():
    """A real-data background contributes to event rate and exposure."""
    observation = EventwisePeObservation(
        event_ids=np.asarray((0,), dtype=np.int32),
        pmt_ids=np.asarray((0,), dtype=np.int32),
        pe_times_ns=np.asarray((1.0,), dtype=np.float32),
        n_events=2,
        n_sensors=1,
        time_grid=TIME_GRID,
    )
    terms = gaussian_eventwise_poisson_terms(
        observation,
        jnp.zeros((1, 2)),
        sigma_ns=0.3,
        background_rate_per_ns=0.5,
    )
    assert float(terms.integral) == pytest.approx(4.0)
    assert float(terms.negative_log_rate) == pytest.approx(-np.log(0.5))
    assert np.isfinite(float(terms.nll))


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
    """PEs in the same transport bin carry different time information."""
    masses = jnp.asarray(((2.0, 0.0),))
    rates = gaussian_mixture_rate(
        masses,
        jnp.asarray((0, 0)),
        jnp.asarray((1.0, 1.8)),
        time_grid=TIME_GRID,
        sigma_ns=0.4,
    )
    assert float(rates[0]) > float(rates[1])


def test_gaussian_point_process_has_the_expected_scale_gradient():
    """Kernel normalization preserves the analytic Poisson scale score."""
    base = jnp.asarray(((2.0, 4.0), (1.0, 3.0)))

    def ensemble(theta, _keys):
        return (jnp.exp(theta[0]) * base,)

    objective = make_gaussian_eventwise_objective(
        ensemble,
        (_observation(),),
        sigma_ns=0.4,
        normalize=False,
    )
    value, gradient = jax.value_and_grad(objective)(
        jnp.zeros(1), jnp.zeros((1, 2), dtype=jnp.uint32)
    )
    assert np.isfinite(float(value))
    terms = gaussian_eventwise_poisson_terms(
        _observation(), base, sigma_ns=0.4
    )
    expected_score = float(terms.integral) - _observation().n_pe
    assert float(gradient[0]) == pytest.approx(expected_score, rel=1e-5)


def test_gaussian_density_is_not_renormalized_at_window_edge():
    """The event term uses the original full-axis Gaussian density."""
    grid = TimeGrid(0.0, 2.0, 2.0)
    rate = gaussian_mixture_rate(
        jnp.asarray(((2.0,),)),
        jnp.asarray((0,)),
        jnp.asarray((1.0,)),
        time_grid=grid,
        sigma_ns=1.0,
    )
    expected = 2.0 / np.sqrt(2.0 * np.pi)
    assert float(rate[0]) == pytest.approx(expected, rel=1e-6)


def test_gaussian_integral_uses_cdf_window_probability():
    """The PPP integral counts only full kernels inside the data window."""
    grid = TimeGrid(0.0, 2.0, 2.0)
    observation = EventwisePeObservation(
        event_ids=np.empty(0, dtype=np.int32),
        pmt_ids=np.empty(0, dtype=np.int32),
        pe_times_ns=np.empty(0, dtype=np.float32),
        n_events=1,
        n_sensors=1,
        time_grid=grid,
        event_exposures=np.asarray((2.0,)),
    )
    probability = gaussian_window_probability(
        time_grid=grid,
        sigma_ns=1.0,
        n_sensors=1,
    )
    terms = gaussian_eventwise_poisson_terms(
        observation,
        jnp.asarray(((3.0,),)),
        sigma_ns=1.0,
    )
    expected = 2.0 * 3.0 * float(probability[0, 0])
    assert float(terms.integral) == pytest.approx(expected, rel=1e-6)


def test_event_exposure_enters_rate_and_integral():
    """Known laser exposure scales both terms of the strict PPP."""
    observation = EventwisePeObservation(
        event_ids=np.asarray((0, 1), dtype=np.int32),
        pmt_ids=np.asarray((0, 0), dtype=np.int32),
        pe_times_ns=np.asarray((1.0, 1.0), dtype=np.float32),
        n_events=2,
        n_sensors=1,
        time_grid=TimeGrid(0.0, 2.0, 2.0),
        event_exposures=np.asarray((1.0, 3.0)),
    )
    terms = gaussian_eventwise_poisson_terms(
        observation,
        jnp.asarray(((2.0,),)),
        sigma_ns=0.5,
    )
    assert float(terms.pe_rate[1] / terms.pe_rate[0]) == pytest.approx(3.0)
    probability = gaussian_window_probability(
        time_grid=observation.time_grid,
        sigma_ns=0.5,
        n_sensors=1,
    )
    expected = 4.0 * 2.0 * float(probability[0, 0])
    assert float(terms.integral) == pytest.approx(expected, rel=1e-6)


def test_event_id_selects_time_offset_in_both_ppp_terms():
    """Each PE and window integral use its physical event timing state."""
    grid = TimeGrid(0.0, 4.0, 2.0)
    observation = EventwisePeObservation(
        event_ids=np.asarray((0, 1), dtype=np.int32),
        pmt_ids=np.asarray((0, 0), dtype=np.int32),
        pe_times_ns=np.asarray((1.0, 2.0), dtype=np.float32),
        n_events=2,
        n_sensors=1,
        time_grid=grid,
        event_exposures=np.asarray((1.0, 2.0)),
        event_time_offsets_ns=np.asarray((0.0, 1.0)),
    )
    terms = gaussian_eventwise_poisson_terms(
        observation,
        jnp.asarray(((3.0, 0.0),)),
        sigma_ns=0.2,
    )
    assert float(terms.pe_rate[1] / terms.pe_rate[0]) == pytest.approx(2.0)
    expected_probability = (
        ndtr((4.0 - 1.0) / 0.2) - ndtr((0.0 - 1.0) / 0.2)
        + 2.0
        * (
            ndtr((4.0 - 1.0 - 1.0) / 0.2)
            - ndtr((0.0 - 1.0 - 1.0) / 0.2)
        )
    )
    assert float(terms.integral) == pytest.approx(
        3.0 * expected_probability, rel=1e-6
    )


def test_window_diagnostics_recovers_mass_when_window_expands():
    """The stored stability check exposes rather than hides edge leakage."""
    summary = gaussian_window_diagnostics(
        jnp.asarray(((2.0,),)),
        time_grid=TimeGrid(0.0, 2.0, 2.0),
        sigma_ns=1.0,
        expansion_ns=10.0,
    )
    assert summary["current_window_fraction"] < 1.0
    assert summary["expanded_window_fraction"] > 0.999


def test_multi_source_objective_uses_one_global_pe_normalization():
    """Sources form one joint PPP objective rather than equal source means."""
    first = EventwisePeObservation(
        event_ids=np.asarray((0,), dtype=np.int32),
        pmt_ids=np.asarray((0,), dtype=np.int32),
        pe_times_ns=np.asarray((1.0,), dtype=np.float32),
        n_events=1,
        n_sensors=1,
        time_grid=TIME_GRID,
    )
    second = EventwisePeObservation(
        event_ids=np.asarray((0, 0, 0), dtype=np.int32),
        pmt_ids=np.asarray((0, 0, 0), dtype=np.int32),
        pe_times_ns=np.asarray((1.0, 1.1, 1.2), dtype=np.float32),
        n_events=1,
        n_sensors=1,
        time_grid=TIME_GRID,
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
        for observation, prediction in zip(
            (first, second), predictions
        )
    ) / 4.0
    value = objective(jnp.zeros(1), jnp.zeros((1, 2), dtype=jnp.uint32))
    assert float(value) == pytest.approx(expected, rel=1e-6)


def test_chunked_objective_matches_one_multi_source_objective():
    """Source chunking must preserve the joint globally normalized PPP."""
    first = _observation()
    second = _observation()
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
        together, (first, second), sigma_ns=0.4
    )
    chunked = make_chunked_gaussian_eventwise_objective(
        (first_only, second_only),
        ((first,), (second,)),
        sigma_ns=0.4,
    )
    theta = jnp.zeros(1)
    keys = jnp.zeros((1, 2), dtype=jnp.uint32)
    np.testing.assert_allclose(
        chunked(theta, keys), unchunked(theta, keys), rtol=1e-6
    )


def test_host_accumulated_objective_matches_fused_loss_and_gradient():
    """Host source accumulation preserves one joint PPP and its gradient."""
    first = EventwisePeObservation(
        event_ids=np.asarray((0,), dtype=np.int32),
        pmt_ids=np.asarray((0,), dtype=np.int32),
        pe_times_ns=np.asarray((1.0,), dtype=np.float32),
        n_events=1,
        n_sensors=1,
        time_grid=TIME_GRID,
    )
    second = EventwisePeObservation(
        event_ids=np.asarray((0, 0, 0), dtype=np.int32),
        pmt_ids=np.asarray((0, 0, 0), dtype=np.int32),
        pe_times_ns=np.asarray((1.0, 1.1, 1.2), dtype=np.float32),
        n_events=1,
        n_sensors=1,
        time_grid=TIME_GRID,
    )
    first_base = jnp.asarray(((2.0, 0.5),))
    second_base = jnp.asarray(((4.0, 1.0),))

    def together(theta, _keys):
        scale = jnp.exp(theta[0])
        return scale * first_base, scale * second_base

    def first_only(theta, _keys):
        return (jnp.exp(theta[0]) * first_base,)

    def second_only(theta, _keys):
        return (jnp.exp(theta[0]) * second_base,)

    fused = make_gaussian_eventwise_objective(
        together, (first, second), sigma_ns=0.4
    )
    split = make_host_accumulated_gaussian_eventwise_objective(
        (first_only, second_only),
        ((first,), (second,)),
        sigma_ns=0.4,
    )
    theta = jnp.asarray((0.2,))
    keys = jnp.zeros((1, 2), dtype=jnp.uint32)
    fused_value, fused_gradient = jax.value_and_grad(fused)(theta, keys)
    split_value, split_gradient = split.value_and_grad(theta, keys)
    assert split.denominator == 4
    np.testing.assert_allclose(split_value, fused_value, rtol=1e-6)
    np.testing.assert_allclose(
        split_gradient, fused_gradient, rtol=1e-6
    )


def test_gaussian_sampler_is_reproducible_and_continuous():
    """Fixed-sigma synthetic timestamps are sampled from continuous kernels."""
    masses = np.asarray(((4.0, 2.0), (1.0, 3.0)))
    first = sample_gaussian_mixture_events(
        masses,
        n_events=4,
        time_grid=TIME_GRID,
        sigma_ns=0.3,
        seed=9,
        event_exposure=np.asarray((1.0, 2.0, 3.0, 4.0)),
    )
    second = sample_gaussian_mixture_events(
        masses,
        n_events=4,
        time_grid=TIME_GRID,
        sigma_ns=0.3,
        seed=9,
        event_exposure=np.asarray((1.0, 2.0, 3.0, 4.0)),
    )
    np.testing.assert_array_equal(first.event_ids, second.event_ids)
    np.testing.assert_array_equal(first.pmt_ids, second.pmt_ids)
    np.testing.assert_array_equal(first.pe_times_ns, second.pe_times_ns)
    np.testing.assert_array_equal(
        first.event_exposures, (1.0, 2.0, 3.0, 4.0)
    )
    coordinates = first.pe_times_ns / 2.0 - 0.5
    assert np.any(np.abs(coordinates - np.round(coordinates)) > 1.0e-4)


def test_gaussian_sampler_applies_event_time_offsets():
    """Synthetic raw PE times retain the event-wise timing shift."""
    observation = sample_gaussian_mixture_events(
        np.asarray(((1000.0, 0.0),)),
        n_events=2,
        time_grid=TIME_GRID,
        sigma_ns=0.05,
        seed=17,
        event_time_offset_ns=np.asarray((0.0, 1.0)),
    )
    means = [
        np.mean(observation.pe_times_ns[observation.event_ids == event])
        for event in range(2)
    ]
    assert means[1] - means[0] == pytest.approx(1.0, abs=0.02)


def test_gaussian_sampler_generates_explicit_uniform_background():
    """Synthetic background follows the same PPP used by the objective."""
    grid = TimeGrid(0.0, 10.0, 2.0)
    observation = sample_gaussian_mixture_events(
        np.zeros((2, grid.n_bins)),
        n_events=1000,
        time_grid=grid,
        sigma_ns=0.2,
        seed=23,
        background_rate_per_ns=np.asarray((0.1, 0.3)),
    )
    expected = 1000 * 10.0 * (0.1 + 0.3)
    assert observation.n_pe == pytest.approx(expected, rel=0.05)
    fractions = np.bincount(
        observation.pmt_ids, minlength=2
    ) / observation.n_pe
    assert fractions[1] == pytest.approx(0.75, abs=0.03)
    terms = gaussian_eventwise_poisson_terms(
        observation,
        jnp.zeros((2, grid.n_bins)),
        sigma_ns=0.2,
        background_rate_per_ns=np.asarray((0.1, 0.3)),
    )
    assert np.all(np.asarray(terms.pe_rate) > 0.0)
    assert float(terms.integral) == pytest.approx(expected)


def test_gaussian_sampler_rejects_invalid_background_rate():
    """Background rates must be non-negative and match sensor count."""
    grid = TimeGrid(0.0, 10.0, 2.0)
    with pytest.raises(ValueError, match="background rate"):
        sample_gaussian_mixture_events(
            np.zeros((2, grid.n_bins)),
            n_events=1,
            time_grid=grid,
            sigma_ns=0.2,
            seed=23,
            background_rate_per_ns=np.asarray((0.1,)),
        )


def test_gaussian_rate_accepts_one_sigma_per_pmt():
    """A wider PMT timing kernel lowers its central peak density."""
    masses = jnp.asarray(((2.0,), (2.0,)))
    rates = gaussian_mixture_rate(
        masses,
        jnp.asarray((0, 1)),
        jnp.asarray((1.0, 1.0)),
        time_grid=TimeGrid(0.0, 2.0, 2.0),
        sigma_ns=np.asarray((0.2, 0.8)),
    )
    assert float(rates[0]) > float(rates[1])
