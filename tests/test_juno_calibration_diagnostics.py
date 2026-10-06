"""Tests for calibration-gradient and random-key credibility checks."""

import jax.numpy as jnp
import numpy as np
import pytest

from lucid.juno_laser.diagnostics import (
    objective_ad_fd_comparison,
    poisson_basis_fisher,
    response_jacobian,
    split_source_indices,
    summarize_closure_scan,
    summarize_key_closure_pair,
)


def test_ad_fd_comparison_matches_quadratic_gradient():
    """Centered differences must reproduce AD for all five coordinates."""
    matrix = jnp.asarray((1.0, 2.0, 0.5, 1.5, 3.0))

    def objective(theta, keys):
        del keys
        return jnp.sum(matrix * theta**2)

    theta = jnp.asarray((0.4, -0.8, 0.3, -0.2, 0.5))
    result = objective_ad_fd_comparison(
        objective,
        theta,
        jnp.zeros((1, 2), dtype=jnp.uint32),
        relative_step=1.0e-2,
    )
    expected = 2.0 * np.asarray(matrix) * np.asarray(theta)
    np.testing.assert_allclose(result["ad_gradient"], expected, rtol=1e-6)
    np.testing.assert_allclose(
        result["finite_difference_gradient"], expected, rtol=2e-4
    )


def _closure(offset, fit_shift=0.0):
    """Return a minimal closure result mapping for pair tests."""
    names = ["a", "b"]
    return {
        "source_positions_m": [[0.0, 0.0, 0.0]],
        "n_events": 2,
        "n_photons": 100,
        "k": 4,
        "intensity": 1000.0,
        "background_rate_hz_per_pmt": 30000.0,
        "data_importance_normalization": "self_normalized",
        "fit_importance_normalization": "self_normalized",
        "parameter_names": names,
        "truth": {"a": 2.0, "b": 0.0},
        "fit": {"a": 2.2 + fit_shift, "b": 0.1 + fit_shift},
        "final_loss_per_pe": 1.0 + fit_shift,
        "fit_seed_offset": offset,
    }


def test_key_pair_summary_uses_truth_aware_scales():
    """The comparison keeps zero-truth shape coordinates finite."""
    result = summarize_key_closure_pair(_closure(0), _closure(17, 0.2))
    assert result["same_fit_seed_offset"] == 0
    assert result["independent_fit_seed_offset"] == 17
    np.testing.assert_allclose(
        list(result["independent_minus_same_scaled"].values()),
        (0.1, 0.2),
    )


def test_key_pair_rejects_different_physics_configuration():
    """Only the fit transport keys may differ in a paired comparison."""
    independent = _closure(17)
    independent["n_photons"] = 200
    with pytest.raises(ValueError, match="n_photons"):
        summarize_key_closure_pair(_closure(0), independent)


def test_response_jacobian_matches_linear_source_response():
    """Per-source response Jacobians retain output and parameter axes."""
    matrix = jnp.asarray(((1.0, 2.0), (3.0, -1.0)))

    def forward(theta, _keys):
        return (matrix @ theta).reshape(1, 2)

    value, jacobian = response_jacobian(
        forward,
        jnp.asarray((0.5, -0.25)),
        jnp.zeros((1, 2), dtype=jnp.uint32),
    )
    np.testing.assert_allclose(value, ((0.0, 1.75),))
    np.testing.assert_allclose(jacobian.reshape(2, 2), matrix)


def test_poisson_basis_fisher_matches_two_cell_example():
    """Fisher is J transpose diag(1/mu) J for every exposure."""
    masses = np.asarray((2.0, 4.0))
    jacobian = np.asarray(((1.0, 0.0), (0.0, 2.0)))
    result = poisson_basis_fisher(
        masses,
        jacobian,
        event_exposures=np.asarray((1.0, 2.0)),
        background_mass=1.0,
    )
    expected = np.diag((1.0 / 3.0 + 4.0 / 5.0,
                        4.0 / 5.0 + 16.0 / 9.0))
    np.testing.assert_allclose(result["fisher"], expected)
    assert result["regular"]
    assert result["rank"] == 2


def test_poisson_basis_fisher_reports_nonregular_support():
    """A zero-rate cell with a nonzero derivative is not silently floored."""
    with pytest.raises(ValueError, match="non-regular"):
        poisson_basis_fisher(
            np.asarray((0.0, 1.0)),
            np.asarray(((1.0,), (0.0,))),
        )


def test_source_split_requires_train_and_heldout_positions():
    """A held-out validation split cannot consume every source."""
    train, heldout = split_source_indices(5, (1, 4))
    assert train == (0, 2, 3)
    assert heldout == (1, 4)
    with pytest.raises(ValueError, match="at least one training"):
        split_source_indices(2, (0, 1))


def _scan_record(n_photons, ensemble_size, seed, gradient):
    """Return one compact synthetic-closure scan fixture."""
    return {
        "completed": True,
        "n_photons": n_photons,
        "fit_ensemble_size": ensemble_size,
        "seed": 42,
        "fit_seed_offset": seed,
        "parameter_names": ["a", "b"],
        "initial_gradient": {"a": gradient[0], "b": gradient[1]},
        "initial_loss_per_pe": float(seed),
    }


def test_closure_scan_groups_seed_statistics_by_transport_budget():
    """Photon and ensemble settings receive independent seed summaries."""
    records = (
        _scan_record(100, 1, 1, (1.0, 3.0)),
        _scan_record(100, 1, 2, (3.0, 5.0)),
        _scan_record(200, 2, 1, (2.0, 4.0)),
    )
    result = summarize_closure_scan(records)
    first = result["settings"]["photons_100_ensemble_1"]
    np.testing.assert_allclose(first["gradient_mean"], (2.0, 4.0))
    np.testing.assert_allclose(first["gradient_std"], (1.0, 1.0))
    assert first["n_seeds"] == 2
    assert first["fit_seed_offsets"] == [1, 2]
    assert 0.0 < first["pairwise_gradient_cosine_mean"] <= 1.0
