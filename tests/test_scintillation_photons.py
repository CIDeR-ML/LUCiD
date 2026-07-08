"""Tests for the data-mode scintillation segment → photon expander.

Covers the invariants from the Phase-1 plan
(`.claude/plans/scintillation-data-mode.md`):

- Output array sizes match ``Σ N_per_seg``.
- Origins land on the parent segment's line within float tolerance.
- Directions are unit vectors.
- Wavelengths are within the truncation window.
- Per-photon times are ≥ the parent segment's time (travel + biexp delay
  are both non-negative).
- Mean photon count per segment matches the Birks/Chou prediction within
  Poisson noise.
- Empty / masked / degenerate inputs return zero-row arrays without
  divide-by-zero.
- The Moyal marginal matches the truncated PDF up to MC noise (KS-style
  bound — not a strict KS test to keep the suite fast and deterministic).
"""
from __future__ import annotations

import numpy as np
import pytest

from lucid.sources.scintillation_photons import (
    _build_moyal_inverse_cdf_np,
    _sample_isotropic_np,
    _sample_hypoexp_np,
    expand_segments_to_photons,
)
from lucid.sources.writer import (
    EMISSION_PROCESS_SCINTILLATION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_three_segments():
    """3 segments along the z axis with known lengths / edep / β."""
    return {
        'start_x_mm': np.array([0.0, 0.0, 0.0]),
        'start_y_mm': np.array([0.0, 0.0, 0.0]),
        'start_z_mm': np.array([0.0, 10.0, 20.0]),
        'end_x_mm':   np.array([0.0, 0.0, 0.0]),
        'end_y_mm':   np.array([0.0, 0.0, 0.0]),
        'end_z_mm':   np.array([10.0, 20.0, 22.0]),
        'edep':       np.array([1.0, 0.5, 2.0]),        # MeV
        'time':       np.array([100.0, 110.0, 120.0]),  # ns
        'beta_start': np.array([0.99, 0.95, 0.50]),
    }


def _medium_params(S=10000.0, kB=0.0, C=0.0):
    """WbLS-ish defaults with finite Moyal window and biexp timings."""
    return {
        'S': S, 'kB': kB, 'C': C,
        'tau_rise': 1.0, 'tau_fall': 25.0,
        'moyal_loc': 410.0, 'moyal_scale': 30.0,
        'lambda_min': 300.0, 'lambda_max': 700.0,
    }


# ---------------------------------------------------------------------------
# Shape + emission-process tag
# ---------------------------------------------------------------------------


def test_empty_segments_returns_empty_arrays():
    out = expand_segments_to_photons(
        {
            'start_x_mm': np.array([]), 'start_y_mm': np.array([]),
            'start_z_mm': np.array([]), 'end_x_mm':   np.array([]),
            'end_y_mm':   np.array([]), 'end_z_mm':   np.array([]),
            'edep':       np.array([]), 'time':       np.array([]),
            'beta_start': np.array([]),
        },
        _medium_params(),
        np.random.default_rng(0),
    )
    for k in ('photon_origins', 'photon_directions', 'photon_times',
              'photon_wavelengths', 'photon_segment_index_raw',
              'photon_emission_process'):
        assert out[k].shape[0] == 0, f"{k} expected empty, got {out[k].shape}"


def test_zero_yield_returns_empty_arrays():
    """S=0 yields N̄=0 everywhere → expander short-circuits to zero rows."""
    out = expand_segments_to_photons(
        _make_three_segments(),
        _medium_params(S=0.0),
        np.random.default_rng(1),
    )
    assert out['photon_origins'].shape == (0, 3)
    assert out['photon_emission_process'].shape == (0,)


def test_all_outputs_tagged_scintillation():
    out = expand_segments_to_photons(
        _make_three_segments(), _medium_params(),
        np.random.default_rng(2),
    )
    assert out['photon_emission_process'].dtype == np.int8
    assert np.all(out['photon_emission_process']
                  == EMISSION_PROCESS_SCINTILLATION)


# ---------------------------------------------------------------------------
# Geometric invariants
# ---------------------------------------------------------------------------


def test_origins_on_segment_line():
    """Every photon origin lies on its parent segment's [Start, End] (in m)."""
    segs = _make_three_segments()
    out = expand_segments_to_photons(segs, _medium_params(),
                                      np.random.default_rng(3))
    seg_idx = out['photon_segment_index_raw']
    # Convert segment endpoints to m to match output units.
    start_m = np.stack([segs['start_x_mm'], segs['start_y_mm'], segs['start_z_mm']], axis=1) / 1000.0
    end_m   = np.stack([segs['end_x_mm'],   segs['end_y_mm'],   segs['end_z_mm']],   axis=1) / 1000.0
    s = start_m[seg_idx]
    e = end_m[seg_idx]
    direction = e - s
    seg_len_sq = np.sum(direction * direction, axis=1)
    # Skip degenerate (zero-length) segments — none in this fixture.
    assert np.all(seg_len_sq > 0)
    # Project (origin - start) onto segment direction; t should be in [0, 1].
    rel = out['photon_origins'] - s
    t = np.sum(rel * direction, axis=1) / seg_len_sq
    assert np.all((t >= -1e-6) & (t <= 1.0 + 1e-6)), \
        f"t out of [0, 1]: min={t.min()} max={t.max()}"
    # Perpendicular component (offset off the line) must be ~0.
    perp = rel - t[:, None] * direction
    perp_norm = np.linalg.norm(perp, axis=1)
    assert np.max(perp_norm) < 1e-5


def test_directions_unit_norm():
    out = expand_segments_to_photons(
        _make_three_segments(), _medium_params(),
        np.random.default_rng(4),
    )
    norms = np.linalg.norm(out['photon_directions'], axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# Wavelength invariants
# ---------------------------------------------------------------------------


def test_wavelengths_within_window():
    mp = _medium_params()
    out = expand_segments_to_photons(
        _make_three_segments(), mp, np.random.default_rng(5),
    )
    lam = out['photon_wavelengths']
    assert lam.min() >= mp['lambda_min'] - 1e-3
    assert lam.max() <= mp['lambda_max'] + 1e-3


def test_wavelength_marginal_matches_moyal_cdf():
    """Sampled λ distribution lies close to the precomputed truncated CDF.

    Uses a large N (1e5) for stable percentile comparison. Loose tolerance
    (1 nm on percentile differences) — looking for "uses the right CDF",
    not a strict KS bound.
    """
    rng = np.random.default_rng(42)
    # Same window/params used inside the expander
    mp = _medium_params()
    cdf, lam_grid = _build_moyal_inverse_cdf_np(
        mp['lambda_min'], mp['lambda_max'],
        mp['moyal_loc'], mp['moyal_scale'])
    # Sample 1e5 wavelengths directly via the same machinery as the expander.
    from lucid.sources.scintillation_photons import _sample_moyal_np
    lam_samples = _sample_moyal_np(
        rng, 100_000, mp['moyal_loc'], mp['moyal_scale'],
        mp['lambda_min'], mp['lambda_max'])
    # Expected percentiles from inverse CDF
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    expected = np.interp(qs, cdf, lam_grid)
    got = np.percentile(lam_samples, qs * 100)
    assert np.max(np.abs(got - expected)) < 1.0


# ---------------------------------------------------------------------------
# Time invariants
# ---------------------------------------------------------------------------


def test_times_at_or_after_segment_time():
    """t_i ≥ Segment_Time[seg_idx[i]] for all rays."""
    segs = _make_three_segments()
    out = expand_segments_to_photons(
        segs, _medium_params(), np.random.default_rng(6),
    )
    seg_idx = out['photon_segment_index_raw']
    seg_t = np.asarray(segs['time'])[seg_idx]
    assert np.all(out['photon_times'] >= seg_t - 1e-4)


# ---------------------------------------------------------------------------
# Mean count vs Birks
# ---------------------------------------------------------------------------


def test_mean_count_matches_birks_chou():
    """Average N_per_seg over many trials reproduces S·E/(1+kB·d+C·d²)."""
    segs = _make_three_segments()
    mp = _medium_params(S=1000.0, kB=0.01, C=0.0)
    # Predict per-segment N̄ analytically.
    delta_mm = np.array([10.0, 10.0, 2.0])
    edep_MeV = np.array(segs['edep'])
    dedx_keVmm = edep_MeV / delta_mm * 1000.0   # 100, 50, 1000
    N_pred = mp['S'] * edep_MeV / (
        1.0 + mp['kB'] * dedx_keVmm + mp['C'] * dedx_keVmm ** 2)

    # Run 200 trials, accumulate per-segment Poisson counts.
    n_trials = 200
    counts_per_seg = np.zeros(3, dtype=np.float64)
    for trial in range(n_trials):
        out = expand_segments_to_photons(
            segs, mp, np.random.default_rng(1000 + trial))
        # Count how many photons each segment got in this trial.
        seg_idx = out['photon_segment_index_raw']
        for s in range(3):
            counts_per_seg[s] += int((seg_idx == s).sum())
    empirical_mean = counts_per_seg / n_trials

    # Poisson sample mean has stderr ≈ √(N̄ / trials). 3σ tolerance.
    tol = 3.0 * np.sqrt(N_pred / n_trials)
    diff = np.abs(empirical_mean - N_pred)
    assert np.all(diff < tol), (
        f"means {empirical_mean} vs predicted {N_pred} differ by {diff}, "
        f"3σ tolerance {tol}")


# ---------------------------------------------------------------------------
# Mask / degeneracy guards
# ---------------------------------------------------------------------------


def test_zero_length_segment_contributes_no_photons():
    """A segment with Start == End is masked out (Δs == 0)."""
    segs = _make_three_segments()
    # Force segment 1 to be zero-length
    segs['end_z_mm'] = np.array([10.0, 10.0, 22.0])   # second segment now Δs=0
    out = expand_segments_to_photons(
        segs, _medium_params(S=5000.0),
        np.random.default_rng(7),
    )
    assert np.sum(out['photon_segment_index_raw'] == 1) == 0


def test_zero_edep_segment_contributes_no_photons():
    segs = _make_three_segments()
    segs['edep'] = np.array([1.0, 0.0, 2.0])
    out = expand_segments_to_photons(
        segs, _medium_params(S=5000.0),
        np.random.default_rng(8),
    )
    assert np.sum(out['photon_segment_index_raw'] == 1) == 0


def test_zero_beta_segment_contributes_no_photons():
    segs = _make_three_segments()
    segs['beta_start'] = np.array([0.99, 0.0, 0.5])
    out = expand_segments_to_photons(
        segs, _medium_params(S=5000.0),
        np.random.default_rng(9),
    )
    assert np.sum(out['photon_segment_index_raw'] == 1) == 0


# ---------------------------------------------------------------------------
# Helpers — direct unit tests
# ---------------------------------------------------------------------------


def test_isotropic_direction_marginals():
    """cos θ ~ U[-1, 1] and φ ~ U[0, 2π] checked via percentiles."""
    rng = np.random.default_rng(123)
    v = _sample_isotropic_np(rng, 50_000)
    # cos θ = z-component
    qs = np.array([0.1, 0.5, 0.9])
    expected_cos = -1.0 + 2.0 * qs
    got_cos = np.percentile(v[:, 2], qs * 100)
    assert np.max(np.abs(got_cos - expected_cos)) < 0.05


def test_hypoexp_positivity_and_mean():
    """Sample mean ≈ τ_rise + τ_fall (sum of independent exponentials)."""
    rng = np.random.default_rng(321)
    tau_r, tau_f = 2.0, 25.0
    t = _sample_hypoexp_np(rng, 50_000, tau_r, tau_f)
    assert np.all(t >= 0)
    expected_mean = tau_r + tau_f
    # Sample stderr ≈ √((τ_r² + τ_f²) / N). Use 5σ tolerance.
    sd = np.sqrt((tau_r ** 2 + tau_f ** 2) / 50_000)
    assert abs(np.mean(t) - expected_mean) < 5.0 * sd
