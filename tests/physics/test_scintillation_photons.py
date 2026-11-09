"""Scintillation photon generation: the NumPy data-mode expander and the JAX track-mode surrogate.

Oracles: Birks/Chou N = S E / (1 + kB d + C d^2) with d in keV/mm (C nonzero), the particle's
travel time u ds / (beta c) with the emission delay switched off, the closed-form Moyal CDF
erfc(exp(-z/2)/sqrt 2) and the hypoexponential rise/fall CDF (KS tests), and isotropy
E[v v^T] = I/3. The JAX surrogate is tested net-free with a one-bin dE/dx stub, which makes
its total quenched intensity closed form.
"""
from __future__ import annotations

import numpy as np
import pytest

from lucid.sources.scintillation_photons import (
    _sample_isotropic_np,
    _sample_hypoexp_np,
    expand_segments_to_photons,
)
from lucid.dataset.schema import EMISSION_PROCESS_SCINTILLATION

from tests.helpers.stats import ks_distance


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


def test_directions_are_isotropic_unit_vectors():
    """Expander directions are unit vectors with E[v] = 0 and E[v v^T] = I/3 (isotropy).
    With ~3e4 photons the standard error per entry is <= 1/sqrt(3e4 * 5) ~ 3e-3."""
    out = expand_segments_to_photons(
        _make_three_segments(), _medium_params(S=10000.0),
        np.random.default_rng(4),
    )
    v = out['photon_directions'].astype(np.float64)
    assert len(v) > 20000
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=0, atol=1e-5)
    np.testing.assert_allclose(v.mean(axis=0), 0.0, atol=0.015)
    np.testing.assert_allclose(v.T @ v / len(v), np.eye(3) / 3.0, atol=0.012)


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


def _truncated_moyal_cdf(x, loc, scale, lo, hi):
    """Closed-form Moyal CDF F(x) = erfc(exp(-z/2)/sqrt(2)), z = (x - loc)/scale (Moyal 1955;
    scipy.stats.moyal), renormalised to the window [lo, hi]."""
    from scipy.special import erfc

    def F(v):
        return erfc(np.exp(-0.5 * (np.asarray(v, np.float64) - loc) / scale) / np.sqrt(2.0))
    return (F(x) - F(lo)) / (F(hi) - F(lo))


@pytest.mark.parametrize('loc,scale', [(410.0, 30.0), (373.26, 18.0)])
def test_wavelength_marginal_matches_the_closed_form_moyal(loc, scale):
    """The data-mode wavelengths follow the truncated Moyal law, by a KS distance against the
    closed-form CDF. The KS 99.9 % critical value at n = 1e5 is 1.95/sqrt(n) = 0.006; the
    512-knot table adds < 1e-4. A Moyal mirrored about its mode misses by ~0.3. The table is the
    track-mode surrogate's: both modes draw from one inverse CDF."""
    from lucid.sources.scintillation_photons import _sample_moyal_np
    from lucid.sources.siren_rays import _build_moyal_inverse_cdf
    lo, hi = 300.0, 700.0
    lam = np.sort(_sample_moyal_np(np.random.default_rng(42), 100_000, loc, scale, lo, hi)
                  .astype(np.float64))
    F = _truncated_moyal_cdf(lam, loc, scale, lo, hi)
    n = len(lam)
    ks = max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))
    assert ks < 0.006, ks
    # the inverse-CDF table itself, without sampling noise
    cdf, grid = (np.asarray(a) for a in _build_moyal_inverse_cdf(lo, hi, loc, scale))
    np.testing.assert_allclose(cdf, _truncated_moyal_cdf(grid, loc, scale, lo, hi), atol=1e-4)
    u = np.random.default_rng(42).uniform(0.0, 1.0, size=100_000)
    np.testing.assert_array_equal(np.sort(np.interp(u, cdf, grid).astype(np.float32)), lam)


# ---------------------------------------------------------------------------
# Time invariants
# ---------------------------------------------------------------------------


def test_times_at_or_after_segment_time():
    """t_i >= Segment_Time[seg_idx[i]] for all rays."""
    segs = _make_three_segments()
    out = expand_segments_to_photons(
        segs, _medium_params(), np.random.default_rng(6),
    )
    seg_idx = out['photon_segment_index_raw']
    seg_t = np.asarray(segs['time'])[seg_idx]
    assert np.all(out['photon_times'] >= seg_t - 1e-4)


def test_time_without_emission_delay_is_the_particle_travel_time():
    """With tau_rise = tau_fall = 0 the photon leaves at the moment the particle reaches its
    origin: t = t_seg + |origin - start| / (beta c). Float32 times near 100 ns are good to
    ~1e-5 ns; beta*c instead of c/beta would be off by up to 0.07 ns on the beta = 0.5 step."""
    segs = _make_three_segments()
    mp = _medium_params()
    mp.update(tau_rise=0.0, tau_fall=0.0)
    out = expand_segments_to_photons(segs, mp, np.random.default_rng(16))
    i = out['photon_segment_index_raw']
    start_m = np.stack([segs['start_x_mm'], segs['start_y_mm'], segs['start_z_mm']], axis=1)[i] / 1e3
    path_m = np.linalg.norm(out['photon_origins'] - start_m, axis=1)
    c_m_per_ns = 0.299792458
    expected = segs['time'][i] + path_m / (segs['beta_start'][i] * c_m_per_ns)
    np.testing.assert_allclose(out['photon_times'], expected, atol=2e-4)


# ---------------------------------------------------------------------------
# Mean count vs Birks
# ---------------------------------------------------------------------------


def test_mean_count_matches_birks_chou():
    """Average N_per_seg over many trials reproduces S·E/(1+kB·d+C·d²), d in keV/mm.

    C = 1e-5 (mm/keV)^2 makes the Chou term matter on the 1000 keV/mm step (C d^2 = 10,
    comparable to kB d = 10); a linear C d term would raise that segment's count 1.9x."""
    segs = _make_three_segments()
    mp = _medium_params(S=1000.0, kB=0.01, C=1e-5)
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
    qs = np.array([0.1, 0.5, 0.9])
    expected_cos = -1.0 + 2.0 * qs
    got_cos = np.percentile(v[:, 2], qs * 100)
    assert np.max(np.abs(got_cos - expected_cos)) < 0.02
    phi = np.arctan2(v[:, 1], v[:, 0]) % (2 * np.pi)
    got_phi = np.percentile(phi, qs * 100)
    assert np.max(np.abs(got_phi - 2 * np.pi * qs)) < 0.1


def _biexp_cdf(t, tau_r, tau_f):
    """CDF of the sum of independent Exp(tau_r) and Exp(tau_f) (hypoexponential):
    F(t) = 1 - (tau_f e^{-t/tau_f} - tau_r e^{-t/tau_r}) / (tau_f - tau_r).
    Its density is the rise/fall shape (e^{-t/tau_f} - e^{-t/tau_r}) / (tau_f - tau_r)."""
    return 1.0 - (tau_f * np.exp(-t / tau_f) - tau_r * np.exp(-t / tau_r)) / (tau_f - tau_r)


def test_hypoexp_moments_and_shape():
    """Mean tau_r + tau_f, variance tau_r^2 + tau_f^2, and the rise/fall CDF (KS < 0.006,
    the 99.9 % critical value at n = 1e5)."""
    rng = np.random.default_rng(321)
    tau_r, tau_f = 2.0, 25.0
    n = 100_000
    t = _sample_hypoexp_np(rng, n, tau_r, tau_f).astype(np.float64)
    assert np.all(t >= 0)
    var = tau_r ** 2 + tau_f ** 2
    assert abs(t.mean() - (tau_r + tau_f)) < 5.0 * np.sqrt(var / n)
    # Var of the sample variance of a near-exponential is ~8 tau^4/n -> 0.9 % s.e.; 5 %
    assert abs(t.var() / var - 1.0) < 0.05
    assert ks_distance(t, lambda x: _biexp_cdf(x, tau_r, tau_f)) < 0.006


# ---------------------------------------------------------------------------
# The JAX track-mode surrogate (make_scintillation_surrogate_fn), net-free
# ---------------------------------------------------------------------------
#
# A stub stands in for the dE/dx SIREN: it puts all the deposited energy in one dE/dx bin
# (centre D0) at every R, so the surrogate's typical dE/dx is exactly D0 everywhere and the
# total quenched intensity has the closed form S E / (1 + kB D0 + C D0^2).

D0 = 250.0                          # keV/mm, the centre of bin 2 of 4 on [0, 400]


class _OneBinModel:
    """``apply`` returns raw = 1 in the D0 bin (weight 10^-9 - 10^-12) and 0 elsewhere (weight
    10^-12 - 10^-12: the target offset, so an empty bin decodes to 0)."""

    def apply(self, params, x):
        import jax.numpy as jnp
        d_norm = x[:, 1]                                  # D0 normalises to 0.25
        return jnp.where(jnp.abs(d_norm - 0.25) < 0.1, 1.0, 0.0)


def _scint_surrogate(lam_lo=300.0, lam_hi=700.0, loc=373.26, scale=18.0):
    from lucid.siren.emitter import Emitter
    from lucid.sources.siren_rays import make_scintillation_surrogate_fn
    e = Emitter(kind='dedx', net=_OneBinModel(), params=None, lo=(10.0, 0.0, 0.0),
                hi=(1000.0, 400.0, 1.0), log10_range=(-12.0, -9.0), target_offset=1e-12, grid_bins=4,
                n_photons=None, t0=None, c_mm_per_ns=None, emission_band=None, meta={},
                _s_max_mm=lambda E: 4.0 * E)
    return make_scintillation_surrogate_fn(e, lam_lo, lam_hi, loc, scale)


_SCINT = dict(S=1387.3, kB=0.01, C=1e-5, tau_rise=0.13, tau_fall=2.7)
_NRAYS = 100_000


def _scint_rays(fn, energy=500.0, **over):
    import jax
    import jax.numpy as jnp
    p = {**_SCINT, **over}
    origin = jnp.array([1.0, -2.0, 0.5])
    direction = jnp.array([1.0, 2.0, -2.0]) / 3.0
    out = fn(origin, direction, energy, _NRAYS, None, jax.random.PRNGKey(5),
             p['S'], p['kB'], p['C'], p['tau_rise'], p['tau_fall'])
    return origin, direction, out


def test_surrogate_total_intensity_is_the_chou_quenched_yield():
    """sum(intensity) = S E / (1 + kB D0 + C D0^2) at constant dE/dx D0; linear in S."""
    fn = _scint_surrogate()
    E = 500.0
    _, _, (_, _, inten, _, _) = _scint_rays(fn, energy=E)
    p = _SCINT
    expected = p['S'] * E / (1.0 + p['kB'] * D0 + p['C'] * D0 ** 2)
    np.testing.assert_allclose(float(inten.sum()), expected, rtol=1e-4)
    _, _, (_, _, inten2, _, _) = _scint_rays(fn, energy=E, S=2 * p['S'])
    np.testing.assert_allclose(float(inten2.sum()), 2 * expected, rtol=1e-5)


def test_surrogate_quench_gradients_are_analytic():
    """d/dkB and d/dC of the total are -S E D0^k / (1 + kB D0 + C D0^2)^2 (k = 1, 2)."""
    import jax
    import jax.numpy as jnp
    fn = _scint_surrogate()
    E, p = 500.0, _SCINT

    def total(kB, C):
        out = fn(jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]), E, 1000, None,
                 jax.random.PRNGKey(0), p['S'], kB, C, p['tau_rise'], p['tau_fall'])
        return out[2].sum()

    g_kB, g_C = jax.grad(total, argnums=(0, 1))(jnp.asarray(p['kB']), jnp.asarray(p['C']))
    den = 1.0 + p['kB'] * D0 + p['C'] * D0 ** 2
    np.testing.assert_allclose(float(g_kB), -p['S'] * E * D0 / den ** 2, rtol=1e-4)
    np.testing.assert_allclose(float(g_C), -p['S'] * E * D0 ** 2 / den ** 2, rtol=1e-4)


def test_surrogate_emission_delay_is_rise_plus_fall():
    """The JAX hypoexp delay: mean tau_r + tau_f, variance tau_r^2 + tau_f^2, rise/fall CDF."""
    fn = _scint_surrogate()
    _, _, (_, _, _, t, _) = _scint_rays(fn)
    t = np.asarray(t, np.float64)
    tr, tf = _SCINT['tau_rise'], _SCINT['tau_fall']
    var = tr ** 2 + tf ** 2
    assert abs(t.mean() - (tr + tf)) < 5.0 * np.sqrt(var / _NRAYS)
    assert abs(t.var() / var - 1.0) < 0.05
    assert ks_distance(t, lambda x: _biexp_cdf(x, tr, tf)) < 0.006


def test_surrogate_wavelengths_follow_the_truncated_moyal():
    fn = _scint_surrogate(loc=410.0, scale=30.0)
    _, _, (_, _, _, _, lam) = _scint_rays(fn)
    assert ks_distance(lam, lambda x: _truncated_moyal_cdf(x, 410.0, 30.0, 300.0, 700.0)) < 0.006


def test_surrogate_rays_are_isotropic_and_on_the_track():
    """Directions: unit, E[v] = 0, E[v v^T] = I/3. Origins: on the track line, between the
    vertex and s_max(E) = 4 E mm (the stub's), in metres."""
    fn = _scint_surrogate()
    E = 500.0
    origin, direction, (v, o, _, _, _) = _scint_rays(fn, energy=E)
    v = np.asarray(v, np.float64)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(v.mean(axis=0), 0.0, atol=5e-3)
    np.testing.assert_allclose(v.T @ v / len(v), np.eye(3) / 3.0, atol=5e-3)
    rel = np.asarray(o, np.float64) - np.asarray(origin)
    s = rel @ np.asarray(direction)
    np.testing.assert_allclose(rel, s[:, None] * np.asarray(direction)[None, :], atol=1e-5)
    assert s.min() >= -1e-6 and s.max() <= 4.0 * E / 1000.0 + 1e-6
