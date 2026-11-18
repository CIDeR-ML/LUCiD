"""Tests for shotgun waveform and per-photon simulation outputs.

Marked slow: requires building a detector + propagator + running JAX sim.
"""
import os
from tests.helpers.paths import REPO
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import numpy as np
import pytest

from lucid.sources import shotgun_source, stack_shotgun_sources
from lucid.simulation.shotgun import setup_shotgun_simulator

# These tests hit the full propagation pipeline and are slow; treat like the
# other integration tests in the repo.
pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def wcte_paths():
    base = str(REPO)
    return (os.path.join(base, 'config', 'WCTE_like_geom_config.json'),
            os.path.join(base, 'config', 'WCTE_like_physics_config.json'))


@pytest.fixture(scope='module')
def sim_waveform(wcte_paths):
    geom, phys = wcte_paths
    return setup_shotgun_simulator(
        geom, physics_config=phys, n_photons=64,
        hits='waveform', K=3,
    )


@pytest.fixture(scope='module')
def sim_per_photon(wcte_paths):
    geom, phys = wcte_paths
    return setup_shotgun_simulator(
        geom, physics_config=phys, n_photons=64,
        hits='photon_hits', K=3,
    )


def _source(n=64):
    return shotgun_source([0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                          n_photons=n, wavelength=400.0, intensity=1.0)


def test_waveform_determinism(sim_waveform):
    wf1, _, _ = sim_waveform(_source(), jax.random.PRNGKey(0))
    wf2, _, _ = sim_waveform(_source(), jax.random.PRNGKey(0))
    assert np.array_equal(np.asarray(wf1), np.asarray(wf2))


def test_per_photon_shapes_and_range(sim_per_photon):
    det, sid, ht, _ = sim_per_photon(_source(), jax.random.PRNGKey(0))
    det_np = np.asarray(det)
    sid_np = np.asarray(sid)
    ht_np = np.asarray(ht)
    assert det_np.shape == (64,)
    assert sid_np.shape == (64,)
    assert ht_np.shape == (64,)
    assert det_np.dtype == bool
    n_detected = int(det_np.sum())
    assert 0 < n_detected <= 64
    # Non-detected photons have sentinel -1; detected have valid sensor id
    assert np.all(sid_np[det_np] >= 0)
    assert np.all(sid_np[det_np] < sim_per_photon.num_sensors)
    assert np.all(sid_np[~det_np] == -1)
    # Detected times positive
    assert np.all(ht_np[det_np] > 0)


def test_per_photon_determinism(sim_per_photon):
    det1, sid1, ht1, _ = sim_per_photon(_source(), jax.random.PRNGKey(0))
    det2, sid2, ht2, _ = sim_per_photon(_source(), jax.random.PRNGKey(0))
    assert np.array_equal(np.asarray(det1), np.asarray(det2))
    assert np.array_equal(np.asarray(sid1), np.asarray(sid2))
    assert np.allclose(np.asarray(ht1), np.asarray(ht2))


def test_batch_produces_consistent_totals(sim_waveform):
    sources = [_source() for _ in range(3)]
    batched = stack_shotgun_sources(sources)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    wf_b, nd_b, ndet_b = sim_waveform.batch(batched, keys)
    assert wf_b.shape[0] == 3
    assert ndet_b.shape == (3,)
    # Per-case equivalence to single-case call
    wf0, nd0, ndet0 = sim_waveform(sources[0], keys[0])
    assert np.allclose(np.asarray(wf_b[0]), np.asarray(wf0))
    assert int(ndet_b[0]) == int(ndet0)


# ── Expected-value waveform mode ───────────────────────────────────────────

@pytest.fixture(scope='module')
def sim_waveform_expected(wcte_paths):
    geom, phys = wcte_paths
    return setup_shotgun_simulator(
        geom, physics_config=phys, n_photons=64,
        transport='expected', detection='expected', K=3,
    )


def test_expected_waveform_continuous_and_finite(sim_waveform_expected):
    wf, _, ndet = sim_waveform_expected(_source(), jax.random.PRNGKey(0))
    wf_np = np.asarray(wf)
    assert np.isfinite(wf_np).all()
    # Continuous weights → expect non-integer charge somewhere.
    has_fractional = np.any((wf_np > 0) & (wf_np != np.floor(wf_np)))
    assert has_fractional, "expected_value mode should produce fractional charges"
    assert float(wf_np.sum()) > 0
    # ndet in expected mode is total continuous charge, so should equal sum.
    assert abs(float(ndet) - float(wf_np.sum())) < 1e-4


def test_expected_waveform_matches_bernoulli_mean(sim_waveform, sim_waveform_expected):
    """Large-N: expected-value waveform should agree with the mean of many
    Bernoulli runs within a few percent."""
    src = _source(64)
    wf_expected, _, _ = sim_waveform_expected(src, jax.random.PRNGKey(0))
    bernoulli_runs = []
    for k in range(64):
        wf_b, _, _ = sim_waveform(src, jax.random.PRNGKey(k))
        bernoulli_runs.append(np.asarray(wf_b))
    mean_bernoulli = np.mean(bernoulli_runs, axis=0)
    # Compare totals (robust summary).
    tot_exp = float(np.asarray(wf_expected).sum())
    tot_bern = float(mean_bernoulli.sum())
    rel = abs(tot_exp - tot_bern) / max(tot_bern, 1e-6)
    assert rel < 0.10, f"expected total {tot_exp:.3f} vs Bernoulli mean {tot_bern:.3f}"


def test_expected_waveform_plus_cherenkov_qe(wcte_paths):
    """Expected-value + Method B: still finite and positive, shape unchanged.
    Needs a physics config that provides a QE curve (SK's does; WCTE's doesn't)."""
    geom, _ = wcte_paths
    base = str(REPO)
    sk_phys = os.path.join(base, 'config', 'SK_physics_config.json')
    sim = setup_shotgun_simulator(
        geom, physics_config=sk_phys, n_photons=64,
        transport='expected', detection='expected', K=3,
        wavelength_sampling='cherenkov_qe',
    )
    # Force Cherenkov sampling by leaving source.wavelength=None.
    src = shotgun_source([0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                         n_photons=64, wavelength=None, intensity=1.0)
    wf, _, _ = sim(src, jax.random.PRNGKey(0))
    wf_np = np.asarray(wf)
    assert np.isfinite(wf_np).all()
    assert float(wf_np.sum()) > 0


@pytest.mark.parametrize('mode', ['bernoulli', 'expected'])
def test_waveform_layout_and_integral(mode, sim_waveform, sim_waveform_expected):
    """Both waveform modes: (num_sensors, window / bin) = (N, 500), nothing dropped (the 500 ns
    window is far past the ~43 ns latest hit here), and the integral is the detected light.

    Bernoulli: each detected photon deposits a unit charge smeared by the SK-like gain spread
    (sigma about 1 %), so the integral is n_detected to 5 %. Expected value: ``n_detected`` is
    the integral itself. (This replaces two identical shape-only tests, one per mode.)
    """
    sim = sim_waveform if mode == 'bernoulli' else sim_waveform_expected
    wf, nd, ndet = sim(_source(), jax.random.PRNGKey(0))
    wf = np.asarray(wf)
    assert wf.shape == (sim.num_sensors, 500)
    assert np.isfinite(wf).all()
    assert int(nd) == 0
    assert float(ndet) > 0
    if mode == 'bernoulli':
        assert abs(float(wf.sum()) - int(ndet)) / int(ndet) < 0.05
    else:
        assert abs(float(wf.sum()) - float(ndet)) < 1e-4


@pytest.fixture(scope='module')
def sim_aggregated_expected(wcte_paths):
    """The same expected-value propagation as ``sim_waveform_expected``, read out per sensor."""
    from lucid.simulation import setup_event_simulator
    geom, phys = wcte_paths
    return setup_event_simulator(
        geom, n_photons=64, source='light_source', temperature=None, K=3,
        physics_config=phys, default_detector_params=True, wavelength_mode=True,
        wavelength_sampling='cherenkov')


@pytest.mark.parametrize('seed', [0, 1])
def test_expected_waveform_rows_are_the_aggregated_charge_and_peak_at_the_direct_time(
        seed, sim_waveform_expected, sim_aggregated_expected):
    """Row s of the expected-value waveform integrates to sensor s's charge in the aggregated
    readout of the same propagation (same key), and the direct light peaks in the bin of the
    straight-line arrival.

    The two readouts share the propagation and differ only in the binning, so the per-sensor
    closure (measured 4e-6 of a 16-PE total) fails for a waveform filed under the wrong sensor
    or the wrong axis, which the totals cannot see. The 64 photons leave the centre along +z
    and meet the top-cap PMTs after (H/2 - r_s) / (c/n) = 1.96 m / 0.2254 m/ns = 8.70 ns, so
    the summed profile peaks in bin 8 (TTS 1 ns; c instead of c/n would put it in bin 6).
    """
    key = jax.random.PRNGKey(seed)
    wf, _, _ = sim_waveform_expected(_source(), key)
    q, _ = sim_aggregated_expected(_source(), key)
    wf, q = np.asarray(wf), np.asarray(q)
    assert q.sum() > 1.0
    np.testing.assert_allclose(wf.sum(axis=1), q, rtol=1e-5, atol=1e-5)
    t_direct = (2.0 - 0.04) / (0.299792458 / 1.33)
    assert abs(int(np.argmax(wf.sum(axis=0))) - int(np.floor(t_direct))) <= 1


def test_expected_value_rejects_per_photon(wcte_paths):
    """Expected-value + per-photon output is unsupported and should error."""
    geom, phys = wcte_paths
    with pytest.raises(ValueError, match="'photon_hits' with detection='expected' is not"):
        setup_shotgun_simulator(
            geom, physics_config=phys, n_photons=64,
            hits='photon_hits', transport='expected', detection='expected', K=3,
        )
