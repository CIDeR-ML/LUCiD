"""The track (SIREN) forward on the tiny detector: how the emitter's params are bound, the band its
Cherenkov λ is sampled over, and the track's start time.

SK_like's physics (SK QE curve, wavelength mode) on the tiny geometry (``tests/helpers/tiny.py``),
the shipped water muon emitter.
"""
import jax
import numpy as np
import pytest

from lucid.detector_params import ParticleParams
from tests.helpers import tiny
from tests.helpers.paths import REPO

PHYS = str(REPO / 'config' / 'SK_like_physics_config.json')
N = 20_000

pytestmark = pytest.mark.requires_emitter('muon', 'water')


def _track(t0=0.0):
    return ParticleParams.from_cartesian(300.0, [0.0, 0.0, -0.3], [0.3, 0.2, 0.93], t0=t0)


def _sim(**setup):
    from lucid.simulation import setup_event_simulator
    kw = dict(K=2, temperature=None, physics_config=PHYS, default_detector_params=True,
              particle='muon', source='particle', hits='first_arrival')
    return setup_event_simulator(tiny.GEOM, N, **{**kw, **setup})


def test_baked_and_passed_detector_params_give_the_same_event():
    """One binding: the same params baked at setup or passed at call time, the same key, the same
    charges and times, bit for bit (the SIREN params are jit arguments in both)."""
    baked = _sim()
    passed = _sim(default_detector_params=False)
    for k in range(2):
        key = jax.random.PRNGKey(k)
        b = baked(_track(), key)
        p = passed(_track(), baked.default_detector_params, key)
        for x, y in zip(b, p):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_cherenkov_light_is_sampled_over_the_emitters_emission_band():
    """By default the track forward samples Cherenkov λ over the band the emitter's N_photons(E)
    counts (its emission_band_nm, the PhotonSim range), exactly as when that band is passed."""
    from lucid.siren.emitter import load_emitter
    band = load_emitter('muon', 'water', 'cherenkov').emission_band
    assert band == (274.91, 673.83)
    own, given = _sim(), _sim(cherenkov_emission_band=band)
    key = jax.random.PRNGKey(3)
    for x, y in zip(own(_track(), key), given(_track(), key)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_both_wavelength_samplers_count_the_same_charge_over_the_band():
    """λ ~ 1/λ² with weight QE(λ), and λ ~ QE(λ)/λ² with weight <QE>_C, have the same expected
    charge when both run over the emission band: <QE>_C is the band's mean, the QE-dead fringe
    included. Over the QE window instead, <QE>_C would count the fringe as detected (+19%)."""
    band = (274.91, 673.83)
    plain = _sim(cherenkov_emission_band=band)
    weighted = _sim(cherenkov_emission_band=band, wavelength_sampling='cherenkov_qe')
    q = [float(np.mean([np.asarray(s(_track(), jax.random.PRNGKey(k))[0]).sum() for k in range(3)]))
         for s in (plain, weighted)]
    np.testing.assert_allclose(q[1], q[0], rtol=0.02)


def test_the_tracks_t0_shifts_every_hit_time_and_no_charge():
    """A track starting t0 later gives every photon's hit the same time plus t0 (float32 rounding
    of ~10 ns times) and exactly the same charges."""
    sim = _sim(hits='photon_list')
    key = jax.random.PRNGKey(4)
    lw0, t_0, i0, q0 = (np.asarray(x) for x in sim(_track(0.0), key))
    lw1, t_1, i1, q1 = (np.asarray(x) for x in sim(_track(7.5), key))
    for a, b in ((lw0, lw1), (i0, i1), (q0, q1)):
        np.testing.assert_array_equal(a, b)
    hit = t_0 > 0                           # the per-photon mode zeroes the time of a non-deposit
    np.testing.assert_array_equal(t_1 > 0, hit)
    assert hit.sum() > 1000
    np.testing.assert_allclose(t_1[hit] - t_0[hit], 7.5, atol=1e-5)
