"""Unit tests for make_hits_moments — compound-Poisson charge moments + per-PMT response."""
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation.sensor_response import (
    make_hits_simulation, make_hits_moments, _expected_min_normal,
)


def _toy():
    # 3 photons into 2 sensors: sensor 0 gets photons 0,2; sensor 1 gets photon 1.
    flat_weights = jnp.array([0.5, 0.8, 0.3])
    flat_indices = jnp.array([0, 1, 0])
    flat_times = jnp.array([10.0, 12.0, 11.0])
    n = 2
    qe = 0.2
    qe_corr = jnp.ones(n)
    return flat_weights, flat_indices, flat_times, n, qe, qe_corr


class TestMakeHitsMoments:
    def test_neutral_matches_make_hits_simulation(self):
        """gain=1, w=0, t0=0 → mean==charge, var==mean, time==make_hits_simulation."""
        fw, fi, ft, n, qe, qc = _toy()
        charge, time = make_hits_simulation(fw, fi, ft, n, qe=qe, qe_corrections=qc)
        mean, var, tmom = make_hits_moments(fw, fi, ft, n, qe=qe, qe_corrections=qc)
        npt.assert_allclose(mean, charge, rtol=1e-6)
        npt.assert_allclose(var, mean, rtol=1e-6)
        npt.assert_allclose(tmom, time, rtol=1e-6)

    def test_gain_scales_mean_and_var(self):
        """mean = gain·μ ; var = gain²·(1+w²)·μ."""
        fw, fi, ft, n, qe, qc = _toy()
        mu, _ = make_hits_simulation(fw, fi, ft, n, qe=qe, qe_corrections=qc)  # μ at gain=1
        gain = jnp.array([2.0, 3.0])
        w = 0.35
        mean, var, _ = make_hits_moments(fw, fi, ft, n, qe=qe, qe_corrections=qc,
                                         gain=gain, spe_width=w)
        lit = mu > 0
        npt.assert_allclose(mean[lit], (gain * mu)[lit], rtol=1e-5)
        npt.assert_allclose(var[lit], (gain**2 * (1 + w**2) * mu)[lit], rtol=1e-5)
        # Fano v/m = gain·(1+w²)
        npt.assert_allclose((var / mean)[lit], (gain * (1 + w**2))[lit], rtol=1e-5)

    def test_t0_shifts_time(self):
        """A per-PMT t0 adds to the first-arrival time of lit sensors."""
        fw, fi, ft, n, qe, qc = _toy()
        _, _, t_base = make_hits_moments(fw, fi, ft, n, qe=qe, qe_corrections=qc)
        t0 = jnp.array([5.0, -2.0])
        _, _, t_shift = make_hits_moments(fw, fi, ft, n, qe=qe, qe_corrections=qc, t0=t0)
        lit = t_base != 0
        npt.assert_allclose((t_shift - t_base)[lit], t0[lit], atol=1e-5)

    def test_tts_pulls_first_arrival_earlier(self):
        """TTS>0 shifts the first-arrival earlier (occupancy bias ≤0); tts=0 unchanged."""
        # one sensor, several photons → occupancy > 1 so the bias is active
        fw = jnp.full(8, 0.5); fi = jnp.zeros(8, int); ft = jnp.linspace(10.0, 13.0, 8)
        _, _, t0_ = make_hits_moments(fw, fi, ft, 1, qe=1.0, qe_corrections=jnp.ones(1), tts=0.0)
        _, _, t2_ = make_hits_moments(fw, fi, ft, 1, qe=1.0, qe_corrections=jnp.ones(1), tts=2.0)
        assert float(t2_[0]) < float(t0_[0])   # earlier with TTS

    def test_empty_sensor_zeroed(self):
        """A sensor with no photons reports zero mean/var/time."""
        fw = jnp.array([0.5]); fi = jnp.array([0]); ft = jnp.array([10.0])
        mean, var, tmom = make_hits_moments(fw, fi, ft, 3, qe=0.2, qe_corrections=jnp.ones(3))
        npt.assert_allclose(mean[1:], 0.0)
        npt.assert_allclose(var[1:], 0.0)
        npt.assert_allclose(tmom[1:], 0.0)


class TestExpectedMinNormal:
    def test_zero_at_low_occupancy(self):
        npt.assert_allclose(_expected_min_normal(jnp.array([0.1, 0.5, 1.0])), 0.0, atol=1e-5)

    def test_negative_and_monotone(self):
        occ = jnp.array([1.0, 2.0, 5.0, 10.0, 100.0])
        b = _expected_min_normal(occ)
        assert jnp.all(jnp.diff(b) < 0)            # more photons → earlier (more negative)
        # known E[min of 5 N(0,1)] ≈ -1.163 (Blom approx within ~3%)
        npt.assert_allclose(float(_expected_min_normal(jnp.array(5.0))), -1.163, atol=0.05)
