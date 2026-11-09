"""Physics-level tests for loss functions.

Tests verify gradient directions, numerical stability, and mathematical
properties of each loss function.
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.losses import (
    counts_loss, energy_loss,
)


class TestCountsLossGradients:
    """Poisson NLL gradient: d/dpred [pred - true log pred] = 1 - true/pred."""

    def test_gradient_is_one_minus_true_over_pred(self):
        """Both normalisations, over- and under-prediction in one vector. The sign test this
        replaces passed a gradient off by any positive factor (e.g. a lost 1/sum(true))."""
        import numpy as np
        true = jnp.array([10.0, 5.0, 8.0, 3.0])
        pred = jnp.array([15.0, 2.0, 8.0, 4.5])
        want = 1.0 - np.asarray(true) / np.asarray(pred)
        g_raw = jax.grad(lambda p: counts_loss(true, p, normalize=False))(pred)
        g_norm = jax.grad(lambda p: counts_loss(true, p))(pred)
        npt.assert_allclose(g_raw, want, rtol=1e-6, atol=1e-7)
        npt.assert_allclose(g_norm, want / float(jnp.sum(true)), rtol=1e-6, atol=1e-8)

    def test_minimum_near_true(self):
        """Loss should be minimized near pred ≈ true."""
        true = jnp.array([10.0, 5.0, 8.0])
        losses = []
        for scale in [0.5, 0.8, 1.0, 1.2, 1.5]:
            losses.append(float(counts_loss(true, true * scale)))
        # Loss at scale=1.0 (perfect prediction) should be the minimum
        assert losses[2] <= min(losses[0], losses[1], losses[3], losses[4])


class TestEnergyLoss:
    def test_symmetric_in_ratio(self):
        """energy_loss = |log(sum_sim/sum_true)|, so doubling and halving give same loss."""
        true = jnp.array([10.0, 10.0])
        loss_double = energy_loss(true * 2, true)
        loss_half = energy_loss(true * 0.5, true)
        npt.assert_allclose(loss_double, loss_half, atol=1e-5)

    def test_monotonic_with_ratio(self):
        """Loss should increase as prediction deviates from truth."""
        true = jnp.array([10.0, 5.0])
        l1 = energy_loss(true * 1.1, true)
        l2 = energy_loss(true * 1.5, true)
        l3 = energy_loss(true * 2.0, true)
        assert l1 < l2 < l3


class TestFirstArrivalWindowNLL:
    """The recon time term against the order statistic it claims to be.

    For one PMT with photon weights w_i at predicted times t_i, a single photon's arrival time has
    the mixture density sum_i (w_i / mu) N(t_i, sigma^2), mu = sum w_i. With n photons the first
    arrival is the minimum of n draws, so ``exp(-tnll)`` must be the probability that the minimum
    lands in the window ``[t_obs - delta/2, t_obs + delta/2]``.
    """
    W = (2.0, 1.0, 0.5)
    T = (10.0, 12.0, 15.0)
    SIGMA, DELTA = 2.1, 1.0

    def _tnll(self, t_obs, n):
        import numpy as np
        from lucid.losses import first_arrival_window_nll
        K = len(t_obs)                              # one "PMT" per window, same photons each
        w = np.tile(self.W, K); t = np.tile(self.T, K)
        idx = np.repeat(np.arange(K), len(self.W))
        mu = np.full(K, sum(self.W))
        return np.asarray(first_arrival_window_nll(
            jnp.log(jnp.asarray(w)), jnp.asarray(t), jnp.asarray(idx), jnp.asarray(t_obs),
            jnp.asarray(mu), jnp.full(K, float(n)), K, sigma=self.SIGMA, delta=self.DELTA), float)

    def test_window_probabilities_tile_to_one(self):
        """Windows tiling [-15, 45] ns cover the whole first-arrival distribution: sum = 1.

        Measured |sum - 1| = 1.3e-6 for n = 1, 3, 7 (float32); tolerance 1e-4. Dropping the
        order-statistic exponent n (S(lo) in place of S(lo)^n) breaks it for n > 1.
        """
        import numpy as np
        for n in (1, 3, 7):
            centres = np.arange(-15.0, 45.0, self.DELTA) + self.DELTA / 2
            p = np.exp(-self._tnll(centres, n))
            npt.assert_allclose(p.sum(), 1.0, atol=1e-4)

    def test_matches_a_brute_force_minimum_of_n_draws(self):
        """exp(-tnll) against the Monte-Carlo frequency of min-of-n in the window.

        4e5 trials; the tolerance is 5 binomial standard errors (about 3e-3 here).
        """
        import numpy as np
        rng = np.random.default_rng(0)
        n, trials = 3, 400_000
        pw = np.asarray(self.W) / sum(self.W)
        comp = rng.choice(len(self.W), size=(trials, n), p=pw)
        draws = np.asarray(self.T)[comp] + self.SIGMA * rng.standard_normal((trials, n))
        first = draws.min(axis=1)
        centres = np.array([6.5, 9.5, 12.0, 16.0])
        p = np.exp(-self._tnll(centres, n))
        freq = np.array([np.mean(np.abs(first - c) <= self.DELTA / 2) for c in centres])
        se = np.sqrt(freq * (1 - freq) / trials)
        assert np.all(np.abs(p - freq) < 5 * se + 1e-6), (p, freq, se)

    def test_an_unlit_pmt_contributes_nothing(self):
        import numpy as np
        assert np.all(self._tnll(np.array([10.0, 20.0]), 0) == 0.0)
