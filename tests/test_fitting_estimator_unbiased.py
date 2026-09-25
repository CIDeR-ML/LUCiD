"""The Neyman residual is unbiased at truth under a redrawn Monte-Carlo forward; profiled gains are not.

The calibration forward is a Monte-Carlo estimate REDRAWN EVERY STEP, so a residual nonlinear in
the model has `E[f(M)] != f(E[M])` and a displaced fixed point. Neyman weights by the data alone,
so it stays linear in the model; the sqrt residual does not. Profiling the gains,
`k = sum(Q)/sum(M)`, is nonlinear through `1/sum(M)` and reintroduces a bias that grows with the
forward noise (the same Jensen family `profile_gains`' docstring records).

The test checks the FIXED-POINT CONDITION directly: the expected gradient at truth must vanish.
Running fits would be slower, noisier, and confound the estimator with the optimizer. The other
calibration gates use deterministic stubs, where any consistent estimator converges, so they
cannot see this.

The toy runs at 5-25% per-sensor forward noise, far above the published configuration: it measures
the mechanism and its direction, not its size in production.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from lucid.fitting import FieldParams, profile_gains
from lucid.fitting.calib import CalibrationForward, CalibrationJacobian, CalibrationProblem
from tests import _calib_toy as toy
from tests._calib_toy import FIELDS

NS = 40
NOISE_HI, NOISE_LO = 0.25, 0.05
N_DRAWS = 800                       # z scales as sqrt(N); 800 keeps the contrast decisive


def _dp_true():
    return toy.dp_true(NS)


_base = toy.charge


def _make_noisy(noise):
    """Forward whose draws scatter around the truth model, LOGNORMAL and mean-preserving.

    Not `1 + sigma*z`: that goes negative about 3e-5 of the time, and over tens of thousands of
    multipliers a negative charge is expected and makes `sqrt(k*M)` NaN. `exp(sigma*z - sigma^2/2)`
    is positive and has mean exactly 1, so any bias measured is the estimator's and not the toy's.
    """
    def sim(source, dp, key):
        z = jax.random.normal(key, (NS,))
        return _base(source, dp) * jnp.exp(noise * z - 0.5 * noise ** 2), jnp.zeros(NS)
    return sim


def _clean(source, dp, key):
    return _base(source, dp), jnp.zeros(NS)


def _sources():
    return toy.sources(NS, seed=17, b_spread=0.5, b_scale=2.3)


@pytest.fixture(scope='module')
def z_scores():
    """Worst-parameter |mean g| / s.e.m. at truth, for each (noise, residual, gain) arm."""
    params = FieldParams(_dp_true(), FIELDS, n_sensors=NS)
    theta_true = params.theta_from_physical()
    gains = np.exp(0.15 * np.random.default_rng(5).standard_normal(NS))
    gains /= np.exp(np.mean(np.log(gains)))
    th = jnp.asarray(theta_true, dtype=jnp.float32)
    ones = jnp.ones(NS)
    # The DATA is one fixed clean dataset; what is redrawn each step is the MODEL. That is the
    # situation the estimator choice is about.
    data = CalibrationForward(_clean, _sources(), params, NS)(th, 5000, jnp.asarray(gains))

    def arm(noise, residual, gain_mode):
        sim = _make_noisy(noise)
        fwd = CalibrationForward(sim, _sources(), params, NS)
        jac = CalibrationJacobian(sim, _sources(), params, NS, key0=9_000_000)
        prob = CalibrationProblem(fwd, jac, params, data, 1e-6,
                                  n_forward_draws=1, jacobian_draws=1)
        g = []
        for step in range(N_DRAWS):
            mu = fwd.average(th, prob.forward_key(step), ones, 1)
            k = (profile_gains(mu.sum(0) + 1e-12, prob.data_sum) if gain_mode == 'profiled'
                 else jnp.asarray(gains))
            if residual == 'neyman':
                r = (k[None, :] * mu - data) / jnp.sqrt(jnp.clip(data, 1e-6, None))
            else:
                r = jnp.sqrt(k[None, :] * mu + 1e-8) - jnp.sqrt(data + 1e-8)
            j = jac(th, jnp.log(k), step, data, 1e-6, draws=(0,))
            g.append(np.asarray(jnp.einsum('gnp,gn->p', j, r) / prob.n_configs))
        g = np.stack(g)
        return float(np.max(np.abs(g.mean(0)) / (g.std(0, ddof=1) / np.sqrt(len(g)))))

    return {(nz, res, gm): arm(nz, res, gm)
            for nz in (NOISE_HI, NOISE_LO)
            for res, gm in (('neyman', 'fixed'), ('neyman', 'profiled'), ('sqrt', 'fixed'))}


def test_the_neyman_residual_is_unbiased_at_truth(z_scores):
    """The claim: with the gains held fixed, the expected gradient vanishes at truth.

    Checked at BOTH noise levels — a residual whose bias merely happened to be small at one
    forward variance would not be linear in the model.
    """
    for nz in (NOISE_HI, NOISE_LO):
        z = z_scores[(nz, 'neyman', 'fixed')]
        assert z < 4.0, (f'at forward noise {nz}, the Neyman gradient at truth is {z:.1f} sigma '
                         f'from zero — its fixed point is displaced')


def test_the_instrument_can_detect_a_displaced_fixed_point(z_scores):
    """The control. Without it, "consistent with zero" could be a statement about a blind test.

    Uses the PROFILED-gain arm rather than the sqrt residual because it is decisive at this
    ensemble size, while the sqrt arm's displacement is real but weak.
    """
    z = z_scores[(NOISE_HI, 'neyman', 'profiled')]
    assert z > 8.0, (f'no arm of this test shows bias ({z:.1f} sigma), so the null result above '
                     f'says nothing — the ensemble is too small or the forward noise too weak')


def test_the_profiling_bias_scales_with_forward_noise(z_scores):
    """Identifies the MECHANISM rather than just recording a number.

    `k = sum(Q)/sum(M)` is nonlinear in the model, so its Jensen term grows with the forward
    variance. If the bias were instead something fixed — a wrong key, a transposed Jacobian — it
    would not track the noise.
    """
    hi = z_scores[(NOISE_HI, 'neyman', 'profiled')]
    lo = z_scores[(NOISE_LO, 'neyman', 'profiled')]
    assert hi > 2.0 * lo, (f'profiling bias does not scale with forward noise '
                           f'(z {lo:.1f} at {NOISE_LO} vs {hi:.1f} at {NOISE_HI}) — the mechanism '
                           f'is not the Jensen term through 1/sum(M)')


def test_the_sqrt_residuals_displacement_grows_with_noise(z_scores):
    """The sqrt residual's fixed-point displacement grows with forward noise.

    The effect is weak in this toy, so this asserts the ORDERING rather than a threshold.
    """
    hi = z_scores[(NOISE_HI, 'sqrt', 'fixed')]
    lo = z_scores[(NOISE_LO, 'sqrt', 'fixed')]
    assert hi > lo, f'sqrt displacement does not grow with forward noise ({lo:.1f} -> {hi:.1f})'
