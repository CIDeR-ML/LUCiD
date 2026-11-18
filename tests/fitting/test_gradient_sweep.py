"""``lucid.gradient_analysis`` sweeps on the tiny detector: the analytic gradient of a calibration
loss vanishes at the truth, and a non-finite point raises instead of becoming a gap."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.gradient_analysis import SweepParam, find_zero_crossing, sweep_1d
from lucid.sources import isotropic_source
from tests.helpers import tiny

ABSORPTION_M = 10.0


def _loss_and_grad(sim):
    source = isotropic_source([0.21, -0.13, 0.32], intensity=tiny.N)
    key = jax.random.PRNGKey(0)
    truth = tiny.transparent(sim.det_geom.num_sensors, absorption_length=ABSORPTION_M)
    q_true = sim(source, truth, key)[0]

    @jax.jit
    def loss_and_grad(dp):
        return jax.value_and_grad(lambda p: jnp.sum((sim(source, p, key)[0] - q_true) ** 2))(dp)
    return truth, loss_and_grad


def test_the_loss_gradient_crosses_zero_at_the_true_absorption_length(tiny_sim):
    truth, loss_and_grad = _loss_and_grad(tiny_sim)
    sp = SweepParam('Absorption Length', 'absorption_length', half_width=4.0, num_points=5)
    r = sweep_1d(loss_and_grad, truth, [sp])['Absorption Length']

    np.testing.assert_allclose(r.values, [6.0, 8.0, 10.0, 12.0, 14.0])
    assert np.argmin(r.losses) == 2 and r.losses[2] == pytest.approx(0.0, abs=1e-6 * r.losses.max())
    assert np.all(r.gradients[:2] < 0) and np.all(r.gradients[3:] > 0)
    assert np.all(np.sign(r.numerical_grads[[0, 4]]) == np.sign(r.gradients[[0, 4]]))
    assert find_zero_crossing(r.values, r.gradients) == pytest.approx(ABSORPTION_M, abs=0.5)


def test_a_non_finite_point_raises(tiny_sim):
    truth, loss_and_grad = _loss_and_grad(tiny_sim)

    def nan_at_the_truth(dp):
        loss, grads = loss_and_grad(dp)
        return jnp.where(jnp.isclose(dp.absorption.absorption_length, ABSORPTION_M), jnp.nan, loss), grads

    sp = SweepParam('Absorption Length', 'absorption_length', half_width=4.0, num_points=3)
    with pytest.raises(FloatingPointError, match="'Absorption Length' = 10.0"):
        sweep_1d(nan_at_the_truth, truth, [sp])
