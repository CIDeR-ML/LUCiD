"""The JAX damped Gauss-Newton transformation in `lucid.fitting.transforms`.

  CONVENTIONS   the clipped Marquardt diagonal, the filtered Levenberg median, and the refusal to
                invent a base when nothing carries curvature, asserted directly on the values.

  COMPOSITION   chain, scheduled damping, gradient accumulation, jit and vmap. The last three need
                a traceable solve (`optax.MultiSteps` jits internally), which is why the step is
                written in JAX, so they are tested rather than assumed.

Whether a float32 solve is accurate enough at the published conditioning is tested separately in
`tests/test_float32_is_adequate.py`.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

from lucid.fitting.transforms import (
    damped_matrix,
    scale_by_damped_gauss_newton,
    damped_gauss_newton,
)

LAM, MU = 0.01, 0.1


def spd(n, cond, seed=0):
    """Symmetric positive-definite with a PRESCRIBED condition number."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return ((Q * np.logspace(0, np.log10(cond), n)) @ Q.T).astype(np.float32)


def test_the_marquardt_diagonal_is_clipped():
    """A negative curvature entry must contribute NO damping, not negative damping.

    Unclipped, `lam*diag(H)` on a negative entry SUBTRACTS from the damping in exactly the
    direction that most needs it.
    """
    H = np.diag(np.array([-1.0, 2.0, 4.0], dtype=np.float32))
    A = np.asarray(damped_matrix(jnp.asarray(H), lam=LAM, mu=MU))
    base = 3.0                                   # median of the two entries carrying curvature
    assert A[0, 0] == pytest.approx(-1.0 + MU * base, rel=1e-5), \
        'the negative entry received Marquardt damping instead of being clipped out of it'
    assert A[1, 1] == pytest.approx(2.0 + LAM * 2.0 + MU * base, rel=1e-5)


def test_the_levenberg_median_is_filtered_not_floored():
    """The flat direction is EXCLUDED from the median rather than dragging it down.

    Flooring at 1e-12 would give median(1e-12, 2, 4) = 2; filtering gives median(2, 4) = 3. The
    difference is a factor of 1.5 in the isotropic damping on this matrix, and grows to a total
    collapse once more than half the diagonal is flat.
    """
    flat = np.diag(np.array([0.0, 2.0, 4.0], dtype=np.float32))
    A = np.asarray(damped_matrix(jnp.asarray(flat), lam=LAM, mu=MU))
    assert A[0, 0] == pytest.approx(MU * 3.0, rel=1e-5), 'base must be the median of 2 and 4'


def test_nothing_carrying_curvature_yields_NaN_rather_than_a_fabricated_base():
    """With no positive curvature anywhere the Gauss-Newton step is undefined.

    A traced function cannot raise, so it must return NaN rather than a numerically singular `A`
    built from a tiny floor that looks usable.
    """
    Z = np.zeros((3, 3), dtype=np.float32)
    A = np.asarray(damped_matrix(jnp.asarray(Z), lam=LAM, mu=MU))
    assert np.isnan(A).all(), 'an all-flat diagonal must not produce a usable-looking matrix'


@pytest.mark.parametrize('cond', [1e2, 1e3, 6.8e3, 1e5])
def test_the_damped_matrix_is_nonsingular_by_construction(cond):
    """The property the Levenberg term exists for, asserted rather than argued.

    For PSD `H` the isotropic shift moves every eigenvalue by exactly `mu*base`, so the smallest
    eigenvalue is bounded below by it. This is what makes an eigen-floor on top provably inert.
    """
    H = spd(19, cond)
    A = np.asarray(damped_matrix(jnp.asarray(H), lam=LAM, mu=MU))
    dg = np.clip(np.diag(H), 0, None)
    base = float(np.median(dg[dg > 1e-12 * dg.max()]))
    assert np.linalg.eigvalsh(A).min() >= MU * base * 0.99, \
        'the isotropic shift did not lower-bound the spectrum'


def test_the_metric_is_required_and_says_so():
    """A silent identity metric would turn Gauss-Newton into scaled gradient descent."""
    tx = scale_by_damped_gauss_newton(LAM, MU)
    g = jnp.ones(4)
    with pytest.raises(ValueError, match='needs the metric'):
        tx.update(g, tx.init(g), None)


def test_it_is_stateless():
    """No moment buffer means nothing to corrupt when a driver rejects a step."""
    tx = scale_by_damped_gauss_newton(LAM, MU)
    g, H = jnp.ones(4), jnp.eye(4)
    st0 = tx.init(g)
    d1, st1 = tx.update(g, st0, None, metric=H)
    d2, st2 = tx.update(g, st1, None, metric=H)
    np.testing.assert_array_equal(np.asarray(d1), np.asarray(d2))


# --------------------------------------------------------------- composition

def test_it_chains_with_other_transformations():
    """extra_args must survive optax.chain, or the metric cannot reach a composed optimiser."""
    H = spd(9, 1e3, seed=3)
    g = np.random.default_rng(4).standard_normal(9).astype(np.float32)
    plain = damped_gauss_newton(LAM, MU)
    clipped = optax.chain(optax.clip(0.01), scale_by_damped_gauss_newton(LAM, MU),
                          optax.scale_by_learning_rate(1.0))
    a, _ = plain.update(jnp.asarray(g), plain.init(jnp.asarray(g)), None, metric=jnp.asarray(H))
    b, _ = clipped.update(jnp.asarray(g), clipped.init(jnp.asarray(g)), None,
                          metric=jnp.asarray(H))
    assert not np.allclose(np.asarray(a), np.asarray(b)), 'the clip did nothing — chain is inert'


def test_the_damping_itself_can_be_scheduled():
    """`lam` can be driven by an optax schedule; annealing it toward 0 must grow the step."""
    H, g = spd(6, 1e3, seed=5), jnp.ones(6)
    tx = optax.inject_hyperparams(damped_gauss_newton)(
        lam=optax.linear_schedule(1.0, 0.0, 4), mu=MU)
    st = tx.init(g)
    norms = []
    for _ in range(4):
        d, st = tx.update(g, st, None, metric=jnp.asarray(H))
        norms.append(float(jnp.linalg.norm(d)))
    assert norms[-1] > norms[0], f'annealing lam 1.0 -> 0 must grow the step; got {norms}'


def test_gradient_accumulation_works_and_numpy_could_not_do_it():
    """optax.MultiSteps: the lever for a photon budget larger than the card holds.

    MultiSteps jits its inner update, so this needs a traceable solve (`np.linalg.solve` on a
    traced array raises TracerArrayConversionError).
    """
    H, g = spd(6, 1e3, seed=6), jnp.ones(6)
    every = 4
    ms = optax.MultiSteps(damped_gauss_newton(LAM, MU), every_k_schedule=every)
    st = ms.init(g)
    seen = []
    for _ in range(every):
        d, st = ms.update(g, st, None, metric=jnp.asarray(H))
        seen.append(float(jnp.linalg.norm(d)))
    assert seen[-1] > 0.0, 'the accumulated step never fired'
    assert all(s == 0.0 for s in seen[:-1]), f'MultiSteps should hold until step k; got {seen}'


def test_it_jits():
    H, g = spd(6, 1e3, seed=7), jnp.ones(6)
    tx = damped_gauss_newton(LAM, MU)
    st = tx.init(g)

    @jax.jit
    def one(g_, st_, H_):
        return tx.update(g_, st_, None, metric=H_)

    d, _ = one(g, st, jnp.asarray(H))
    ref, _ = tx.update(g, st, None, metric=jnp.asarray(H))
    np.testing.assert_allclose(np.asarray(d), np.asarray(ref), rtol=1e-6, atol=0)


def test_it_vmaps_over_a_batch_of_problems():
    """Batching: many independent fits stepped at once must match stepping each alone."""
    B, P = 5, 6
    Hs = jnp.stack([jnp.asarray(spd(P, 1e3, seed=s)) for s in range(B)])
    gs = jnp.asarray(np.random.default_rng(8).standard_normal((B, P)), dtype=jnp.float32)
    tx = damped_gauss_newton(LAM, MU)
    st = tx.init(gs[0])

    batched = jax.vmap(lambda g_, H_: tx.update(g_, st, None, metric=H_)[0])(gs, Hs)
    for i in range(B):
        one, _ = tx.update(gs[i], st, None, metric=Hs[i])
        np.testing.assert_allclose(np.asarray(batched[i]), np.asarray(one), rtol=1e-6, atol=0)
