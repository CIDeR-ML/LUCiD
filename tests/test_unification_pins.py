"""Byte/value pins gating the unification refactor (MAIN_BRANCH_PLAN §8 test net).

These lock the numerical behavior of the load-bearing primitives so the fit refactor
(U6) and the recon port (U7) cannot silently drift them. Each pin records the exact
reproduction recipe. Fast (CPU), no GPU/forward needed unless noted.
"""
import numpy as np


def test_ridge_inverse_indefinite_pin():
    """ridge_inverse on a FIXED indefinite 4x4 — the damped-inverse core of the GN step.

    Captured 2026-06-11; defaults ridge=0.02, mu=0.3. numpy eigh ⇒ byte-stable on CPU.
    """
    from lucid.fitting.gauss_newton import ridge_inverse
    H = np.array([[2, 1, 0, 0], [1, -3, 1, 0], [0, 1, 5, 2], [0, 0, 2, 1]], float)
    R = ridge_inverse(H)
    expected = np.array([
        [0.45427877, -0.53099229, 0.04921011, 0.00669514],
        [-0.53099229, 3.17969004, -0.36701907, 0.09159118],
        [0.04921011, -0.36701907, 0.34080785, -0.37253675],
        [0.00669514, 0.09159118, -0.37253675, 1.05498724]])
    np.testing.assert_allclose(R, expected, rtol=0, atol=1e-7)
    assert R.shape == (4, 4)
    # symmetric (the metric is symmetrized) and positive-definite (eigen-floor)
    np.testing.assert_allclose(R, R.T, atol=1e-12)
    assert np.linalg.eigvalsh(R).min() > 0


def test_counts_loss_normalize_flag():
    """counts_loss(normalize=) — default preserves the historical ÷Σtrue; False is raw Σnll.

    The raw form is what recon needs (÷total-light kills the energy constraint). Pin the
    exact relationship raw == normalized·(Σtrue+eps) and default==normalize=True.
    """
    import jax.numpy as jnp
    from lucid.losses import counts_loss, poisson_nll
    true = jnp.array([2.0, 0.0, 3.0, 1.0, 5.0])
    pred = jnp.array([1.5, 0.5, 2.0, 1.2, 4.4])
    eps = 1e-8
    norm = float(counts_loss(true, pred, normalize=True))
    raw = float(counts_loss(true, pred, normalize=False))
    default = float(counts_loss(true, pred))
    assert default == norm                                   # default unchanged
    assert float(poisson_nll(true, pred)) == norm            # alias unchanged
    np.testing.assert_allclose(raw, norm * (float(true.sum()) + eps), rtol=1e-6)
    assert raw > norm                                        # Σtrue=11 > 1 ⇒ raw larger
