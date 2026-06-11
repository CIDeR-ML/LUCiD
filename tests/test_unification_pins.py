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


def test_first_arrival_window_nll_pin_and_amp_detach():
    """The ported order-statistic time loss: value pin + the AMP_DETACH gradient routing.

    Synthetic 2-PMT event (3 photons on PMT0, 2 on PMT1; PMT1 unlit in the data). The time
    term's gradient MUST flow only through predicted times — zero through the photon weights
    and the per-PMT total μ (charge owns counts, time owns geometry). CPU-deterministic.
    """
    import jax
    import jax.numpy as jnp
    from lucid.losses import first_arrival_window_nll
    ND = 2
    fi = jnp.array([0, 0, 0, 1, 1]); ft = jnp.array([10., 11., 12., 20., 21.]); lw = jnp.zeros(5)
    ot = jnp.array([10.5, 20.5]); mu = jnp.array([3., 2.]); oc = jnp.array([3., 0.])
    t = first_arrival_window_nll(lw, ft, fi, ot, mu, oc, ND, sigma=2.5, delta=1.0)
    np.testing.assert_allclose(np.asarray(t), [1.9178959, 0.0], atol=1e-5)   # value pin + unlit→0
    f = lambda a, b, c: jnp.sum(first_arrival_window_nll(a, b, fi, ot, c, oc, ND))
    np.testing.assert_allclose(np.asarray(jax.grad(f, 0)(lw, ft, mu)), 0.0, atol=1e-12)  # ∂/∂weights = 0
    np.testing.assert_allclose(np.asarray(jax.grad(f, 2)(lw, ft, mu)), 0.0, atol=1e-12)  # ∂/∂μ = 0
    assert np.abs(np.asarray(jax.grad(f, 1)(lw, ft, mu))).sum() > 0.1        # ∂/∂times ≠ 0


def test_joint_params_bounds_and_helpers():
    """JointParams (D4 umbrella) + particle_bounds/joint_bounds work with the generic
    normalize/denormalize/mask helpers unchanged."""
    import jax
    import jax.numpy as jnp
    from lucid.detector_params import (
        JointParams, DetectorParams, ParticleParams, particle_bounds, joint_bounds,
        normalize_params, denormalize_params, make_optimization_mask)
    NS = 16
    pmin, pmax = particle_bounds(16.9, 36.2)
    assert isinstance(pmin, ParticleParams)
    np.testing.assert_allclose(np.asarray(pmin.position), [-16.9, -16.9, -18.1])
    np.testing.assert_allclose(np.asarray(pmax.position), [16.9, 16.9, 18.1])
    jmin, jmax = joint_bounds(NS, 16.9, 36.2)
    assert isinstance(jmin, JointParams) and isinstance(jmin.detector, DetectorParams)

    # a sample JointParams + normalize/denormalize round-trip through the generic helpers
    dp = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                                  wall_reflection_rate=.2, sensor_reflection_rate=.2,
                                  absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
    track = ParticleParams.from_cartesian(1050., [1., 2., 3.], [0., 0., 1.], 0.)
    jp = JointParams(dp, track)
    back = denormalize_params(normalize_params(jp, jmin, jmax), jmin, jmax)
    np.testing.assert_allclose(np.asarray(back.particle.position), [1., 2., 3.], rtol=1e-4)
    np.testing.assert_allclose(float(back.detector.scattering.scatter_length), 70., rtol=1e-4)

    # mask selects leaves by name across BOTH sub-pytrees
    mask = make_optimization_mask(jp, {'energy', 'scatter_length'})
    assert bool(mask.particle.energy) is True
    assert bool(mask.detector.scattering.scatter_length) is True
    assert bool(mask.detector.response.qe) is False


def test_no_env_reads_in_lucid_package():
    """Ratchet (B6): the lucid/ package must have ZERO os.environ/os.getenv reads.

    Import-time env reads (the TTS leak, U1) break two-Problem coexistence and make the
    forward non-reproducible. After the de-env there are none; this fails the day one
    re-accretes (route physics/model knobs through DetectorParams/SimConfig/args instead).
    """
    import os
    import re
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lucid')
    pat = re.compile(r'os\.environ|os\.getenv|getenv\(')
    offenders = []
    for dirpath, _, files in os.walk(root):
        if '__pycache__' in dirpath:
            continue
        for fn in files:
            if not fn.endswith('.py'):
                continue
            p = os.path.join(dirpath, fn)
            with open(p) as f:
                for i, line in enumerate(f, 1):
                    if pat.search(line) and not line.lstrip().startswith('#'):
                        offenders.append(f'{p}:{i}: {line.strip()}')
    assert not offenders, 'env reads re-accreted in lucid/:\n' + '\n'.join(offenders)
