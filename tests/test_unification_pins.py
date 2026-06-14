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


def test_seed_vertex_time_recovers_point_source():
    """seed_vertex_time (robust multilateration) recovers a POINT source to ~cm/ns.

    The un-robust port put the seed ~15 m off (scattered/late-photon outliers); the robust
    version (bright preselection + RANSAC inlier grid + GN on inliers) must recover a clean
    point source. CPU, no forward. Pins the robustness, not just a loose bound.
    """
    from lucid.fitting import seed_vertex_time
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 3000); zc = rng.uniform(-18, 18, 3000)
    POS = np.column_stack([16.9 * np.cos(th), 16.9 * np.sin(th), zc])          # cylinder wall PMTs
    vtrue = np.array([5., -3., 7.]); t0true = 4.0; VS = 0.2167
    T = t0true + np.linalg.norm(POS - vtrue, axis=1) / VS
    oc = (rng.uniform(size=len(POS)) < 0.4).astype(float) * 3.                 # ~40% lit, q=3
    ot = np.where(oc > 0, T + rng.normal(0, 0.5, len(POS)), 0.)                # 0.5 ns jitter
    v, t0 = seed_vertex_time(POS, oc, ot)
    assert np.linalg.norm(v - vtrue) * 100 < 5.0                              # <5 cm (not 15 m)
    assert abs(t0 - t0true) < 1.0


def test_fit_track_multistart_margin_arbitration():
    """fit_track_multistart margin gate: prefer seed A; switch to B only if B beats A by margin.

    Plain argmin over-selects the (forward-biased) time seed via SIREN-bias ties; the 1% margin
    restricts the switch to decisive rescues. Pins the decision logic with a stubbed fit_track +
    a controlled model.loss (no forward). The seed id is encoded in start[0]."""
    import jax.numpy as jnp
    import lucid.fitting.recon as R
    orig = R.fit_track
    R.fit_track = lambda model, oc, ot, s, **kw: (np.asarray(s, float), {})    # stub: no fitting
    try:
        class M:
            def __init__(self, losses): self.losses = losses                  # {seed_id: loss}
            def loss(self, t9, oc, ot, key): return self.losses[int(round(float(t9[0])))]
        A = np.zeros(9); B = np.zeros(9); B[0] = 1.                            # ids 0, 1
        oc = jnp.zeros(2); ot = jnp.zeros(2)
        for la, lb, want in [(100., 95., 1),    # B 5% lower -> B (decisive)
                             (100., 99.5, 0),   # B 0.5% lower -> A (within margin)
                             (100., 110., 0)]:  # B higher -> A
            _, info = R.fit_track_multistart(M({0: la, 1: lb}), oc, ot, [A, B], nkeys=1, margin=0.01)
            assert info['which'] == want, (la, lb, want, info['which'])
    finally:
        R.fit_track = orig


def test_fitting_contracts_protocols():
    """B5: the closure-surface contracts are executable (Protocol), not just docstrings.

    The two opaque fitting callables (calibration forward, recon per-photon predictor) are
    typed Protocols so they're grep/pyright/IDE-checkable. Pin that they import, are
    runtime_checkable, and a conforming callable satisfies them."""
    from lucid.fitting import CalibForward, PerPhotonPredictor
    assert callable(getattr(CalibForward, '__instancecheck__', None))        # runtime_checkable
    calib = lambda theta, ek, pk: None
    pred = lambda track, key: (None, None, None, None)
    assert isinstance(calib, CalibForward)
    assert isinstance(pred, PerPhotonPredictor)
    assert not isinstance(42, CalibForward)                                  # non-callable rejected


def test_fitting_analysis_seam():
    """lucid.fitting.analysis — the result-analysis seam (moved inside from 5 inline notebook copies).

    Pin the vertex longitudinal/transverse decomposition (deterministic) + resolution_stats keys.
    """
    from lucid.fitting import resolution_stats, vertex_residual, angular_error_deg
    # vertex 0.3 m along +z, 0.4 m in +x; true dir +z -> lon=0.3, tra=0.4 (m -> ×100 cm by caller)
    lon, tra = vertex_residual([0.4, 0.0, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose([lon, tra], [0.3, 0.4], atol=1e-9)
    np.testing.assert_allclose(angular_error_deg([1, 0, 0], [0, 0, 1]), 90.0, atol=1e-6)
    s = resolution_stats(np.array([1., -2., 3., -4., 5.]))
    assert s['n'] == 5 and set(s) >= {'median', 'mean', 'rms', 'containment', 'median_ci'}
    np.testing.assert_allclose(s['rms'], np.sqrt((np.array([1, 4, 9, 16, 25])).mean()), rtol=1e-9)


def test_no_env_reads_in_lucid_package():
    """Ratchet (B6): the lucid/ FORWARD/PHYSICS path must have ZERO os.environ/os.getenv reads.

    Import-time env reads (the TTS leak, U1) break two-Problem coexistence and make the
    forward non-reproducible. After the de-env there are none; this fails the day one
    re-accretes (route physics/model knobs through DetectorParams/SimConfig/args instead).

    EXEMPT: ``lucid/production/`` — the data-production ORCHESTRATION layer is env-driven
    by design (external-binary paths PHOTONSIM_BIN / GENIE_PREFIX / GENIE_XSEC_FILE; cluster
    scheduling). These are infra config, not forward/physics knobs, and never touch the
    differentiable forward. The ratchet stays strict for simulation/sources/geometry/
    wavelength/fitting/siren/optimization.
    """
    import os
    import re
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lucid')
    _EXEMPT = (os.path.join(root, 'production') + os.sep,)
    pat = re.compile(r'os\.environ|os\.getenv|getenv\(')
    offenders = []
    for dirpath, _, files in os.walk(root):
        if '__pycache__' in dirpath:
            continue
        if any((dirpath + os.sep).startswith(e) for e in _EXEMPT):
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
