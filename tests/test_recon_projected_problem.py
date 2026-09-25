"""The projected reconstruction fit, moved onto the shared loop without changing what it computes.

``analysis/paper/utils/pipeline.py`` carried a hand-written damped Gauss-Newton loop —
`_fit_track_projected`, the fourth copy in the tree — whose only genuine difference from
``fit_track`` was a projector applied to the time term. A projector is a property of the PROBLEM,
so the copy came out as :class:`lucid.fitting.recon.ProjectedReconProblem` and the loop is now the
shared one.

This file is the gate for that move. It transcribes the ORIGINAL loop verbatim as a reference
implementation and requires the library composition to reproduce it exactly, on a problem posed
analytically — no simulator, no detector, no SIREN weights, so it runs in CI, which is where the
tripwire's lesson says a gate has to run.

The reference is kept in this file on purpose. The original will be deleted from `pipeline.py`, and
a gate that compares the new code against nothing is not a gate; a gate that compares it against a
copy of the old code, written down where a reviewer can read both, is.

Design notes, each of which is a way this could have been useless:

* the model is posed in the same SCALE9-scaled spirit as ``tests/_recon_analytic_model.py``: the projector is
  built in scaled coordinates, so a problem posed directly in theta-space would exercise a
  projector pointing somewhere else;
* the Fishers are PSD by construction. The original damps with ``lam * np.diag(Fs)`` UNCLIPPED
  while :func:`lucid.fitting.transforms.damped_matrix` clips the Marquardt diagonal at zero — identical
  for any PSD metric, which is what a Gauss-Newton Fisher is, and the difference would otherwise
  be attributed to the refactor rather than to the damping convention;
* the charge and time Fishers DIFFER, and the time one is deliberately near-singular along the
  soft direction. If they were equal, or if the time term were well conditioned in that direction,
  the projector would have almost nothing to do and the comparison would pass with it removed —
  ``test_the_projector_actually_bites`` is the guard against exactly that.
"""
import numpy as np
import pytest

from lucid.fitting.gn import gauss_newton
from lucid.fitting.recon import ProjectedReconProblem, SCALE9

NITERS, REFRESH, POLYAK = 24, 8, 6
LR, LR_FINAL, LAM, RIDGE_I, TRUST = 1.2, 0.6, 0.01, 0.1, 3.0
SOFT_LEN = 0.285

_THETA_STAR = np.array([1000.0, 0.5, -0.3, 2.0, 0.4, -0.9, 0.3, 0.15, 5.0])
_START = np.array([1400.0, 1.1, 0.4, 1.2, 0.2, -0.6, 0.5, -0.1, 8.0])


def _psd(seed, n=9, cond=30.0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.logspace(0, -np.log10(cond), n)
    return q @ np.diag(ev) @ q.T


_FQ = _psd(1)
# Time Fisher: strong across the soft direction, nearly flat along it. Built in scaled
# coordinates and mapped back, so the flat direction really is the one the projector removes.
def _ft_matrix():
    th = _THETA_STAR
    st, ct, sp, cp = th[4], th[5], th[6], th[7]
    u = np.array([st / np.hypot(st, ct) * cp / np.hypot(sp, cp),
                  st / np.hypot(st, ct) * sp / np.hypot(sp, cp),
                  ct / np.hypot(st, ct)])
    v = np.zeros(9)
    v[1:4] = SOFT_LEN * u
    v[8] = 1.0
    vs = v / SCALE9
    vs = vs / np.linalg.norm(vs)
    base = _psd(2)
    soft = np.outer(vs, vs)
    return base - 0.98 * (soft @ base @ soft)      # nearly no curvature along vs


_FT = _ft_matrix()


def _grads(theta):
    d = np.asarray(theta, float) - _THETA_STAR
    return _FQ @ (d / SCALE9) / SCALE9, _FT @ (d / SCALE9) / SCALE9


def _fishers(theta):
    return _FQ / np.outer(SCALE9, SCALE9), _FT / np.outer(SCALE9, SCALE9)


def _soft_P(th):
    """The original's projector, transcribed from pipeline.py."""
    st, ct, sp, cp = th[4], th[5], th[6], th[7]
    nt = np.hypot(st, ct)
    npp = np.hypot(sp, cp)
    stn, ctn, spn, cpn = st / nt, ct / nt, sp / npp, cp / npp
    u = np.array([stn * cpn, stn * spn, ctn])
    v = np.zeros(9)
    v[1:4] = SOFT_LEN * u
    v[8] = 1.0
    vs = v / SCALE9
    vs = vs / np.linalg.norm(vs)
    return np.eye(9) - np.outer(vs, vs)


def _original(start, niters=NITERS, refresh=REFRESH, polyak_w=POLYAK,
              lr=LR, lr_final=LR_FINAL, lam=LAM, ridge_i=RIDGE_I, trust=TRUST):
    """`_fit_track_projected` as it stood, verbatim but for the injected grads/fishers."""
    S = SCALE9
    th = np.asarray(start, float)
    gq, gt = _grads(th)
    traj = [th.copy()]
    gnorms = []
    fq = ft = None
    since = 0
    for it in range(niters):
        if fq is None or since >= refresh:
            fq, ft = _fishers(th)
            since = 0
        since += 1
        p = _soft_P(th)
        gs = S * gq + p @ (S * gt)
        fs = (S[:, None] * fq * S[None, :]) + p @ (S[:, None] * ft * S[None, :]) @ p
        marq = np.diag(lam * np.diag(fs))
        r_i = ridge_i * np.median(np.clip(np.diag(fs), 1e-12, None)) * np.eye(9)
        lr_it = lr + (lr_final - lr) * (it / max(1, niters - 1))
        du = -lr_it * np.linalg.solve(fs + marq + r_i + 1e-9 * np.eye(9), gs)
        du = np.clip(du, -trust, trust)
        th_new = th + S * du
        gq_n, gt_n = _grads(th_new)
        if (np.isfinite(th_new).all() and np.isfinite(gq_n).all() and np.isfinite(gt_n).all()):
            th, gq, gt = th_new, gq_n, gt_n
        gnorms.append(float(np.linalg.norm(S * (gq + gt))))
        traj.append(th.copy())
    out = np.mean(np.array(traj)[-polyak_w:], axis=0)
    return out, np.array(traj)


def _library(start, **kw):
    prob = ProjectedReconProblem(_grads, _fishers, scale=SCALE9, soft_length=SOFT_LEN)
    res = gauss_newton(prob, np.asarray(start, float), kw.get('niters', NITERS),
                       lam=LAM, mu=RIDGE_I, jitter=1e-9, lr=LR, lr_final=LR_FINAL,
                       scale=None, max_step=TRUST, refresh=kw.get('refresh', REFRESH),
                       readout='polyak', polyak=kw.get('polyak_w', POLYAK),
                       reject_nonfinite=True)
    return res['theta'], res['history']


# `_original` is a float64 numpy transcription of the pre-move loop. The library's step is now the
# float32 JAX transformation, so the comparison is at float32 tolerance rather than exact --
# MEASURED at 3.82e-06 on this problem, bounded here at 1e-5.
#
# What the comparison is for has not changed: it asks whether moving the projector into
# `ProjectedReconProblem` preserved the ALGORITHM, and a difference of 4e-06 over 24 steps is the
# arithmetic, not the algorithm. A projector applied in the wrong place, or a Fisher assembled in
# the wrong order, moves the trajectory by far more than this -- which is what
# `test_the_projector_actually_bites` demonstrates by removing it and watching the answer change.
ORIGINAL_RTOL = 1e-5


def test_the_library_reproduces_the_original_trajectory():
    ref_out, ref_traj = _original(_START)
    lib_out, lib_traj = _library(_START)
    np.testing.assert_allclose(lib_traj, ref_traj, rtol=ORIGINAL_RTOL, atol=0,
                               err_msg='trajectory differs from the original loop')
    np.testing.assert_allclose(lib_out, ref_out, rtol=ORIGINAL_RTOL, atol=0,
                               err_msg='Polyak readout differs from the original loop')


@pytest.mark.parametrize('niters,refresh,polyak_w', [(24, 8, 6), (13, 3, 4), (7, 1, 7)])
def test_it_reproduces_across_cadences(niters, refresh, polyak_w):
    """The refresh cadence and the Polyak window are where the two loops are most likely to
    disagree by one, so they are varied rather than pinned at a single convenient value."""
    ref_out, ref_traj = _original(_START, niters=niters, refresh=refresh, polyak_w=polyak_w)
    lib_out, lib_traj = _library(_START, niters=niters, refresh=refresh, polyak_w=polyak_w)
    np.testing.assert_allclose(lib_traj, ref_traj, rtol=ORIGINAL_RTOL, atol=0)
    np.testing.assert_allclose(lib_out, ref_out, rtol=ORIGINAL_RTOL, atol=0)


def test_the_comparison_could_have_failed():
    """The fit must actually move, or exact equality is a statement about two static arrays."""
    _, traj = _original(_START)
    assert np.abs(np.diff(traj, axis=0)).max() > 1e-6
    assert np.abs(traj[-1] - traj[0]).max() > 1e-3


def test_the_projector_actually_bites():
    """Removing the projector must change the answer, or this gates nothing about projection."""
    prob = ProjectedReconProblem(_grads, _fishers, scale=SCALE9, soft_length=SOFT_LEN)
    unprojected = ProjectedReconProblem(_grads, _fishers, scale=SCALE9, soft_length=SOFT_LEN)
    unprojected.soft_projector = lambda theta: np.eye(9)
    kw = dict(lam=LAM, mu=RIDGE_I, jitter=1e-9, lr=LR, lr_final=LR_FINAL, scale=None,
              max_step=TRUST, refresh=REFRESH, readout='polyak', polyak=POLYAK,
              reject_nonfinite=True)
    a = gauss_newton(prob, np.asarray(_START, float), NITERS, **kw)['theta']
    b = gauss_newton(unprojected, np.asarray(_START, float), NITERS, **kw)['theta']
    assert np.abs(a - b).max() > 1e-3, 'the projector makes no difference on this problem'


class TestProjector:
    def test_is_an_orthogonal_projector_of_rank_eight(self):
        p = ProjectedReconProblem(_grads, _fishers).soft_projector(_THETA_STAR)
        np.testing.assert_allclose(p, p.T, atol=1e-12)
        np.testing.assert_allclose(p @ p, p, atol=1e-12)
        assert abs(np.trace(p) - 8.0) < 1e-9        # exactly one direction removed

    def test_it_removes_the_soft_direction_and_nothing_else(self):
        prob = ProjectedReconProblem(_grads, _fishers, soft_length=SOFT_LEN)
        p = prob.soft_projector(_THETA_STAR)
        th = _THETA_STAR
        u = np.array([th[4] / np.hypot(th[4], th[5]) * th[7] / np.hypot(th[6], th[7]),
                      th[4] / np.hypot(th[4], th[5]) * th[6] / np.hypot(th[6], th[7]),
                      th[5] / np.hypot(th[4], th[5])])
        v = np.zeros(9)
        v[1:4] = SOFT_LEN * u
        v[8] = 1.0
        vs = v / SCALE9
        np.testing.assert_allclose(p @ vs, 0.0, atol=1e-10)      # the ray is annihilated
        w = np.zeros(9)
        w[0] = 1.0                                               # energy is untouched
        assert np.linalg.norm(p @ w - w) < 0.5

    def test_it_tracks_the_current_direction(self):
        """The degenerate ray points along the track, so the projector must rotate with the fit."""
        prob = ProjectedReconProblem(_grads, _fishers)
        turned = _THETA_STAR.copy()
        turned[4], turned[5] = turned[5], turned[4]
        assert np.abs(prob.soft_projector(_THETA_STAR) - prob.soft_projector(turned)).max() > 1e-3
