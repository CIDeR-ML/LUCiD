"""Fast, deterministic tests for the GN+Schur+Fisher core (lucid.fitting).

Uses a TOY per-sensor charge model M(theta) (deterministic — the SourceModel forward
ignores the noise keys) so the linear-algebra machinery (cached-J Gauss-Newton, the
gauge-constrained per-PMT Schur complement, the ridge inverse, and the Fisher/CRB) is
exercised without the heavy photon forward.
"""
import numpy as np
import jax.numpy as jnp

from lucid.fitting import fit, crb, SourceModel
from lucid.fitting.gauss_newton import make_constrained_schur, ridge_inverse


# Toy log-linear charge model: log M_src[s] = A_src[s] · theta  → M = exp(A theta), so
# √(k·M) is smooth in (theta, log k). TWO sources with DIFFERENT per-sensor response
# matrices share ONE per-PMT k — this is what breaks the global↔per-PMT-k degeneracy
# (single source = degenerate, since a free per-sensor k absorbs any pattern; the shared
# k across diverse sources over-determines and pins theta — the source-diversity result).
NS, NP = 40, 4
_rngA = np.random.default_rng(7)
_A1 = np.linspace(0.2, 1.0, NS)[:, None] * np.cos(np.arange(NP)[None, :] + 1.0)
_A2 = _rngA.standard_normal((NS, NP)) * 0.6        # a structurally different source
THETA_TRUE = np.array([0.3, -0.5, 0.8, -0.2])
_rng = np.random.default_rng(0)
K_TRUE = np.exp(0.15 * _rng.standard_normal(NS)); K_TRUE /= np.exp(np.mean(np.log(K_TRUE)))  # gauge mean(log k)=0


def _make_toy_forward(A):
    def fwd(theta, ek, pk):
        return jnp.exp(jnp.asarray(A) @ theta)     # (NS,) mean charge at k=1; deterministic
    return fwd


def _truth_charges():
    return [np.asarray(K_TRUE) * np.array(_make_toy_forward(A)(jnp.asarray(THETA_TRUE), None, None))
            for A in (_A1, _A2)]


def _toy_forward(theta, ek, pk):
    return jnp.exp(jnp.asarray(_A1) @ theta)


class TestSchurAndRidge:
    def test_constrained_schur_kills_constant_mode(self):
        """Minv applied to a constant vector returns ~0 (the gauged sum-of-log-k mode)."""
        Hkk = np.linspace(1.0, 3.0, NS)
        Minv = make_constrained_schur(Hkk)
        out = Minv(np.ones(NS))
        assert abs(float(np.sum(out))) < 1e-9   # constant mode projected out

    def test_ridge_inverse_spd(self):
        H = _A1.T @ _A1
        Hinv = ridge_inverse(H, ridge=0.02, mu=0.3)
        ev = np.linalg.eigvalsh(Hinv)
        assert np.all(ev > 0)   # damped inverse is SPD


class TestGaussNewtonRecovery:
    def test_recovers_toy_globals_and_k_two_sources(self):
        srcs = [SourceModel(_make_toy_forward(_A1)), SourceModel(_make_toy_forward(_A2))]
        truth = _truth_charges()
        start = THETA_TRUE + np.array([0.4, -0.3, 0.5, 0.3])   # perturbed init
        res = fit(srcs, truth, start, NS, steps=200, refresh=5,
                  nb_h=1, nb_r=1, step_max=0.3, ridge=1e-4, mu=0.02)
        # globals recovered (theta is the toy's linear param == log_theta in the fitter)
        frac = np.abs(res['log_theta'] - THETA_TRUE)
        assert frac.max() < 0.05, f"global recovery off: {res['log_theta']} vs {THETA_TRUE}"
        # per-PMT k recovered up to the mean(log k)=0 gauge
        k_ratio = res['k'] / K_TRUE
        k_ratio /= np.exp(np.mean(np.log(k_ratio)))
        assert np.std(np.log(k_ratio)) < 0.05

    def test_single_source_is_degenerate(self):
        """One source + free per-PMT k → global θ is NOT identifiable (degenerate);
        documents why source diversity is required."""
        src = SourceModel(_make_toy_forward(_A1))
        truth = [_truth_charges()[0]]
        start = THETA_TRUE + np.array([0.4, -0.3, 0.5, 0.3])
        res = fit([src], truth, start, NS, steps=120, refresh=5,
                  nb_h=1, nb_r=1, step_max=0.3, ridge=1e-3, mu=0.05)
        # converges (residual minimised) but to a WRONG theta — the degeneracy
        assert np.abs(res['log_theta'] - THETA_TRUE).max() > 0.1


class TestStabilizers:
    def test_bake_k_recovers_globals_two_sources(self):
        """Closed-form k=ΣQ/ΣM bake (GN on globals only) recovers θ + k on diverse sources."""
        srcs = [SourceModel(_make_toy_forward(_A1)), SourceModel(_make_toy_forward(_A2))]
        truth = _truth_charges()
        start = THETA_TRUE + np.array([0.4, -0.3, 0.5, 0.3])
        res = fit(srcs, truth, start, NS, steps=200, refresh=5, nb_h=1,
                  step_max=0.3, ridge=1e-4, mu=0.02, bake_k=True)
        assert np.abs(res['log_theta'] - THETA_TRUE).max() < 0.05
        kr = res['k'] / K_TRUE; kr /= np.exp(np.mean(np.log(kr)))
        assert np.std(np.log(kr)) < 0.05

    def test_polyak_returns_averaged_iterate(self):
        srcs = [SourceModel(_make_toy_forward(_A1)), SourceModel(_make_toy_forward(_A2))]
        truth = _truth_charges()
        start = THETA_TRUE + np.array([0.4, -0.3, 0.5, 0.3])
        res = fit(srcs, truth, start, NS, steps=200, refresh=5, nb_h=1,
                  step_max=0.3, ridge=1e-4, mu=0.02, polyak=20)
        assert np.abs(res['log_theta'] - THETA_TRUE).max() < 0.05


class TestFisherCRB:
    def test_crb_positive_and_honesty_factor(self):
        src = SourceModel(_toy_forward)
        c_raw = crb([src], THETA_TRUE, NS, nb_h=1, honesty=1.0)
        c = crb([src], THETA_TRUE, NS, nb_h=1)   # default √12
        assert np.all(c['sigma'] > 0)
        # honesty inflates sigma by exactly √12
        np.testing.assert_allclose(c['sigma'] / c_raw['sigma'], np.sqrt(12.0), rtol=1e-6)
