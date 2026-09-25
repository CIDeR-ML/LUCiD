"""Seed fusion and the margin gate: the two seeding rules in `lucid.fitting`.

`fuse_seeds` encodes a physical claim about WHICH seeder is trustworthy in WHICH direction: time
multilateration is excellent transverse to the track and poor along it, while the charge grid is
longitudinally unbiased and gives the better direction. The fused seed must therefore take the
transverse part from the time seed and the longitudinal part from the charge seed. Getting the
decomposition backwards gives a seed that looks reasonable and is systematically worse, so the
tests check the decomposition itself, not the output shape.
"""
import numpy as np
import pytest

from lucid.fitting import fuse_seeds, pick_by_margin, vec9_dir, vec9_from_track

TRUE_VTX = np.array([1.0, 2.0, -0.5])
TRUE_DIR = np.array([0.6, 0.0, 0.8])                 # unit
TRUE = vec9_from_track(500.0, TRUE_VTX, TRUE_DIR, t0=3.0)


def _offset(base, along=0.0, across=0.0, dE=0.0, dt0=0.0):
    """A seed displaced from truth by given longitudinal and transverse amounts."""
    perp = np.cross(TRUE_DIR, [0.0, 1.0, 0.0])
    perp = perp / np.linalg.norm(perp)
    vtx = TRUE_VTX + along * TRUE_DIR + across * perp
    return vec9_from_track(base[0] + dE, vtx, vec9_dir(base), t0=base[8] + dt0)


class TestFuseSeeds:
    def test_it_takes_transverse_from_b_and_longitudinal_from_a(self):
        """The whole point of fusing, stated as a measurement rather than a shape check.

        seed A is put 2 m off ACROSS the track and right along it; seed B the reverse. The fused
        vertex must inherit A's (good) longitudinal placement and B's (good) transverse one, so it
        should sit essentially at truth even though NEITHER input does.
        """
        a = _offset(TRUE, along=0.0, across=2.0)     # charge grid: longitudinally right
        b = _offset(TRUE, along=1.5, across=0.0)     # time seed: transversely right
        f = fuse_seeds(a, b)

        d = vec9_dir(a)
        err = f[1:4] - TRUE_VTX
        lon = float(err @ d)
        tra = float(np.linalg.norm(err - lon * d))
        assert abs(lon) < 1e-9, f'longitudinal error {lon:.3f} m — A''s good component was lost'
        assert tra < 1e-9, f'transverse error {tra:.3f} m — B''s good component was lost'
        # and neither input was already this good
        assert np.linalg.norm(a[1:4] - TRUE_VTX) > 1.9
        assert np.linalg.norm(b[1:4] - TRUE_VTX) > 1.4

    def test_swapping_the_arguments_is_not_the_same(self):
        """The roles are asymmetric. If this passed, the decomposition would be doing nothing."""
        a = _offset(TRUE, across=2.0)
        b = _offset(TRUE, along=1.5)
        assert np.abs(fuse_seeds(a, b)[1:4] - fuse_seeds(b, a)[1:4]).max() > 0.5

    def test_direction_and_energy_come_from_a(self):
        a = _offset(TRUE, across=2.0, dE=120.0)
        b = _offset(TRUE, along=1.5, dE=-300.0)
        f = fuse_seeds(a, b)
        assert f[0] == a[0]
        np.testing.assert_allclose(f[4:8], a[4:8], rtol=0, atol=0)

    @pytest.mark.parametrize('mode,expect', [('avg', 2.0), ('A', 1.0), ('B', 3.0)])
    def test_t0_modes(self, mode, expect):
        a = _offset(TRUE, dt0=1.0 - TRUE[8])
        b = _offset(TRUE, dt0=3.0 - TRUE[8])
        assert fuse_seeds(a, b, t0_mode=mode)[8] == pytest.approx(expect)

    def test_two_identical_seeds_fuse_to_themselves(self):
        a = _offset(TRUE, along=0.7, across=0.3)
        np.testing.assert_allclose(fuse_seeds(a, a), a, rtol=0, atol=1e-12)

    def test_it_uses_no_truth(self):
        """Signature check with teeth: fusing depends only on the two seeds."""
        import inspect
        params = list(inspect.signature(fuse_seeds).parameters)
        assert params == ['seed_a', 'seed_b', 't0_mode']


class TestPickByMargin:
    def test_the_preferred_seed_is_kept_on_a_tie(self):
        assert pick_by_margin([1.0, 1.0], prefer=0, margin=0.01) == 0

    def test_a_marginal_win_does_not_switch(self):
        """0.5% better, against a 1% margin — the case the gate exists for."""
        assert pick_by_margin([1.0, 0.995], prefer=0, margin=0.01) == 0

    def test_a_decisive_win_does_switch(self):
        assert pick_by_margin([1.0, 0.5], prefer=0, margin=0.01) == 1

    def test_the_best_decisive_winner_is_chosen(self):
        assert pick_by_margin([1.0, 0.6, 0.4], prefer=0, margin=0.01) == 2

    def test_margin_zero_is_plain_argmin_among_challengers(self):
        assert pick_by_margin([1.0, 0.999], prefer=0, margin=0.0) == 1

    def test_a_negative_base_loss_still_gates_in_the_right_direction(self):
        """The threshold is `base - margin*|base|`, so it must tighten for negative losses too —
        using `base*(1-margin)` instead would LOOSEN it and silently invert the gate."""
        assert pick_by_margin([-1.0, -1.005], prefer=0, margin=0.01) == 0
        assert pick_by_margin([-1.0, -1.5], prefer=0, margin=0.01) == 1

    def test_prefer_can_be_any_index(self):
        assert pick_by_margin([0.5, 1.0], prefer=1, margin=0.01) == 0
        assert pick_by_margin([0.999, 1.0], prefer=1, margin=0.01) == 1


def test_it_is_the_rule_fit_track_multistart_applies():
    """`fit_track_multistart` must CALL `pick_by_margin` rather than restate the rule, so the two
    cannot drift. Asserted on the source."""
    import inspect
    from lucid.fitting import fit_track_multistart
    src = inspect.getsource(fit_track_multistart)
    assert 'pick_by_margin(' in src
    assert 'margin * abs(' not in src, 'the margin rule is written out again inside multistart'


# --------------------------------------------------------------------------------------------
# `pipeline.seed_errors` and `pipeline.seed_event` compose the library's primitives; nothing
# else gates them, so each is checked against the inline implementation it replaced.
# --------------------------------------------------------------------------------------------

def _seed_errors_before_consolidation(seed, th9, d):
    """Verbatim, as it stood in pipeline.py, with its own copy of vec9_dir."""
    seed = np.asarray(seed, float)
    th9 = np.asarray(th9, float)
    v = seed
    st, ct, sp, cp = v[4], v[5], v[6], v[7]
    nt, npp = np.hypot(st, ct), np.hypot(sp, cp)
    sdir = np.array([st / nt * cp / npp, st / nt * sp / npp, ct / nt])
    dv = seed[1:4] - th9[1:4]
    lon = float(np.dot(dv, d))
    tra = float(np.linalg.norm(dv - lon * d))
    ddeg = float(np.degrees(np.arccos(np.clip(sdir @ d, -1, 1))))
    return np.array([np.linalg.norm(dv) * 100, tra * 100, lon * 100,
                     ddeg, float(seed[0] - th9[0]), float(seed[8] - th9[8])])


def test_seed_errors_still_reports_what_it_did_before():
    """`seed_errors` matches the inline reference except for the library's `+1e-12` norm guard.

    `vertex_residual` and `angular_error_deg` add that guard against a zero-norm direction; on a
    unit direction it moves the centimetre-valued outputs by ~1e-8. The test asserts it is the
    ONLY difference and cannot grow.
    """
    from analysis.paper.utils.pipeline import seed_errors
    rng = np.random.default_rng(0)
    scale = np.array([500, 3, 3, 3, 1, 1, 1, 1, 5])
    worst = 0.0
    for _ in range(500):
        seed = rng.standard_normal(9) * scale
        th9 = rng.standard_normal(9) * scale
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        worst = max(worst, float(np.abs(_seed_errors_before_consolidation(seed, th9, d)
                                        - seed_errors(seed, th9, d)).max()))
    assert worst < 1e-7, f'seed_errors moved by {worst:.3e}, far beyond the 1e-12 norm guard'
    assert worst > 0.0, 'expected the norm guard to show; if it vanished, check what changed'


def _pick_before_consolidation(lossA, lossB, lossF, sel_margin):
    """Verbatim, as `pipeline.seed_event` computed the two picks before the rewrite."""
    thr = lossA - sel_margin * abs(lossA)
    pick_gated = 1 if lossB < thr else 0
    losses3 = [lossA, lossB, lossF]
    cand = [i for i in (1, 2) if losses3[i] < thr]
    pick3 = min(cand, key=lambda i: losses3[i]) if cand else 0
    return pick_gated, pick3


def test_the_seed_event_pick_rewrite_is_equivalent():
    """`pick_by_margin` reproduces `seed_event`'s inline 2-way (A vs B) and 3-way (A vs B vs
    fused) picks, negative losses included.

    `seed_event` needs PhotonSim ROOT input and trained SIREN weights, neither of which ships with
    the repo, so the call is checked against the code it replaced over random loss triples.
    """
    rng = np.random.default_rng(11)
    for _ in range(20000):
        scale = 10.0 ** rng.uniform(-3, 3)
        a, b, f = (rng.standard_normal(3) * scale)
        margin = float(rng.choice([0.0, 0.001, 0.01, 0.1]))
        want_g, want_3 = _pick_before_consolidation(a, b, f, margin)
        got_g = pick_by_margin([a, b], prefer=0, margin=margin)
        got_3 = pick_by_margin([a, b, f], prefer=0, margin=margin)
        assert (got_g, got_3) == (want_g, want_3), (
            f'losses {(a, b, f)} margin {margin}: got {(got_g, got_3)}, want {(want_g, want_3)}')
