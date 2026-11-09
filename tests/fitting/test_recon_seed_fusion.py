"""Seed fusion and the margin gate: the two seeding rules in `lucid.fitting`.

`fuse_seeds` encodes a physical claim about WHICH seeder is trustworthy in WHICH direction: time
multilateration is excellent transverse to the track and poor along it, while the first-light grid
is longitudinally unbiased and gives the better direction. The fused seed must therefore take the
transverse part from the time seed and the longitudinal part from the grid seed. Getting the
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
        a = _offset(TRUE, along=0.0, across=2.0)     # first-light grid: longitudinally right
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

    def test_t0_is_the_mean(self):
        a = _offset(TRUE, dt0=1.0 - TRUE[8])
        b = _offset(TRUE, dt0=3.0 - TRUE[8])
        assert fuse_seeds(a, b)[8] == pytest.approx(2.0)

    def test_two_identical_seeds_fuse_to_themselves(self):
        a = _offset(TRUE, along=0.7, across=0.3)
        np.testing.assert_allclose(fuse_seeds(a, a), a, rtol=0, atol=1e-12)

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
