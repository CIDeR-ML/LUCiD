"""Data pipeline of the 2-D loss-geometry calibration figure.

`compute_2d_neyman.py` -> `combine_2d.py` -> `calib_plots.loss_geometry`. The published surface is
computed in shards, each filling its own round-robin slice (`idx % NSHARDS == SHARD`) and leaving
zeros elsewhere; `combine_2d` then SUMS the partials. That is exact only while the slices are
disjoint: a point claimed by two shards would be silently doubled. So sharded must equal
unsharded, exactly.

Not asserted: "the minimum sits at truth". Truth data and model use independent key streams, so
the minimum displaces by residual noise that grows as the photon budget shrinks; at this test's
N_PH it can sit in a grid corner. The properties below are budget-independent instead.

Needs no downloaded data (laser sources and in-repo geometry, not the SIREN emitter). Runs three
subprocesses on a reduced grid; compile and detector construction dominate, not the grid points.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
COMPUTE = REPO / 'analysis' / 'paper' / 'utils' / 'compute_2d_neyman.py'
COMBINE = REPO / 'analysis' / 'paper' / 'utils' / 'combine_2d.py'
FIGURE = 'calib_loss_geometry'

# Reduced from LANDSCAPE_RECIPE: enough grid to have interior points and a Fisher at each, cheap
# enough to run. Every knob the path reads is pinned so nothing leaks in from the environment.
SETTINGS = {'NGRID': '5', 'HWF': '0.4', 'K': '3', 'N_PH': '1e5',
            'NK_TRUTH': '2', 'NK_SIM': '2',
            'INTEN': '1e8', 'NCAP': '100', 'NANG': '150', 'NHGT': '100'}
FIELDS = ('L', 'Gx', 'Gy', 'Fxx', 'Fxy', 'Fyy')


def _run(script, out_dir, **extra):
    env = {**os.environ, **SETTINGS, 'LUCID_PAPER_OUTPUT': str(out_dir), **extra}
    r = subprocess.run([sys.executable, str(script)], env=env, cwd=str(REPO),
                       capture_output=True, text=True)
    assert r.returncode == 0, f'{script.name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}'
    return r


@pytest.fixture(scope='module')
def surfaces(tmp_path_factory):
    """The same grid computed twice: in one process, and split across two shards then combined."""
    whole_dir = tmp_path_factory.mktemp('whole')
    _run(COMPUTE, whole_dir, NSHARDS='1', SHARD_ID='0')
    whole = dict(np.load(whole_dir / 'data' / FIGURE / 'landscape2d_neyman.npz'))

    shard_dir = tmp_path_factory.mktemp('sharded')
    parts = []
    for s in ('0', '1'):
        _run(COMPUTE, shard_dir, NSHARDS='2', SHARD_ID=s)
        parts.append(dict(np.load(shard_dir / 'data' / FIGURE / f'landscape2d_neyman_part{s}.npz')))
    _run(COMBINE, shard_dir)
    combined = dict(np.load(shard_dir / 'data' / FIGURE / 'landscape2d_neyman.npz'))
    return whole, combined, parts


class TestSharding:
    def test_sharded_equals_unsharded_exactly(self, surfaces):
        """The published surface is built from 10 shards. If that is not identical to computing
        it in one process, the figure is an artifact of how it was parallelised."""
        whole, combined, _ = surfaces
        for f in FIELDS:
            np.testing.assert_allclose(combined[f], whole[f], rtol=0, atol=0,
                                       err_msg=f'{f}: sharded result differs from unsharded')

    def test_the_shards_cover_every_point_exactly_once(self, surfaces):
        """`combine_2d` SUMS the partials, so overlap would double-count silently.

        Checked on the Fisher diagonal rather than the loss: Fxx is strictly positive wherever a
        point was computed, so its nonzero pattern IS the shard's support.
        """
        _, _, parts = surfaces
        support = [np.asarray(p['Fxx']) != 0.0 for p in parts]
        overlap = support[0] & support[1]
        assert not overlap.any(), f'{overlap.sum()} point(s) computed by BOTH shards'
        covered = support[0] | support[1]
        assert covered.all(), f'{(~covered).sum()} point(s) computed by NEITHER shard'

    def test_each_shard_did_a_fair_share(self, surfaces):
        """Round-robin over 25 points and 2 shards is 13/12 — a shard doing all or none would
        still satisfy disjoint-and-complete."""
        _, _, parts = surfaces
        counts = sorted(int((np.asarray(p['Fxx']) != 0.0).sum()) for p in parts)
        assert counts == [12, 13], f'shard point counts {counts}, expected [12, 13]'


class TestTheSurface:
    def test_the_fisher_is_positive_semidefinite_everywhere(self, surfaces):
        """F = 2 sum J J / Q is PSD by construction, so this catches a sign error or a transposed
        einsum rather than a physics mistake — which is exactly the failure a refactor introduces."""
        whole, _, _ = surfaces
        fxx, fxy, fyy = (np.asarray(whole[k], float) for k in ('Fxx', 'Fxy', 'Fyy'))
        assert (fxx >= 0).all(), 'Fxx has negative entries'
        assert (fyy >= 0).all(), 'Fyy has negative entries'
        det = fxx * fyy - fxy ** 2
        assert (det >= -1e-6 * np.abs(fxx * fyy).max()).all(), \
            'the 2x2 Fisher is indefinite at some grid point'
        assert (fxx > 0).all() and (fyy > 0).all(), 'a direction has no curvature at all'

    def test_the_ad_gradient_agrees_with_the_surface_it_belongs_to(self, surfaces):
        """`L` is evaluated and `Gx, Gy` come from AD — two independent computations of the same
        thing, so finite-differencing the surface must reproduce the gradient's SIGN.

        Sign, not magnitude: the grid spans +-40% of truth in five points, far too coarse for a
        finite difference to match a derivative numerically. A swapped or negated gradient — the
        realistic refactor error — inverts the sign everywhere.
        """
        whole, _, _ = surfaces
        L = np.asarray(whole['L'], float)
        x, y = np.asarray(whole['X'], float), np.asarray(whole['Y'], float)
        fd_x, fd_y = np.gradient(L, x, y)
        gx, gy = np.asarray(whole['Gx'], float), np.asarray(whole['Gy'], float)
        inner = (slice(1, -1), slice(1, -1))
        assert np.sign(fd_x[inner]) == pytest.approx(np.sign(gx[inner])), \
            'dL/dX from AD disagrees in sign with the surface'
        assert np.sign(fd_y[inner]) == pytest.approx(np.sign(gy[inner])), \
            'dL/dY from AD disagrees in sign with the surface'

    def test_the_surface_is_not_degenerate(self, surfaces):
        """Every assertion above would pass on a constant surface."""
        whole, _, _ = surfaces
        L = np.asarray(whole['L'], float)
        assert np.isfinite(L).all()
        assert L.min() > 0, 'a Neyman chi-square went non-positive'
        assert L.max() / L.min() > 1.2, 'the surface is nearly flat over +-40% of truth'

    def test_the_grid_is_centred_on_truth(self, surfaces):
        """The axes are built as truth*(1 +- HWF), so truth must sit at the centre — this is what
        makes the figure's axes mean what the caption says, independent of where the MINIMUM is."""
        whole, _, _ = surfaces
        x, y = np.asarray(whole['X'], float), np.asarray(whole['Y'], float)
        assert x[len(x) // 2] == pytest.approx(float(whole['truth_x']), rel=1e-6)
        assert y[len(y) // 2] == pytest.approx(float(whole['truth_y']), rel=1e-6)
