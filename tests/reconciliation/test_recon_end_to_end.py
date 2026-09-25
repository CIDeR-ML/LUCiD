"""The reconstruction chain, executed. Not a resolution measurement — a wiring gate.

Everything on the live recon path is verified as a FUNCTION: `fuse_seeds` was moved verbatim,
`seed_errors` matches its predecessor to 9.7e-09, `pick_by_margin` matches over 20,000 loss
triples, `ProjectedReconProblem` reproduces the loop it replaced to 1e-5. None of that executes
the CHAIN — `_prepare_event` -> data simulation -> both seeders -> `fuse_seeds` -> the margin pick
-> `fit_track_multistart` -> reported errors — where a wrong argument order, a renamed dict key or
a `None` slipping through would live.

That gap was real: during the restructure, three functions on this path were changed and the only
thing that ever ran the chain was a scratch script, once, by hand.

WHAT THIS ASSERTS is structure and sanity, not physics. The settings below are deliberately far
below the published working point (5k rays against 250k, 8 iterations against 150, one key against
eight), because the point is to reach every line, not to converge. A reconstruction from this
configuration is EXPECTED to be poor; asserting a good one would make the test fail for a reason
that has nothing to do with correctness. The published performance is measured elsewhere, on GPU,
at the real working point: vertex 12.60 cm and direction 0.89 deg over 15 events, against the
documented ~12 cm / ~1.0 deg.

COST: roughly 5-10 minutes, CPU-only. `conftest.py` hides the GPU before importing jax — a wedged
NVIDIA driver once put the whole suite into uninterruptible sleep — so this cannot use the card
even when one is present. That is the price of the only test that runs this path at all.

DATA: needs the trained SIREN emitter and a PhotonSim ROOT, neither of which is in the checkout
(`data/` is gitignored, so every worktree lacks them). It skips when they are absent, and
`conftest.py` prints a banner naming what was skipped rather than letting the count pass quietly.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
_SIREN = REPO / 'data' / 'water' / 'muon' / 'siren_training' / 'trained_model'
_ROOT = REPO / 'data' / 'water' / 'muon' / '1000MeV_100events.root'

requires_data = pytest.mark.skipif(
    not (_SIREN.is_dir() and _ROOT.exists()),
    reason=f'trained SIREN emitter or PhotonSim ROOT not present ({_SIREN}, {_ROOT}) '
           f'— run ./scripts/download_data.sh')


@pytest.fixture(scope='module')
def record(tmp_path_factory):
    """One reconstructed event, at the cheapest settings that still reach every line."""
    from analysis.paper.utils import studies
    from analysis.paper.utils.run import run_local
    import h5py

    cfg = studies.base_config(particle='muon', energy_mev=1000, n_rays=5000, n_events=1,
                              root_file=str(_ROOT), name='recon_e2e_gate')
    # Not the published recipe: 8 iterations and one key, because this gate is about reaching the
    # code, not converging. See the module docstring.
    cfg['gn'] = {**cfg['gn'], 'niters': 8, 'nkeys': 1}
    out = run_local(cfg, tmp_path_factory.mktemp('recon'), events=[0], verbose=False)
    with h5py.File(out, 'r') as h5:
        grp = h5['events/ev0000']
        rec = {k: np.asarray(grp[k]) for k in grp}
        rec.update({k: grp.attrs[k] for k in grp.attrs})
    return rec


@requires_data
class TestTheChainRuns:
    def test_it_reports_every_field_the_figures_read(self, record):
        """A renamed or dropped key breaks the plotting code, not this pipeline — so it is
        asserted here, where it is cheap to see."""
        for k in ('fit_err', 'truth_vec9', 'fit_vec9', 'seedA_vec9', 'seedB_vec9', 'seedF_vec9'):
            assert k in record, f'{k} missing from the reconstruction record'

    def test_both_seeders_produced_a_distinct_seed(self, record):
        """The charge grid and the time multilateration must not be returning the same thing —
        if they were, `fuse_seeds` and the margin gate would both be no-ops."""
        a, b = record['seedA_vec9'], record['seedB_vec9']
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.abs(a - b).max() > 1e-6, 'the two seeders returned the same seed'

    def test_the_fused_seed_takes_from_both(self, record):
        """`fuse_seeds` on the real path: the vertex must differ from BOTH inputs, and the
        direction must come from seed A."""
        a, b, f = record['seedA_vec9'], record['seedB_vec9'], record['seedF_vec9']
        assert np.isfinite(f).all()
        assert np.abs(f[1:4] - a[1:4]).max() > 1e-9, 'fused vertex is just seed A'
        assert np.abs(f[1:4] - b[1:4]).max() > 1e-9, 'fused vertex is just seed B'
        np.testing.assert_allclose(f[4:8], a[4:8], rtol=0, atol=0)   # direction inherited from A

    def test_the_fit_moved_away_from_its_start(self, record):
        """A fit that returns its seed unchanged would pass every structural check above."""
        starts = np.stack([record['seedA_vec9'], record['seedB_vec9'], record['seedF_vec9']])
        fit = record['fit_vec9']
        assert np.isfinite(fit).all()
        assert np.abs(starts - fit[None, :]).max() > 1e-6, 'the fit never moved off a seed'

    def test_the_reported_errors_are_finite_and_not_absurd(self, record):
        """Sanity, not resolution. The bands are wide on purpose — at 5k rays and 8 iterations a
        poor fit is the expected outcome, and a tight band here would fail for the wrong reason.
        What these catch is a broken coordinate transform or a unit error, which lands orders of
        magnitude out, not tens of percent."""
        vtx_cm, dir_deg, dE_MeV, dt0_ns = record['fit_err'][:4]
        assert np.isfinite(record['fit_err']).all()
        assert 0.0 <= vtx_cm < 2000.0, f'vertex error {vtx_cm} cm is outside the detector'
        assert 0.0 <= dir_deg <= 180.0, f'direction error {dir_deg} deg is not an angle'
        assert abs(dE_MeV) < 5000.0, f'energy error {dE_MeV} MeV exceeds the beam energy 5x'
        assert abs(dt0_ns) < 100.0, f't0 error {dt0_ns} ns is far outside the search window'

    def test_the_energy_stayed_positive(self, record):
        assert float(record['fit_vec9'][0]) > 0.0, 'fitted energy went negative'
