"""Reconstruction recovers truth at the published resolution. A physics gate, not a wiring gate.

`test_recon_end_to_end.py` checks that the chain runs, at settings too cheap to converge. This
file runs the PUBLISHED working point (`analysis.paper.utils.studies.named_config('nrays_mu_250k')`:
250k rays, 150 Gauss-Newton iterations, 8 keys, the paper seeder) on the repo's 1 GeV muon ROOT
and compares the resolution it reaches with the measured one. A factor-2 error anywhere in the
recon physics (a Fisher block, the energy routing, a t0 sign, a unit) moves at least one of these
medians far outside its band; the end-to-end gate's 20 m / 5 GeV bands cannot see that.

Reference (the oracle): 100 events of `nrays_mu_250k` on this ROOT, measured 2026-10-31 with the
vertex-pivot placement:

    vertex   median 13.8 +- 0.9 cm, 68 % containment 17.4 +- 1.3 cm
    direction median 0.80 deg      |dE| median 10.7 MeV      |dt0| median 0.28 ns

Band. With N events here the gate's own median has a statistical error ``s_N``, taken from a
bootstrap of the gate's events. The reference carries ``s_ref`` (quoted for the vertex; for the
other three, which were quoted without one, 6.5 % of the value, the vertex median's relative
error at 100 events). The gate asserts ``|gate - ref| < 3 sqrt(s_N^2 + s_ref^2)``. At N = 20 that
is about +-45 % of the reference, so a doubling or halving fails. The events are fixed (the
placement and data seeds are deterministic in the config), so the gate is reproducible, not
flaky.

Cost: ~1 min per event on an A100; far too slow for CPU. Run it on a GPU node with

    LUCID_GATE_GPU=1 pytest --slow tests/paper/test_published_resolution.py

``LUCID_GATE_EVENTS`` (default 20) sets N. The reconstruction runs in a subprocess because
`tests/conftest.py` pins this process to the CPU; ``LUCID_GATE_GPU=1`` unpins the subprocess so
it sees the GPU (and the CPU backend the sensor map is built on). Without ``LUCID_GATE_GPU=1``
the ``gpu`` marker deselects this file (``tests/conftest.py``). Needs the trained SIREN emitter
and the 100-event ROOT under the gitignored `data/`; skips without them.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from lucid.fitting import bootstrap
from tests.helpers.paths import DATA, REPO

# 20 events: the lucid-data 5-event slice does not serve it, only the 100-event file.
_ROOT_REL = 'water/muon/1000MeV_100events.root'
_ROOT = DATA / _ROOT_REL

pytestmark = [pytest.mark.slow, pytest.mark.gpu, pytest.mark.requires_emitter('muon', 'water'),
              pytest.mark.requires_data(_ROOT_REL)]
N_EVENTS = int(os.environ.get('LUCID_GATE_EVENTS', '20'))

# (name, reference value, reference error or None, statistic of the per-event errors)
REFERENCE = {
    'vertex_median_cm': (13.8, 0.9),
    'vertex_p68_cm': (17.4, 1.3),
    'direction_median_deg': (0.80, None),
    'abs_dE_median_MeV': (10.7, None),
    'abs_dt0_median_ns': (0.28, None),
}
_REL_ERR_UNQUOTED = 0.9 / 13.8          # the vertex median's relative error at 100 events

_RUNNER = """
import json, sys
from analysis.paper.utils.run import run_local
cfg = json.load(open(sys.argv[1]))
run_local(cfg, sys.argv[2], events=json.loads(sys.argv[3]), verbose=True)
"""


def published_config(n_events):
    from analysis.paper.utils import studies
    cfg = studies.named_config('nrays_mu_250k', n_events)
    cfg['root_file'] = str(_ROOT)
    cfg['name'] = 'recon_resolution_gate'
    return cfg


def _stats(fe):
    """The five gate statistics from the (N, 4) fit_err array [vtx cm, dir deg, dE MeV, dt0 ns]."""
    return {
        'vertex_median_cm': np.median(fe[:, 0]),
        'vertex_p68_cm': np.percentile(fe[:, 0], 68),
        'direction_median_deg': np.median(fe[:, 1]),
        'abs_dE_median_MeV': np.median(np.abs(fe[:, 2])),
        'abs_dt0_median_ns': np.median(np.abs(fe[:, 3])),
    }


@pytest.fixture(scope='module')
def fit_errors(tmp_path_factory):
    import h5py
    out = tmp_path_factory.mktemp('recon_gate')
    cfg_path = out / 'cfg.json'
    cfg_path.write_text(json.dumps(published_config(N_EVENTS)))
    env = dict(os.environ, PYTHONPATH=str(REPO))
    if os.environ.get('LUCID_GATE_GPU') == '1':
        # Unpin rather than pin to cuda: the sensor map is built on the CPU backend
        # (propagation/cylinder.py), so the subprocess needs both, as in a production run.
        env.pop('JAX_PLATFORMS', None)
        env.pop('JAX_PLATFORM_NAME', None)
        if env.get('CUDA_VISIBLE_DEVICES', '') == '':
            env.pop('CUDA_VISIBLE_DEVICES', None)
    subprocess.run([sys.executable, '-c', _RUNNER, str(cfg_path), str(out),
                    json.dumps(list(range(N_EVENTS)))], cwd=REPO, env=env, check=True)
    with h5py.File(out / 'recon_resolution_gate.h5', 'r') as h5:
        fe = np.stack([np.asarray(h5['events'][k]['fit_err']) for k in sorted(h5['events'])])
    assert fe.shape == (N_EVENTS, 4) and np.isfinite(fe).all()
    return fe


@pytest.mark.parametrize('name', list(REFERENCE))
def test_resolution_matches_the_published_measurement(fit_errors, name):
    ref, ref_err = REFERENCE[name]
    if ref_err is None:
        ref_err = _REL_ERR_UNQUOTED * ref
    got = float(_stats(fit_errors)[name])
    s_n = float(bootstrap(fit_errors, lambda fe: _stats(fe)[name], n_boot=2000, seed=0).std())
    band = 3.0 * np.hypot(s_n, ref_err)
    assert abs(got - ref) < band, (
        f'{name}: {got:.3g} over {len(fit_errors)} events vs the published {ref:.3g} '
        f'(band +-{band:.3g}: gate bootstrap {s_n:.3g}, reference {ref_err:.3g})')
