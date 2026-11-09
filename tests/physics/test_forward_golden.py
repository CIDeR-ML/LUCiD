"""The one committed golden: the tiny detector's composed forward at one working point.

Every other test states a property that must always hold. This one states what the forward
produced when it was last captured on purpose, so a change meant to be numerically neutral that is
not (a refactor that reorders the transport, a constant that moved) fails before merge. The
per-change oracle makes the same comparison at SK scale, but it lives outside the repository.

The working point uses every stage of the calibration forward: an isotropic source off every
symmetry axis, Rayleigh and Mie scattering, absorption, reflecting walls and PMTs, four transport
steps, the soft (temperature 0.2) deposit, expected-value hits. The digest is the total charge,
the charge norm, the charge-weighted mean time and the gradient of the total charge in the seven
optical fields.

The bound is 1e-3 relative (the gradient: of each value, plus 1e-4 of the gradient's norm, so a
near-zero column is not held to its own rounding). It is a guess at the cross-CPU and
toolchain floor, not a measurement: the SK-scale tripwire this file replaces drifted 1.5e-4 to
1.1e-3 between an Intel jax 0.4.38 capture and an AMD jax 0.5.3 run. A re-capture is a numerics
change: make it in its own commit, saying why, with

    LUCID_CAPTURE_GOLDEN=1 pytest tests/physics/test_forward_golden.py
"""
import json
import os
import platform
from pathlib import Path

import jax
import numpy as np
import pytest

from tests.helpers import tiny

pytestmark = pytest.mark.golden

GOLDEN = Path(tiny.__file__).with_name('tiny_forward_golden.json')
FIELDS = ('g', 'scatter_length', 'mie_scatter_length', 'absorption_length',
          'wall_reflection_rate', 'sensor_reflection_rate', 'qe')
RTOL = 1e-3


def _digest():
    from lucid.detector_params import _flatten_detector_params
    from lucid.sources import isotropic_source
    sim = tiny.simulator(K=4, temperature=0.2)
    params = tiny.transparent(sim.det_geom.num_sensors, scatter_length=5.0,
                              mie_scatter_length=20.0, g=0.9, absorption_length=10.0,
                              wall_reflection_rate=0.2, sensor_reflection_rate=0.25)
    source = isotropic_source([0.21, -0.13, 0.32], intensity=tiny.N)
    key = jax.random.PRNGKey(20261101)

    def total(dp):
        return sim(source, dp, key)[0].sum()

    q, t = (np.asarray(a, np.float64) for a in sim(source, params, key))
    grad = _flatten_detector_params(jax.grad(total)(params))
    return dict(q_sum=float(q.sum()), q_l2=float(np.linalg.norm(q)),
                t_mean_ns=float((q * t).sum() / q.sum()),
                grad_q_sum={f: float(grad[f]) for f in FIELDS})


def _cpu():
    try:
        with open('/proc/cpuinfo') as f:
            return next(line.split(':', 1)[1].strip() for line in f if line.startswith('model name'))
    except (OSError, StopIteration):
        return platform.machine()


def test_the_tiny_forward_matches_its_golden():
    got = _digest()
    if os.environ.get('LUCID_CAPTURE_GOLDEN') == '1':
        got['captured'] = dict(jax=jax.__version__, numpy=np.__version__, cpu=_cpu())
        GOLDEN.write_text(json.dumps(got, indent=2) + '\n')
        pytest.skip(f'captured {GOLDEN.name}; commit it with the reason')
    ref = json.loads(GOLDEN.read_text())
    for k in ('q_sum', 'q_l2', 't_mean_ns'):
        np.testing.assert_allclose(got[k], ref[k], rtol=RTOL, err_msg=k)
    g_ref = np.array([ref['grad_q_sum'][f] for f in FIELDS])
    g_got = np.array([got['grad_q_sum'][f] for f in FIELDS])
    bound = RTOL * np.abs(g_ref) + 1e-4 * np.linalg.norm(g_ref)
    bad = np.abs(g_got - g_ref) > bound
    assert not bad.any(), 'gradient drift: ' + ', '.join(
        f'{FIELDS[i]} {g_got[i]:.6g} vs {g_ref[i]:.6g}' for i in np.nonzero(bad)[0])
