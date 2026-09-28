"""The singleton source must get a device OUTSIDE the `pmap`-ed range.

`CalibrationForward` maps its grouped sources over devices 0..n-1 with `pmap`. A source alone in
its structure group is called directly; left on JAX's default device 0 it serialises behind pmap
shard 0 instead of running alongside it. Placement (`self._laser_dev` and the `jax.device_put`
calls in `__call__`) is value-neutral, so no numerical test can catch a regression.

`tests/conftest.py` sets `JAX_PLATFORMS=cpu`, so the main suite sees ONE device, where the rule
`devs[min(n_sources - 1, len(devs) - 1)]` resolves to that device and every `device_put` is a
no-op. This file therefore forces eight CPU devices in a SUBPROCESS: `XLA_FLAGS` must be set
before JAX initialises, and conftest has already initialised it, so an in-process
`monkeypatch.setenv` would silently do nothing.

Covered: the selection rule on a real `CalibrationForward` with a stub simulator (`sim` is only
referenced inside a lazily-jitted body, so no photons are simulated), and that `device_put`
commits on this backend. Not covered: that the jitted singleton physically executes on that
device (operand devices are not introspectable under `jit`, and the output's device after
`jnp.stack` reflects the stack, not the placement); wall-clock overlap is a profiler question.
"""
import os
import subprocess
import sys

import pytest

# Run in a subprocess with 8 forced CPU devices. Everything the child needs is in this string:
# keeping it self-contained is deliberate, so the parent's already-initialised JAX cannot leak in.
PROG = r'''
import sys, json
sys.path.insert(0, %r)
import jax, jax.numpy as jnp
from collections import namedtuple

devs = jax.devices()
from lucid.fitting.calib import CalibrationForward

NS = 5
Solo = namedtuple('Solo', 'a b c d e')      # 5 fields -- the laser's shape
Many = namedtuple('Many', 'a b c')          # 3 fields -- the isotropic shape

class _Params:
    W = 1
    def to_dp(self, theta, wl, gains):
        return theta

def _sim(src, dp, key):
    return (jnp.ones(NS) * jnp.sum(jnp.asarray(src[0])),)

# The published layout: ONE structurally distinct source followed by seven identical ones.
sources = ([Solo(*[jnp.float32(1.0)] * 5)]
           + [Many(*[jnp.float32(i)] * 3) for i in range(7)])

fwd = CalibrationForward(_sim, sources, _Params(), NS)

out = {
    'n_devices': len(devs),
    'n_sources': fwd.n_sources,
    'placed_index': fwd._placed,
    'laser_dev_id': fwd._laser_dev.id if fwd._laser_dev is not None else None,
    'largest_group': fwd._largest,
    # does device_put actually commit on this backend? Targets the LAST device, so this is
    # meaningful at 8 devices and still well-defined at 1.
    'commits': list(jax.device_put(jnp.ones(3), devs[-1]).devices())[0].id,
}
print('RESULT ' + json.dumps(out))
'''


def _run(n_devices):
    """Run PROG under `n_devices` forced CPU devices and return its parsed RESULT."""
    import json
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env['XLA_FLAGS'] = (env.get('XLA_FLAGS', '') +
                        f' --xla_force_host_platform_device_count={n_devices}').strip()
    env['JAX_PLATFORMS'] = 'cpu'
    env.pop('CUDA_VISIBLE_DEVICES', None)
    p = subprocess.run([sys.executable, '-c', PROG % repo], capture_output=True, text=True,
                       env=env, cwd=repo, timeout=600)
    if p.returncode != 0:
        pytest.fail(f'child failed (rc={p.returncode})\nstdout:\n{p.stdout}\nstderr:\n{p.stderr[-3000:]}')
    line = next((l for l in p.stdout.splitlines() if l.startswith('RESULT ')), None)
    assert line, f'no RESULT line\nstdout:\n{p.stdout}\nstderr:\n{p.stderr[-2000:]}'
    return json.loads(line[len('RESULT '):])


@pytest.fixture(scope='module')
def placed():
    return _run(8)


def test_the_forced_device_count_actually_took(placed):
    """The control. If the flag were ignored the child would see one device and everything below
    would pass vacuously."""
    assert placed['n_devices'] == 8


def test_the_singleton_source_is_the_one_that_gets_placed(placed):
    """`_placed` must identify the source alone in its structure group (it runs unstacked, so it
    needs its own device). `largest_group` confirms the other seven really grouped together, since
    in this layout the singleton is also index 0.
    """
    assert placed['placed_index'] == 0
    assert placed['largest_group'] == 7, (
        'the seven same-structure sources did not group; the layout under test is not the '
        'published one and the placement question it is asking is the wrong one')


def test_the_singleton_gets_a_device_outside_the_mapped_range(placed):
    """`pmap` over the group of seven occupies devices 0..6; the singleton must not land on any of
    them, or it queues behind a shard. Expected: device 7, from
    `devs[min(n_sources - 1, len(devs) - 1)]` with 8 sources and 8 devices.
    """
    assert placed['laser_dev_id'] is not None, 'no device was chosen for the singleton source'
    assert placed['laser_dev_id'] >= placed['largest_group'], (
        f"singleton placed on device {placed['laser_dev_id']}, inside the range "
        f"0..{placed['largest_group'] - 1} that pmap occupies -- it will serialise behind a shard")
    assert placed['laser_dev_id'] == 7


def test_device_put_commits_on_this_backend(placed):
    """Placement is only real if `device_put` binds. A backend where it silently no-ops would make
    every assertion above true and the fix ineffective."""
    assert placed['commits'] == 7


def test_one_device_degrades_without_crashing():
    """The fallback. Most machines are not eight-card nodes, and the rule must not raise or pick a
    device that does not exist when there are fewer devices than sources -- it clamps to the last
    one, which is where the no-op behaviour in the main suite comes from.
    """
    solo = _run(1)
    assert solo['n_devices'] == 1
    assert solo['laser_dev_id'] == 0, 'with one device the rule must clamp to it, not raise'
