"""The singleton source must get a device OUTSIDE the mapped range. Asserted, not profiled.

`CalibrationForward` maps its grouped sources over devices 0..n-1 with `pmap`. A source that is
alone in its structure group is called directly instead, and if it is left on JAX's default
device -- device 0 -- it SERIALISES behind pmap shard 0 rather than running alongside it.

That is not hypothetical. It shipped: measured at the published recipe, the forward took 1.121 s
against 0.612 s for the mapped group alone, 45% of every forward wasted, and on a ten-GPU node
devices 7, 8 and 9 sat idle for an entire campaign. The fix (`self._laser_dev`, and the
`jax.device_put` calls in `__call__`) is worth 1.76x on the forward and moves no number --
placement is value-neutral, and the engine and this class were bit-identical on all six pinned
fields while placing this source differently.

WHY THIS TEST DID NOT EXIST, which is the interesting part. `tests/conftest.py` sets
`JAX_PLATFORMS=cpu` for the whole suite, so every test sees exactly ONE device. With one device
the placement rule `devs[min(n_sources - 1, len(devs) - 1)]` resolves to that same device and
every `jax.device_put` is a no-op. The existing equivalence tests DO construct a
`CalibrationForward` with the real 1-laser + 7-isotropic layout, so this code runs in CI --
it just cannot fail there. A regression reverting the placement would have been invisible to the
entire suite, and the only thing that would have caught it is re-running a profiler by hand.

So this file forces eight CPU devices in a SUBPROCESS. `XLA_FLAGS` must be set before JAX
initialises, and conftest has already initialised it by the time any test body runs, so an
in-process `monkeypatch.setenv` would silently do nothing -- itself an instance of the failure
mode this file exists to catch.

WHAT IT COVERS: the selection rule, on a real `CalibrationForward` built with a stub simulator
(cheap -- `sim` is only referenced inside a lazily-jitted body, so no photons are simulated), and
that `device_put` genuinely commits on this backend. WHAT IT DOES NOT COVER: that the jitted
singleton's computation physically executes on that card. Under `jit` the operand devices are not
introspectable from inside the trace, and asserting on the output's device after `jnp.stack` would
be asserting on the stack, not the placement. Wall-clock overlap remains a profiler question, and
is not tested here.
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
    would pass vacuously -- which is precisely how this gap survived in the main suite."""
    assert placed['n_devices'] == 8


def test_the_singleton_source_is_the_one_that_gets_placed(placed):
    """`_placed` must identify the structurally-distinct source, not simply index 0.

    The rule is about GROUP SIZE, not about being first or being a laser: a source alone in its
    structure group is executed unstacked, so it is the one that needs a card of its own. The
    published layout happens to put the laser first, which makes "index 0" and "the singleton"
    coincide and would let a wrong rule pass unnoticed here -- so the assertion is written against
    the group structure, and `largest_group` is checked to confirm the other seven really did
    group together rather than each becoming its own singleton.
    """
    assert placed['placed_index'] == 0
    assert placed['largest_group'] == 7, (
        'the seven same-structure sources did not group; the layout under test is not the '
        'published one and the placement question it is asking is the wrong one')


def test_the_singleton_gets_a_device_outside_the_mapped_range(placed):
    """THE REGRESSION. `pmap` over the group of seven occupies devices 0..6; the singleton must
    not land on any of them, or it queues behind a shard instead of overlapping it.

    Device 7 specifically, from `devs[min(n_sources - 1, len(devs) - 1)]` with 8 sources and 8
    devices. The bug this replaces left it unset, so it ran on JAX's default -- device 0, i.e.
    exactly on top of pmap shard 0.
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
