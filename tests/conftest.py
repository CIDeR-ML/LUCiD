"""Shared fixtures for the LUCiD test suite."""
import os
from tests.helpers.paths import REPO
# Force CPU-only BEFORE importing jax. JAX_PLATFORM_NAME alone does NOT stop
# jaxlib's CUDA plugin from probing the GPU at import: on a node whose NVIDIA
# UVM driver is wedged or heavily contended (e.g. concurrent GPU jobs), that
# probe blocks the process in uninterruptible D-state (wchan
# uvm_gpu_retain_by_uuid) — which looks like the whole suite "hanging forever"
# and gets it killed. Hiding the GPU and setting the authoritative JAX_PLATFORMS
# var makes the fast suite never touch the driver. setdefault leaves an explicit
# CUDA_VISIBLE_DEVICES (e.g. a deliberate GPU run) untouched.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # back-compat with older jaxlib

import pytest
import jax

def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="Run the slow tests too (detector builds, propagator compiles, simulations)")


def pytest_configure(config):
    """Make ``--slow`` an alias for "drop the default -m 'not slow'".

    There is one slow mechanism: the ``slow`` marker, deselected by ``addopts``
    in pyproject.toml. ``--slow`` survives because every script and doc in the
    tree spells it, and it now means exactly what ``-m ''`` means, so
    ``pytest -m slow`` and ``--slow`` can no longer disagree about which tests
    are slow. The previous mechanism -- a hardcoded list of *basenames* refused
    at collection -- was directory-blind and silently skipped whole files that
    ``-m slow`` would have selected.

    An explicit ``-m`` on the command line wins: ``--slow -m slow`` runs the
    slow tests only, which is what it looks like it should do.
    """
    if config.getoption("--slow") and config.option.markexpr == "not slow":
        config.option.markexpr = ""


# Tests deselected this run (slow, gpu), so the terminal summary can say so. Otherwise the suite
# reports a pass count with no sign of what it left out.
_DESELECTED = []


def pytest_deselected(items):
    _DESELECTED.extend(items)


def pytest_collection_modifyitems(config, items):
    """``gpu`` tests are deselected unless ``LUCID_GATE_GPU=1``, whatever ``-m`` says.

    They need a GPU node (the published-resolution gate is ~1 min per event on an A100), and this
    process is pinned to the CPU above, so a ``--slow`` run on a CPU host must not start them.
    ``LUCID_GATE_GPU=1`` is also what unpins their subprocess.
    """
    if os.environ.get('LUCID_GATE_GPU') == '1':
        return
    gpu = [item for item in items if item.get_closest_marker('gpu')]
    if gpu:
        config.hook.pytest_deselected(items=gpu)
        items[:] = [item for item in items if not item.get_closest_marker('gpu')]


# --------------------------------------------------------------------------------------------
# Data a test needs: one marker per kind, resolved here and nowhere else.
#
# `data/` is gitignored, so a fresh clone and every git worktree hold only its tracked JSON
# pointers; the trained SIREN emitters' weights and the example PhotonSim events arrive with
# scripts/download_data.py (the lucid-data release asset). A test says what it needs:
#
#   @pytest.mark.requires_data('water/muon/1000MeV_100events.root')   # files under data/
#   @pytest.mark.requires_data(paths.MUON_1GEV_EVENTS)                 # a tuple: any one of them
#   @pytest.mark.requires_emitter('muon', 'wbls', dedx=True)           # a trained SIREN emitter
#
# An emitter is resolved the way the simulator resolves it (the material JSON's "siren" set, then
# lucid.siren.emitter.emitter_dir), so a guard follows the loader wherever the material sends it.
#
# Missing data is a skip with a generated reason, listed in a banner at the end of the run;
# LUCID_REQUIRE_DATA=1 makes it a failure, for a job that is supposed to have the data.
# --------------------------------------------------------------------------------------------
def _missing_emitter_files(particle, material, dedx=False):
    from lucid.siren.emitter import emitter_dir
    from lucid.wavelength.medium import load_medium
    siren_set = load_medium(material).model['siren']
    files = [emitter_dir(particle, siren_set, kind) / name
             for kind in (('cherenkov', 'dedx') if dedx else ('cherenkov',))
             for name in ('emitter.json', 'weights.npz')]
    return [f for f in files if not f.is_file()]


def _missing_data(item):
    """What ``item``'s requires_data / requires_emitter markers name and data/ lacks."""
    from tests.helpers.paths import DATA, REPO
    missing = []
    for mark in item.iter_markers('requires_data'):
        for need in mark.args:
            options = need if isinstance(need, tuple) else (need,)
            if not any((DATA / p).is_file() for p in options):
                missing.append(' or '.join(f'data/{p}' for p in options))
    for mark in item.iter_markers('requires_emitter'):
        missing += [os.path.relpath(f, REPO) for f in _missing_emitter_files(*mark.args,
                                                                            **mark.kwargs)]
    return missing


_DATA_SKIPS = {}       # reason -> node ids


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    missing = _missing_data(item)
    if not missing:
        return
    reason = f'needs {", ".join(missing)} (python scripts/download_data.py)'
    if os.environ.get('LUCID_REQUIRE_DATA') == '1':
        pytest.fail(f'LUCID_REQUIRE_DATA=1 and this test {reason}', pytrace=False)
    _DATA_SKIPS.setdefault(reason, []).append(item.nodeid)
    pytest.skip(reason)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    w = terminalreporter
    # Reported first and unconditionally: a green count is misleading in proportion to what it
    # left out.
    if _DESELECTED:
        w.write_sep('=', f'{len(_DESELECTED)} test(s) DESELECTED — --slow adds the slow ones, '
                         'LUCID_GATE_GPU=1 the gpu ones', yellow=True, bold=True)
        for name in sorted({item.location[0] for item in _DESELECTED}):
            w.write_line(f'  {name}')
        w.write_line('  These did NOT run. A green suite here does not cover them.')
    if not _DATA_SKIPS:
        return
    n = sum(len(v) for v in _DATA_SKIPS.values())
    w.write_sep('=', f'{n} test(s) SKIPPED because data is missing', red=True, bold=True)
    for reason, nodes in sorted(_DATA_SKIPS.items()):
        w.write_line(f'  {len(nodes)} test(s): {reason}')
        for nid in nodes[:3]:
            w.write_line(f'      {nid}')
        if len(nodes) > 3:
            w.write_line(f'      ... and {len(nodes) - 3} more')
    w.write_line('')
    w.write_line('  These did NOT run. A green suite here does not cover them.')
    w.write_line('  Set LUCID_REQUIRE_DATA=1 to make missing data a FAILURE instead.')


@pytest.fixture(scope="session")
def key():
    """Fixed PRNGKey for reproducible stochastic tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def small_cylinder_config():
    """Path to a small generic cylinder detector config (WCTE-shaped,
    algorithmic placement). The real-WCTE config (``WCTE_geom_config``)
    loads measured PMT positions from a separate npz file and is not
    suitable as a generic cylinder fixture."""
    return os.path.join(str(REPO),
                        "config", "WCTE_like_geom_config.json")


# --------------------------------------------------------------------------------------------
# The tiny detector (tests/helpers/tiny.py): the composed forward model, each simulator built and
# compiled once per session, so the tests that run the whole chain are in the default lane. The
# simulators only the physics tests use are in tests/physics/conftest.py.
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_sim():
    """Calibration source, expected-value transport, one step."""
    from tests.helpers import tiny
    return tiny.simulator()
