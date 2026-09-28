"""Shared fixtures for the LUCiD test suite."""
import os
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
import jax.numpy as jnp

# Files marked @pytest.mark.slow — skip importing them unless --slow is passed.
_SLOW_FILES = [
    "test_containers.py",
    "test_integration.py",
    "test_optics_physics.py",
    "test_photon_step.py",
    "test_photon_step_physics.py",
    "test_propagation_differentiability.py",
    "test_propagator_output.py",
    "test_reflection_integration.py",
    "test_ray_intersection.py",
    "test_sensor_map_validation.py",
    "test_shared_propagator.py",
    "test_shared_propagator_differentiability.py",
    "test_shotgun_waveform.py",
    "test_sk_like_integration.py",
    "test_wavelength_integration.py",
    "test_qe_importance_sampling.py",
    "test_qe_setup_broadcast.py",
    "test_tripwire.py",
    # e2e simulation smoke tests (build detectors + run sims; ~90s total)
    "test_e2e_calibration.py",
    "test_e2e_track_and_data.py",
    "test_e2e_superk.py",
    "test_e2e_gradients.py",
    "test_e2e_edge_cases.py",
]


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="Include slow tests (detector/propagator/simulation)")


# Slow files skipped this run, so the terminal summary can name them. Otherwise the suite reports
# a pass count with no sign that whole files (test_tripwire.py among them) were never collected.
_IGNORED = set()

def pytest_ignore_collect(collection_path, config):
    if config.getoption("--slow", default=False):
        return False
    if collection_path.name in _SLOW_FILES:
        _IGNORED.add(collection_path.name)
        return True
    return False


# --------------------------------------------------------------------------------------------
# Skips that mean "the data is missing", reported loudly.
#
# `data/` is gitignored, so a fresh clone and EVERY git worktree materialise almost none of it.
# Tests that need a downloaded asset therefore skip — correctly, since erroring would read like a
# defect — but pytest prints only a count, and a suite with silently skipped tests looks green.
#
# So: a banner at the end of the run naming what was skipped and what would restore it, and
# LUCID_REQUIRE_DATA=1 to turn those skips into failures for a CI job that is supposed to have the
# data. Nothing here changes which tests run by default.
# --------------------------------------------------------------------------------------------
_DATA_HINTS = ('not present', 'download_data', 'no wbls', 'not found', 'missing')


def _is_data_skip(reason):
    r = str(reason).lower()
    return any(h in r for h in _DATA_HINTS)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # --- files that were never COLLECTED -------------------------------------------------
    # Reported first and unconditionally: a green count is misleading in proportion to what it
    # left out.
    if _IGNORED:
        w = terminalreporter
        w.write_sep('=', f'{len(_IGNORED)} FILE(S) NOT COLLECTED (slow) — pass --slow to include',
                    yellow=True, bold=True)
        for name in sorted(_IGNORED):
            w.write_line(f'  {name}')
        w.write_line('  These did NOT run. A green suite here does not cover them.')

    skipped = terminalreporter.stats.get('skipped', [])
    data_skips = {}
    for rep in skipped:
        reason = rep.longrepr[2] if isinstance(rep.longrepr, tuple) else str(rep.longrepr)
        if _is_data_skip(reason):
            data_skips.setdefault(reason.replace('Skipped: ', ''), []).append(rep.nodeid)
    if not data_skips:
        return
    n = sum(len(v) for v in data_skips.values())
    w = terminalreporter
    w.write_sep('=', f'{n} test(s) SKIPPED because data is missing', red=True, bold=True)
    for reason, nodes in sorted(data_skips.items()):
        w.write_line(f'  {len(nodes)} test(s): {reason}')
        for nid in nodes[:3]:
            w.write_line(f'      {nid}')
        if len(nodes) > 3:
            w.write_line(f'      ... and {len(nodes) - 3} more')
    w.write_line('')
    w.write_line('  These did NOT run. A green suite here does not cover them.')
    w.write_line('  Set LUCID_REQUIRE_DATA=1 to make missing data a FAILURE instead.')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """With LUCID_REQUIRE_DATA=1, a data skip becomes a failure.

    A hookwrapper that rewrites the finished report, NOT a `pytest.fail()` inside the hook: this
    hook's job is to build the report, and raising from it aborts the whole session with
    INTERNALERROR rather than failing the test.
    """
    outcome = yield
    if os.environ.get('LUCID_REQUIRE_DATA') != '1':
        return
    rep = outcome.get_result()
    if not rep.skipped:
        return
    reason = rep.longrepr[2] if isinstance(rep.longrepr, tuple) else str(rep.longrepr)
    if _is_data_skip(reason):
        rep.outcome = 'failed'
        rep.longrepr = (f'LUCID_REQUIRE_DATA=1 and the data this test needs is missing.\n'
                        f'{reason.replace("Skipped: ", "")}')


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
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "config", "WCTE_like_geom_config.json")


@pytest.fixture(scope="session")
def cylinder_detector(small_cylinder_config):
    """Build the small generic cylinder detector once per session."""
    from lucid.geometry import generate_detector
    return generate_detector(small_cylinder_config)


@pytest.fixture(scope="session")
def fixed_flat_hits():
    """Fixed flat arrays for make_hits tests."""
    return dict(
        flat_weights=jnp.array([0.5, 0.3, 0.8, 0.1, 0.6]),
        flat_indices=jnp.array([0, 2, 5, 5, 10]),
        flat_times=jnp.array([10.0, 15.0, 12.0, 20.0, 8.0]),
        num_detectors=20,
        qe_corrections=jnp.ones(20),
    )
