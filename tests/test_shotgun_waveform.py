"""Tests for shotgun waveform and per-photon simulation outputs.

Marked slow: requires building a detector + propagator + running JAX sim.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.sources import shotgun_source, stack_shotgun_sources
from lucid.simulation.shotgun import setup_shotgun_simulator

# These tests hit the full propagation pipeline and are slow; treat like the
# other integration tests in the repo.
pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def wcte_paths():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return (os.path.join(base, 'config', 'WCTE_geom_config.json'),
            os.path.join(base, 'config', 'WCTE_physics_config.json'))


@pytest.fixture(scope='module')
def sim_waveform(wcte_paths):
    geom, phys = wcte_paths
    return setup_shotgun_simulator(
        geom, physics_config=phys, n_photons=64,
        output_mode='waveform', K=3, detector_type='Cylinder',
    )


@pytest.fixture(scope='module')
def sim_per_photon(wcte_paths):
    geom, phys = wcte_paths
    return setup_shotgun_simulator(
        geom, physics_config=phys, n_photons=64,
        output_mode='per_photon', K=3, detector_type='Cylinder',
    )


def _source(n=64):
    return shotgun_source([0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                          n_photons=n, wavelength=400.0, intensity=1.0)


def test_waveform_shapes(sim_waveform):
    wf, nd, ndet = sim_waveform(_source(), jax.random.PRNGKey(0))
    assert wf.ndim == 2
    assert wf.shape[1] == 500  # window 500 / bin 1
    assert wf.shape[0] == sim_waveform.num_sensors
    assert nd.ndim == 0 and ndet.ndim == 0


def test_waveform_determinism(sim_waveform):
    wf1, _, _ = sim_waveform(_source(), jax.random.PRNGKey(0))
    wf2, _, _ = sim_waveform(_source(), jax.random.PRNGKey(0))
    assert np.array_equal(np.asarray(wf1), np.asarray(wf2))


def test_waveform_nonzero_and_finite(sim_waveform):
    wf, nd, ndet = sim_waveform(_source(), jax.random.PRNGKey(0))
    wf_np = np.asarray(wf)
    assert np.isfinite(wf_np).all()
    assert int(ndet) > 0
    assert int(nd) == 0  # 500ns window >> 43ns max hit time at this scale
    assert float(wf_np.sum()) > 0


def test_per_photon_shapes_and_range(sim_per_photon):
    det, sid, ht = sim_per_photon(_source(), jax.random.PRNGKey(0))
    det_np = np.asarray(det)
    sid_np = np.asarray(sid)
    ht_np = np.asarray(ht)
    assert det_np.shape == (64,)
    assert sid_np.shape == (64,)
    assert ht_np.shape == (64,)
    assert det_np.dtype == bool
    n_detected = int(det_np.sum())
    assert 0 < n_detected <= 64
    # Non-detected photons have sentinel -1; detected have valid sensor id
    assert np.all(sid_np[det_np] >= 0)
    assert np.all(sid_np[det_np] < sim_per_photon.num_sensors)
    assert np.all(sid_np[~det_np] == -1)
    # Detected times positive
    assert np.all(ht_np[det_np] > 0)


def test_per_photon_determinism(sim_per_photon):
    det1, sid1, ht1 = sim_per_photon(_source(), jax.random.PRNGKey(0))
    det2, sid2, ht2 = sim_per_photon(_source(), jax.random.PRNGKey(0))
    assert np.array_equal(np.asarray(det1), np.asarray(det2))
    assert np.array_equal(np.asarray(sid1), np.asarray(sid2))
    assert np.allclose(np.asarray(ht1), np.asarray(ht2))


def test_batch_produces_consistent_totals(sim_waveform):
    sources = [_source() for _ in range(3)]
    batched = stack_shotgun_sources(sources)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    wf_b, nd_b, ndet_b = sim_waveform.batch(batched, keys)
    assert wf_b.shape[0] == 3
    assert ndet_b.shape == (3,)
    # Per-case equivalence to single-case call
    wf0, nd0, ndet0 = sim_waveform(sources[0], keys[0])
    assert np.allclose(np.asarray(wf_b[0]), np.asarray(wf0))
    assert int(ndet_b[0]) == int(ndet0)


def test_waveform_charge_approximates_detected_count(sim_waveform):
    """With unit per-hit charge and gain smearing σ≈1%, summed charge ≈ n_detected."""
    wf, _, ndet = sim_waveform(_source(), jax.random.PRNGKey(0))
    assert abs(float(np.asarray(wf).sum()) - int(ndet)) / max(int(ndet), 1) < 0.05
