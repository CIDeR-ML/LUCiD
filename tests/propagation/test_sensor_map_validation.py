"""The sensor map's overflow: when it happens, and that it is reported.

``create_propagator`` keeps ``max_candidates_per_ray`` (K) sensors per grid cell. A cell that
more than K sensors' disks overlap loses the excess, and the rays landing there can never find
those sensors. ``validate_sensor_map`` (lucid/propagation/shared.py) is the only report of it.
This file checks

* that the grid every shipped surface detector gets by default, and the explicit grid the paper
  passes to every cylinder, does not overflow -- counted by brute force from the forward map
  and, separately, by the validator's silence;
* that a grid which does overflow produces EACH warning that describes it, with the number brute
  force gives, so silencing either one fails;
* that ``make_grid`` derives what it is not given and keeps what it is.

Whether an overflow should raise rather than warn is a library decision; see the strict xfails.
The forward map's own blind spot (a sensor wanting more than its 4 recorded cells) is tested
end to end in tests/propagation/test_propagation_aimed_rays.py.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from lucid.geometry import generate_detector
from lucid.propagation.shared import create_propagator
from tests.helpers.sensor_map import forward_cell_counts, sensor_map_warnings
from tests.helpers.paths import REPO

pytestmark = pytest.mark.slow


# The explicit grid the published pipeline passes to every cylinder
# (analysis/paper/utils/studies.py PIPELINE['grid']; examples/seed_reconstruct.py uses the
# same), without its max_candidates_per_ray=8.
PAPER_GRID = dict(n_cap=80, n_angular=120, n_height=80)

SURFACE_CONFIGS = ['WCTE_like', 'SK_like', 'IWCD', 'EOS', 'BigHK', 'SK', 'HK', 'WCTE',
                   'JUNO', 'TAO', 'MidBox', 'nuSCOPE']
CYLINDER_CONFIGS = ['WCTE_like', 'SK_like', 'IWCD', 'EOS', 'SK', 'HK', 'WCTE', 'BigHK']


def _det(name):
    return generate_detector(os.path.join(REPO, 'config', f'{name}_geom_config.json'))


def _build(det, **grid):
    # temperature=None: the step overlap needs no lookup table, and the map does not depend on it
    return lambda: create_propagator(det, jnp.asarray(det.all_points, jnp.float32),
                                     float(det.S_radius), temperature=None, **grid)


# ── The grids in use do not overflow ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", SURFACE_CONFIGS)
def test_default_grid_drops_no_sensor(name):
    """At the default grid no sensor is left out of the forward map, the slots hold every cell's
    sensors (the auto grid aims at 4; the tightest layouts, WCTE and TAO, need 5), and the
    validator reports nothing beyond the slot raise."""
    det = _det(name)
    msgs, (prop, grid) = sensor_map_warnings(_build(det))
    counts, orphans = forward_cell_counts(det, grid)
    assert orphans == 0, f"{name}: {orphans} sensors assigned to no grid cell"
    k = np.asarray(prop(jnp.zeros((1, 3)), jnp.eye(3)[:1])['sensor_indices']).shape[0]
    assert k >= counts.max(), f"{name}: {k} slots for cells holding up to {counts.max()}"
    assert counts.max() <= 5, f"{name}: the auto grid lets a cell hold {counts.max()} sensors"
    assert [m for m in msgs if "raising max_candidates_per_ray" not in m] == [], f"{name}: {msgs}"


@pytest.mark.parametrize("name", CYLINDER_CONFIGS)
def test_paper_grid_drops_no_sensor(name):
    """The paper's explicit grid is coarser than some detectors' PMT spacing (BigHK: cells hold up
    to 6). The propagator must then size its slots to fit, so nothing is dropped and every sensor
    stays reachable; it says so in one warning."""
    det = _det(name)
    msgs, (prop, grid) = sensor_map_warnings(_build(det, **PAPER_GRID))
    counts, orphans = forward_cell_counts(det, grid)
    assert orphans == 0, f"{name}: {orphans} sensors assigned to no grid cell"
    k = np.asarray(prop(jnp.zeros((1, 3)), jnp.eye(3)[:1])['sensor_indices']).shape[0]
    assert k >= counts.max(), f"{name}: {k} slots for cells holding up to {counts.max()}"
    assert [m for m in msgs if "raising max_candidates_per_ray" not in m] == [], f"{name}: {msgs}"


# ── An overflowing grid is reported, each symptom by its own message ──────────────────────────

# WCTE_like on a 5 x 5 x 5 grid with K = 2: every cell is crowded.
_COARSE = dict(n_cap=5, n_angular=5, n_height=5, max_candidates_per_ray=2)


@pytest.fixture(scope="module")
def overflowed():
    det = _det('WCTE_like')
    msgs, (prop, grid) = sensor_map_warnings(_build(det, **_COARSE))
    counts, _ = forward_cell_counts(det, grid)
    return det, msgs, prop, counts


def _one(msgs, text):
    hits = [m for m in msgs if text in m]
    assert len(hits) == 1, f"expected one warning containing {text!r}, got {msgs}"
    return hits[0]


def test_overflow_raises_the_slots_to_the_true_maximum(overflowed):
    """The most sensors forward-assigned to one cell, by brute force, becomes the slot count."""
    _, msgs, prop, counts = overflowed
    k = int(counts.max())
    assert k > 2
    msg = _one(msgs, "raising max_candidates_per_ray")
    assert f"holds {k} sensors; raising max_candidates_per_ray from 2 to {k}" in msg, msg
    out = prop(jnp.zeros((3, 3)), jnp.eye(3))
    assert out['sensor_weights'].shape == (k, 3)
    assert out['sensor_indices'].shape == (k, 3)


def test_after_raising_nothing_is_dropped_and_every_sensor_is_reachable(overflowed):
    """With the slots raised, no geometric assignment is missing from the inverse map and no
    sensor is unreachable, so the validator reports neither."""
    _, msgs, _, _ = overflowed
    assert not [m for m in msgs if "missing from the inverse map" in m], msgs
    assert not [m for m in msgs if "do not appear in any cell's inverse map" in m], msgs


# ── make_grid ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ['WCTE_like', 'JUNO', 'nuSCOPE'])
def test_auto_grid_bounds_the_forward_count(name):
    """make_grid alone (no propagator) already sizes the cells for K = 4: the cylinder from
    the nearest-neighbour spacing, the sphere and the box from the sensor count."""
    det = _det(name)
    counts, orphans = forward_cell_counts(det, det.make_grid(max_candidates_per_ray=4))
    assert counts.max() <= 4, f"{name}: max {counts.max()} sensors in one cell"
    assert orphans == 0


def test_explicit_overrides_auto():
    """Explicit grid params should override auto-derivation."""
    det = _det('WCTE_like')
    assert tuple(det.make_grid(n_cap=10, n_angular=20, n_height=15)) == (10, 20, 15)


def test_partial_override():
    """Only the given parameter changes; the others take exactly their auto-derived values."""
    det = _det('WCTE_like')
    auto = det.make_grid()
    assert 100 != auto.n_angular
    assert det.make_grid(n_angular=100) == auto._replace(n_angular=100)
