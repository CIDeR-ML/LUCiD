"""
Tests for :meth:`lucid.geometry.Cylinder.from_pmt_file` — the unified
loader that builds a :class:`Cylinder` from a PMT-positions ``.npz``
file. See ``docs/reference/pmt-npz-schema.md`` for the schema.
"""

import os

import numpy as np
import numpy.testing as npt
import pytest

from lucid.geometry import Cylinder, generate_detector
from tests.helpers.paths import REPO



REPO_CONFIG = os.path.abspath(
    os.path.join(REPO, 'config')
)

REAL_DETECTORS = [
    # (label,             npz file,                       n_active, has_inactive)
    ('SK',                'sk_geometry.npz',              11096,    False),
    ('HK',                'hk_geometry.npz',              19746,    True),
    ('WCTE',              'wcte_geometry.npz',            1995,     False),
]


# ── Schema validation ─────────────────────────────────────────────────


def _write_npz(tmp_path, **arrays):
    p = os.path.join(tmp_path, 'tmp.npz')
    np.savez_compressed(p, **arrays)
    return p


def _minimal_valid_arrays(n=4):
    """A tiny but schema-valid PMT array: 2 barrel + 1 top + 1 bottom."""
    return dict(
        positions_mm=np.array([
            [1000.0,    0.0,    0.0],   # barrel
            [   0.0, 1000.0,    0.0],   # barrel
            [   0.0,    0.0, 1000.0],   # top
            [   0.0,    0.0,-1000.0],   # bottom
        ]),
        directions=np.array([
            [-1.0, 0.0,  0.0],
            [ 0.0,-1.0,  0.0],
            [ 0.0, 0.0, -1.0],
            [ 0.0, 0.0,  1.0],
        ]),
        surfaces=np.array(['barrel', 'barrel', 'top', 'bottom'], dtype='<U6'),
        pmt_id=np.array([10, 20, 30, 40], dtype=np.int32),
        radius=np.float64(1.0),
        height=np.float64(2.0),
        sensor_radius=np.float64(0.05),
    )


@pytest.mark.parametrize("missing_key", [
    'positions_mm', 'directions', 'surfaces', 'pmt_id',
    'radius', 'height', 'sensor_radius',
])
def test_missing_required_key_raises(tmp_path, missing_key):
    arrs = _minimal_valid_arrays()
    arrs.pop(missing_key)
    p = _write_npz(str(tmp_path), **arrs)
    with pytest.raises(KeyError, match=missing_key):
        Cylinder.from_pmt_file(p)


def test_minimal_valid_npz_loads(tmp_path):
    p = _write_npz(str(tmp_path), **_minimal_valid_arrays())
    det = Cylinder.from_pmt_file(p)
    assert isinstance(det, Cylinder)
    assert len(det.all_points) == 4
    assert det.r == 1.0
    assert det.H == 2.0
    assert det.S_radius == 0.05


def test_directions_shape_mismatch_raises(tmp_path):
    arrs = _minimal_valid_arrays()
    arrs['directions'] = np.zeros((3, 3))  # wrong N
    p = _write_npz(str(tmp_path), **arrs)
    with pytest.raises(ValueError, match="directions shape"):
        Cylinder.from_pmt_file(p)


def test_surfaces_shape_mismatch_raises(tmp_path):
    arrs = _minimal_valid_arrays()
    arrs['surfaces'] = np.array(['barrel', 'top'], dtype='<U6')  # wrong N
    p = _write_npz(str(tmp_path), **arrs)
    with pytest.raises(ValueError, match="surfaces shape"):
        Cylinder.from_pmt_file(p)


def test_unknown_surface_label_raises(tmp_path):
    arrs = _minimal_valid_arrays()
    arrs['surfaces'] = np.array(['barrel', 'side', 'top', 'bottom'], dtype='<U6')
    p = _write_npz(str(tmp_path), **arrs)
    with pytest.raises(ValueError, match="unknown surface labels"):
        Cylinder.from_pmt_file(p)


# ── Reordering and snap behaviour ─────────────────────────────────────


# Shuffle (top, barrel, bottom, barrel): input row i ends up at output row _EXPECTED_SLOT[i].
_SHUFFLE = [2, 0, 3, 1]


def _shuffled_arrays():
    arrs = _minimal_valid_arrays()
    for k in ('positions_mm', 'directions', 'surfaces', 'pmt_id'):
        arrs[k] = arrs[k][_SHUFFLE]
    return arrs


def test_active_block_reordered_to_barrel_top_bottom(tmp_path):
    """The active arrays in the npz can be in any order; the loader
    reorders them to (barrel, top, bottom) for compatibility with
    Cylinder's ID_to_case convention -- and every per-PMT array moves with its row."""
    p = _write_npz(str(tmp_path), **_shuffled_arrays())
    det = Cylinder.from_pmt_file(p)
    # The shuffled file holds, in order: top(30), barrel(10), bottom(40), barrel(20).
    # Barrel first in file order, then top, then bottom:
    npt.assert_array_equal([det.ID_to_case[i] for i in range(4)], [0, 0, 1, 2])
    npt.assert_array_equal(det.pmt_id, [10, 20, 30, 40])
    # Positions, in metres, travel with their pmt_id (rows of _minimal_valid_arrays).
    npt.assert_allclose(det.raw_positions,
                        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, -1.0]], atol=1e-12)
    npt.assert_allclose(det.pmt_directions,
                        [[-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0], [0, 0, 1.0]], atol=0)
    assert {pid: det.pmt_id_to_idx[pid] for pid in (10, 20, 30, 40)} == \
        {10: 0, 20: 1, 30: 2, 40: 3}


def test_per_pmt_metadata_reordered(tmp_path):
    """A per-active-PMT metadata array (length N) must be reordered
    consistently with positions/directions.

    The rows are shuffled first: on input already in (barrel, top, bottom) order the reorder is
    the identity, and a loader that skipped it would pass.
    """
    arrs = _shuffled_arrays()
    # metadata keyed to each row's pmt_id, so it can be traced through the reorder
    arrs['custom_meta'] = 10 * arrs['pmt_id']
    p = _write_npz(str(tmp_path), **arrs)
    det = Cylinder.from_pmt_file(p)
    npt.assert_array_equal(det.custom_meta, 10 * det.pmt_id)
    npt.assert_array_equal(det.custom_meta, [100, 200, 300, 400])


def test_snap_moves_cap_pmts_to_the_cap_planes(tmp_path):
    """Cap PMTs snap axially onto z = +-H/2 (H = 2 m here), and keep their x, y."""
    arrs = _minimal_valid_arrays()
    arrs['positions_mm'][2] = [100.0, -50.0, 900.0]     # top, 100 mm below the cap
    arrs['positions_mm'][3] = [0.0, 30.0, -1040.0]      # bottom, 40 mm beyond it
    det = Cylinder.from_pmt_file(_write_npz(str(tmp_path), **arrs))
    npt.assert_allclose(det.tcap_points, [[0.1, -0.05, 1.0]], atol=1e-12)
    npt.assert_allclose(det.bcap_points, [[0.0, 0.03, -1.0]], atol=1e-12)
    npt.assert_allclose(det.raw_positions[2:], [[0.1, -0.05, 0.9], [0.0, 0.03, -1.04]],
                        atol=1e-12)


def test_snap_to_wall_default_on(tmp_path):
    """A PMT pushed inward should be snapped to r_xy=R by default."""
    arrs = _minimal_valid_arrays()
    arrs['positions_mm'][0] = [800.0, 0.0, 0.0]  # 200 mm inside wall
    p = _write_npz(str(tmp_path), **arrs)
    det = Cylinder.from_pmt_file(p)
    # PMT 0 is barrel → r_xy must equal R after snap
    r_xy = float(np.linalg.norm(det.barr_points[0, :2]))
    assert abs(r_xy - det.r) < 1e-9
    # Raw position is preserved
    assert abs(np.linalg.norm(det.raw_positions[0, :2]) - 0.8) < 1e-9


def test_snap_to_wall_off_keeps_raw_positions(tmp_path):
    arrs = _minimal_valid_arrays()
    arrs['positions_mm'][0] = [800.0, 0.0, 0.0]
    p = _write_npz(str(tmp_path), **arrs)
    det = Cylinder.from_pmt_file(p, snap_to_wall=False)
    r_xy = float(np.linalg.norm(det.barr_points[0, :2]))
    assert abs(r_xy - 0.8) < 1e-9


# ── Inactive PMT block ────────────────────────────────────────────────


def test_inactive_block_attached_unmodified(tmp_path):
    arrs = _minimal_valid_arrays()
    arrs['inactive_positions_mm'] = np.array([[1500.0, 0.0, 0.0]])
    arrs['inactive_directions']   = np.array([[-1.0, 0.0, 0.0]])
    arrs['inactive_surfaces']     = np.array(['barrel'], dtype='<U6')
    arrs['inactive_pmt_id']       = np.array([999], dtype=np.int32)
    arrs['inactive_categories']   = np.array(['od'], dtype='<U8')
    p = _write_npz(str(tmp_path), **arrs)
    det = Cylinder.from_pmt_file(p)
    # Active count unaffected by inactive entries
    assert len(det.all_points) == 4
    # Inactive arrays are attached as-is (NOT reordered, NOT mm→m converted)
    npt.assert_array_equal(det.inactive_positions_mm[0], [1500.0, 0.0, 0.0])
    npt.assert_array_equal(det.inactive_pmt_id, [999])


# ── Real detector files ───────────────────────────────────────────────


@pytest.mark.slow                 # builds SK / HK / WCTE detectors
@pytest.mark.parametrize("label,npz_name,n_active,has_inactive", REAL_DETECTORS)
def test_real_detector_loads(label, npz_name, n_active, has_inactive):
    det = Cylinder.from_pmt_file(os.path.join(REPO_CONFIG, npz_name))
    assert isinstance(det, Cylinder)
    assert len(det.all_points) == n_active
    assert det.r > 0 and det.H > 0 and det.S_radius > 0


@pytest.mark.slow                 # builds SK / HK / WCTE detectors
@pytest.mark.parametrize("label,npz_name,n_active,has_inactive", REAL_DETECTORS)
def test_real_detector_snap_lands_on_geometry(label, npz_name, n_active, has_inactive):
    det = Cylinder.from_pmt_file(os.path.join(REPO_CONFIG, npz_name))
    barrel_rxy = np.linalg.norm(det.barr_points[:, :2], axis=1)
    npt.assert_allclose(barrel_rxy, det.r, atol=1e-9)
    npt.assert_allclose(det.tcap_points[:, 2],  det.H / 2,  atol=1e-9)
    npt.assert_allclose(det.bcap_points[:, 2], -det.H / 2, atol=1e-9)


@pytest.mark.slow                 # builds SK / HK / WCTE detectors
@pytest.mark.parametrize("label,npz_name,n_active,has_inactive", REAL_DETECTORS)
def test_real_detector_pmt_directions_shape(label, npz_name, n_active, has_inactive):
    det = Cylinder.from_pmt_file(os.path.join(REPO_CONFIG, npz_name))
    assert det.pmt_directions.shape == (n_active, 3)
    norms = np.linalg.norm(det.pmt_directions, axis=1)
    npt.assert_allclose(norms, 1.0, atol=1e-6)


@pytest.mark.slow                 # builds SK / HK / WCTE detectors
@pytest.mark.parametrize("label,npz_name,n_active,has_inactive", REAL_DETECTORS)
def test_real_detector_pmt_id_lookup_roundtrips(label, npz_name, n_active, has_inactive):
    """pmt_id -> index lands on that PMT's own row of the file.

    The reference is the raw npz, read here independently: the file row holding a pmt_id gives
    its position in mm, which must equal (in metres) the pre-snap position at the index the
    lookup returns.
    """
    path = os.path.join(REPO_CONFIG, npz_name)
    raw = np.load(path, allow_pickle=True)
    det = Cylinder.from_pmt_file(path)
    raw_ids = np.asarray(raw['pmt_id'])
    assert len(np.unique(raw_ids)) == len(raw_ids) == n_active
    rows = np.random.default_rng(0).choice(n_active, size=min(200, n_active), replace=False)
    for row in rows:
        idx = det.pmt_id_to_idx[int(raw_ids[row])]
        npt.assert_allclose(det.raw_positions[idx], raw['positions_mm'][row] * 1e-3, atol=1e-9)
        assert det.ID_to_case[idx] == {'barrel': 0, 'top': 1, 'bottom': 2}[str(raw['surfaces'][row])]


@pytest.mark.slow                 # builds SK / HK / WCTE detectors
def test_hk_inactive_block_present():
    det = Cylinder.from_pmt_file(os.path.join(REPO_CONFIG, 'hk_geometry.npz'))
    assert hasattr(det, 'inactive_positions_mm')
    assert hasattr(det, 'inactive_categories')
    # 19036 inactive (15504 mPMT + 3532 OD)
    assert len(det.inactive_positions_mm) == 19036
    cats = sorted(set(det.inactive_categories.tolist()))
    assert cats == ['id_type2', 'od']


@pytest.mark.slow                 # builds SK / HK / WCTE detectors
def test_wcte_metadata_present():
    det = Cylinder.from_pmt_file(os.path.join(REPO_CONFIG, 'wcte_geometry.npz'))
    assert hasattr(det, 'mpmt_id')
    assert hasattr(det, 'pmt_id_in_mpmt')
    assert det.mpmt_id.shape == (1995,)
    # NuPRISM-derived WCTE has 105 mPMTs (one id in 0..105 is skipped)
    # and 19 PMT slots numbered 1..19 within each mPMT.
    assert len(np.unique(det.mpmt_id)) == 105
    assert int(det.pmt_id_in_mpmt.min()) == 1 and int(det.pmt_id_in_mpmt.max()) == 19


# Note: SK_official and WCTE_official are intentionally NOT covered
# here because their underlying source data (ConnectionTable_SK5.root
# and the WCTE Geometry-package extract) are not distributed broadly
# — tests must work for users who only have the public .txt geofiles.


# ── generate_detector dispatch ────────────────────────────────────────


@pytest.mark.parametrize("config_name,n_active", [
    ('SK_geom_config.json',            11096),
    ('HK_geom_config.json',            19746),
    ('WCTE_geom_config.json',           1995),
])
@pytest.mark.slow                 # builds SK / HK / WCTE detectors
def test_generate_detector_loads_via_npz_path(config_name, n_active):
    det = generate_detector(os.path.join(REPO_CONFIG, config_name))
    assert isinstance(det, Cylinder)
    assert len(det.all_points) == n_active
