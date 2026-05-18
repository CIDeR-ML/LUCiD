"""
Tests for :meth:`lucid.geometry.Cylinder.from_pmt_file` — the unified
loader that builds a :class:`Cylinder` from a PMT-positions ``.npz``
file. See ``lucid/geometry/PMT_NPZ_SCHEMA.md`` for the schema.
"""

import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

from lucid.geometry import Cylinder, generate_detector


REPO_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'config')
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


def test_active_block_reordered_to_barrel_top_bottom(tmp_path):
    """The active arrays in the npz can be in any order; the loader
    reorders them to (barrel, top, bottom) for compatibility with
    Cylinder's ID_to_case convention."""
    arrs = _minimal_valid_arrays()
    # Shuffle: top, barrel, bottom, barrel
    perm = [2, 0, 3, 1]
    for k in ('positions_mm', 'directions', 'surfaces', 'pmt_id'):
        arrs[k] = arrs[k][perm]
    p = _write_npz(str(tmp_path), **arrs)
    det = Cylinder.from_pmt_file(p)
    # First two are barrel, then top, then bottom
    assert det.ID_to_case[0] == 0 and det.ID_to_case[1] == 0  # barrel
    assert det.ID_to_case[2] == 1                              # top
    assert det.ID_to_case[3] == 2                              # bottom


def test_per_pmt_metadata_reordered(tmp_path):
    """A per-active-PMT metadata array (length N) must be reordered
    consistently with positions/directions."""
    arrs = _minimal_valid_arrays()
    # custom metadata: integers that we can trace through the reorder
    arrs['custom_meta'] = np.array([100, 200, 300, 400], dtype=np.int32)
    p = _write_npz(str(tmp_path), **arrs)
    det = Cylinder.from_pmt_file(p)
    # In input order: barrel(100), barrel(200), top(300), bottom(400)
    # After reorder (barrel, top, bottom): 100, 200, 300, 400 (same here)
    npt.assert_array_equal(det.custom_meta, [100, 200, 300, 400])


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


@pytest.mark.parametrize("label,npz_name,n_active,has_inactive", REAL_DETECTORS)
def test_real_detector_loads(label, npz_name, n_active, has_inactive):
    det = Cylinder.from_pmt_file(os.path.join(REPO_CONFIG, npz_name))
    assert isinstance(det, Cylinder)
    assert len(det.all_points) == n_active
    assert det.r > 0 and det.H > 0 and det.S_radius > 0


@pytest.mark.parametrize("label,npz_name,n_active,has_inactive", REAL_DETECTORS)
def test_real_detector_snap_lands_on_geometry(label, npz_name, n_active, has_inactive):
    det = Cylinder.from_pmt_file(os.path.join(REPO_CONFIG, npz_name))
    barrel_rxy = np.linalg.norm(det.barr_points[:, :2], axis=1)
    npt.assert_allclose(barrel_rxy, det.r, atol=1e-9)
    npt.assert_allclose(det.tcap_points[:, 2],  det.H / 2,  atol=1e-9)
    npt.assert_allclose(det.bcap_points[:, 2], -det.H / 2, atol=1e-9)


@pytest.mark.parametrize("label,npz_name,n_active,has_inactive", REAL_DETECTORS)
def test_real_detector_pmt_directions_shape(label, npz_name, n_active, has_inactive):
    det = Cylinder.from_pmt_file(os.path.join(REPO_CONFIG, npz_name))
    assert det.pmt_directions.shape == (n_active, 3)
    norms = np.linalg.norm(det.pmt_directions, axis=1)
    npt.assert_allclose(norms, 1.0, atol=1e-6)


@pytest.mark.parametrize("label,npz_name,n_active,has_inactive", REAL_DETECTORS)
def test_real_detector_pmt_id_lookup_roundtrips(label, npz_name, n_active, has_inactive):
    det = Cylinder.from_pmt_file(os.path.join(REPO_CONFIG, npz_name))
    # Pick a few PMTs and verify the lookup map agrees with the array
    for i in (0, n_active // 2, n_active - 1):
        pid = int(det.pmt_id[i])
        assert det.pmt_id_to_idx[pid] == i


def test_hk_inactive_block_present():
    det = Cylinder.from_pmt_file(os.path.join(REPO_CONFIG, 'hk_geometry.npz'))
    assert hasattr(det, 'inactive_positions_mm')
    assert hasattr(det, 'inactive_categories')
    # 19036 inactive (15504 mPMT + 3532 OD)
    assert len(det.inactive_positions_mm) == 19036
    cats = sorted(set(det.inactive_categories.tolist()))
    assert cats == ['id_type2', 'od']


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
def test_generate_detector_loads_via_npz_path(config_name, n_active):
    det = generate_detector(os.path.join(REPO_CONFIG, config_name))
    assert isinstance(det, Cylinder)
    assert len(det.all_points) == n_active


def test_generate_detector_algorithmic_path_unchanged():
    """The algorithmic Cylinder construction path (no npz_file_path
    in the JSON) must keep working — used by SK_like / WCTE_like /
    BigHK / IWCD / MidBox / EOS / TAO."""
    det = generate_detector(os.path.join(REPO_CONFIG, 'WCTE_like_geom_config.json'))
    assert isinstance(det, Cylinder)
    # algorithmic placement targets 2500 sensors; actual is 2444
    assert det.n_sensors == 2500
    assert len(det.all_points) == 2444
