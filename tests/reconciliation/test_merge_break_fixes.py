"""Regression tests for two merge breaks fixed after the unification↔refactor-v2 audit:

1. event_io schema routing — `lucid.sources.event_io.read_photon_data_from_photonsim` (the
   monolith reader bound by `lucid-optimize`/`generate.py`) must handle BOTH the legacy
   `OpticalPhotons` schema AND the current chunked `OpticalPhotonsRaw` schema, returning
   positions in its historical CM convention either way. Before the fix it raised
   `KeyInFileError: 'PhotonPosX'` on new ROOTs.

2. scintillation data-mode — the WbLS data path reads scintillation scalars off the NESTED
   `DetectorParams.scintillation` sub-tuple; the flat `dp.S` access raised AttributeError.
"""
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")
ak = pytest.importorskip("awkward")


def _write_legacy_root(path, n=50, energy=1000.0):
    """Legacy schema: per-photon branches live directly on OpticalPhotons (positions in mm)."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(-100, 4700, (n, 3))   # mm
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    with uproot.recreate(path) as f:
        f["OpticalPhotons"] = {
            "PrimaryEnergy": np.array([energy], np.float64),
            "PhotonPosX": ak.Array([pos[:, 0]]), "PhotonPosY": ak.Array([pos[:, 1]]),
            "PhotonPosZ": ak.Array([pos[:, 2]]),
            "PhotonDirX": ak.Array([d[:, 0]]), "PhotonDirY": ak.Array([d[:, 1]]),
            "PhotonDirZ": ak.Array([d[:, 2]]), "PhotonTime": ak.Array([rng.uniform(0, 10, n)]),
        }
    return pos, d


def _write_raw_root(path, chunks=(30, 20), energy=1000.0):
    """Current schema: OpticalPhotons holds only metadata; per-photon scalars are chunked in
    OpticalPhotonsRaw (positions in mm, with PhotonWavelength)."""
    rng = np.random.default_rng(1)
    px, py, pz, dx, dy, dz, tt, wl, eid, csid = ([] for _ in range(10))
    start = 0
    allpos = []
    for k in chunks:
        p = rng.uniform(-100, 4500, (k, 3)); allpos.append(p)
        dd = rng.normal(size=(k, 3)); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
        px.append(p[:, 0]); py.append(p[:, 1]); pz.append(p[:, 2])
        dx.append(dd[:, 0]); dy.append(dd[:, 1]); dz.append(dd[:, 2])
        tt.append(rng.uniform(0, 10, k)); wl.append(rng.uniform(275, 674, k))
        eid.append(0); csid.append(start); start += k
    with uproot.recreate(path) as f:
        f["OpticalPhotons"] = {"PrimaryEnergy": np.array([energy], np.float64),
                               "NOpticalPhotons": np.array([sum(chunks)], np.int64)}
        f["OpticalPhotonsRaw"] = {
            "EventID": np.array(eid, np.int64), "ChunkStartID": np.array(csid, np.int64),
            "PhotonPosX": ak.Array(px), "PhotonPosY": ak.Array(py), "PhotonPosZ": ak.Array(pz),
            "PhotonDirX": ak.Array(dx), "PhotonDirY": ak.Array(dy), "PhotonDirZ": ak.Array(dz),
            "PhotonTime": ak.Array(tt), "PhotonWavelength": ak.Array(wl),
        }
    return np.concatenate(allpos)   # mm


def test_event_io_reads_legacy_schema_cm(tmp_path):
    from lucid.sources.event_io import read_photon_data_from_photonsim as mono
    pos_mm, _ = _write_legacy_root(str(tmp_path / "legacy.root"))
    r = mono(str(tmp_path / "legacy.root"), 0)
    o = np.asarray(r["photon_origins"])
    assert o.shape == pos_mm.shape
    np.testing.assert_allclose(o, pos_mm / 10.0, rtol=1e-5)   # mm -> cm
    assert "wavelengths" not in r and float(r["energy"]) == 1000.0


def test_event_io_routes_to_raw_schema_no_keyerror(tmp_path):
    """The bug: monolith reader raised KeyInFileError 'PhotonPosX' on new ROOTs. Now it routes
    to the chunked reader and returns CM + wavelengths."""
    from lucid.sources.event_io import read_photon_data_from_photonsim as mono
    from lucid.sources.root_reader import read_photon_data_from_photonsim as canon
    pos_mm = _write_raw_root(str(tmp_path / "raw.root"))
    r = mono(str(tmp_path / "raw.root"), 0)              # must NOT raise
    o = np.asarray(r["photon_origins"])
    assert o.shape[0] == pos_mm.shape[0]
    np.testing.assert_allclose(o, pos_mm / 10.0, rtol=1e-4)   # CM convention preserved
    assert "wavelengths" in r
    # consistency with the canonical (meters) reader: monolith == 100x canonical
    oc = np.asarray(canon(str(tmp_path / "raw.root"), 0)["photon_origins"])
    np.testing.assert_allclose(o, oc * 100.0, rtol=1e-4)


def test_particle_reader_rejects_chunked_schema(tmp_path):
    """The legacy particle-genealogy reader read_particle_data_from_photonsim only
    supports the legacy (per-photon-on-OpticalPhotons) schema. On a chunked
    OpticalPhotonsRaw ROOT it must raise a clear, actionable error pointing at the
    v3 chain — NOT a cryptic uproot KeyInFileError deep in tree.arrays()."""
    from lucid.sources.event_io import read_particle_data_from_photonsim
    _write_raw_root(str(tmp_path / "raw.root"))
    with pytest.raises(KeyError, match="chunked PhotonSim schema|OpticalPhotonsRaw"):
        read_particle_data_from_photonsim(str(tmp_path / "raw.root"), 0)


def test_scintillation_params_nested_access():
    """The WbLS data path reads scintillation scalars via .scintillation.*; the flat alias does
    not exist (and must not be reintroduced)."""
    import os
    from lucid.detector_params import load_detector_params
    cfg = "config/SK_like_wbls_physics_config.json"
    if not os.path.exists(cfg):
        pytest.skip("no wbls physics config")
    dp = load_detector_params(cfg, num_sensors=64)
    # the seven scalars event_generation.py reads must be reachable on the nested sub-tuple
    for k in ("S", "kB", "C", "tau_rise", "tau_fall", "moyal_loc", "moyal_scale"):
        assert np.isfinite(float(getattr(dp.scintillation, k)))
    assert float(dp.scintillation.S) > 0.0          # wbls actually scintillates
    with pytest.raises(AttributeError):              # the old (broken) flat access
        _ = dp.S
