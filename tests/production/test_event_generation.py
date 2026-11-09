"""Production event generation, run end to end on CPU with a stub kernel.

Before this file no test ran the three drivers production runs
(``lucid.production.dataprod.drivers.run_{particles,pileup,supernova}``): every writer
test fed a hand-built event dict, so the code that builds that dict (reader,
categorizer, segment filter, translation, rotation, t0 shift, digitizer,
trigger / selection, pile-up merge, supernova chunking, containment) had no
test at all.

Two things are stubbed, nothing else:

* ``uproot``: an in-memory PhotonSim file that serves exactly the branches
  ``root_reader.read_event_raw`` reads (in mm, as PhotonSim writes them), so
  the real reader and its mm -> m conversion run.
* the optical kernel: every photon is detected with charge 1 at the sensor
  nearest to the point 1 m along its direction, after a fixed time of flight
  ``TOF_NS``. It records the photons it was handed.

Both stubs are trivial on purpose: the expected file contents then follow by
hand from the event built in each test (photon counts, first-arrival times,
segment geometry), which is the oracle. No SIREN, no GPU, no downloaded data.
"""
import json
import sys
import types

import h5py
import numpy as np
import pytest

import jax.numpy as jnp

from lucid.geometry.cylinder import Cylinder
from lucid.production.dataprod import drivers
from lucid.simulation.readout import Readout
from lucid.simulation.trigger import TriggerConfig, find_trigger_gates
from lucid.sources.placement import detector_bounds, random_rotation, sample_vertex
from lucid.dataset.schema import EMISSION_PROCESS_DARK

TOF_NS = np.float32(7.25)        # stub time of flight; exact in float32
CHER_COS = np.cos(np.deg2rad(42.0))
R_DET, H_DET, S_RADIUS = 2.0, 4.0, 0.04


# --------------------------------------------------------------------------
# fake PhotonSim ROOT file
# --------------------------------------------------------------------------
class _Branch:
    def __init__(self, values):
        self._values = values

    def array(self, library="np", entry_start=None, entry_stop=None):
        return np.asarray(self._values[entry_start:entry_stop])


class _Tree:
    def __init__(self, rows):
        self._rows = rows

    @property
    def num_entries(self):
        return len(self._rows)

    def keys(self):
        return list(self._rows[0].keys()) if self._rows else []

    def __getitem__(self, name):
        return _Branch([r[name] for r in self._rows])

    def arrays(self, branches, entry_start=0, entry_stop=None, library="np"):
        rows = self._rows[entry_start:entry_stop]
        return {b: [r[b] for r in rows] for b in branches}


class _RootFile:
    def __init__(self, events):
        self.trees = {"OpticalPhotons": _Tree([e["tree"] for e in events]),
                      "OpticalPhotonsRaw": _Tree([c for e in events for c in e["chunks"]])}

    def __getitem__(self, name):
        return self.trees[name]

    def __contains__(self, name):
        return name in self.trees

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_root(monkeypatch):
    """Register in-memory PhotonSim files: ``fake_root(path, [event, ...])``."""
    files = {}
    mod = types.SimpleNamespace(open=lambda p: files[str(p)])
    monkeypatch.setitem(sys.modules, "uproot", mod)

    def _register(path, events):
        for i, ev in enumerate(events):
            for c in ev["chunks"]:
                c["EventID"] = i
        files[str(path)] = _RootFile(events)
        return str(path)
    return _register


def _cone(axis, n):
    """``n`` unit vectors at 42 degrees around ``axis``, evenly spread in azimuth."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, h); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    phi = 2 * np.pi * (np.arange(n) + 0.5) / n
    s = np.sqrt(1 - CHER_COS ** 2)
    return CHER_COS * a + s * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)


def _event(tracks, segments, split_chunks=False):
    """One PhotonSim entry.

    ``tracks``: (tid, parent, pdg, KE MeV, creator process).
    ``segments``: (tid, start_m, end_m, time_ns, n_cherenkov), grouped by
    ascending tid as PhotonSim writes them. Each segment emits ``n_cherenkov``
    photons at fractions (k+1/2)/n along it, on a 42 degree cone around it, at
    ``time + 0.25 k`` ns (binary fractions, so float32 sums stay exact).

    Returns the branch rows and a truth dict (photons in emission order, m).
    """
    seg_start = np.array([s[1] for s in segments], float)
    seg_end = np.array([s[2] for s in segments], float)
    d = seg_end - seg_start
    seg_dir = d / np.linalg.norm(d, axis=1, keepdims=True)
    ph_o, ph_d, ph_t, ph_seg, ph_frac = [], [], [], [], []
    for j, (_tid, _a, _b, t, nch) in enumerate(segments):
        if nch == 0:
            continue
        f = (np.arange(nch) + 0.5) / nch
        ph_o.append(seg_start[j] + f[:, None] * d[j])
        ph_d.append(_cone(seg_dir[j], nch))
        ph_t.append(t + 0.25 * np.arange(nch))
        ph_seg.append(np.full(nch, j))
        ph_frac.append(f)
    ph_o = np.concatenate(ph_o) if ph_o else np.zeros((0, 3))
    ph_d = np.concatenate(ph_d) if ph_d else np.zeros((0, 3))
    ph_t = np.concatenate(ph_t) if ph_t else np.zeros(0)
    ph_seg = np.concatenate(ph_seg).astype(np.int64) if ph_seg else np.zeros(0, np.int64)
    mm = ph_o * 1000.0
    pieces = [slice(0, len(ph_t))]
    if split_chunks and len(ph_t) > 1:
        k = len(ph_t) // 2
        pieces = [slice(k, None), slice(0, k)]     # stored out of order on purpose
    chunks = []
    for sl in pieces:
        chunks.append({
            "ChunkStartID": sl.start,
            "PhotonPosX": mm[sl, 0], "PhotonPosY": mm[sl, 1], "PhotonPosZ": mm[sl, 2],
            "PhotonDirX": ph_d[sl, 0], "PhotonDirY": ph_d[sl, 1], "PhotonDirZ": ph_d[sl, 2],
            "PhotonTime": ph_t[sl], "PhotonWavelength": np.full(len(ph_t[sl]), 400.0),
        }) if len(ph_t) else None
    chunks = [c for c in chunks if c is not None]
    tid = np.array([t[0] for t in tracks], np.int64)
    first = {}
    for j, s in enumerate(segments):
        first.setdefault(s[0], j)
    tpos = np.array([seg_start[first[t]] * 1000.0 if t in first else np.zeros(3) for t in tid])
    tdir = np.array([seg_dir[first[t]] if t in first else [0, 0, 1.0] for t in tid])
    tree = {
        "PrimaryEnergy": float(tracks[0][3]),
        "TrackInfo_TrackID": tid,
        "TrackInfo_PosX": tpos[:, 0], "TrackInfo_PosY": tpos[:, 1], "TrackInfo_PosZ": tpos[:, 2],
        "TrackInfo_DirX": tdir[:, 0], "TrackInfo_DirY": tdir[:, 1], "TrackInfo_DirZ": tdir[:, 2],
        "TrackInfo_Energy": np.array([t[3] for t in tracks], float),
        "TrackInfo_Time": np.zeros(len(tracks)),
        "TrackInfo_ParentTrackID": np.array([t[1] for t in tracks], np.int64),
        "TrackInfo_PDG": np.array([t[2] for t in tracks], np.int64),
        "TrackInfo_CreatorProcess": [t[4] for t in tracks],
        "NSegments": len(segments),
        "Segment_StartX": seg_start[:, 0] * 1000, "Segment_StartY": seg_start[:, 1] * 1000,
        "Segment_StartZ": seg_start[:, 2] * 1000,
        "Segment_EndX": seg_end[:, 0] * 1000, "Segment_EndY": seg_end[:, 1] * 1000,
        "Segment_EndZ": seg_end[:, 2] * 1000,
        "Segment_DirX": seg_dir[:, 0], "Segment_DirY": seg_dir[:, 1], "Segment_DirZ": seg_dir[:, 2],
        "Segment_Edep": 2.0 * np.linalg.norm(d, axis=1),
        "Segment_Time": np.array([s[3] for s in segments], float),
        "Segment_BetaStart": np.full(len(segments), 0.99),
        "Segment_NCherenkov": np.array([s[4] for s in segments], np.int64),
        "Segment_TrackID": np.array([s[0] for s in segments], np.int64),
        "Photon_SegmentIndex": ph_seg,
        "RooTrackerEntryID": -1, "IncomingNuPdg": 0, "IncomingNuKE": 0.0,
    }
    truth = dict(origins=ph_o, dirs=ph_d, times=ph_t, seg=ph_seg, frac=ph_frac,
                 seg_start=seg_start, seg_end=seg_end, seg_dir=seg_dir,
                 seg_tid=np.array([s[0] for s in segments]),
                 seg_nch=np.array([s[4] for s in segments]))
    return {"tree": tree, "chunks": chunks, "truth": truth}


# --------------------------------------------------------------------------
# stub kernel + simulator
# --------------------------------------------------------------------------
def _nearest_sensor(origins_m, dirs, sensors):
    p = np.asarray(origins_m, float) + np.asarray(dirs, float)
    d2 = ((p[:, None, :] - sensors[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1).astype(np.int32)


class _StubSimulator:
    """Kernel stand-in with the fields of the ``Simulator`` record ``setup_event_simulator`` returns."""

    def __init__(self, detector):
        self.sensors = np.asarray(detector.all_points, float)
        self.det_geom = types.SimpleNamespace(
            detector_type="cylinder", detector=detector, sensor_points=detector.all_points,
            medium=types.SimpleNamespace(material="water", emission_processes=("cherenkov",)))
        self.reflection_model = "scalar_mix"
        self.default_detector_params = object()     # baked in: called as (track, key, photons)
        self.calls = []

    def __call__(self, key, photonsim_data):
        n = int(photonsim_data["N"])
        o_all = np.asarray(photonsim_data["photon_origins"], np.float64)     # metres
        b = o_all.shape[0]
        o = o_all[:n]
        dirs = np.asarray(photonsim_data["photon_directions"], np.float64)[:n]
        t = np.asarray(photonsim_data["photon_times"], np.float32)[:n]
        sen = _nearest_sensor(o, dirs, self.sensors)
        if n:
            self.calls.append(dict(origins=o, dirs=dirs, times=t))
        ns = self.sensors.shape[0]
        qe_w = np.zeros(b, np.float32); qe_w[:n] = 1.0
        qe_t = np.full(b, np.inf, np.float32); qe_t[:n] = t + TOF_NS
        sen_full = np.zeros(b, np.int32); sen_full[:n] = sen
        pe = np.bincount(sen, minlength=ns).astype(np.float32)
        tmin = np.full(ns, np.inf, np.float32)
        np.minimum.at(tmin, sen, t + TOF_NS)
        tmin = np.where(np.isfinite(tmin), tmin, 0.0).astype(np.float32)
        seg = np.asarray(photonsim_data["photon_segment_index"], np.int32)
        return tuple(jnp.asarray(a) for a in (pe, tmin, tmin, qe_w, qe_t, qe_t, sen_full, seg))

    def photons(self):
        """Every photon the kernel saw, concatenated in call order."""
        return {k: np.concatenate([c[k] for c in self.calls]) for k in ("origins", "dirs", "times")}


@pytest.fixture(scope="module")
def detector():
    return Cylinder(R_DET, H_DET, 120, S_RADIUS)


def _read(path):
    """Whole HDF5 file as {'attrs': config attrs, '<group path>': array}."""
    out = {}
    with h5py.File(path, "r") as f:
        out["config"] = dict(f["config"].attrs)
        f.visititems(lambda name, obj: out.__setitem__(name, obj[()])
                     if isinstance(obj, h5py.Dataset) else None)
    return out


def _files(out_dir, idx=0):
    return {k: _read(out_dir / k / f"wc_{k}_{idx:04d}.h5") for k in ("sensor", "hits", "step", "labl")}


def _check_digit_fk(sensor, hits, step, ev="event_000"):
    """hits / step decompositions point at a digit on the same sensor and sum to its charge."""
    sd_s, sd_pe = sensor[f"{ev}/sensor_idx"], sensor[f"{ev}/PE"]
    h_d, h_s, h_pe = hits[f"{ev}/digit_idx"], hits[f"{ev}/sensor_idx"], hits[f"{ev}/PE"]
    assert ((h_d >= 0) & (h_d < sd_s.size)).all()
    np.testing.assert_array_equal(h_s, sd_s[h_d])
    np.testing.assert_allclose(np.bincount(h_d, weights=h_pe, minlength=sd_s.size), sd_pe)
    s_d, s_s = step[f"{ev}/sensor_hits/digit_idx"], step[f"{ev}/sensor_hits/sensor_idx"]
    np.testing.assert_array_equal(s_s, sd_s[s_d])
    np.testing.assert_allclose(
        np.bincount(s_d, weights=step[f"{ev}/sensor_hits/PE"], minlength=sd_s.size), sd_pe)


def _check_hits_equal_step_rollup(hits, step, labl, ev="event_000"):
    """hits.h5 is step/sensor_hits rolled up segment -> track -> particle, per digit.

    Keyed by (particle, sensor, digit, emission process); PE sums, T and T_reco
    are minima. Dark rows (no segment) are excluded from the hits side.
    """
    seg_part = labl[f"{ev}/per_track/particle_idx"][step[f"{ev}/track_idx"]]
    sh = {k: step[f"{ev}/sensor_hits/{k}"] for k in
          ("segment_idx", "sensor_idx", "digit_idx", "emission_process", "PE", "T", "T_reco")}
    roll = {}
    for j, s, d, e, pe, t, tr in zip(*sh.values()):
        key = (int(seg_part[j]), int(s), int(d), int(e))
        r = roll.setdefault(key, [0.0, np.inf, np.inf])
        r[0] += pe; r[1] = min(r[1], t); r[2] = min(r[2], tr)
    dark = hits[f"{ev}/emission_process"] == EMISSION_PROCESS_DARK
    got = {(int(p), int(s), int(d), int(e)): [pe, t, tr] for p, s, d, e, pe, t, tr in zip(
        *(hits[f"{ev}/{k}"][~dark] for k in
          ("particle_idx", "sensor_idx", "digit_idx", "emission_process", "PE", "T", "T_reco")))}
    assert set(got) == set(roll)
    for k in got:
        assert got[k][0] == pytest.approx(roll[k][0], rel=1e-6)
        assert (got[k][1], got[k][2]) == (roll[k][1], roll[k][2]), k


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------
def _muon_michel_event():
    """Muon along +x leaving the R=2 m tank, a sub-Cherenkov neutron, a Michel at 2 us.

    Segment containment by construction (R = 2 m, |z| <= 2 m, vertex at origin):
    muon [in, in, straddles (start in, end out), out]; Michel [in, in].
    """
    tracks = [(1, 0, 13, 500.0, "primary"),
              (2, 1, 2112, 5.0, "muMinusCaptureAtRest"),
              (3, 1, 11, 30.0, "Decay")]
    y, z = 0.1, 0.2
    segs = [(1, (0.00, y, z), (0.75, y, z), 0.0, 6),
            (1, (0.75, y, z), (1.50, y, z), 2.5, 6),
            (1, (1.50, y, z), (2.25, y, z), 5.0, 6),
            (1, (2.25, y, z), (3.00, y, z), 7.5, 6),
            (2, (0.3, -0.5, 0.1), (0.4, -0.5, 0.1), 1.0, 0),
            (2, (0.4, -0.5, 0.1), (0.5, -0.5, 0.1), 1.5, 0),
            (3, (0.5, -0.4, -0.3), (0.5, -0.1, -0.3), 2000.0, 8),
            (3, (0.5, -0.1, -0.3), (0.5, 0.2, -0.3), 2001.0, 8)]
    return _event(tracks, segs, split_chunks=True)


def _dim_muon_event():
    """Two photons: below a 4-hit trigger, so the event is dropped."""
    return _event([(1, 0, 13, 300.0, "primary")],
                  [(1, (0.0, 0.3, 0.0), (0.5, 0.3, 0.0), 0.0, 2)])


def _expected_digits(truth, sim, t0):
    """(sensor, T, PE, particle) per digit, from the event truth and the stub rule.

    One digit per (sensor, light cluster): every cluster in these events is
    shorter than the 200 ns integration window and clusters are > 1 us apart.
    """
    sen = _nearest_sensor(truth["origins"], truth["dirs"], sim.sensors)
    arr = truth["times"].astype(np.float32) + TOF_NS
    tid = truth["seg_tid"][truth["seg"]]
    rows = {}
    for s, a, p in zip(sen, arr, tid):
        cluster = int(a // 1000)
        key = (int(s), cluster)
        r = rows.setdefault(key, [np.inf, 0, set()])
        r[0] = min(r[0], float(a)); r[1] += 1; r[2].add(int(p))
    return {k: (np.float64(v[0]) + np.float64(t0), v[1], v[2]) for k, v in rows.items()}


# --------------------------------------------------------------------------
# single-vertex driver
# --------------------------------------------------------------------------
TRIG = {"window_ns": 50.0, "n_thr": 4, "pad_before_ns": 10.0, "pad_after_ns": 20.0}
SKI_NO_DARK = {"model": "ski", "dark_rate_khz": 0.0}


def test_particles_driver_writes_what_the_event_implies(tmp_path, fake_root, detector):
    ev0, ev1 = _muon_michel_event(), _dim_muon_event()
    path = fake_root(tmp_path / "ps.root", [ev0, ev1])
    sim = _StubSimulator(detector)
    drivers.run_particles(
        sim, path, output_dir=str(tmp_path / "out"), file_index=0, n_events=2, master_seed=7,
        job_id=1, readout=Readout.from_blocks(SKI_NO_DARK, TRIG), dataset_name="t",
        apply_smearing=False, apply_translation=False, pad_size_buckets=(64,))
    f = _files(tmp_path / "out")
    sensor, hits, step, labl = f["sensor"], f["hits"], f["step"], f["labl"]

    # selection: the 2-photon event cannot fire a 4-hit trigger and is dropped
    for k in f:
        cfg = f[k]["config"]
        assert cfg["complete"] and cfg["n_events"] == 1 and cfg["n_events_requested"] == 2
        assert cfg["selection"] == "trigger" and cfg["trigger_n_thr"] == 4
    assert sensor["config"]["digitizer_model"] == "ski"
    np.testing.assert_array_equal(sensor["config/source_event_idx"], [0])
    assert "event_001/PE" not in sensor

    # reader: the kernel saw every photon of event 0, in m (PhotonSim writes mm),
    # in global photon order although the chunks were stored out of order
    tr = ev0["truth"]
    seen = sim.photons()
    np.testing.assert_allclose(seen["origins"][:len(tr["times"])], tr["origins"], atol=1e-6)

    t0 = labl["event_000/per_interaction/t0"]
    assert t0.shape == (1,) and abs(t0[0]) <= 250.0
    assert labl["event_000/per_event/t0"] == t0[0]

    # sensor.h5: one digit per (sensor, cluster), T = first arrival + t0 in
    # float64 (a float32 shift would be off by ~1e-5 ns at |t0| ~ 100 ns)
    exp = _expected_digits(tr, sim, t0[0])
    got = {(int(s), int((t - t0[0]) // 1000)): (t, pe)
           for s, t, pe in zip(sensor["event_000/sensor_idx"], sensor["event_000/T"],
                               sensor["event_000/PE"])}
    assert set(got) == set(exp)
    for k, (t, pe) in got.items():
        assert abs(t - exp[k][0]) < 1e-9, k
        assert pe == exp[k][1]
    assert sensor["event_000/T"].dtype == np.float64

    # per_window: two gates (muon, Michel). For a burst of n digits spread less
    # than W the count first reaches n_thr at the n_thr-th digit and falls below
    # it W after the (n - n_thr + 1)-th: gate = [T[n_thr-1] - pb, T[n-n_thr] + W + pa].
    pw_s, pw_e = labl["event_000/per_window/window_start"], labl["event_000/per_window/window_end"]
    off = labl["event_000/per_window/digit_offsets"]
    assert pw_s.shape == (2,)
    for w, cluster in enumerate((0, 2)):
        ts = np.sort([v[0] for k, v in exp.items() if k[1] == cluster])
        n, thr = ts.size, TRIG["n_thr"]
        assert pw_s[w] == pytest.approx(ts[thr - 1] - TRIG["pad_before_ns"], abs=1e-9)
        assert pw_e[w] == pytest.approx(ts[n - thr] + TRIG["window_ns"] + TRIG["pad_after_ns"],
                                        abs=1e-9)
        assert off[w + 1] - off[w] == n
        win_t = sensor["event_000/T"][off[w]:off[w + 1]]
        assert ((win_t >= pw_s[w]) & (win_t <= pw_e[w])).all()

    # hits.h5: per (particle, sensor, digit) photon counts; particle 0 = muon (1,),
    # particle 1 = Michel (1, 3); the neutron makes no light and no particle
    _check_digit_fk(sensor, hits, step)
    _check_hits_equal_step_rollup(hits, step, labl)
    sen = _nearest_sensor(tr["origins"], tr["dirs"], sim.sensors)
    part = np.where(tr["seg_tid"][tr["seg"]] == 1, 0, 1)
    exp_counts = {}
    for s, p in zip(sen, part):
        exp_counts[(int(p), int(s))] = exp_counts.get((int(p), int(s)), 0) + 1
    got_counts = {}
    for p, s, pe in zip(hits["event_000/particle_idx"], hits["event_000/sensor_idx"],
                        hits["event_000/PE"]):
        got_counts[(int(p), int(s))] = got_counts.get((int(p), int(s)), 0) + pe
    assert got_counts == exp_counts
    assert hits["event_000/T"].dtype == np.float64

    # step.h5: the sub-Cherenkov neutron's segments are filtered out; positions
    # in m; per-segment containment is AND of both endpoints
    keep = tr["seg_tid"] != 2
    np.testing.assert_allclose(
        np.stack([step["event_000/start_x"], step["event_000/start_y"], step["event_000/start_z"]], 1),
        tr["seg_start"][keep], atol=1e-6)
    np.testing.assert_allclose(
        np.stack([step["event_000/end_x"], step["event_000/end_y"], step["event_000/end_z"]], 1),
        tr["seg_end"][keep], atol=1e-6)
    np.testing.assert_array_equal(step["event_000/track_idx"], [0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(step["event_000/contained"], [1, 1, 0, 0, 1, 1])
    np.testing.assert_allclose(step["event_000/time"] - t0[0], [0, 2.5, 5, 7.5, 2000, 2001],
                               atol=1e-9)
    # step sensor_hits: segment index in the filtered table (raw 6, 7 -> 4, 5)
    raw_to_filt = np.cumsum(keep) - 1
    exp_seg = {}
    for s, j in zip(sen, tr["seg"]):
        exp_seg[(int(raw_to_filt[j]), int(s))] = exp_seg.get((int(raw_to_filt[j]), int(s)), 0) + 1
    got_seg = {}
    for j, s, pe in zip(step["event_000/sensor_hits/segment_idx"],
                        step["event_000/sensor_hits/sensor_idx"], step["event_000/sensor_hits/PE"]):
        got_seg[(int(j), int(s))] = got_seg.get((int(j), int(s)), 0) + pe
    assert got_seg == exp_seg

    # labl.h5: categories, genealogy CSR, per-track Cherenkov totals, containment roll-up
    np.testing.assert_array_equal(labl["event_000/per_particle/category"], [0, 1])
    np.testing.assert_array_equal(labl["event_000/per_particle/genealogy_offsets"], [0, 1, 3])
    np.testing.assert_array_equal(labl["event_000/per_particle/genealogy_data"], [1, 1, 3])
    np.testing.assert_array_equal(labl["event_000/per_track/track_id"], [1, 3])
    np.testing.assert_array_equal(labl["event_000/per_track/n_cherenkov"], [24, 16])
    np.testing.assert_array_equal(labl["event_000/per_track/particle_idx"], [0, 1])
    np.testing.assert_array_equal(labl["event_000/per_particle/contained"], [False, True])
    assert not labl["event_000/per_interaction/contained"][0]
    assert not labl["event_000/per_event/contained"]
    # untranslated, the event stays at the vertex PhotonSim recorded: the muon's start
    np.testing.assert_allclose(
        [labl["event_000/per_interaction/vertex_x"][0], labl["event_000/per_interaction/vertex_y"][0],
         labl["event_000/per_interaction/vertex_z"][0]], tr["seg_start"][0], atol=1e-7)


def test_rotation_and_translation_keep_photons_on_their_segments(tmp_path, fake_root, detector):
    """With rotation + translation on, photons and segments move together.

    Invariants of a rigid motion about the recorded vertex (the muon's start): every
    photon still sits at its fraction along its own segment, still makes 42 degrees with
    it, and every segment endpoint keeps its distance from the vertex. Rotating positions but
    not directions, or translating photons (m) and segments (mm) by different
    amounts, breaks one of them.
    """
    ev = _muon_michel_event()
    path = fake_root(tmp_path / "ps.root", [ev])
    sim = _StubSimulator(detector)
    drivers.run_particles(
        sim, path, output_dir=str(tmp_path / "out"), file_index=0, n_events=1, master_seed=11,
        job_id=1, readout=Readout.from_blocks(SKI_NO_DARK), dataset_name="t",
        apply_smearing=False, apply_translation=True, apply_rotation=True,
        pad_size_buckets=(64,))
    f = _files(tmp_path / "out")
    step, labl = f["step"], f["labl"]
    tr = ev["truth"]
    v = np.array([labl[f"event_000/per_interaction/vertex_{c}"][0] for c in "xyz"], float)
    assert np.hypot(v[0], v[1]) <= 0.9 * R_DET + 1e-6 and abs(v[2]) <= 0.9 * H_DET / 2 + 1e-6
    assert np.linalg.norm(v) > 0.05                     # really translated

    st = np.stack([step[f"event_000/start_{c}"] for c in "xyz"], 1).astype(float)
    en = np.stack([step[f"event_000/end_{c}"] for c in "xyz"], 1).astype(float)
    dr = np.stack([step[f"event_000/dir_{c}"] for c in "xyz"], 1).astype(float)
    keep = tr["seg_tid"] != 2
    v0 = tr["seg_start"][0]
    np.testing.assert_allclose(np.linalg.norm(st - v, axis=1),
                               np.linalg.norm(tr["seg_start"][keep] - v0, axis=1), atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(en - v, axis=1),
                               np.linalg.norm(tr["seg_end"][keep] - v0, axis=1), atol=1e-5)
    # the rotation really happened (the muon no longer points along +x)
    assert np.dot(dr[0], [1, 0, 0]) < 0.999
    # segment directions were rotated with the endpoints (dir is float16 on disk)
    chord = (en - st) / np.linalg.norm(en - st, axis=1, keepdims=True)
    np.testing.assert_allclose(dr, chord, atol=2e-3)

    seen = sim.photons()
    raw_to_filt = np.cumsum(keep) - 1
    j = raw_to_filt[tr["seg"]]
    frac = np.concatenate(tr["frac"])
    on_seg = st[j] + frac[:, None] * (en[j] - st[j])
    np.testing.assert_allclose(seen["origins"], on_seg, atol=1e-5)
    np.testing.assert_allclose((seen["dirs"] * chord[j]).sum(1), CHER_COS, atol=2e-3)


# --------------------------------------------------------------------------
# pile-up driver
# --------------------------------------------------------------------------
def _single_muon_event():
    return _event([(1, 0, 13, 400.0, "primary")],
                  [(1, (0.0, 0.2, 0.1), (0.6, 0.2, 0.1), 0.0, 10)])


def test_pileup_driver_pools_two_vertices_into_shared_digits(tmp_path, fake_root, detector):
    """Two identical vertices at different t0 light the same sensors.

    With the infinite-window ``basic`` digitizer each sensor is one digit whose
    charge is the sum over both vertices and whose time is the earlier vertex's
    arrival; the hits decomposition splits it back per vertex, and the second
    stream's track ids are offset past the first's.
    """
    ev_a, ev_b = _single_muon_event(), _single_muon_event()
    pa = fake_root(tmp_path / "a.root", [ev_a])
    pb = fake_root(tmp_path / "b.root", [ev_b])
    sim = _StubSimulator(detector)
    drivers.run_pileup(
        sim, [pa, pb], ["particles", "genie"], output_dir=str(tmp_path / "out"), file_index=0,
        n_events=1, master_seed=3, job_id=1, readout=Readout.from_blocks(), dataset_name="t",
        apply_smearing=False, apply_translation=False)
    f = _files(tmp_path / "out")
    sensor, hits, step, labl = f["sensor"], f["hits"], f["step"], f["labl"]
    t0 = labl["event_000/per_interaction/t0"]
    assert t0.shape == (2,) and t0[0] != t0[1]
    np.testing.assert_array_equal(labl["event_000/per_interaction/source_type"], [0, 1])
    assert labl["event_000/per_event/t0"] == t0.min()
    np.testing.assert_array_equal(labl["event_000/per_track/track_id"], [1, 3])
    np.testing.assert_array_equal(labl["event_000/per_particle/interaction_idx"], [0, 1])

    tr = ev_a["truth"]
    sen = _nearest_sensor(tr["origins"], tr["dirs"], sim.sensors)
    arr = tr["times"].astype(np.float32) + TOF_NS
    for s in np.unique(sen):
        first = float(arr[sen == s].min())
        cnt = int((sen == s).sum())
        row = np.flatnonzero(sensor["event_000/sensor_idx"] == s)
        assert row.size == 1
        assert sensor["event_000/PE"][row[0]] == 2 * cnt
        assert abs(sensor["event_000/T"][row[0]] - (first + t0.min())) < 1e-9
        for p in (0, 1):
            h = (hits["event_000/sensor_idx"] == s) & (hits["event_000/particle_idx"] == p)
            assert hits["event_000/PE"][h].sum() == cnt
            assert abs(hits["event_000/T"][h].min() - (first + t0[p])) < 1e-9
    assert set(sensor["event_000/sensor_idx"]) == set(np.unique(sen))
    _check_digit_fk(sensor, hits, step)
    _check_hits_equal_step_rollup(hits, step, labl)


# --------------------------------------------------------------------------
# supernova driver
# --------------------------------------------------------------------------
def _positron(n_photons, y0):
    return _event([(1, 0, -11, 20.0, "primary")],
                  [(1, (0.0, y0, 0.0), (0.0, y0 + 0.1, 0.05), 0.0, n_photons)])


def _distinct_sensors(ev, sim):
    tr = ev["truth"]
    return np.unique(_nearest_sensor(tr["origins"], tr["dirs"], sim.sensors)).size


def test_supernova_driver_chunks_selects_and_keeps_float64_time(tmp_path, fake_root, detector):
    """Five interactions: three 150 ns apart (one chunk), one dim, one at +1 s.

    * Chunking measures the gap between consecutive interactions (150 < 200 ns
      coupling), so the first three form ONE readout window.
    * The dim interaction (2 physics digits < min_physics_hits = 3) is dropped.
    * The one at +1 s has exactly 3 physics digits, so it is kept, and its
      digit times minus its t0 equal the stub arrival times to 1e-6 ns
      (float32 at 1e9 ns has a 64 ns step).
    """
    sim = _StubSimulator(detector)
    evs = [_positron(6, 0.0), _positron(6, 0.3), _positron(6, -0.3), _positron(2, 0.6),
           _positron(3, -0.6)]
    assert _distinct_sensors(evs[3], sim) == 2
    assert _distinct_sensors(evs[4], sim) == 3
    path = fake_root(tmp_path / "sn.root", evs)
    times_ms = [0.0, 150e-6, 300e-6, 500.0, 1000.0]
    drivers.run_supernova(
        sim, path, times_ms, interaction_channels=[0, 1, 2, 3, 4],
        sn_ordering="NMO", sn_transformation="AdiabaticMSW_NMO",
        output_dir=str(tmp_path / "out"), file_index=0, master_seed=5, job_id=1,
        readout=Readout.from_blocks(SKI_NO_DARK, None, min_physics_hits=3), dataset_name="t",
        apply_smearing=False, apply_translation=False)
    f = _files(tmp_path / "out")
    sensor, hits, step, labl = f["sensor"], f["hits"], f["step"], f["labl"]

    np.testing.assert_array_equal(labl["event_000/per_interaction/interaction_channel"],
                                  [0, 1, 2, 4])
    t0 = labl["event_000/per_interaction/t0"]
    np.testing.assert_allclose(t0 - t0[0], [0.0, 150.0, 300.0, 1e9], atol=1e-6)
    assert labl["event_000/per_window/window_start"].shape == (2,)
    assert sensor["config"]["selection"] == "min_physics_hits"
    _check_digit_fk(sensor, hits, step)
    _check_hits_equal_step_rollup(hits, step, labl)

    # the +1 s interaction: last window, 3 digits, exact float64 times
    off = labl["event_000/per_window/digit_offsets"]
    np.testing.assert_array_equal(np.diff(off), [off[1], 3])
    tr = evs[4]["truth"]
    sen = _nearest_sensor(tr["origins"], tr["dirs"], sim.sensors)
    arr = tr["times"].astype(np.float32) + TOF_NS
    exp = {int(s): float(arr[sen == s].min()) for s in np.unique(sen)}
    got_s = sensor["event_000/sensor_idx"][off[1]:]
    got_t = sensor["event_000/T"][off[1]:]
    assert {int(s) for s in got_s} == set(exp)
    for s, t in zip(got_s, got_t):
        assert abs((t - t0[3]) - exp[int(s)]) < 1e-6

    # every kept interaction's photons were all detected and attributed to it
    pi = labl["event_000/per_particle/interaction_idx"]
    for i, n_ph in enumerate((6, 6, 6, 3)):
        rows = np.isin(hits["event_000/particle_idx"], np.flatnonzero(pi == i))
        assert hits["event_000/PE"][rows].sum() == n_ph


def test_supernova_vertices_move_photons_and_segments_together(tmp_path, fake_root, detector):
    """SN translation (vertices redrawn out of the PMTs, pooled digitization).

    Vertices fill r <= 0.995 R; each interaction's segments (stored in mm) and
    photons (m) move together, its recorded vertex (the positron's start) onto the
    vertex labl records.
    """
    sim = _StubSimulator(detector)
    evs = [_positron(6, 0.0), _positron(6, 0.3)]
    path = fake_root(tmp_path / "sn.root", evs)
    drivers.run_supernova(
        sim, path, [0.0, 10.0], output_dir=str(tmp_path / "out"), file_index=0, master_seed=9,
        sn_ordering="NMO", sn_transformation="AdiabaticMSW_NMO",
        job_id=1, readout=Readout.from_blocks(SKI_NO_DARK, None, min_physics_hits=1),
        dataset_name="t", apply_smearing=False, apply_translation=True)
    f = _files(tmp_path / "out")
    step, labl = f["step"], f["labl"]
    vx = np.stack([labl[f"event_000/per_interaction/vertex_{c}"] for c in "xyz"], 1).astype(float)
    assert vx.shape == (2, 3)
    assert (np.hypot(vx[:, 0], vx[:, 1]) <= 0.995 * R_DET + 1e-6).all()
    assert (np.abs(vx[:, 2]) <= 0.995 * H_DET / 2 + 1e-6).all()
    assert np.linalg.norm(vx, axis=1).min() > 0.05
    st = np.stack([step[f"event_000/start_{c}"] for c in "xyz"], 1)
    np.testing.assert_allclose(st, vx, atol=1e-5)
    seen = sim.photons()["origins"]
    exp = np.concatenate([e["truth"]["origins"] - e["truth"]["seg_start"][0] + vx[i]
                          for i, e in enumerate(evs)])
    np.testing.assert_allclose(seen, exp, atol=1e-5)


def test_a_supernova_burst_with_no_kept_interaction_writes_a_complete_empty_batch(
        tmp_path, fake_root, monkeypatch, detector):
    """No interaction reaches min_physics_hits: the job wrote no files, so its own check and
    verify_jobs failed it, and the resubmission (same seeds) failed again, forever. Now it writes
    a 0-event batch flagged complete, which every reader of a batch accepts."""
    from lucid.dataset.layout import check_batch
    from lucid.dataset.reader import iter_events, list_events
    from lucid.production.dataprod import run_job
    from lucid.production.dataprod.scripts.verify_jobs import is_complete
    from lucid.production.qa import validations

    out = tmp_path / "out"
    out.mkdir()
    (out / "output_job_000001.root").touch()
    fake_root(out / "output_job_000001.root", [_positron(6, 0.0), _positron(2, 0.3)])
    (out / "gntp_job_000001.times.json").write_text(json.dumps([0.0, 10.0]))
    cfg = {"name": "t", "material": "water", "digitizer": SKI_NO_DARK, "trigger": None,
           "primary_source": "supernova", "supernova": {"orderings": ["NMO"]},
           "selection": {"mode": "min_physics_hits", "n": 50}}
    (out / "cfg.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(run_job, "_simulator", lambda name: _StubSimulator(detector))
    rc = run_job.main(["--config", str(out / "cfg.json"), "--detector", "SK_like",
                       "--output-dir", str(out), "--job-id", "1", "--master-seed", "3",
                       "--skip-photonsim", "--skip-genie"])
    assert rc == run_job.EXIT_OK

    ok, messages = check_batch(out, 0, expected_dataset_name="t")
    assert ok, messages
    assert any(m.startswith("EMPTY_KEPT") for m in messages)
    assert is_complete(out, 0, None) == (True, "")
    for sub, f in _files(out).items():
        c = f["config"]
        assert (c["n_events"], c["n_events_requested"], bool(c["complete"])) == (0, 1, True), sub
        assert c["selection"] == "min_physics_hits"
        path = out / sub / f"wc_{sub}_0000.h5"
        assert list(iter_events(path)) == [] and list_events(path).size == 0
    passed, detail, _ = validations.check_integrity(out)
    assert passed and detail.startswith("1 shards, 0 events"), detail
    assert all(a.size == 0 for a in validations.load_response_samples(out, 4))
    assert validations.load_primaries(out)[0].size == 0


@pytest.mark.parametrize("orderings, flag, rc, stamped", [
    (["IMO"], None, 0, ("IMO", "AdiabaticMSW_IMO")),
    (["NMO", "noosc"], "noosc", 0, ("noosc", "NoTransformation")),
    (["NMO", "IMO"], None, 10, None),       # which one? --sn-ordering must say
    (["IMO"], "NMO", 10, None),             # not an ordering the config lists
    (None, None, 10, None),                 # no ordering at all: no default
])
def test_a_direct_supernova_job_runs_and_records_the_ordering_its_config_lists(
        tmp_path, fake_root, monkeypatch, detector, orderings, flag, rc, stamped):
    """``lucid-run-job`` without ``--sn-ordering`` ran NMO whatever ``supernova.orderings`` said,
    and no batch recorded the ordering. It is now stamped as ``sn_ordering`` and
    ``sn_transformation`` on all four files."""
    from lucid.production.dataprod import run_job

    out = tmp_path / "out"
    out.mkdir()
    (out / "output_job_000001.root").touch()
    fake_root(out / "output_job_000001.root", [_positron(6, 0.0), _positron(6, 0.3)])
    (out / "gntp_job_000001.times.json").write_text(json.dumps([0.0, 10.0]))
    sn = {} if orderings is None else {"orderings": orderings}
    cfg = {"name": "t", "material": "water", "digitizer": SKI_NO_DARK, "trigger": None,
           "primary_source": "supernova", "supernova": sn,
           "selection": {"mode": "min_physics_hits", "n": 1}}
    (out / "cfg.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(run_job, "_simulator", lambda name: _StubSimulator(detector))
    argv = ["--config", str(out / "cfg.json"), "--detector", "SK_like", "--output-dir", str(out),
            "--job-id", "1", "--master-seed", "3", "--skip-photonsim", "--skip-genie"]
    assert run_job.main(argv + (["--sn-ordering", flag] if flag else [])) == rc
    if stamped is None:
        assert not (out / "sensor").exists()
        return
    for sub, f in _files(out).items():
        assert (f["config"]["sn_ordering"], f["config"]["sn_transformation"]) == stamped, sub


# --------------------------------------------------------------------------
# lucid-run-job: one master seed for every path
# --------------------------------------------------------------------------
_PARTICLES = {"particles": [{"pdg": 13}], "energy_distribution": "uniform"}
_PATHS = {
    "particles": ({**_PARTICLES}, ["output_job_000001.root"]),
    "pileup": ({"pile_up": True, "vertices": [dict(_PARTICLES), dict(_PARTICLES)]},
               ["output_job_000001_v0.root", "output_job_000001_v1.root"]),
    "supernova": ({"primary_source": "supernova", "supernova": {"orderings": ["NMO"]},
                   "selection": {"mode": "min_physics_hits", "n": 1}},
                  ["output_job_000001.root"]),
}


def _run_job(tmp_path, fake_root, monkeypatch, detector, path, seed):
    """``lucid-run-job --skip-photonsim --skip-genie --master-seed seed`` on one path, with the
    stub kernel; returns the written sensor and hits files."""
    from lucid.production.dataprod import run_job
    extra, roots = _PATHS[path]
    out = tmp_path / f"{path}_{seed}"
    out.mkdir()
    events = ([_positron(6, 0.0), _positron(6, 0.3)] if path == "supernova"
              else [_single_muon_event()])
    for name in roots:
        (out / name).touch()
        fake_root(out / name, events)
    if path == "supernova":
        (out / "gntp_job_000001.times.json").write_text(json.dumps([0.0, 10.0]))
    cfg = {"name": "t", "material": "water", "n_events_per_job": 1, "digitizer": SKI_NO_DARK,
           "trigger": None, **extra}
    (out / "cfg.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(run_job, "_simulator", lambda name: _StubSimulator(detector))
    rc = run_job.main(["--config", str(out / "cfg.json"), "--detector", "SK_like",
                       "--output-dir", str(out), "--job-id", "1", "--master-seed", str(seed),
                       "--skip-photonsim", "--skip-genie"])
    assert rc == run_job.EXIT_OK
    f = _files(out)
    return {k: f[k] for k in ("sensor", "hits")}


@pytest.mark.parametrize("path", sorted(_PATHS))
def test_every_path_resolves_the_master_seed_mod_2_31_minus_1(tmp_path, fake_root, monkeypatch,
                                                              detector, path):
    """A master seed and the same seed plus 2**31 - 1 are one seed on every path: the same smeared
    digits and the same recorded ``lucid_master_seed``. The kernel keys always reduced it; the
    single-vertex digitizer stream once took it unreduced."""
    small = 7
    a = _run_job(tmp_path, fake_root, monkeypatch, detector, path, small)
    b = _run_job(tmp_path, fake_root, monkeypatch, detector, path, small + 2 ** 31 - 1)
    for kind in a:
        assert a[kind]["config"]["lucid_master_seed"] == b[kind]["config"]["lucid_master_seed"] == small
        assert sorted(a[kind]) == sorted(b[kind])
        for k in a[kind]:
            if k != "config":
                np.testing.assert_array_equal(a[kind][k], b[kind][k], err_msg=f"{kind}/{k}")
    assert a["sensor"]["event_000/T"].size > 0


# --------------------------------------------------------------------------
# helpers the drivers rely on
# --------------------------------------------------------------------------
def test_interaction_chunks_split_on_the_gap_between_neighbours():
    # consecutive gaps 150, 150, 1000, 50 ns with a 200 ns coupling distance
    t = np.array([0.0, 150.0, 300.0, 1300.0, 1350.0])
    assert drivers._group_interactions_by_gap(t, 200.0) == [[0, 1, 2], [3, 4]]
    assert drivers._group_interactions_by_gap(np.array([]), 200.0) == []
    # a gap of exactly the coupling distance still couples (strict >)
    assert drivers._group_interactions_by_gap(np.array([0.0, 200.0]), 200.0) == [[0, 1]]


def test_min_physics_hits_keeps_exactly_the_threshold_and_ignores_dark():
    # 4 digits: digits 0-2 have physics light, digit 3 is dark only
    sd = {"sensor_idx": np.array([5, 1, 3, 2], np.uint16),
          "PE": np.array([1, 2, 1, 1], np.float32),
          "T": np.array([10.0, 30.0, 20.0, 40.0])}
    hits = {"particle_idx": np.array([0, 0, 1, -1], np.int32),
            "digit_idx": np.array([0, 1, 2, 3], np.int32),
            "emission_process": np.array([0, 0, 0, EMISSION_PROCESS_DARK], np.int8)}
    assert Readout({}, min_physics_hits=4).select(sd, hits, None) is None   # dark does not count
    out = Readout({}, min_physics_hits=3).select(sd, hits, None)            # exactly 3 physics digits
    assert out is not None
    sd2, h2, _, pw = out
    np.testing.assert_array_equal(sd2["sensor_idx"], [1, 2, 3, 5])   # canonical (sensor, T)
    # each remapped row still points at the digit it described
    np.testing.assert_array_equal(sd2["sensor_idx"][h2["digit_idx"]], sd["sensor_idx"])
    np.testing.assert_array_equal(pw["digit_offsets"], [0, 4])
    assert (pw["window_start"][0], pw["window_end"][0]) == (10.0, 40.0)


def test_dark_noise_span_covers_every_gate_edge():
    """Dark is generated over [min - pad, max + pad]; every gate must sit inside.

    Worst case: exactly n_thr hits packed tighter than W. The gate then opens
    pb before the last hit and closes W + pa after the first one, i.e. up to
    W + pa past the last hit, so pad must reach max(pb, pa) + W.
    """
    for pb, pa in ((30.0, 30.0), (10.0, 80.0), (90.0, 5.0)):
        cfg = TriggerConfig(window_ns=200.0, n_thr=5, pad_before_ns=pb, pad_after_ns=pa)
        t = 1000.0 + np.linspace(0.0, 4.0, 5)
        gates = find_trigger_gates(t, cfg)
        pad = Readout({}, trigger=cfg).dark_pad_ns()
        assert gates.shape == (1, 2)
        assert gates[0, 0] >= t.min() - pad and gates[0, 1] <= t.max() + pad
        assert gates[0, 1] > t.max() + max(pb, pa)        # the window term is needed
    assert Readout({}).dark_pad_ns() == 100.0


def test_a_trigger_with_the_min_physics_hits_selection_is_refused():
    """min_physics_hits replaces the trigger in ``select``, yet ``meta`` recorded the trigger as
    applied and ``dark_pad_ns`` sized the dark span from it."""
    trig = {"window_ns": 200.0, "n_thr": 25, "pad_before_ns": 300.0, "pad_after_ns": 300.0}
    with pytest.raises(ValueError, match="replaces the trigger"):
        Readout.from_blocks(SKI_NO_DARK, trig, min_physics_hits=3)
    meta = Readout.from_blocks(SKI_NO_DARK, None, min_physics_hits=3).meta()
    assert (meta["trigger"], meta["selection"]) == ("none", "min_physics_hits")
    assert Readout.from_blocks(SKI_NO_DARK, trig).meta()["trigger_n_thr"] == 25


# --------------------------------------------------------------------------
# detector bounds and vertex sampling
# --------------------------------------------------------------------------
def test_supernova_vertices_keep_the_detectors_pmt_radius_from_every_sensor(tmp_path, fake_root):
    """The supernova path redraws a vertex within the detector's own PMT radius (``S_radius``)
    of a sensor. 0.45 m PMTs on a 2 m cylinder: about a third of the volume lies 0.25-0.45 m from
    a sensor, so 40 vertices rejected at any smaller radius (0.25 m was the old fixed value) land
    there."""
    det = Cylinder(R_DET, H_DET, 120, 0.45)
    sim = _StubSimulator(det)
    n = 40
    path = fake_root(tmp_path / "sn.root", [_positron(6, 0.0) for _ in range(n)])
    drivers.run_supernova(
        sim, path, list(np.arange(n) * 10.0), output_dir=str(tmp_path / "out"), file_index=0,
        sn_ordering="NMO", sn_transformation="AdiabaticMSW_NMO", master_seed=9, job_id=1,
        readout=Readout.from_blocks(SKI_NO_DARK, None, min_physics_hits=1), dataset_name="t",
        apply_smearing=False, apply_translation=True)
    labl = _files(tmp_path / "out")["labl"]
    vx = np.stack([labl[f"event_000/per_interaction/vertex_{c}"] for c in "xyz"], 1).astype(float)
    assert vx.shape == (n, 3)
    d = np.sqrt(((vx[:, None, :] - sim.sensors[None]) ** 2).sum(-1)).min(1)
    assert (d > 0.45).all()


def test_detector_bounds_shapes():
    cyl = types.SimpleNamespace(detector_type="Cylinder",
                                detector=types.SimpleNamespace(r=3.0, H=5.0, S_radius=0.1))
    b = detector_bounds(cyl)
    assert (b["type"], b["radius"], b["height"]) == ("cylinder", 3.0, 5.0)
    sph = types.SimpleNamespace(detector_type="sphere", detector=types.SimpleNamespace(r=4.0))
    assert detector_bounds(sph) == {"type": "sphere", "radius": 4.0}
    box = types.SimpleNamespace(detector_type="box",
                                detector=types.SimpleNamespace(L=1.0, W=2.0, H=3.0))
    assert detector_bounds(box) == {
        "type": "box", "length": 1.0, "width": 2.0, "height": 3.0}
    string = types.SimpleNamespace(detector_type="string", detector=object())
    assert detector_bounds(string) is None
    assert detector_bounds(None) is None


N_VTX = 4000


def test_fiducial_vertices_are_volume_uniform():
    """Uniform in volume: (r/r_max)^2 ~ U(0,1) in a cylinder, (r/r_max)^3 in a sphere.

    The mean of a U(0,1) sample of 4000 has sigma 0.0046; a linear-in-r draw
    gives E[(r/r_max)^2] = 1/3, 36 sigma away. Tolerance 0.02 (4.3 sigma).
    """
    rng = np.random.default_rng(0)
    cyl = {"type": "cylinder", "radius": 10.0, "height": 20.0}
    v = np.array([sample_vertex(cyl, rng) for _ in range(N_VTX)])
    r = np.hypot(v[:, 0], v[:, 1]) / (0.9 * 10.0)
    z = v[:, 2] / (0.9 * 10.0)
    assert r.max() <= 1.0 + 1e-6 and np.abs(z).max() <= 1.0 + 1e-6
    assert np.mean(r ** 2) == pytest.approx(0.5, abs=0.02)
    assert np.mean(z) == pytest.approx(0.0, abs=0.03)          # sigma 0.009
    assert np.mean(z ** 2) == pytest.approx(1 / 3, abs=0.02)   # uniform in z
    sph = {"type": "sphere", "radius": 5.0}
    v = np.array([sample_vertex(sph, rng) for _ in range(N_VTX)])
    r = np.linalg.norm(v, axis=1) / (0.9 * 5.0)
    assert r.max() <= 1.0 + 1e-6
    assert np.mean(r ** 3) == pytest.approx(0.5, abs=0.02)
    assert np.abs(v.mean(0)).max() < 0.1                       # isotropic: sigma 0.04


def test_supernova_vertices_are_redrawn_out_of_pmts():
    """With PMT rejection on no vertex lands within pmt_radius of a sensor.

    A 4x4x4 grid of 'sensors' inside a 1 m cylinder with 0.12 m spheres covers
    ~15% of the volume, so without (or with an inverted) rejection hundreds of
    the 2000 draws would land inside one.
    """
    rng = np.random.default_rng(1)
    g = np.linspace(-0.6, 0.6, 4)
    sensors = np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3)
    b = {"type": "cylinder", "radius": 1.0, "height": 2.0}
    v = np.array([sample_vertex(b, rng, r_frac=0.995, z_frac=0.995,
                                            sensor_positions=sensors, pmt_radius=0.12)
                  for _ in range(2000)])
    d = np.sqrt(((v[:, None, :] - sensors[None]) ** 2).sum(-1)).min(1)
    assert (d > 0.12).all()
    free = np.array([sample_vertex(b, rng, r_frac=0.995, z_frac=0.995)
                     for _ in range(2000)])
    d_free = np.sqrt(((free[:, None, :] - sensors[None]) ** 2).sum(-1)).min(1)
    assert (d_free <= 0.12).mean() > 0.05       # the test geometry can tell


def test_random_rotations_are_proper_and_uniform():
    """Haar-uniform SO(3): det +1, orthogonal, E[R] = 0, R z uniform on the sphere.

    Each entry of a Haar rotation has mean 0 and variance 1/3, so the mean of
    3000 draws has sigma 0.011; the z-component of R e_z is U(-1, 1), so
    E[c^2] = 1/3 (sigma 0.0054).
    """
    rng = np.random.default_rng(2)
    Rs = np.array([random_rotation(rng) for _ in range(3000)])
    np.testing.assert_allclose(np.linalg.det(Rs), 1.0, atol=1e-12)
    np.testing.assert_allclose(Rs @ Rs.transpose(0, 2, 1), np.broadcast_to(np.eye(3), Rs.shape),
                               atol=1e-12)
    assert np.abs(Rs.mean(0)).max() < 0.05
    c = Rs[:, 2, 2]
    assert np.mean(c ** 2) == pytest.approx(1 / 3, abs=0.025)
