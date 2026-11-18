"""Unit tests for lucid.simulation.trigger (sliding-window readout trigger)."""
import numpy as np
import pytest

from lucid.simulation.trigger import (
    TriggerConfig, find_trigger_gates, assign_windows, hits_in_gates, apply_trigger,
)


def test_single_burst_one_gate():
    cfg = TriggerConfig(window_ns=200.0, n_thr=10, pad_before_ns=30.0, pad_after_ns=30.0)
    # 15 hits in a fast ~5 ns rise (physical Cherenkov burst) -> the count clears
    # threshold near the burst start, so the gate brackets it. Sub-threshold
    # noise far away is ignored. (A burst that rises slower than pad_ns would
    # lose its leading edge — the trailing-window pre-trigger is only pad_ns.)
    burst = np.linspace(1000.0, 1005.0, 15)
    noise = np.array([5000.0, 5001.0, 9000.0])          # only 2-in-W, below n_thr=10
    gates = find_trigger_gates(np.concatenate([burst, noise]), cfg)
    assert gates.shape == (1, 2)
    # By hand (hit k at 1000 + k*5/14): the count reaches 10 at the 10th hit
    # (k = 9) and drops below 10 when the 6th hit (k = 5) expires at +200 ns.
    # Gate = [1003.214 - 30, 1201.786 + 30] = [973.214, 1231.786].
    np.testing.assert_allclose(gates[0], [1000.0 + 9 * 5 / 14 - 30.0,
                                          1000.0 + 5 * 5 / 14 + 200.0 + 30.0], atol=1e-9)
    assert hits_in_gates(np.concatenate([burst, noise]), gates)[:15].all()


def test_exactly_n_thr_hits_fire_and_one_fewer_does_not():
    cfg = TriggerConfig(window_ns=200.0, n_thr=10, pad_before_ns=30.0, pad_after_ns=30.0)
    # 9 hits in-window, threshold 10 -> never fires
    gates = find_trigger_gates(np.linspace(0.0, 100.0, 9), cfg)
    assert gates.shape[0] == 0
    assert not hits_in_gates(np.array([50.0]), gates).any()
    # 10 hits in-window -> fires: opens at the 10th hit (100 ns), closes when
    # the first one expires (0 + 200 ns)
    gates = find_trigger_gates(np.linspace(0.0, 100.0, 10), cfg)
    np.testing.assert_allclose(gates, [[100.0 - 30.0, 200.0 + 30.0]])


def test_hits_exactly_one_window_apart_coincide():
    """The count is over [t - W, t]: at a tie the arrival is counted before the expiry.

    Two hits exactly W apart therefore reach n_thr = 2 (for an instant);
    two hits W + 1 ps apart do not.
    """
    cfg = TriggerConfig(window_ns=200.0, n_thr=2, pad_before_ns=10.0, pad_after_ns=10.0)
    gates = find_trigger_gates(np.array([0.0, 200.0]), cfg)
    np.testing.assert_allclose(gates, [[190.0, 210.0]])
    assert find_trigger_gates(np.array([0.0, 200.001]), cfg).shape == (0, 2)


def test_overlapping_padded_gates_merge():
    """Bursts whose gates overlap only through their pads become one window."""
    cfg = TriggerConfig(window_ns=10.0, n_thr=3, pad_before_ns=30.0, pad_after_ns=30.0)
    # burst 1: gate [2 - 30, 0 + 10 + 30] = [-28, 40]; burst 2: [62 - 30, 60 + 40] = [32, 100]
    t = np.array([0.0, 1.0, 2.0, 60.0, 61.0, 62.0])
    np.testing.assert_allclose(find_trigger_gates(t, cfg), [[-28.0, 100.0]])
    # without the overlap (burst 2 later by 20 ns) they stay two gates
    t2 = np.array([0.0, 1.0, 2.0, 80.0, 81.0, 82.0])
    np.testing.assert_allclose(find_trigger_gates(t2, cfg), [[-28.0, 40.0], [52.0, 120.0]])


def test_two_separated_bursts_two_gates():
    cfg = TriggerConfig(window_ns=200.0, n_thr=10, pad_before_ns=30.0, pad_after_ns=30.0)
    b1 = np.linspace(1000.0, 1040.0, 15)
    b2 = np.linspace(6000.0, 6040.0, 15)                # >> W+pad away
    gates = find_trigger_gates(np.concatenate([b1, b2]), cfg)
    assert gates.shape[0] == 2
    assert gates[0, 1] < gates[1, 0]


def test_assign_windows_and_keep_mask():
    cfg = TriggerConfig(window_ns=200.0, n_thr=10, pad_before_ns=30.0, pad_after_ns=30.0)
    b1 = np.linspace(1000.0, 1040.0, 15)
    b2 = np.linspace(6000.0, 6040.0, 15)
    t = np.concatenate([b1, b2, [3000.0]])              # last hit is between gates
    gates = find_trigger_gates(t, cfg)
    win = assign_windows(t, gates)
    assert (win[:15] == 0).all()                        # first burst -> window 0
    assert (win[15:30] == 1).all()                      # second burst -> window 1
    assert win[-1] == -1                                # the lone in-between hit -> no window
    keep = hits_in_gates(t, gates)
    assert keep[:30].all() and not keep[-1]
    # gate edges are inclusive
    g = gates[0]
    np.testing.assert_array_equal(
        assign_windows(np.array([g[0] - 1e-6, g[0], g[1], g[1] + 1e-6]), gates), [-1, 0, 0, -1])


def _synth_event(sensor, T):
    """One-hit-per-digit synthetic (sensor_digits, hits, seg) for apply_trigger."""
    n = len(T)
    base = dict(sensor_idx=np.asarray(sensor, np.uint16), PE=np.ones(n, np.float32),
                T=np.asarray(T, np.float32))
    dec = dict(particle_idx=np.zeros(n, np.int32), digit_idx=np.arange(n, dtype=np.int32),
               sensor_idx=base["sensor_idx"], PE=base["PE"], T=base["T"],
               T_reco=base["T"], emission_process=np.zeros(n, np.int8))
    seg = dict(dec, segment_idx=np.zeros(n, np.int32))
    return base, dec, seg


def test_apply_trigger_filters_sorts_remaps():
    cfg = TriggerConfig(window_ns=200.0, n_thr=10, pad_before_ns=30.0, pad_after_ns=30.0)
    # burst A (12 hits ~1000 ns) + burst B (12 hits ~6000 ns) + 1 isolated (3000 ns)
    sensor = np.concatenate([np.arange(12), np.arange(12), [5]])
    T = np.concatenate([1000 + np.arange(12) * 2.0, 6000 + np.arange(12) * 2.0, [3000.0]])
    sd, hits, seg = _synth_event(sensor, T)
    res = apply_trigger(sd, hits, seg, cfg)
    assert res is not None
    nsd, nhits, nseg, pw = res
    assert nsd["sensor_idx"].shape[0] == 24                     # isolated digit dropped
    assert pw["window_start"].shape[0] == 2
    assert list(pw["digit_offsets"]) == [0, 12, 24]            # CSR: 12 per window
    # canonical sort: window 0 first (T~1000), window 1 (T~6000); sensor-sorted within
    assert (nsd["T"][:12] < 2000).all() and (nsd["T"][12:] > 5000).all()
    assert (np.diff(nsd["sensor_idx"][:12]) >= 0).all()
    # digit_idx remapped to a clean 0..23 bijection; the dropped digit's row is gone
    assert nhits["PE"].shape[0] == 24
    assert sorted(nhits["digit_idx"].tolist()) == list(range(24))
    assert nseg["digit_idx"].max() < 24
    # each surviving decomposition row still describes the digit it pointed at:
    # in _synth_event row i belongs to old digit i, so its (sensor, T) must be
    # the new digit's (sensor, T)
    for tbl in (nhits, nseg):
        np.testing.assert_array_equal(tbl["sensor_idx"], nsd["sensor_idx"][tbl["digit_idx"]])
        np.testing.assert_array_equal(tbl["T"], nsd["T"][tbl["digit_idx"]])


def test_apply_trigger_no_trigger_returns_none():
    cfg = TriggerConfig(window_ns=200.0, n_thr=10, pad_before_ns=30.0, pad_after_ns=30.0)
    sd, hits, seg = _synth_event(np.arange(5), np.linspace(0, 1000, 5))   # <=2 in any W
    assert apply_trigger(sd, hits, seg, cfg) is None


def test_config_from_block():
    assert TriggerConfig.from_block(None) is None
    c = TriggerConfig.from_block({"n_thr": 25, "pad_ns": 40})              # pad_ns shorthand
    assert c.n_thr == 25 and c.pad_before_ns == 40 and c.pad_after_ns == 40
    c2 = TriggerConfig.from_block({"pad_before_ns": 10, "pad_after_ns": 50})
    assert c2.pad_before_ns == 10 and c2.pad_after_ns == 50


def test_empty():
    cfg = TriggerConfig()
    assert find_trigger_gates(np.array([]), cfg).shape == (0, 2)
    assert assign_windows(np.array([]), np.empty((0, 2))).shape == (0,)


def test_unknown_trigger_key_raises():
    """SIM-32: from_block dropped unknown keys, so "n_threshold" ran n_thr=30.

    Every production trigger block reaches from_block through
    ``Readout.from_blocks``, so the strict check sits there.
    """
    from lucid.simulation.readout import Readout

    assert Readout.from_blocks(trigger=None).trigger is None
    canonical = {"window_ns": 200.0, "n_thr": 25,
                 "pad_before_ns": 300.0, "pad_after_ns": 300.0}
    assert Readout.from_blocks(trigger=canonical).trigger == TriggerConfig(**canonical)
    # `_`-prefixed comment keys stay allowed: the shipped physics configs use
    # "_note" for their validation provenance.
    assert Readout.from_blocks(trigger={**canonical, "_note": "frozen"}).trigger is not None
    # The symmetric shorthand from_block understands is allowed too.
    assert Readout.from_blocks(trigger={"pad_ns": 40}).trigger.pad_after_ns == 40

    with pytest.raises(ValueError, match="n_threshold"):
        Readout.from_blocks(trigger={"window_ns": 200.0, "n_threshold": 45})
    with pytest.raises(ValueError, match="did you mean 'n_thr'"):
        Readout.from_blocks(trigger={"n_thrr": 45})


def _study_dataset(root, trigger_attrs):
    """One-event dataset: 40 physics digits in 20 ns, as the trigger study reads it."""
    import h5py
    for sub in ("sensor", "hits"):
        (root / sub).mkdir(parents=True)
    with h5py.File(root / "sensor" / "wc_sensor_0000.h5", "w") as f:
        f.require_group("config").attrs.update(trigger_attrs)
        f.create_dataset("event_000/T", data=np.linspace(100.0, 120.0, 40))
    with h5py.File(root / "hits" / "wc_hits_0000.h5", "w") as f:
        f.create_dataset("event_000/digit_idx", data=np.arange(40, dtype=np.int32))
        f.create_dataset("event_000/emission_process", data=np.zeros(40, np.int8))
    return str(root)


def test_trigger_study_needs_a_recorded_or_given_trigger(tmp_path, capsys):
    """An untriggered dataset records no trigger_*; the study used to run W=200, N_thr=30,
    pad 30 (production runs N_thr 25 / 45, pad 300) and report that as the dataset's trigger."""
    from lucid.production.qa import trigger_study
    bare = _study_dataset(tmp_path / "bare", {"trigger": "none"})
    with pytest.raises(SystemExit) as e:
        trigger_study.main([bare])
    assert e.value.code == 2
    assert "records no trigger window_ns, n_thr, pad_before_ns, pad_after_ns" in \
        capsys.readouterr().err
    trigger_study.main([bare, "-W", "200", "-N", "45", "--pad", "300"])
    assert "N_thr=45  pad=300/300ns" in capsys.readouterr().out
    stamped = _study_dataset(tmp_path / "stamped", {
        "trigger": "sliding_window", "trigger_window_ns": 200.0, "trigger_n_thr": 25,
        "trigger_pad_before_ns": 300.0, "trigger_pad_after_ns": 300.0})
    trigger_study.main([stamped])
    assert "W=200ns  N_thr=25  pad=300/300ns" in capsys.readouterr().out
