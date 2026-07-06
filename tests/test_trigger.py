"""Unit tests for lucid.simulation.trigger (sliding-window readout trigger)."""
import numpy as np

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
    assert gates[0, 0] <= 1000.0 and gates[0, 1] >= 1005.0
    assert gates[0, 1] < 5000.0
    assert hits_in_gates(np.concatenate([burst, noise]), gates)[:15].all()


def test_below_threshold_no_gate():
    cfg = TriggerConfig(window_ns=200.0, n_thr=10, pad_before_ns=30.0, pad_after_ns=30.0)
    # 9 hits in-window, threshold 10 -> never fires
    gates = find_trigger_gates(np.linspace(0.0, 100.0, 9), cfg)
    assert gates.shape[0] == 0
    assert not hits_in_gates(np.array([50.0]), gates).any()


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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} trigger tests passed.")


if __name__ == "__main__":
    _run_all()
