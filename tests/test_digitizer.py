"""Unit tests for lucid.simulation.digitizer (Phase 1: standalone, not yet wired).

Runnable directly (``python tests/test_digitizer.py``) or under pytest.
"""
import numpy as np

from lucid.simulation.digitizer import (
    MODEL_PRESETS, resolve_model_config, digitize_event, generate_dark_noise,
    charge_resolution_sigma, apply_readout_resolution, digitize_and_decompose,
    EMISSION_PROCESS_DARK,
)


def _rng():
    return np.random.default_rng(1234)


def test_resolve_model_config():
    assert resolve_model_config(None)["model"] == "basic"
    assert resolve_model_config("ski")["integration_window_ns"] == 200.0
    m = resolve_model_config({"model": "hk", "dark_rate_khz": 5.0, "threshold_pe": 0.3})
    assert m["model"] == "hk" and m["dark_rate_khz"] == 5.0 and m["threshold_pe"] == 0.3
    # untouched preset defaults still present
    assert m["deadtime_ns"] == MODEL_PRESETS["hk"]["deadtime_ns"]
    for bad in ("nope", {"model": "sk9"}):
        try:
            resolve_model_config(bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_basic_collapses_to_one_digit_per_sensor():
    # Two sensors; sensor 0 sees a bunch AND a far-separated late bunch.
    # basic (infinite window) must still yield exactly one digit per sensor,
    # with PE = sum and T = first arrival.
    sensor = np.array([0, 0, 0, 1, 1])
    times = np.array([100.0, 101.0, 5000.0, 200.0, 202.0])
    charges = np.array([1.0, 2.0, 3.0, 0.5, 0.5])
    r = digitize_event(sensor, times, charges, n_sensors=2,
                       model=resolve_model_config("basic"))
    assert r.n_digits == 2
    # sensor 0: sum 6, first arrival 100; sensor 1: sum 1.0, first 200
    order = np.argsort(r.digit_sensor_idx)
    assert list(r.digit_sensor_idx[order]) == [0, 1]
    np.testing.assert_allclose(r.digit_pe_true[order], [6.0, 1.0])
    np.testing.assert_allclose(r.digit_time[order], [100.0, 200.0])
    # every photon assigned (no drops in basic)
    assert (r.photon_digit_idx >= 0).all()


def test_multi_hit_and_deadtime_veto():
    # ski: window 200 ns, deadtime 0 → a bunch beyond the window opens a fresh
    # digit and nothing is vetoed. The same photons under a deadtime override
    # veto the in-deadtime bunch — exercising the deadtime code path.
    sensor = np.array([7, 7, 7, 7])
    times = np.array([1000.0, 1100.0, 1600.0, 3000.0])
    charges = np.array([1.0, 1.0, 5.0, 1.5])
    # window [1000,1200] integrates the first two (pe=2); 1600 opens a digit
    # (pe=5); 3000 opens a third (pe=1.5). deadtime 0 → all three kept.
    r = digitize_event(sensor, times, charges, n_sensors=8,
                       model=resolve_model_config("ski"))
    assert r.n_digits == 3
    np.testing.assert_allclose(sorted(r.digit_pe_true), [1.5, 2.0, 5.0])
    assert (r.photon_digit_idx >= 0).all()
    # deadtime override: after [1000,1200] the (1200, 2100] window is dead, so
    # the 1600 bunch is vetoed; 3000 opens a new digit.
    model_dt = resolve_model_config({"model": "ski", "deadtime_ns": 900.0})
    r2 = digitize_event(sensor, times, charges, n_sensors=8, model=model_dt)
    assert r2.n_digits == 2
    np.testing.assert_allclose(sorted(r2.digit_pe_true), [1.5, 2.0])
    assert r2.photon_digit_idx[2] == -1


def test_threshold_drops_subthreshold_digit():
    model = resolve_model_config("ski")  # threshold 0.25 pe
    sensor = np.array([3, 3])
    times = np.array([500.0, 2000.0])
    charges = np.array([0.1, 1.0])   # first digit below threshold, second above
    r = digitize_event(sensor, times, charges, n_sensors=4, model=model)
    assert r.n_digits == 1
    np.testing.assert_allclose(r.digit_pe_true, [1.0])
    assert r.photon_digit_idx[0] == -1 and r.photon_digit_idx[1] == 0


def test_photon_digit_idx_conserves_charge():
    # For every emitted digit, the summed charge of its member photons equals
    # digit_pe_true — the invariant the hits.h5 decomposition relies on.
    model = resolve_model_config("ski")
    rng = _rng()
    sensor = rng.integers(0, 20, size=500)
    times = rng.uniform(0, 4000, size=500)
    charges = rng.uniform(0.3, 2.0, size=500)
    r = digitize_event(sensor, times, charges, n_sensors=20, model=model)
    for d in range(r.n_digits):
        member_charge = charges[r.photon_digit_idx == d].sum()
        np.testing.assert_allclose(member_charge, r.digit_pe_true[d], rtol=1e-5)


def test_charge_sigma_models():
    # sk_like piecewise fractional resolution
    np.testing.assert_allclose(charge_resolution_sigma(np.array([10.0]), "sk_like"), [0.12])
    np.testing.assert_allclose(charge_resolution_sigma(np.array([50.0]), "sk_like"), [0.375])
    np.testing.assert_allclose(charge_resolution_sigma(np.array([200.0]), "sk_like"), [1.0])
    # float-override path (legacy/basic only): single-pe sigma f, scales f*sqrt(Q)
    np.testing.assert_allclose(charge_resolution_sigma(np.array([1.0]), 0.1), [0.1])
    np.testing.assert_allclose(charge_resolution_sigma(np.array([4.0]), 0.1), [0.2])


def test_dark_noise_generation_and_labelling():
    rng = np.random.default_rng(7)
    n_sensors = 1000
    # 10 kHz over a 1 ms window → mu=10 per sensor → ~10k hits
    s, t, q = generate_dark_noise(n_sensors, rate_khz=10.0,
                                  t_start_ns=0.0, t_end_ns=1_000_000.0, rng=rng)
    assert s.size > 8000 and s.size < 12000        # Poisson around 10k
    assert (t >= 0).all() and (t <= 1_000_000.0).all()
    np.testing.assert_allclose(q, 1.0)
    # disabled → empty
    s0, _, _ = generate_dark_noise(100, 0.0, 0.0, 1000.0, rng)
    assert s0.size == 0
    # concatenating dark with real photons and digitizing: dark photons that
    # fall alone on an otherwise-empty sensor form their own (dark) digits.
    real_s = np.array([5]); real_t = np.array([100.0]); real_q = np.array([2.0])
    dark_s = np.array([5]); dark_t = np.array([100000.0]); dark_q = np.array([1.0])
    cat_s = np.concatenate([real_s, dark_s])
    cat_t = np.concatenate([real_t, dark_t])
    cat_q = np.concatenate([real_q, dark_q])
    is_dark = np.array([False, True])
    r = digitize_event(cat_s, cat_t, cat_q, n_sensors=10,
                       model=resolve_model_config("ski"))
    assert r.n_digits == 2
    # the dark photon lands in a distinct (later) digit; caller can tag it
    dark_digit = r.photon_digit_idx[is_dark][0]
    assert dark_digit >= 0
    np.testing.assert_allclose(r.digit_pe_true[dark_digit], 1.0)


def test_apply_readout_resolution():
    rng = _rng()
    pe_true = np.array([1.0, 5.0, 100.0])
    t = np.array([10.0, 20.0, 30.0])
    # basic (legacy sk_like, no time model): pe within a few sigma, time unchanged
    pr, tr = apply_readout_resolution(pe_true, t, resolve_model_config("basic"), rng)
    assert (pr >= 0).all()
    np.testing.assert_allclose(tr, t)  # time_model "none" for basic
    # ski applies the SPE charge + charge-dependent Gaussian time jitter -> both move
    pr2, tr2 = apply_readout_resolution(pe_true, t, resolve_model_config("ski"), _rng())
    assert not np.allclose(tr2, t)
    assert not np.allclose(pr2, pe_true) and (pr2 >= 0).all()


def test_decompose_basic_single_digit_and_conserves():
    # basic: one digit per sensor; hits/seg decomposition sums to the digit.
    sensor = np.array([0, 0, 1])
    charge = np.array([1.0, 2.0, 3.0])
    t_true = np.array([10.0, 11.0, 20.0])
    t_reco = np.array([10.5, 11.5, 20.5])
    particle = np.array([0, 0, 1])
    segment = np.array([0, 0, 5])
    emp = np.array([0, 0, 0])
    sd, hits, seg = digitize_and_decompose(
        sensor_idx=sensor, charge=charge, t_true=t_true, t_reco=t_reco,
        particle_idx=particle, segment_idx=segment, emission_process=emp,
        n_sensors=2, model=resolve_model_config("basic"), rng=_rng())
    assert sd["sensor_idx"].shape[0] == 2          # one digit per sensor
    # hits: (p0,s0,d?) PE=3 ; (p1,s1,d?) PE=3
    assert hits["particle_idx"].tolist() == [0, 1]
    np.testing.assert_allclose(sorted(hits["PE"]), [3.0, 3.0])
    # digit_idx maps to the sensor's digit; every hits row has a valid digit
    assert (hits["digit_idx"] >= 0).all() and hits["digit_idx"].max() < 2
    # seg decomposition mirrors it (real segments)
    assert sorted(seg["segment_idx"].tolist()) == [0, 5]
    np.testing.assert_allclose(sorted(seg["PE"]), [3.0, 3.0])


def test_decompose_multihit_splits_digit_idx():
    # ski: one particle, one sensor, two bunches >200 ns apart -> two digits,
    # two hits rows with different digit_idx.
    sensor = np.array([4, 4])
    charge = np.array([2.0, 1.5])
    t = np.array([1000.0, 3000.0])
    sd, hits, seg = digitize_and_decompose(
        sensor_idx=sensor, charge=charge, t_true=t, t_reco=t,
        particle_idx=np.array([0, 0]), segment_idx=np.array([0, 1]),
        emission_process=np.array([0, 0]),
        n_sensors=8, model=resolve_model_config("ski"), rng=_rng())
    assert sd["sensor_idx"].shape[0] == 2
    assert (hits["sensor_idx"] == 4).all()
    assert sorted(hits["digit_idx"].tolist()) == [0, 1]   # split across digits
    np.testing.assert_allclose(sorted(hits["PE"]), [1.5, 2.0])


def test_decompose_dark_is_labelled():
    rng = np.random.default_rng(0)
    # one real deposit + dark noise on a big detector over a long window
    sensor = np.array([2]); charge = np.array([3.0])
    t = np.array([500.0])
    sd, hits, seg = digitize_and_decompose(
        sensor_idx=sensor, charge=charge, t_true=t, t_reco=t,
        particle_idx=np.array([0]), segment_idx=np.array([0]),
        emission_process=np.array([0]),
        n_sensors=2000, model=resolve_model_config("ski"), rng=rng,
        dark_rate_khz=50.0, readout_pad_ns=1e6)   # force plenty of dark hits
    dark_rows = hits["emission_process"] == EMISSION_PROCESS_DARK
    assert dark_rows.any(), "expected dark-labelled hits rows"
    # dark rows carry particle_idx = -1 and never appear in the segment table
    assert (hits["particle_idx"][dark_rows] == -1).all()
    assert (seg["segment_idx"] >= 0).all()   # seg table has no dark
    # the real deposit is still present and correctly attributed
    real = (hits["particle_idx"] == 0)
    assert real.any() and np.isclose(hits["PE"][real].sum(), 3.0)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} digitizer tests passed.")


if __name__ == "__main__":
    _run_all()
