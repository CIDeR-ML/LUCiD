"""Unit tests for lucid.simulation.digitizer."""
import numpy as np
import pytest

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
        with pytest.raises(ValueError):
            resolve_model_config(bad)


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
    """ski windows: [t0, t0 + 200] inclusive, then the deadtime runs from the window END.

    Hits at 1000, 1100, 1200 (exactly t0 + 200), 1600, 2000, 3000 ns on one sensor.
    deadtime 0: windows [1000, 1200] (three hits), [1600, 1800], [2000, 2200], [3000, ...].
    deadtime 900: dead until 1200 + 900 = 2100, so 1600 AND 2000 are vetoed;
    a deadtime counted from the window open (1900) would let 2000 through.
    """
    sensor = np.full(6, 7)
    times = np.array([1000.0, 1100.0, 1200.0, 1600.0, 2000.0, 3000.0])
    charges = np.array([1.0, 1.0, 0.5, 5.0, 0.75, 1.5])
    r = digitize_event(sensor, times, charges, n_sensors=8,
                       model=resolve_model_config("ski"))
    order = np.argsort(r.digit_time)
    np.testing.assert_allclose(r.digit_time[order], [1000.0, 1600.0, 2000.0, 3000.0])
    np.testing.assert_allclose(r.digit_pe_true[order], [2.5, 5.0, 0.75, 1.5])
    np.testing.assert_array_equal(r.photon_digit_idx[:3], r.photon_digit_idx[0])
    model_dt = resolve_model_config({"model": "ski", "deadtime_ns": 900.0})
    r2 = digitize_event(sensor, times, charges, n_sensors=8, model=model_dt)
    order = np.argsort(r2.digit_time)
    np.testing.assert_allclose(r2.digit_time[order], [1000.0, 3000.0])
    np.testing.assert_allclose(r2.digit_pe_true[order], [2.5, 1.5])
    np.testing.assert_array_equal(r2.photon_digit_idx[3:5], [-1, -1])


def test_windowing_does_not_apply_the_discriminator():
    """digitize_event windows; the discriminator is a readout stage.

    A discriminator fires on the analogue pulse, so it has to run after the SPE
    spectrum is sampled. Cutting here instead would act on the integrated
    photoelectron count, where it is inert -- every hit sensor has at least one
    photoelectron.
    """
    model = resolve_model_config("ski")  # threshold 0.25 pe
    sensor = np.array([3, 3])
    times = np.array([500.0, 2000.0])
    charges = np.array([0.1, 1.0])
    r = digitize_event(sensor, times, charges, n_sensors=4, model=model)
    assert r.n_digits == 2
    np.testing.assert_allclose(sorted(r.digit_pe_true), [0.1, 1.0])

    # The cut lives here instead, on the digitised charge.
    from lucid.simulation.digitizer import apply_discriminator
    keep = apply_discriminator(np.array([0.1, 1.0]), model)
    np.testing.assert_array_equal(keep, [False, True])
    # basic has no discriminator, so nothing is ever dropped.
    np.testing.assert_array_equal(
        apply_discriminator(np.array([0.0, 0.1]), resolve_model_config("basic")),
        [True, True])


def test_discriminator_drops_digits_and_remaps_digit_idx():
    """Surviving digits are renumbered and every deposit's digit_idx follows.

    hits.h5 / step's digit_idx is a foreign key into sensor.h5; a stale index
    after the cut would silently point at the wrong digit.
    """
    rng = _rng()
    n = 400
    sd, hits, seg = digitize_and_decompose(
        sensor_idx=rng.integers(0, 30, size=n),
        charge=np.ones(n), t_true=rng.uniform(0, 3000, size=n),
        t_reco=rng.uniform(0, 3000, size=n),
        particle_idx=np.zeros(n, np.int64), segment_idx=np.zeros(n, np.int64),
        emission_process=np.zeros(n, np.int64),
        n_sensors=30, model=resolve_model_config("ski"), rng=rng)

    n_digits = sd["PE"].size
    # Every surviving digit clears the threshold ...
    assert (sd["PE"] >= 0.25).all()
    # ... and every digit_idx still indexes a real digit.
    for tbl in (hits, seg):
        assert tbl["digit_idx"].size
        assert tbl["digit_idx"].min() >= 0
        assert tbl["digit_idx"].max() < n_digits
        # ... and the digit it points at is on the same sensor
        np.testing.assert_array_equal(tbl["sensor_idx"], sd["sensor_idx"][tbl["digit_idx"]])


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
    # sk_like piecewise fractional resolution, arXiv:1307.0162 Table 2:
    # 1.2% below 20 p.e., 0.75% from 20 to 130 p.e., 0.5% above (edges at 20 and 130)
    q = np.array([10.0, 19.9, 20.0, 50.0, 129.9, 130.0, 200.0])
    np.testing.assert_allclose(charge_resolution_sigma(q, "sk_like"),
                               q * np.array([0.012, 0.012, 0.0075, 0.0075, 0.0075, 0.005, 0.005]))
    # float-override path (legacy/basic only): single-pe sigma f, scales f*sqrt(Q)
    np.testing.assert_allclose(charge_resolution_sigma(np.array([1.0]), 0.1), [0.1])
    np.testing.assert_allclose(charge_resolution_sigma(np.array([4.0]), 0.1), [0.2])


def test_dark_noise_generation_and_labelling():
    rng = np.random.default_rng(7)
    n_sensors = 1000
    # 10 kHz over a 1 ms window → mu=10 per sensor → ~10k hits
    s, t, q = generate_dark_noise(n_sensors, rate_khz=10.0,
                                  t_start_ns=0.0, t_end_ns=1_000_000.0, rng=rng)
    # total ~ Poisson(10^4): sigma 100, so +-5 sigma = [9500, 10500] (a 20% rate
    # error is 20 sigma out)
    assert 9500 < s.size < 10500
    counts = np.bincount(s, minlength=n_sensors)
    # Poisson per sensor: variance = mean (sample var/mean has sigma ~0.045 here)
    assert counts.var(ddof=1) / counts.mean() == pytest.approx(1.0, abs=0.2)
    assert (t >= 0).all() and (t <= 1_000_000.0).all()
    # uniform in the window: mean 5e5 ns, sigma of the mean ~2.9e3 ns
    assert t.mean() == pytest.approx(5e5, abs=1.5e4)
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
    # basic (legacy sk_like, no time model): charge >= 0, time unchanged
    pr, tr = apply_readout_resolution(pe_true, t, resolve_model_config("basic"), rng)
    assert (pr >= 0).all()
    np.testing.assert_allclose(tr, t)  # time_model "none" for basic


@pytest.mark.parametrize("name", ["ski", "hk"])
@pytest.mark.parametrize("q", [1.0, 5.0, 50.0])
def test_spe_charge_is_unbiased(name, q):
    """Mean reco charge = photoelectron count (the fit_mean normalisation).

    The SPE mixture mean is w*tau + (1-w)*mu (closed form), which fit_mean must
    match; dividing by it makes E[pe_reco] = Q. Clipping the sum at 0 adds
    <0.4% at Q = 1 (Gaussian core 2 sigma above 0). n = 200k gives a
    statistical sigma below 0.2%; tolerance 1%. Dropping the normalisation is
    -15% (SK).
    """
    model = resolve_model_config({"model": name, "tdc_ns": 0.0})
    spe = model["spe"]
    assert spe["fit_mean"] == pytest.approx(
        spe["w"] * spe["tau"] + (1 - spe["w"]) * spe["mu"], rel=5e-3)
    n = 200_000
    pr, _ = apply_readout_resolution(np.full(n, q), np.zeros(n), model, _rng())
    assert pr.mean() / q == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("name,expected", [("ski", 0.77), ("hk", 0.92)])
def test_single_pe_survives_the_discriminator_at_the_documented_rate(name, expected):
    """P(pe_reco >= 0.25 | 1 p.e.) is ~77% for the SK SPE and ~92% for HK.

    Documented in the digitizer module docstring; the closed form from the
    mixture, (1-w) Phi((mu - x)/sig) + w exp(-x/tau) with x = 0.25 fit_mean,
    gives 0.769 (SK) and 0.924 (HK). Binomial sigma at n = 200k is 0.001;
    tolerance 0.01 (the two PMTs are 0.15 apart).
    """
    from lucid.simulation.digitizer import apply_discriminator
    model = resolve_model_config(name)
    n = 200_000
    pr, _ = apply_readout_resolution(np.ones(n), np.zeros(n), model, _rng())
    assert apply_discriminator(pr, model).mean() == pytest.approx(expected, abs=0.01)


def test_tdc_truncates_times_to_its_step():
    """With tdc_ns = 0.4 (WCSim SKI timing precision) every time is a multiple of 0.4 ns."""
    model = resolve_model_config("ski")
    assert model["tdc_ns"] == 0.4
    t_in = np.linspace(100.0, 101.0, 1000)
    _, t = apply_readout_resolution(np.full(1000, 3.0), t_in, model, _rng())
    k = t / 0.4
    np.testing.assert_allclose(k, np.round(k), atol=1e-6)
    assert np.unique(t).size > 3          # jitter still spreads them over several steps


def test_time_jitter_follows_the_digitised_charge():
    """A photoelectron is timed by the pulse it produced: the jitter is evaluated on the
    digitised charge (floored at the model's jitter_floor_pe), the charge the discriminator sees."""
    model = resolve_model_config({"model": "ski", "tdc_ns": 0.0})
    n = 400_000
    q, t = apply_readout_resolution(np.ones(n), np.zeros(n), model, _rng())
    q = q.astype(np.float64)
    sigma = np.maximum(0.58, 0.33 + np.sqrt(10.0 / np.maximum(q, model["jitter_floor_pe"])))
    for lo, hi in ((0.25, 0.5), (1.0, 1.5), (2.0, 3.0)):
        m = (q >= lo) & (q < hi)
        z = t[m] / sigma[m]
        assert abs(np.std(z) - 1.0) < 0.02, (lo, hi, np.std(z))
        # unbiased: the mean of z is 0 within 5 standard errors (a 1 ns bias is
        # 0.2-1.7 sigma, i.e. tens of standard errors)
        assert abs(np.mean(z)) < 5.0 / np.sqrt(m.sum()), (lo, hi, np.mean(z))


def test_hk_jitter_is_a_gaussian_with_a_late_exponential_tail():
    """hk_emg = Normal(-0.2, sigma(Q)) + Exp(1/lambda(Q)), WCSim BoxandLine20inchHQE.

    lambda(Q) = 0.4113 + 0.07827 Q (WCSim lambda_param), so at Q = 5 p.e. the
    exponentially-modified Gaussian has mean -0.2 + 1/lambda = 1.046 ns, and its
    late tail makes it right-skewed. Standard error of the mean at n = 200k is
    0.003 ns; tolerance 0.02 ns.
    """
    from lucid.simulation.digitizer import _sample_time_jitter
    model = resolve_model_config({"model": "hk", "tdc_ns": 0.0})
    n = 200_000
    t = _sample_time_jitter(np.zeros(n), np.full(n, 5.0), model, _rng())
    lam = 0.4113 + 0.07827 * 5.0
    assert t.mean() == pytest.approx(-0.2 + 1.0 / lam, abs=0.02)
    skew = np.mean((t - t.mean()) ** 3) / t.std() ** 3
    assert skew > 0.3


def test_single_pe_that_digitises_higher_is_timed_better():
    n = 400_000
    for name in ("ski", "hk"):
        model = resolve_model_config({"model": name, "tdc_ns": 0.0})
        q, t = apply_readout_resolution(np.ones(n), np.zeros(n), model, _rng())
        low, high = t[(q >= 0.25) & (q < 0.5)], t[(q >= 2.0) & (q < 3.0)]
        assert np.std(low) > 1.1 * np.std(high), (name, np.std(low), np.std(high))


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
    # T / T_reco are the FIRST arrival of each group (min over its deposits)
    np.testing.assert_allclose(hits["T"], [10.0, 20.0])
    np.testing.assert_allclose(hits["T_reco"], [10.5, 20.5])
    o = np.argsort(seg["segment_idx"])
    np.testing.assert_allclose(seg["T"][o], [10.0, 20.0])
    np.testing.assert_allclose(seg["T_reco"][o], [10.5, 20.5])


def test_late_light_cap_is_measured_from_each_interaction():
    """Deposits later than 1e5 ns after THEIR interaction are dropped.

    Two interactions pooled as in a supernova chunk: t0 = 0 and t0 = 1e9 ns.
    With t_ref (per-deposit interaction t0) both interactions keep their prompt
    light and each loses only its own late (+2e5 ns) deposit. A cap on absolute
    time would drop the whole second interaction.
    """
    t = np.array([5.0, 2e5, 1e9 + 5.0, 1e9 + 2e5])
    ref = np.array([0.0, 0.0, 1e9, 1e9])
    kw = dict(sensor_idx=np.array([0, 1, 2, 3]), charge=np.ones(4), t_true=t, t_reco=t,
              particle_idx=np.array([0, 0, 1, 1]), segment_idx=np.array([0, 0, 1, 1]),
              emission_process=np.zeros(4, np.int64), n_sensors=4,
              model=resolve_model_config("basic"), rng=_rng(), apply_resolution=False)
    sd, hits, _ = digitize_and_decompose(t_ref=ref, **kw)
    np.testing.assert_array_equal(np.sort(sd["sensor_idx"]), [0, 2])
    np.testing.assert_allclose(np.sort(sd["T"]), [5.0, 1e9 + 5.0])
    np.testing.assert_array_equal(np.sort(hits["particle_idx"]), [0, 1])
    # single-vertex call (no t_ref): the earliest deposit is the reference
    sd1, _, _ = digitize_and_decompose(**kw)
    np.testing.assert_array_equal(sd1["sensor_idx"], [0])


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


# --- per-photon scatter tag (fiTQun scattering table) -------------------------

def test_resolve_first_detection_tags_only_detected_photons():
    """The tag is meaningful only where a photon was actually detected."""
    import jax.numpy as jnp
    import jax
    from lucid.simulation.hits import _resolve_first_detection

    # Two photons, one slot each: the first is detectable, the second is not.
    flat_weights = jnp.array([1.0, 0.0])
    flat_indices = jnp.array([3, 7])
    flat_times = jnp.array([5.0, 5.0])
    qe = jnp.array([1.0, 1.0])
    flat_indirect = jnp.array([True, True])

    detected, sensor_id, hit_time, indirect = _resolve_first_detection(
        flat_weights, flat_indices, flat_times, n_photons=2,
        per_photon_qe=qe, qe_key=jax.random.PRNGKey(0), threshold=1e-10,
        flat_indirect=flat_indirect)

    assert bool(detected[0]) and not bool(detected[1])
    # An undetected photon is never tagged, whatever the propagation said.
    assert bool(indirect[0]) and not bool(indirect[1])

    # Omitting the tag keeps the old 3-value behaviour, all-False.
    *_, none_dev = _resolve_first_detection(
        flat_weights, flat_indices, flat_times, n_photons=2,
        per_photon_qe=qe, qe_key=jax.random.PRNGKey(0), threshold=1e-10)
    assert not bool(none_dev.any())
