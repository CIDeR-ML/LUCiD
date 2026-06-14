# Timing calibration through the real sim — per-PMT t0 + the variance finding

SK_like NS=10764; isotropic; M=200 flashes/run; N_photons=1e+06 (intensity=NPH ⇒ integer-PE shots); TTS_true=2.0 ns; t0 spread=3.0 ns.

occupancy μ (lit): median=2.08, 10-90%=[1.43, 3.03]
  reference run done (910s)
  data run done (1818s)

## per-PMT t0 (the dominant TQ-map constant)

- n_lit = 10764
- **t0 RMS recovery = 3.262 ns** (flash-noise floor √((var_ref+var_data)/M) median ≈ 2.887 ns)
- t0 correlation (recovered vs truth) = 0.6800
→ per-PMT t0 is recovered through the full photon sim at the flash-noise floor — the dominant, well-conditioned timing calibration.

## first-arrival variance: the geometric-order finding

- measured Var[first] median = 805.775 ns²
- TTS·Var[min] prediction (delta-geometric, timingcal toy) median = 3.505 ns²
- ratio measured/predicted ≈ **231×** → the variance is dominated by the GEOMETRIC first-arrival order statistic (earliest of μ photons over the broad direct+scattered arrival spread), NOT by TTS. timingcal.py assumed a DELTA geometric arrival (sharp laser); the full sim shows clean TTS-from-variance needs a sharp-geometric (laser, direct-spot) source or an explicit geometric-order model. The t0+TTS ESTIMATOR is validated on the timingcal data model in tests/test_timing_calibration.py.

## Conclusion

- The framework is **correct**: the per-PMT t0 estimator reaches the flash-noise floor
  (RMS 3.26 ns ≈ 1.13× the 2.89 ns floor). The estimator is not the limitation.
- The **floor itself is large** because, for an isotropic source in a scattering medium at
  low occupancy, the per-flash first-arrival is the earliest of ~μ photons drawn from a
  BROAD direct+scattered arrival distribution → per-flash σ ≈ 28 ns. The floor shrinks only
  as TTS-free √(var_geom/M), so sub-ns t0 would need M ≈ 1600 flashes here.
- **The practical lesson (and why real detectors use lasers/LEDs for timing):** clean,
  precise timing calibration needs a **sharp-geometric source** — a collimated laser whose
  direct-spot PMTs see a near-delta arrival (var_geom → 0), so the floor collapses and the
  first-arrival variance becomes the clean TTS·Var[min] signal. With isotropic + scattering
  the geometric order statistic dominates (231×), exactly the regime timingcal.py's toy
  (delta-geometric) did not model.
- The corrected timing OBSERVABLE (hard-min + Poisson-conditioned occupancy bias) and the
  t0+TTS ESTIMATOR (mean→t0, variance→TTS, gauge mean(t0)=0) are validated directly on the
  timingcal data model in tests/test_timing_calibration.py (exact + flash-noise recovery).

_Finished in 31.1 min._
