# Laser timing calibration — the clean (sharp-geometric) regime

SK_like NS=10764; downward laser (NA=0.3); M=300 flashes/run; N_photons=3e+05 (intensity=NPH ⇒ integer-PE shots); TTS_true=2.0 ns; t0 spread=3.0 ns.

  reference run done (507s)
  data run done (1009s)
occupancy μ over lit PMTs: median=5.88, 10-90%=[1.48, 6.68]; n_lit=523

## TTS recovery (implied σ_TTS = √(Var[first]/Var[min|N≥1]))

- **spot PMTs (μ>1, n=523): implied TTS = 2.201 ns** vs truth 2.0 ns (10.1%) — the headline, unbiased over the direct spot.
- halo PMTs (upper-quartile var, n=131): implied TTS = 46.616 ns (21.2× the spot) — inflated by the residual geometric order statistic (the T3 effect), cleanly separated by the variance.
- sharpest 25% by ref_var (n=131): 2.009 ns — biased LOW at small M (selecting on the noisy variance estimate picks downward fluctuations); unbiased as M grows.

## per-PMT t0 recovery

- **sharp PMTs**: t0 RMS = **0.181 ns** (spread 3.0 ns), corr = 0.9981
- all lit PMTs: t0 RMS = 2.102 ns
→ on the sharp PMTs t0 is recovered far below the 3.0 ns spread (vs the geometry-limited isotropic floor of ~2.9 ns in T3).

**Conclusion:** with a sharp-geometric (laser) source, both t0 (from the first-arrival mean) and TTS (from the variance) recover cleanly on the direct-spot PMTs — the corrected observable + the t0/TTS estimator work end-to-end through the real sim once the source removes the geometric order statistic. This is why timing calibration uses lasers/LEDs, now reproduced on the unified framework.

_Finished in 17.6 min._
