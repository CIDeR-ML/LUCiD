# Capstone — joint charge + timing calibration + TQ map (gap 4)

SK_like NS=10764, N_photons=5e+05, K=8, grid={'n_cap': 100, 'n_angular': 150, 'n_height': 100}, moments mode, sources=[laser_down, iso], steps=120, w_time=0.3, start ±15%.
Truth: optics(L_R70/L_abs60/wall=sensor.2/qe.07) + tts=2.0ns + per-PMT k spread 12% + per-PMT t0 spread 3.0ns.

joint fit_charge_time of ['scatter_length', 'absorption_length', 'wall_reflection_rate', 'sensor_reflection_rate', 'qe', 'tts'] + per-PMT k (Q-map) + per-PMT t0 (T-map) …
  fit done (912s)

| global | truth | start | recovered | frac err |
|---|---|---|---|---|
| scatter_length | 70 | 66.9 | 66.66 | 4.8% |
| absorption_length | 60 | 53.7 | 59.18 | 1.4% |
| wall_reflection_rate | 0.2 | 0.188 | 0.1786 | 10.7% |
| sensor_reflection_rate | 0.2 | 0.178 | 0.1804 | 9.8% |
| qe | 0.07 | 0.0652 | 0.07069 | 1.0% |
| tts | 2 | 1.96 | 1.937 | 3.1% |

**Q-map** (per-PMT k): corr(k̂, k_true) = 0.7013, RMS frac = 12.3% (spread 12%)
**T-map** (per-PMT t0): corr(t̂0, t0_true) = 0.6631, RMS = 3.357 ns (spread 3.0ns)

Joint charge+timing in ONE fit: charge → optics + Q-map; timing → T-map + tts. The unified multi-observable capstone on the exact GN recipe (charge Schur k + time Schur t0).

_Finished in 15.2 min._
