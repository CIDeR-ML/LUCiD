# C1 scalar-charge calibration — pytree-native optax fit on the real SK_like forward

NS=10764, N_photons=2e+05, K=6, sources=[laser_down, iso], steps=600, lr=0.03, optimizer=adam, project=gauge_mean_log+clip.
Truth per-PMT k: log-normal spread 12% (gauged mean(log k)=0). Global start perturbation ±20%.

loss at truth = 0.000e+00; at start = 1.883e+02

  step    0: loss=3.207e+02  L_abs= 72.03  qe=0.0860  L_R= 56.03  (29s)
  step   60: loss=8.333e+01  L_abs= 71.72  qe=0.0613  L_R= 57.48  (171s)
  step  120: loss=8.199e+01  L_abs= 71.36  qe=0.0603  L_R= 59.00  (302s)
  step  180: loss=8.052e+01  L_abs= 70.91  qe=0.0604  L_R= 60.52  (433s)
  step  240: loss=7.745e+01  L_abs= 70.38  qe=0.0608  L_R= 62.00  (564s)
  step  300: loss=7.489e+01  L_abs= 69.79  qe=0.0608  L_R= 63.44  (694s)
  step  360: loss=7.247e+01  L_abs= 69.15  qe=0.0608  L_R= 64.81  (825s)
  step  420: loss=6.706e+01  L_abs= 68.48  qe=0.0616  L_R= 66.09  (954s)
  step  480: loss=5.981e+01  L_abs= 67.79  qe=0.0625  L_R= 67.27  (1083s)
  step  540: loss=4.664e+01  L_abs= 67.14  qe=0.0629  L_R= 68.30  (1214s)
  step  599: loss=2.755e+01  L_abs= 66.59  qe=0.0633  L_R= 69.09  (1342s)

## Recovery (optax) vs fit_gn campaign reference

| param | truth | start | optax fit | frac err | fit_gn ref |
|---|---|---|---|---|---|
| L_R | 70 | 56 | 69.09 | 1.3% | tight |
| L_M | 3e+03 | 3.6e+03 | 3614 | 20.5% | ~17% (thin Mie, hard) |
| g | 0.9 | 0.72 | 0.999 | 11.0% | loose |
| L_abs | 60 | 72 | 66.59 | 11.0% | ~4.9% |
| wall_refl | 0.2 | 0.16 | 0.255 | 27.5% | ~3% |
| sensor_refl | 0.2 | 0.24 | 0.1685 | 15.8% | ~2.8% |
| qe | 0.07 | 0.056 | 0.06331 | 9.6% | ~1.9% |

per-PMT k: corr(k̂, k_true) = 0.6992, RMS frac = 12.3% (truth spread 12%)

_Finished in 22.4 min._
