# Per-PMT QE via closed-form k = Q/M (gap 5)

SK_like NS=10764, N_photons=1e+06, K=8, grid={'n_cap': 100, 'n_angular': 150, 'n_height': 100}, isotropic source, optics known. k̂_s = Q_s/M_s(k=1), gauge mean(log k)=0, independent data/model keys.
Random spread 12%; smooth z-trend amplitude 14%.

| truth k | n_lit | slope(k̂ vs k) | corr | RMS frac |
|---|---|---|---|---|
| random only | 10764 | 1.0003 | 0.8185 | 8.4% |
| smooth trend only | 10764 | 1.0038 | 0.8550 | 8.4% |
| random + trend | 10764 | 1.0021 | 0.9095 | 8.4% |

Slope ≈ 1.000 confirms the closed-form k̂=Q/M is unbiased and captures both the white per-PMT spread and the SMOOTH position-correlated trend (the case that, if ignored, biases the reflectivities — here recovered directly). Gauge mean(log k)=0 fixes the qe↔mean(k) degeneracy. Scatter is the forward-key (≈photon) noise floor.

_Finished in 0.6 min._
