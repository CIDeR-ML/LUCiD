# Angular (complex) reflection FIT — unified GN recipe (gap 1)

SK_like NS=10764, N_photons=5e+05, K=8, grid={'n_cap': 100, 'n_angular': 150, 'n_height': 100}, reflection_model=angular, λ_refl=400.0nm, sources=[laser_down, laser_wall, iso], steps=100, start ±15%.
Truth angular params (deviated from defaults): {'wall_R0': 0.1, 'wall_p': 2.0, 'wall_fspec': 0.3, 'cathode_nr': 2.5, 'cathode_nk': 1.2, 'sensor_fspec': 0.3}. Optics held at truth.

GN fit of ['wall_R0', 'wall_p', 'cathode_nr', 'cathode_nk', 'wall_fspec', 'sensor_fspec'] …
  fit + CRB done (1508s)

| param | truth | start | recovered | frac err | CRB σ |
|---|---|---|---|---|---|
| wall_R0 | 0.1 | 0.104 | 0.1133 | 13.3% | 5.05% |
| wall_p | 2 | 1.87 | 3.202 | 60.1% | 10.06% |
| cathode_nr | 2.5 | 2.18 | 2.72 | 8.8% | 65.48% |
| cathode_nk | 1.2 | 1.04 | 1.148 | 4.4% | 52.13% |
| wall_fspec | 0.3 | 0.33 | 0.2892 | 3.6% | 0.47% |
| sensor_fspec | 0.3 | 0.34 | 0.3134 | 4.5% | 0.40% |

Angular reflection is PARTIALLY charge-identifiable (read the CRB column): wall_R0 (normal reflectance) and the spec/diff fractions wall_fspec/sensor_fspec are constrained — the fractions tightly (CRB ~0.4%), because the per-sensor reflected-light PATTERN depends on the spec/diff split (so they are NOT charge-blind as one might assume). The Schlick exponent wall_p is weakly constrained (recovered 60% off), and the cathode Fresnel indices cathode_nr↔cathode_nk are NEAR-DEGENERATE (CRB ~50-65% — they trade in producing the cathode reflectance magnitude), recovered near truth only because they started near it. Pinning wall_p / the cathode pair needs more incidence-angle diversity or the timing observable.

_Finished in 25.1 min._
