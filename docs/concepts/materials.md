# Materials

Every detector pairs its geometry with a **material** — a JSON in `config/materials/`
carrying the medium's optical reference curves and, where relevant, its scintillation
parameters. Three ship with LUCiD:

| Material | File | Notes |
|---|---|---|
| Water | `config/materials/water.json` | Rayleigh 1/λ⁴ scattering; absorption = SK-calibration power law (blue) joined onto Pope & Fry (1997) data (red), smoothly blended at the ~464 nm seam |
| WbLS | `config/materials/wbls.json` | water-based liquid scintillator: inherits water's bulk optics (negligible difference at ~10% mass fraction) and adds the scintillation block below |
| Ice | `config/materials/ice.json` | placeholder using the water functional form with ice-tuned parameters; suitable for geometry/workflow studies, not ice-optics precision work |

How the curves are consumed per photon — and which parts are fittable — is described in
[wavelength physics](wavelength.md). The rest of this page collects the measured
scintillation parametrizations behind the WbLS material.

## WbLS scintillation

### Light yield

Chou parametrization (Birks + bimolecular term):

```
dL/dx  =  S · (dE/dx) / [1 + kB · (dE/dx) + C · (dE/dx)²]
```

Birks parametrization: set `C = 0`.

**S** — scintillation yield in the unquenched limit. Linear in WbLS mass fraction `c` for `c ∈ [1, 10] %`:

```
S(c)  =  (127.9 ± 17.0) · c  +  (108.3 ± 51.0)   ph / MeV
```

Source: Caravaca 2020 (arXiv:2006.00173), linear fit through 1%, 5%, 10% datapoints from CHESS with cosmic muons + ⁹⁰Sr, scintillation channel separated from Cherenkov.

**kB, C** — ionization quenching. Measured for 5% WbLS only, on proton recoils 2–20 MeV:

| Parametrization | kB                                                 | C                                                       |
| --------------- | -------------------------------------------------- | ------------------------------------------------------- |
| Chou            | (1.65 ± 0.81) cm/GeV = (1.65 ± 0.81) × 10⁻⁵ mm/keV | (13.30 ± 2.70) cm²/GeV² = (1.33 ± 0.27) × 10⁻⁹ mm²/keV² |
| Birks-only      | (5.95 ± 0.43) cm/GeV = (5.95 ± 0.43) × 10⁻⁵ mm/keV | —                                                       |

S–kB correlation in the Chou fit is 87.4% (full covariance in Callaghan 2023 Table 7).

!!! note "What LUCiD implements"
    The simulation samples a single rise/fall hypoexponential (`tau_rise`, `tau_fall`);
    the slow second decay component (τ₂ ≈ 27 ns, R₁ ≈ 0.06) from the measured
    two-component fit below is dropped — `wbls.json`'s own comments record this
    simplification.

    Scintillation emission requires **`wavelength_mode=True`** at
    `setup_event_simulator` — the surrogate draws per-photon wavelengths from the
    Moyal spectrum, so `setup_event_simulator` raises if a scintillating medium is
    paired with `wavelength_mode=False`.

Source: Callaghan 2023 (arXiv:2210.03876), Table 6. Sample oxygenated (atmospheric).

### Emission timing

Two-decay biexponential with shared rise time:

```
p(t)  =  R₁ · g(t; τ_r, τ₁)  +  (1 − R₁) · g(t; τ_r, τ₂)

g(t; τ_r, τ_d)  =  (e^(−t/τ_d) − e^(−t/τ_r)) / (τ_d − τ_r),   t ≥ 0
```

Measured at c ∈ {1, 5, 10} %:

| Parameter | 1% WbLS         | 5% WbLS         | 10% WbLS        |
| --------- | --------------- | --------------- | --------------- |
| τ_r [ns]  | 0.00 ± 0.06     | 0.06 ± 0.11     | 0.13 ± 0.12     |
| τ₁ [ns]   | 2.25 ± 0.15     | 2.35 ± 0.13     | 2.70 ± 0.16     |
| τ₂ [ns]   | 15.10 ± 7.47    | 23.21 ± 3.28    | 27.05 ± 4.20    |
| R₁        | 0.96 ± 0.01     | 0.94 ± 0.01     | 0.94 ± 0.01     |

Quoted uncertainties include statistical and systematic. Source: Caravaca 2020 (arXiv:2006.00173), Table 2. Excitation by ⁹⁰Sr betas.