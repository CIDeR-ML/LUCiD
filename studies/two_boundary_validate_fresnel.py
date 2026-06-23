"""Validate the interface against ANALYTIC Fresnel on a stripped-down case.

Monochromatic mode (wavelength_mode=False) ⇒ BOTH media use the same DetectorParams bulk
scalars, so contrast (labls/water, n_out=1.33) vs matched (labls/water_n148, n_out=1.48)
differ by ONLY the interface index. Turn scattering OFF (L_scat=L_mie=1e8). Then:

  * Center source ⇒ every ray hits the interface at NORMAL incidence ⇒ the contrast/matched
    charge ratio must equal the analytic normal-incidence Fresnel transmission
        T0 = 1 - ((n_LS - n_W)/(n_LS + n_W))^2 = 0.99715   (R0 = 0.285%)
    (up to a small multi-pass recovery of the reflected 0.285%, which absorption suppresses).
  * The ratio must be FLAT in cosγ (symmetry) for the center source.

We sweep absorption (multi-pass control) and K, and also check off-center sources to see
whether the 'lensing' ratio>1 is geometric (persists with no scattering) or scattering-coupled.
"""
import numpy as np
import jax, jax.numpy as jnp
from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source
from lucid.geometry import generate_detector

CONTRAST = "config/JUNO_nested_labls_geom_config.json"
MATCHED = "config/JUNO_nested_labls_matched_geom_config.json"
PHYS = "config/JUNO_nested_labls_physics_config.json"
R_OUT, N_W, N_LS = 19.5, 1.33, 1.48
R0 = ((N_LS - N_W) / (N_LS + N_W))**2
T0 = 1 - R0
INT, NR, NB = 50_000_000.0, 500_000, 12
SENS = np.asarray(generate_detector(CONTRAST).all_points)
COSG = SENS[:, 2] / R_OUT


def dp(scatter, absorption):
    return DetectorParams.from_flat(
        scatter_length=scatter, mie_scatter_length=scatter, absorption_length=absorption,
        wall_reflection_rate=0.2, sensor_reflection_rate=0.2, qe=0.065,
        qe_corrections=jnp.ones(10000))


def sim(geom, K):
    # MONOCHROMATIC: wavelength_mode=False ⇒ both media share DetectorParams bulk scalars.
    return setup_event_simulator(geom, NR, temperature=None, K=K, is_calibration=True,
                                 detector_type='nested_sphere', wavelength_mode=False, physics_config=PHYS)


def total(s, d, r):
    src = isotropic_source(position=[0.0, 0.0, float(r)], intensity=INT, wavelength=420.0)
    return np.mean([np.asarray(s(src, d, jax.random.PRNGKey(11 + b))[0]).sum() for b in range(NB)])


def main():
    print(f"Analytic normal-incidence: R0={R0*100:.4f}%  T0={T0:.5f}\n")

    print("=== TEST A: center source, NO scattering — ratio must -> T0=0.99715, flat ===")
    print(f"{'K':>3} {'L_abs':>7} {'ratio con/mat':>14} {'vs T0':>10}")
    for K in [8, 24, 48]:
        sc, sm = sim(CONTRAST, K), sim(MATCHED, K)
        for L_abs in [1e8, 100.0, 30.0]:
            d = dp(1e8, L_abs)
            qc, qm = total(sc, d, 0.0), total(sm, d, 0.0)
            print(f"{K:>3} {L_abs:>7.0f} {qc/qm:>14.5f} {(qc/qm-T0)*100:>+9.3f}%")

    print("\n=== center: ratio flat over cosγ? (no scatter, K=48, L_abs=30) ===")
    sc, sm = sim(CONTRAST, 48), sim(MATCHED, 48)
    d = dp(1e8, 30.0)
    src = isotropic_source(position=[0., 0., 0.], intensity=INT, wavelength=420.0)
    cc = np.mean([np.asarray(sc(src, d, jax.random.PRNGKey(11+b))[0]) for b in range(NB)], 0)
    cm = np.mean([np.asarray(sm(src, d, jax.random.PRNGKey(11+b))[0]) for b in range(NB)], 0)
    edges = np.linspace(-1, 1, 20); idx = np.clip(np.digitize(COSG, edges)-1, 0, 18)
    prof = np.array([(cc[idx==b].sum())/(cm[idx==b].sum()) for b in range(19)])
    print("  cosγ-binned ratio:", np.round(prof, 4))
    print(f"  spread (max-min) = {np.nanmax(prof)-np.nanmin(prof):.4f}  (≈0 ⇒ symmetric, pure Fresnel)")

    print("\n=== TEST B: off-center, NO scattering — is the lensing ratio>1 geometric? ===")
    print(f"{'r_s':>5} {'ratio con/mat':>14}  (>1 ⇒ refraction concentrates onto sensors)")
    for r in [0.0, 8.0, 13.0, 16.5, 17.3]:
        d = dp(1e8, 30.0)
        qc, qm = total(sc, d, r), total(sm, d, r)
        print(f"{r:>5.1f} {qc/qm:>14.5f}")


if __name__ == "__main__":
    main()
