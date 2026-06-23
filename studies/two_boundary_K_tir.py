"""K-convergence in the TIR regime (sources PAST the 15.73 m threshold).

TIR-trapped photons are whispering-gallery modes: in a sphere a reflected ray keeps a
constant incidence angle every bounce, so a super-critical photon stays trapped until it
SCATTERS (Rayleigh L=28 m) to a sub-critical angle or is absorbed. Near-grazing photons
have short chords ⇒ many bounces ⇒ many K iterations before they escape and deposit. This
checks whether K=24 (used in the interface study) is converged for the TIR sources, or
whether truncation is over-estimating the interface charge loss.
"""
import os, json
import numpy as np
import jax, jax.numpy as jnp
from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source

CONTRAST = "config/JUNO_nested_labls_geom_config.json"
MATCHED = "config/JUNO_nested_labls_matched_geom_config.json"
PHYS = "config/JUNO_nested_labls_physics_config.json"
WL, INT, NR, NB = 430.0, 50_000_000.0, 500_000, 8
K_LIST = [8, 16, 24, 32, 48, 64, 96]
SRC_R = [16.5, 17.3]            # past TIR threshold
OUTDIR = os.path.join(os.path.dirname(__file__), "out")


def dp():
    return DetectorParams.from_flat(
        scatter_length=50.0, wall_reflection_rate=0.2, sensor_reflection_rate=0.2,
        absorption_length=50.0, qe=0.065, qe_corrections=jnp.ones(10000))


def total(geom, K, r, d):
    sim = setup_event_simulator(geom, NR, temperature=None, K=K, is_calibration=True,
                                detector_type='nested_sphere', wavelength_mode=True, physics_config=PHYS)
    src = isotropic_source(position=[0.0, 0.0, r], intensity=INT, wavelength=WL)
    return float(np.mean([np.asarray(sim(src, d, jax.random.PRNGKey(7000 + b))[0]).sum() for b in range(NB)]))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = dp()
    res = {}
    for r in SRC_R:
        print(f"--- r={r} (contrast=labls/water, matched=labls/water_n148) ---")
        con = {K: total(CONTRAST, K, r, d) for K in K_LIST}
        mat = {K: total(MATCHED, K, r, d) for K in K_LIST}
        res[str(r)] = {"K": K_LIST, "con": con, "mat": mat}
        for K in K_LIST:
            loss = 1.0 - con[K] / mat[K]
            print(f"  K={K:3d}  contrast Q={con[K]:.4e} ({con[K]/con[K_LIST[-1]]*100:5.1f}% of K{K_LIST[-1]})"
                  f"  matched Q={mat[K]:.4e} ({mat[K]/mat[K_LIST[-1]]*100:5.1f}%)  interface loss={loss*100:5.2f}%")
    json.dump({r: {"K": v["K"], "con": [v["con"][K] for K in v["K"]],
                   "mat": [v["mat"][K] for K in v["K"]]} for r, v in res.items()},
              open(os.path.join(OUTDIR, "two_boundary_K_tir.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
