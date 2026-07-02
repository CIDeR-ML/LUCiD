#!/usr/bin/env python3
"""Differentiability showcase for the scintillation light yield S (ph/MeV) --
the ACTUAL thing: the forward simulator is re-run at every scan point with a
different S, and the loss gradient dNLL/dS is taken by automatic differentiation
THROUGH the simulator (not by rescaling a precomputed prediction).

S enters as a runtime DetectorParams field, so a single jax.value_and_grad of the
loss is JIT-compiled ONCE and evaluated at each S (no recompile per point).

Loss: per-sensor Poisson NLL on the TOTAL observed charge, comparing the
re-simulated prediction at yield S to the fixed data-like event (injected
PhotonSim photons, cached by juno_cher_scint_fraction.py):
    mu_i(S) = forward_sim(track, S)_i           # both emission processes enabled
    n_i     = qc_data_i + qs_data_i             # observed total charge
    NLL(S)  = sum_i [ mu_i - n_i * ln(mu_i) ]

Env: NPHOT (default 1e6), NPOINTS (default 11), SREL_LO/SREL_HI, SEED.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LUCID_DIR = Path(os.environ.get("LUCID_DIR", HERE.parents[1]))   # analysis/juno_wbls -> repo root
if not (LUCID_DIR / "lucid").is_dir():
    LUCID_DIR = Path("/opt/LUCiD")
sys.path.insert(0, str(LUCID_DIR))
CONFIG = LUCID_DIR / "config"

import jax
import jax.numpy as jnp
import lucid.geometry.detector_geometry as _detgeom
from lucid.detector_params import ParticleParams, load_physics_config
from lucid.simulation import setup_event_simulator

DET = "JUNO_wbls"
GEOM = str(CONFIG / f"{DET}_geom_config.json")
PHYS = str(CONFIG / f"{DET}_physics_config.json")
THETA, PHI, ENERGY = np.pi / 4, np.pi / 6, 1000.0
NPHOT = int(os.environ.get("NPHOT", 1_000_000))
NPOINTS = int(os.environ.get("NPOINTS", 11))
SREL_LO = float(os.environ.get("SREL_LO", 0.5))
SREL_HI = float(os.environ.get("SREL_HI", 1.5))
SEED = int(os.environ.get("SEED", 6))
EPS = 1e-6
CHARGES = HERE / "data" / "juno_cher_scint_charges.npz"
OUTNPZ = HERE / "data" / "juno_S_loss_scan.npz"

# Forward prediction must emit BOTH processes so total charge depends on S.
_ORIG_MAKE_MEDIUM = _detgeom.make_medium
_detgeom.make_medium = (lambda mat, *a, **k:
    _ORIG_MAKE_MEDIUM(mat, *a, **k)._replace(
        emission_processes=("cherenkov", "scintillation")))


def main():
    sim = setup_event_simulator(
        GEOM, NPHOT, temperature=0.0, K=6, is_data=False,
        detector_type="Sphere", max_candidates_per_ray=4, physics_config=PHYS,
        default_detector_params=False, hit_mode="aggregated")
    dp0, _, _ = load_physics_config(PHYS)
    S0 = float(dp0.scintillation.S)
    key = jax.random.PRNGKey(SEED)
    track = ParticleParams(
        energy=jnp.array(ENERGY, jnp.float32), position=jnp.zeros(3, jnp.float32),
        theta=jnp.array(THETA, jnp.float32), phi=jnp.array(PHI, jnp.float32),
        t0=jnp.array(0.0, jnp.float32))

    z = np.load(CHARGES)
    mask = (z["qc_p"] + z["qs_p"]) > 0.0           # predicted-active sensors
    midx = jnp.asarray(np.where(mask)[0])
    n = jnp.asarray((z["qc_d"] + z["qs_d"])[mask])  # observed total charge (data)
    print(f"likelihood over {int(mask.sum())} sensors; NPHOT={NPHOT}")

    def nll(S):
        dp = dp0._replace(scintillation=dp0.scintillation._replace(S=S))
        charges, _ = sim(track, dp, key)            # re-simulate prediction at S
        mu = jnp.clip(charges[midx], EPS, None)
        return jnp.sum(mu - n * jnp.log(mu))

    vg = jax.jit(jax.value_and_grad(nll))           # JIT once
    S_scan = np.linspace(SREL_LO * S0, SREL_HI * S0, NPOINTS)
    losses = np.empty(NPOINTS); grads = np.empty(NPOINTS)
    for i, S in enumerate(S_scan):
        l, g = vg(jnp.asarray(S, jnp.float32))
        losses[i] = float(l); grads[i] = float(g)
        print(f"  S={S:8.1f} ({S/S0:4.2f} S0)  NLL={losses[i]:.5e}  "
              f"dNLL/dS={grads[i]:+.5e}", flush=True)

    OUTNPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTNPZ, S=S_scan, S0=S0, loss=losses, grad=grads)
    print(f"cached -> {OUTNPZ}")


if __name__ == "__main__":
    main()
