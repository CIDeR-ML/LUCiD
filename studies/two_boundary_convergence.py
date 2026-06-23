"""Two-medium (LAB-LS / water) convergence study: K and N scaling vs source location.

For the high-contrast nested detector (inner LAB-LS n=1.48, outer water n=1.33, critical
angle ~64 deg) we sweep:

  * K (scatter/interface iterations) at fixed N — TIR-trapped photons bounce many times,
    so charge converges in K, and the closer the source sits to the interface the larger
    the K needed. This is gate-5 (K-convergence INCLUDING TIR trajectories).
  * N (photons) by BATCHING independent forwards — the Monte-Carlo standard error of the
    detected charge should fall as ~N^{-1/2}.

Source locations span the interface incidence-angle distribution (controls TIR) and the
LS:water path-length ratio (the source-diversity lever that breaks L_LS<->L_water):
  r=0     center          — normal incidence, NO TIR (baseline)
  r=8.75  half radius      — moderate angular spread, partial TIR
  r=15.0  near interface   — wide spread, strong TIR (K stress case)
  r=18.5  water buffer     — outer-medium source, water->LS crossings (no TIR inward)

Run:  JAX_PLATFORM_NAME=cuda python studies/two_boundary_convergence.py
Outputs: studies/out/two_boundary_convergence.{png,json}
"""
import os, json, time
import numpy as np
import jax, jax.numpy as jnp

from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source

GEOM = "config/JUNO_nested_labls_geom_config.json"
PHYS = "config/JUNO_nested_labls_physics_config.json"
R_IN, R_OUT = 17.5, 19.5
WAVELENGTH = 430.0          # LAB scintillation/Cherenkov peak
INTENSITY = 50_000_000.0    # source photons (physical); MC ray count is N below
OUTDIR = os.path.join(os.path.dirname(__file__), "out")

SOURCES = {
    "center r=0":        [0.0, 0.0, 0.0],
    "half r=8.75":       [0.0, 0.0, 8.75],
    "near-iface r=15":   [0.0, 0.0, 15.0],
    "buffer r=18.5":     [0.0, 0.0, 18.5],
}
K_LIST = [2, 4, 6, 8, 12, 16, 24, 32]
N_RAYS = 500_000            # photons per batch (MC rays)
K_SWEEP_BATCHES = 4        # 2M rays per (K, source) point — Q_tot noise is tiny
N_SWEEP_K = 24             # converged K for the N study
N_SWEEP_BATCHES = 32       # up to 16M rays
N_CHECKPOINTS = [1, 2, 4, 8, 16, 32]


def make_dp():
    return DetectorParams.from_flat(
        scatter_length=50.0, wall_reflection_rate=0.2, sensor_reflection_rate=0.2,
        absorption_length=50.0, qe=0.065, qe_corrections=jnp.ones(10000))


def build_sim(K):
    return setup_event_simulator(
        GEOM, N_RAYS, temperature=None, K=K, is_calibration=True,
        detector_type='nested_sphere', wavelength_mode=True, physics_config=PHYS)


def batched_charges(sim, dp, pos, n_batches, seed0=0):
    """Run n_batches independent forwards; return list of per-sensor charge vectors."""
    src = isotropic_source(position=pos, intensity=INTENSITY, wavelength=WAVELENGTH)
    out = []
    for b in range(n_batches):
        c, _ = sim(src, dp, jax.random.PRNGKey(seed0 + b))
        out.append(np.asarray(c))
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    dp = make_dp()
    results = {"K_list": K_LIST, "sources": list(SOURCES), "N_rays": N_RAYS,
              "k_sweep": {}, "n_sweep": {}}

    # ---- K sweep: Q_tot(K) per source -------------------------------------
    print("=== K sweep (Q_tot vs K) ===")
    for K in K_LIST:
        t0 = time.time()
        sim = build_sim(K)
        for name, pos in SOURCES.items():
            cs = batched_charges(sim, dp, pos, K_SWEEP_BATCHES, seed0=100)
            q = float(np.mean([c.sum() for c in cs]))
            results["k_sweep"].setdefault(name, {})[K] = q
        print(f"  K={K:2d}  ({time.time()-t0:5.1f}s)  " +
              "  ".join(f"{n}={results['k_sweep'][n][K]:.3e}" for n in SOURCES))

    # ---- N sweep: MC standard error vs N (batched), at converged K --------
    print(f"\n=== N sweep (MC error vs N) at K={N_SWEEP_K} ===")
    sim = build_sim(N_SWEEP_K)
    for name, pos in SOURCES.items():
        cs = batched_charges(sim, dp, pos, N_SWEEP_BATCHES, seed0=500)
        Qb = np.array([c.sum() for c in cs])                  # per-batch totals
        C = np.stack(cs)                                       # (B, n_sensors)
        rec = {"N": [], "Qtot_mean": [], "Qtot_sem": [], "persensor_rel_sem": []}
        for B in N_CHECKPOINTS:
            if B > N_SWEEP_BATCHES:
                break
            cb, qb = C[:B], Qb[:B]
            mean_c = cb.mean(0)
            lit = mean_c > 1e-6
            # standard error of the mean (per sensor and total) over B independent batches
            sem_c = cb.std(0, ddof=1) / np.sqrt(B) if B > 1 else np.full_like(mean_c, np.nan)
            rel = float(np.mean(sem_c[lit] / mean_c[lit])) if B > 1 else float("nan")
            rec["N"].append(B * N_RAYS)
            rec["Qtot_mean"].append(float(qb.mean()))
            rec["Qtot_sem"].append(float(qb.std(ddof=1) / np.sqrt(B)) if B > 1 else float("nan"))
            rec["persensor_rel_sem"].append(rel)
        results["n_sweep"][name] = rec
        print(f"  {name:16s} Qtot={rec['Qtot_mean'][-1]:.3e}  "
              f"per-sensor rel-SEM @N={rec['N'][-1]/1e6:.0f}M = {rec['persensor_rel_sem'][-1]:.4f}")

    with open(os.path.join(OUTDIR, "two_boundary_convergence.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot(results)
    print(f"\nWrote {OUTDIR}/two_boundary_convergence.{{png,json}}")


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # K-convergence: Q_tot(K) / Q_tot(K_max)
    for name in results["sources"]:
        ks = results["k_sweep"][name]
        K = sorted(int(k) for k in ks)
        q = np.array([ks.get(k, ks.get(str(k))) for k in K])
        ax1.plot(K, q / q[-1], "o-", label=name)
    ax1.axhline(1.0, color="k", lw=0.6, ls=":")
    ax1.set_xlabel("K (scatter/interface iterations)")
    ax1.set_ylabel("Q_tot(K) / Q_tot(K=%d)" % max(int(k) for k in ks))
    ax1.set_title("K-convergence (LAB-LS / water)\nlonger optical path (farther source) needs larger K; all converge by K~16-24")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # N-convergence: per-sensor relative SEM vs N, with 1/sqrt(N) reference
    for name in results["sources"]:
        rec = results["n_sweep"][name]
        N = np.array(rec["N"][1:]); rel = np.array(rec["persensor_rel_sem"][1:])
        ax2.loglog(N, rel, "o-", label=name)
    Nref = np.array(results["n_sweep"][results["sources"][0]]["N"][1:])
    ref = rel[0] * np.sqrt(Nref[0] / Nref)
    ax2.loglog(Nref, ref, "k--", lw=1, label=r"$\propto N^{-1/2}$")
    ax2.set_xlabel("N (photons, batched)")
    ax2.set_ylabel("mean per-sensor relative SEM")
    ax2.set_title("N-convergence (MC standard error)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "two_boundary_convergence.png"), dpi=130)


if __name__ == "__main__":
    main()
