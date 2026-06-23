"""Two-medium (LAB-LS / water): (1) expected-value vs sampling-mode convergence overlay,
and (2) per-sensor hit-pattern maps per source — done the symmetry-correct way.

The four sources sit on the z-axis, so the detector response is AZIMUTHALLY SYMMETRIC: the
per-sensor charge depends only on the polar angle θ from the source axis (z), not on φ. The
correct, low-noise hit-pattern representation is therefore the azimuthal average vs cosθ
(Fibonacci sphere ⇒ equal sensors per cosθ band ⇒ unbiased average that denoises by √N_φ).
We also VALIDATE the symmetry (split the φ<π and φ≥π hemispheres — their cosθ profiles must
agree within MC noise; a refraction/interface bug would break this).

Run:  python studies/two_boundary_modes_hitpattern.py
Outputs: studies/out/two_boundary_modes.png, studies/out/two_boundary_hitpattern.png
"""
import os, json
import numpy as np
import jax, jax.numpy as jnp

from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source
from lucid.geometry import generate_detector

GEOM = "config/JUNO_nested_labls_geom_config.json"
PHYS = "config/JUNO_nested_labls_physics_config.json"
R_OUT = 19.5
WAVELENGTH = 430.0
INTENSITY = 50_000_000.0
N_RAYS = 500_000
OUTDIR = os.path.join(os.path.dirname(__file__), "out")

SOURCES = {
    "center r=0":      [0.0, 0.0, 0.0],
    "half r=8.75":     [0.0, 0.0, 8.75],
    "near-iface r=15": [0.0, 0.0, 15.0],
    "buffer r=18.5":   [0.0, 0.0, 18.5],
}
SENSORS = np.asarray(generate_detector(GEOM).all_points)   # (10000,3), index == charge index
COST = SENSORS[:, 2] / R_OUT                                # polar cosθ about the source (z) axis
PHI = np.arctan2(SENSORS[:, 1], SENSORS[:, 0])             # azimuth


def make_dp():
    return DetectorParams.from_flat(
        scatter_length=50.0, wall_reflection_rate=0.2, sensor_reflection_rate=0.2,
        absorption_length=50.0, qe=0.065, qe_corrections=jnp.ones(10000))


def build_sim(K, expected):
    return setup_event_simulator(
        GEOM, N_RAYS, temperature=None, K=K, is_calibration=True,
        detector_type='nested_sphere', wavelength_mode=True, physics_config=PHYS,
        use_expected_value=expected)


def batched(sim, dp, pos, n_batches, seed0):
    src = isotropic_source(position=pos, intensity=INTENSITY, wavelength=WAVELENGTH)
    return [np.asarray(sim(src, dp, jax.random.PRNGKey(seed0 + b))[0]) for b in range(n_batches)]


# ----------------------------------------------------------------------------
# Part 1 — expected vs sampling convergence overlay
# ----------------------------------------------------------------------------
def convergence_modes():
    dp = make_dp()
    K_LIST = [2, 4, 8, 12, 16, 24, 32]
    res = {"k": {}, "n": {}}
    for expected in (True, False):
        mode = "expected" if expected else "sampling"
        print(f"--- mode={mode}: K sweep ---")
        res["k"][mode] = {n: {} for n in SOURCES}
        for K in K_LIST:
            sim = build_sim(K, expected)
            for n, pos in SOURCES.items():
                cs = batched(sim, dp, pos, 4, seed0=100)
                res["k"][mode][n][K] = float(np.mean([c.sum() for c in cs]))
            print(f"  K={K:2d} done")
        # N sweep at K=24
        print(f"--- mode={mode}: N sweep (K=24) ---")
        sim = build_sim(24, expected)
        res["n"][mode] = {}
        for n, pos in SOURCES.items():
            C = np.stack(batched(sim, dp, pos, 16, seed0=500))
            rec = {"N": [], "rel_sem": []}
            for B in [2, 4, 8, 16]:
                cb = C[:B]; mean_c = cb.mean(0); lit = mean_c > 1e-6
                sem = cb.std(0, ddof=1) / np.sqrt(B)
                rec["N"].append(B * N_RAYS)
                rec["rel_sem"].append(float(np.mean(sem[lit] / mean_c[lit])))
            res["n"][mode][n] = rec
            print(f"  {n}: relSEM@8M(N) = {rec['rel_sem'][-1]:.4f}")
    return res, K_LIST


# ----------------------------------------------------------------------------
# Part 2 — hit-pattern maps (azimuthal-average + symmetry check)
# ----------------------------------------------------------------------------
def hit_patterns(n_bins=40, n_batches=16):
    dp = make_dp()
    sim = build_sim(24, expected=True)     # converged K, clean expected-value pattern
    edges = np.linspace(-1, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(COST, edges) - 1, 0, n_bins - 1)
    patt = {}
    for n, pos in SOURCES.items():
        C = np.stack(batched(sim, dp, pos, n_batches, seed0=900))   # (B, n_sensors)
        c_mean = C.mean(0)
        # azimuthal average per cosθ band (equal sensors/band → unbiased)
        prof = np.array([c_mean[bin_idx == b].mean() for b in range(n_bins)])
        # symmetry validation: φ-hemisphere split must agree within MC noise
        h1, h2 = PHI < 0, PHI >= 0
        p1 = np.array([c_mean[(bin_idx == b) & h1].mean() for b in range(n_bins)])
        p2 = np.array([c_mean[(bin_idx == b) & h2].mean() for b in range(n_bins)])
        lit = prof > prof.max() * 1e-3
        sym_dev = float(np.max(np.abs(p1 - p2)[lit] / prof[lit]))   # max relative φ-asymmetry
        patt[n] = {"cos": centers, "prof": prof, "sym_dev": sym_dev,
                   "c_mean": c_mean}
        print(f"  {n:16s} peak={prof.max():.3e}  φ-asymmetry(max rel)={sym_dev:.3f}")
    return patt


def plot_modes(res, K_LIST):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {n: c for n, c in zip(SOURCES, ["C0", "C1", "C2", "C3"])}
    for mode, ls, mk in [("expected", "-", "o"), ("sampling", "--", "s")]:
        for n in SOURCES:
            ks = res["k"][mode][n]; q = np.array([ks[k] for k in K_LIST]); q = q / q[-1]
            a1.plot(K_LIST, q, ls, marker=mk, ms=4, color=colors[n],
                    label=f"{n} ({mode})" if mode == "expected" else None)
    a1.axhline(1, color="k", lw=0.5, ls=":")
    a1.set_xlabel("K"); a1.set_ylabel("Q_tot(K)/Q_tot(32)")
    a1.set_title("K-convergence: expected (—) vs sampling (- -)")
    a1.legend(fontsize=7); a1.grid(alpha=0.3)
    for mode, ls, mk in [("expected", "-", "o"), ("sampling", "--", "s")]:
        for n in SOURCES:
            rec = res["n"][mode][n]
            a2.loglog(rec["N"], rec["rel_sem"], ls, marker=mk, ms=4, color=colors[n],
                      label=f"{n} ({mode})" if mode == "sampling" else None)
    N0 = res["n"]["expected"][list(SOURCES)[0]]["N"]
    r0 = res["n"]["expected"][list(SOURCES)[0]]["rel_sem"][0]
    a2.loglog(N0, r0 * np.sqrt(N0[0] / np.array(N0)), "k:", lw=1, label=r"$\propto N^{-1/2}$")
    a2.set_xlabel("N (photons)"); a2.set_ylabel("per-sensor relative SEM")
    a2.set_title("N-convergence: sampling (- -, ~shot noise) vs expected (—, DiCE)")
    a2.legend(fontsize=7); a2.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "two_boundary_modes.png"), dpi=130)


def plot_patterns(patt):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 5))
    # Panel A — azimuthally-averaged charge vs cosθ (the symmetry-correct hit pattern)
    a1 = fig.add_subplot(1, 2, 1)
    for n in SOURCES:
        a1.semilogy(patt[n]["cos"], patt[n]["prof"], "o-", ms=3, label=n)
    a1.set_xlabel(r"$\cos\theta$  (sensor polar angle about source/z axis; +1 = nearest pole)")
    a1.set_ylabel("mean charge / sensor (azimuthal average)")
    a1.set_title("Hit pattern vs polar angle (LAB-LS/water)\n+1 = toward source")
    a1.legend(fontsize=8); a1.grid(alpha=0.3, which="both")
    # Panel B — 2D φ vs cosθ scatter for the near-interface source (visual φ-symmetry: bands)
    a2 = fig.add_subplot(1, 2, 2)
    n = "near-iface r=15"; c = patt[n]["c_mean"]
    sc = a2.scatter(PHI, COST, c=c, s=4, cmap="viridis",
                    vmax=np.percentile(c, 99.5))
    a2.set_xlabel(r"$\phi$ (azimuth)"); a2.set_ylabel(r"$\cos\theta$")
    a2.set_title(f"per-sensor charge, {n}\nhorizontal bands ⇒ azimuthal symmetry (max φ-asym {patt[n]['sym_dev']:.1%})")
    fig.colorbar(sc, ax=a2, label="charge")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "two_boundary_hitpattern.png"), dpi=130)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("=== Part 1: mode overlay ===")
    res, K_LIST = convergence_modes()
    plot_modes(res, K_LIST)
    print("\n=== Part 2: hit patterns (azimuthal + symmetry) ===")
    patt = hit_patterns()
    plot_patterns(patt)
    # persist the numeric profiles (drop the big per-sensor array)
    dump = {n: {"cos": patt[n]["cos"].tolist(), "prof": patt[n]["prof"].tolist(),
                "sym_dev": patt[n]["sym_dev"]} for n in SOURCES}
    json.dump(dump, open(os.path.join(OUTDIR, "two_boundary_hitpattern.json"), "w"), indent=2)
    print(f"\nWrote {OUTDIR}/two_boundary_modes.png, two_boundary_hitpattern.png")


if __name__ == "__main__":
    main()
