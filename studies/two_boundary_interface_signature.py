"""Isolate and visualise the LS<->water INTERFACE signature (Fresnel + TIR), separated
from the bulk optics, with source-position-resolved structure.

Why a controlled comparison: the LAB/water charge deficit is mostly BULK (LAB Rayleigh
28 m vs water ~250 m), not the interface. To see the interface ALONE we toggle only the
index step while holding the outer bulk optics fixed:

  contrast : labls (n=1.48) / water        (n_out=1.33)  -> Fresnel + TIR at the interface
  matched  : labls (n=1.48) / water_n148    (n_out=1.48)  -> NO index step, SAME water bulk

The ratio  contrast / matched  (same RNG keys = common random numbers, so MC noise cancels)
is the pure interface transfer function. TIR only exists for a source beyond the threshold
radius r* = R_in*sin(theta_c) = 17.5*(1.33/1.48) = 15.73 m (critical angle 64 deg); a source
reaches max interface incidence arcsin(r_s/R_in). We scan r_s across the threshold.

Plotting (axisymmetric source on z-axis):
  (A) cosθ ratio profiles per source, with the TIR-threshold sources highlighted;
  (B) orthographic projection down the source axis (near + far hemispheres) of the ratio
      for the strongest-TIR source — concentric rings are the symmetry-correct view of an
      axisymmetric pattern, and the TIR edge shows as a ring.

Run:  python studies/two_boundary_interface_signature.py
Out:  studies/out/two_boundary_interface_signature.png (+ .json)
"""
import os, json
import numpy as np
import jax, jax.numpy as jnp

from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source
from lucid.geometry import generate_detector

CONTRAST = "config/JUNO_nested_labls_geom_config.json"          # labls / water (n-step)
MATCHED  = "config/JUNO_nested_labls_matched_geom_config.json"  # labls / water_n148 (no step)
PHYS     = "config/JUNO_nested_labls_physics_config.json"
R_IN, R_OUT = 17.5, 19.5
N_W, N_LS = 1.33, 1.48
THETA_C = np.arcsin(N_W / N_LS)
R_STAR = R_IN * np.sin(THETA_C)            # 15.73 m TIR threshold
# K=32: the matched/non-TIR side converges by K=16, but the TIR (contrast) source needs
# K~32-48 to fully converge (whispering-gallery trapped photons; see two_boundary_K_tir.py).
# K=24 over-estimates the interface loss by ~0.2 pp; K=32 is 99.9% converged.
WAVELENGTH, INTENSITY, N_RAYS, K = 430.0, 50_000_000.0, 500_000, 32
N_BATCH = 24
OUTDIR = os.path.join(os.path.dirname(__file__), "out")

# Sources spanning the TIR threshold (radius on +z axis).
SRC_R = [0.0, 13.0, 16.5, 17.0, 17.3]
SENSORS = np.asarray(generate_detector(CONTRAST).all_points)
COST = SENSORS[:, 2] / R_OUT
PHI = np.arctan2(SENSORS[:, 1], SENSORS[:, 0])


def make_dp():
    return DetectorParams.from_flat(
        scatter_length=50.0, wall_reflection_rate=0.2, sensor_reflection_rate=0.2,
        absorption_length=50.0, qe=0.065, qe_corrections=jnp.ones(10000))


def sim_for(geom):
    return setup_event_simulator(
        geom, N_RAYS, temperature=None, K=K, is_calibration=True,
        detector_type='nested_sphere', wavelength_mode=True, physics_config=PHYS)


def charge_crn(sim, dp, r_s):
    """Per-sensor mean charge over N_BATCH batches at fixed CRN seeds."""
    src = isotropic_source(position=[0.0, 0.0, r_s], intensity=INTENSITY, wavelength=WAVELENGTH)
    acc = np.zeros(SENSORS.shape[0])
    for b in range(N_BATCH):
        acc += np.asarray(sim(src, dp, jax.random.PRNGKey(1000 + b))[0])
    return acc / N_BATCH


def cos_profile(c, n_bins=48):
    edges = np.linspace(-1, 1, n_bins + 1)
    idx = np.clip(np.digitize(COST, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([c[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])
    return centers, prof


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    dp = make_dp()
    sc, sm = sim_for(CONTRAST), sim_for(MATCHED)
    out = {}
    print(f"theta_c={np.rad2deg(THETA_C):.1f} deg, TIR threshold r*={R_STAR:.2f} m")
    for r in SRC_R:
        c_con = charge_crn(sc, dp, r)
        c_mat = charge_crn(sm, dp, r)
        cc, pc = cos_profile(c_con)        # azimuthally-averaged (denoised) profiles
        _, pm = cos_profile(c_mat)
        loss = 1.0 - c_con.sum() / c_mat.sum()
        out[str(r)] = dict(cos=cc.tolist(), con=pc.tolist(), mat=pm.tolist(),
                           ratio=(pc / pm).tolist(), loss=float(loss),
                           timax=float(np.rad2deg(np.arcsin(min(r / R_IN, 1.0)))),
                           tir=bool(r > R_STAR))
        print(f"  r={r:5.2f} ({'TIR' if r>R_STAR else 'no TIR':6s}) "
              f"max-incidence={out[str(r)]['timax']:4.1f} deg  interface loss={loss*100:5.2f}%")
    json.dump(out, open(os.path.join(OUTDIR, "two_boundary_interface_signature.json"), "w"), indent=2)
    plot_from_json()
    print(f"\nWrote {OUTDIR}/two_boundary_interface_signature.png")


def _ring_image(cos_centers, prof, npx=300):
    """Render an axisymmetric profile prof(cosθ) as a near-hemisphere orthographic disk
    (radius rho = sinθ), filled from the DENOISED 1D profile — no per-sensor noise."""
    g = np.linspace(-1, 1, npx)
    X, Y = np.meshgrid(g, g)
    rho = np.sqrt(X**2 + Y**2)
    inside = rho <= 1.0
    cth = np.sqrt(np.clip(1 - rho**2, 0, 1))           # near hemisphere: cosθ from disk radius
    img = np.full((npx, npx), np.nan)
    img[inside] = np.interp(cth[inside], cos_centers, prof)
    return img


def plot_from_json():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = json.load(open(os.path.join(OUTDIR, "two_boundary_interface_signature.json")))
    Rs = sorted((float(k) for k in d), key=float)
    fig = plt.figure(figsize=(15, 4.6))

    # (A) Interface transfer vs polar angle — azimuthally averaged, robust y-range.
    a1 = fig.add_subplot(1, 3, 1)
    for r in Rs:
        e = d[str(r)]; ls = "-" if e["tir"] else "--"
        a1.plot(np.array(e["cos"]), np.array(e["ratio"]), ls, lw=1.8,
                label=f"r={r:g}  ({'TIR '+('%.0f'%e['timax'])+'°' if e['tir'] else 'Fresnel'})")
    a1.axhline(1.0, color="k", lw=0.6, ls=":")
    a1.set_ylim(0.4, 1.25)                              # clip near-pole caustic spikes (noted in text)
    a1.set_xlabel(r"$\cos\theta$  (+1 = outer pole toward source)")
    a1.set_ylabel("interface transfer  =  contrast / matched")
    a1.set_title("Pure interface signature (bulk divided out)\nsolid = source past TIR threshold (r*=15.73 m)")
    a1.legend(fontsize=8, loc="lower center"); a1.grid(alpha=0.3)

    # (B) Absolute azimuthal profiles for the strongest TIR source — where the light is.
    r = Rs[-1]; e = d[str(r)]
    a2 = fig.add_subplot(1, 3, 2)
    a2.semilogy(e["cos"], e["mat"], "C7--", lw=1.8, label="matched (no index step)")
    a2.semilogy(e["cos"], e["con"], "C3-", lw=1.8, label="contrast (Fresnel+TIR)")
    a2.set_xlabel(r"$\cos\theta$"); a2.set_ylabel("mean charge / sensor")
    a2.set_title(f"Absolute pattern, source r={r:g} (max incidence {e['timax']:.0f}°)\n"
                 f"interface removes {e['loss']*100:.0f}% of the light")
    a2.legend(fontsize=8); a2.grid(alpha=0.3, which="both")

    # (C) Smooth ring map of the interface transfer (near hemisphere) for that source.
    a3 = fig.add_subplot(1, 3, 3)
    img = _ring_image(np.array(e["cos"]), np.array(e["ratio"]))
    im = a3.imshow(img, extent=[-1, 1, -1, 1], origin="lower", cmap="RdBu_r",
                   vmin=0.5, vmax=1.5)
    th = np.linspace(0, 2*np.pi, 300)
    a3.plot(np.sin(THETA_C)*np.cos(th), np.sin(THETA_C)*np.sin(th), "k--", lw=1.2)
    a3.plot(np.cos(th), np.sin(th), "k-", lw=0.8)
    a3.set_aspect("equal"); a3.set_xticks([]); a3.set_yticks([])
    a3.set_title(f"interface transfer, near hemisphere (r={r:g})\n"
                 f"center=pole toward source; dashed=critical-angle ring")
    fig.colorbar(im, ax=a3, label="contrast/matched", fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "two_boundary_interface_signature.png"), dpi=130)


if __name__ == "__main__":
    main()
