"""Single-sensor response field g(r_s, γ) over source location, exploiting spherical symmetry.

By rotational symmetry the charge a FIXED outer-sphere sensor collects from a source depends
only on (r_s = source radius, γ = source–sensor opening angle). And by reciprocity the
azimuthally-averaged hit pattern of an on-axis source at radius r_s IS the clean (denoised)
estimator of that one sensor's response vs γ (=sensor polar angle θ). So we stack azimuthal
profiles over a scan of source radii to fill the 2D response map.

The interface writes a TIR 'shadow' onto the map: (r_s, γ) whose DIRECT refracted path needs
super-critical incidence get only scattered light. We overlay that geometric boundary
(ray-traced, no MC) and divide by the matched (no-index-step) control to isolate it.

Out: studies/out/two_boundary_sensor_response.png (+ .json)
"""
import os, json
import numpy as np
import jax, jax.numpy as jnp
from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source
from lucid.geometry import generate_detector

CONTRAST = "config/JUNO_nested_labls_geom_config.json"
MATCHED = "config/JUNO_nested_labls_matched_geom_config.json"
PHYS = "config/JUNO_nested_labls_physics_config.json"
R_IN, R_OUT, N_W, N_LS = 17.5, 19.5, 1.33, 1.48
THETA_C = np.arcsin(N_W / N_LS)
R_STAR = R_IN * np.sin(THETA_C)
WL, INT, NR, NB, K = 430.0, 50_000_000.0, 500_000, 8, 32
NBIN = 60
OUTDIR = os.path.join(os.path.dirname(__file__), "out")

# Source radii: coarse in the bulk, dense near the interface where the TIR action is.
SRC_R = np.concatenate([np.linspace(0.0, 15.0, 9), np.linspace(15.5, 17.4, 11)])
SENS = np.asarray(generate_detector(CONTRAST).all_points)
COSG = SENS[:, 2] / R_OUT


def dp():
    return DetectorParams.from_flat(
        scatter_length=50.0, wall_reflection_rate=0.2, sensor_reflection_rate=0.2,
        absorption_length=50.0, qe=0.065, qe_corrections=jnp.ones(10000))


def az_profile(sim, d, r, edges):
    """Azimuthally-averaged charge vs cosγ for an on-axis source at radius r (CRN-batched)."""
    src = isotropic_source(position=[0.0, 0.0, float(r)], intensity=INT, wavelength=WL)
    acc = np.zeros(SENS.shape[0])
    for b in range(NB):
        acc += np.asarray(sim(src, d, jax.random.PRNGKey(3000 + b))[0])
    acc /= NB
    idx = np.clip(np.digitize(COSG, edges) - 1, 0, NBIN - 1)
    return np.array([acc[idx == b].mean() if np.any(idx == b) else np.nan for b in range(NBIN)])


def tir_shadow_edge(a):
    """cosγ where the GRAZING (just-critical) transmitted ray lands — the forward edge of
    the direct transmitted beam from an on-axis source at radius a. Rays more tangential
    than this TIR (no direct light), so by chord symmetry the source directly illuminates a
    forward cap (cosγ > this) and a backward cap, leaving a direct-shadow band between.
    Returns cosγ_edge or None (a below threshold)."""
    if a <= R_STAR:
        return None
    psi = np.arcsin(R_STAR / a) * 0.999            # just sub-critical → transmits, near-grazing
    P = np.array([0.0, 0.0, a])
    d = np.array([np.sin(psi), 0.0, np.cos(psi)])
    b2 = 2 * P @ d; c2 = P @ P - R_IN**2
    Q = P + ((-b2 + np.sqrt(b2*b2 - 4*c2)) / 2) * d
    nrm = Q / R_IN
    ci = abs(d @ nrm); eta = N_LS / N_W
    ct = np.sqrt(max(0.0, 1 - eta*eta*(1 - ci*ci)))
    s = np.sign(d @ nrm); m = -s * nrm
    dt = eta * d + (eta * ci - ct) * m; dt /= np.linalg.norm(dt)
    b3 = 2 * Q @ dt; c3 = Q @ Q - R_OUT**2
    Sout = Q + ((-b3 + np.sqrt(b3*b3 - 4*c3)) / 2) * dt
    return float(Sout[2] / R_OUT)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = dp()
    edges = np.linspace(-1, 1, NBIN + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sc = setup_event_simulator(CONTRAST, NR, temperature=None, K=K, is_calibration=True,
                               detector_type='nested_sphere', wavelength_mode=True, physics_config=PHYS)
    sm = setup_event_simulator(MATCHED, NR, temperature=None, K=K, is_calibration=True,
                               detector_type='nested_sphere', wavelength_mode=True, physics_config=PHYS)
    Mc = np.array([az_profile(sc, d, r, edges) for r in SRC_R])     # (n_r, n_cos) contrast
    Mm = np.array([az_profile(sm, d, r, edges) for r in SRC_R])     # matched
    edge = np.array([tir_shadow_edge(r) for r in SRC_R], dtype=float)
    print("source radii:", np.round(SRC_R, 2))
    print("shadow cosγ edge:", np.round(edge, 3))
    json.dump({"r": SRC_R.tolist(), "cos": centers.tolist(),
               "contrast": Mc.tolist(), "matched": Mm.tolist(), "shadow_cos": edge.tolist()},
              open(os.path.join(OUTDIR, "two_boundary_sensor_response.json"), "w"))
    _plot(SRC_R, centers, Mc, Mm, edge)
    print(f"Wrote {OUTDIR}/two_boundary_sensor_response.png")


def _gamma_axis(ax):
    """Add a secondary top axis: source–sensor opening angle γ in degrees (γ = arccos cosγ)."""
    top = ax.secondary_xaxis("top",
                             functions=(lambda c: np.degrees(np.arccos(np.clip(c, -1, 1))),
                                        lambda g: np.cos(np.radians(g))))
    top.set_xticks([0, 30, 60, 90, 120, 150, 180])
    top.set_xlabel(r"opening angle $\gamma$ between source and sensor (deg)", fontsize=9)


def _plot(R, cos, Mc, Mm, edge):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.8))
    ext = [cos[0], cos[-1], R[0], R[-1]]
    XLAB = (r"$\cos\gamma$    ($+1$: sensor directly over source  →  $-1$: sensor on far side)")

    # ---- Panel 1: the single-sensor response field g(r_s, γ) ----
    im1 = a1.imshow(Mc, origin="lower", aspect="auto", extent=ext, cmap="turbo",
                    norm=LogNorm(vmin=max(Mc[Mc > 0].min(), Mc.max() * 1e-3), vmax=Mc.max()))
    a1.set_xlabel(XLAB); a1.set_ylabel(r"source radius $r_s$ (m)")
    a1.set_title(r"Charge collected by one sensor vs. source position", fontsize=11)
    _gamma_axis(a1)
    fig.colorbar(im1, ax=a1, label="mean charge per sensor (log)")

    # ---- Panel 2: interface signature = ratio to the no-index-step control ----
    ratio = Mc / np.maximum(Mm, 1e-9)
    im2 = a2.imshow(ratio, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=0.4, vmax=1.6)
    a2.axhline(R_STAR, color="k", ls="--", lw=1.0)
    a2.text(-0.97, R_STAR + 0.2, r"$r^*$", color="k", fontsize=10)
    a2.set_xlabel(XLAB); a2.set_ylabel(r"source radius $r_s$ (m)")
    a2.set_title(r"Interface effect (ratio to matched-index control)", fontsize=11)
    _gamma_axis(a2)
    fig.colorbar(im2, ax=a2, label="contrast / matched")

    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "two_boundary_sensor_response.png"), dpi=130)


def plot_from_json():
    """Re-make the figure from the saved JSON (no simulation)."""
    d = json.load(open(os.path.join(OUTDIR, "two_boundary_sensor_response.json")))
    _plot(np.array(d["r"]), np.array(d["cos"]), np.array(d["contrast"]),
          np.array(d["matched"]), np.array(d["shadow_cos"], dtype=float))


if __name__ == "__main__":
    main()
