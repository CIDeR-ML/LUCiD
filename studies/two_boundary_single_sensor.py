"""One fixed PMT (top pole), source moved through the LS — charge that PMT collects.

SELF-CONTAINED: this script computes its OWN single-sensor charge field g(r_s, γ) by running
the simulator (on-axis source-radius scan + azimuthal average, using spherical symmetry +
reciprocity — the same computation as two_boundary_sensor_response.py), then remaps it onto
the physical (x, z) source plane. It no longer depends on a pre-computed .json.

  source at (x, z)  →  r_s = √(x²+z²),  cosγ = z / r_s   →  charge = g(r_s, γ)

Left:  charge in the fixed top PMT vs source position.
Right: interface effect (contrast / matched-index control) — where Fresnel/TIR change it.

Out: studies/out/two_boundary_single_sensor.png
Run (GPU):  python studies/two_boundary_single_sensor.py
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source
from lucid.geometry import generate_detector

CONTRAST = "config/JUNO_nested_labls_geom_config.json"           # LAB-LS (n=1.48) inside water (n=1.33)
MATCHED = "config/JUNO_nested_labls_matched_geom_config.json"    # same bulk, no index step (control)
PHYS = "config/JUNO_nested_labls_physics_config.json"
R_IN, R_OUT = 17.5, 19.5
WL, INT, NR, NB, K = 430.0, 5.0e7, 500_000, 8, 32
NBIN = 60
# Source radii: coarse in the bulk, dense near the interface where the TIR action is.
SRC_R = np.concatenate([np.linspace(0.0, 15.0, 9), np.linspace(15.5, 17.4, 11)])
SENS = np.asarray(generate_detector(CONTRAST).all_points)
COSG = SENS[:, 2] / R_OUT
OUTDIR = os.path.join(os.path.dirname(__file__), "out")


def dp():
    return DetectorParams.from_flat(
        scatter_length=50.0, wall_reflection_rate=0.2, sensor_reflection_rate=0.2,
        absorption_length=50.0, qe=0.065, qe_corrections=jnp.ones(10000))


def az_profile(sim, d, r, edges):
    """Azimuthally-averaged charge vs cosγ for an on-axis source at radius r (CRN-batched).

    By reciprocity this IS the (denoised) response of a fixed top-pole sensor vs the
    source-sensor opening angle γ (= sensor polar angle), for a source at radius r."""
    src = isotropic_source(position=[0.0, 0.0, float(r)], intensity=INT, wavelength=WL)
    acc = np.zeros(SENS.shape[0])
    for b in range(NB):
        acc += np.asarray(sim(src, d, jax.random.PRNGKey(3000 + b))[0])
    acc /= NB
    idx = np.clip(np.digitize(COSG, edges) - 1, 0, NBIN - 1)
    return np.array([acc[idx == b].mean() if np.any(idx == b) else np.nan for b in range(NBIN)])


def compute_fields():
    """Run the sims → single-sensor charge field g(r_s, cosγ) for the contrast + matched configs."""
    d = dp()
    edges = np.linspace(-1, 1, NBIN + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sc = setup_event_simulator(CONTRAST, NR, temperature=None, K=K, is_calibration=True,
                               detector_type='nested_sphere', wavelength_mode=True, physics_config=PHYS)
    sm = setup_event_simulator(MATCHED, NR, temperature=None, K=K, is_calibration=True,
                               detector_type='nested_sphere', wavelength_mode=True, physics_config=PHYS)
    Mc = np.array([az_profile(sc, d, r, edges) for r in SRC_R])     # contrast (index step)
    Mm = np.array([az_profile(sm, d, r, edges) for r in SRC_R])     # matched (no index step)
    return SRC_R, centers, Mc, Mm


def _circle(ax, rad, **kw):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(rad * np.cos(t), rad * np.sin(t), **kw)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("backend:", jax.default_backend())
    r, cos, Mc, Mm = compute_fields()                              # <-- generates its own data

    # Remap g(r_s, cosγ) → the physical (x, z) source plane.
    fc = RegularGridInterpolator((r, cos), Mc, bounds_error=False, fill_value=np.nan)
    fm = RegularGridInterpolator((r, cos), Mm, bounds_error=False, fill_value=np.nan)
    n = 500
    g = np.linspace(-R_IN, R_IN, n)
    X, Z = np.meshgrid(g, g)
    RS = np.sqrt(X ** 2 + Z ** 2)
    COSGG = np.divide(Z, RS, out=np.ones_like(Z), where=RS > 1e-9)
    inside = RS <= r.max()
    pts = np.stack([np.clip(RS, r.min(), r.max()), np.clip(COSGG, cos.min(), cos.max())], axis=-1)
    charge = fc(pts); matched = fm(pts); ratio = charge / matched
    for A in (charge, ratio):
        A[~inside] = np.nan

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 6.4))
    for ax in (a1, a2):
        _circle(ax, R_OUT, color="k", lw=1.5)            # sensor sphere
        _circle(ax, R_IN, color="k", lw=1.0, ls="--")    # LS / water interface
        ax.plot(0, R_OUT, marker="v", ms=14, color="yellow", mec="k", mew=1.2, zorder=5)
        ax.text(0, R_OUT + 1.1, "sensor", ha="center", fontsize=8)
        ax.set_aspect("equal"); ax.set_xlim(-R_OUT - 1, R_OUT + 1); ax.set_ylim(-R_OUT - 1, R_OUT + 2)
        ax.set_xlabel("source x (m)"); ax.set_ylabel("source z (m)")

    cm = a1.pcolormesh(X, Z, charge, cmap="turbo", shading="auto",
                       norm=LogNorm(vmin=np.nanmax(charge) * 1e-2, vmax=np.nanmax(charge)))
    a1.set_title("Charge in the sensor vs. source position")
    fig.colorbar(cm, ax=a1, fraction=0.046, label="charge in that sensor (log)")

    rm = a2.pcolormesh(X, Z, ratio, cmap="RdBu_r", shading="auto", vmin=0.4, vmax=1.6)
    a2.set_title("Interface effect (ratio to matched-index control)")
    fig.colorbar(rm, ax=a2, fraction=0.046, label="contrast / matched")

    fig.suptitle("Fix one PMT (top), move the source through the LS, read that PMT", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "two_boundary_single_sensor.png"), dpi=130)
    print(f"Wrote {OUTDIR}/two_boundary_single_sensor.png")


if __name__ == "__main__":
    main()
