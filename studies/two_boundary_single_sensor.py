"""One fixed sensor, source moved through the detector — charge that sensor collects.

Pick a single PMT at the top pole (0, 0, R_out). Move the source anywhere in the LS and
plot, in the SOURCE's physical (x, z) plane, how much charge that one sensor records. By
rotational symmetry this equals the azimuthally-averaged response field g(r_s, γ) already
computed (γ = opening angle between the source and the fixed sensor), so we just remap that
data onto (x, z) — same information, but in the intuitive geometric picture (no new sim, and
denoised).

  source at (x, z)  →  r_s = √(x²+z²),  cosγ = z / r_s   →  charge = g(r_s, γ)

Left:  charge in the fixed sensor vs source position.
Right: interface effect (contrast / matched-index control) — where Fresnel/TIR change it.

Out: studies/out/two_boundary_single_sensor.png
"""
import os, json
import numpy as np
from scipy.interpolate import RegularGridInterpolator

R_IN, R_OUT = 17.5, 19.5
OUTDIR = os.path.join(os.path.dirname(__file__), "out")


def _circle(ax, rad, **kw):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(rad * np.cos(t), rad * np.sin(t), **kw)


def main():
    d = json.load(open(os.path.join(OUTDIR, "two_boundary_sensor_response.json")))
    r = np.array(d["r"]); cos = np.array(d["cos"])
    Mc = np.array(d["contrast"]); Mm = np.array(d["matched"])
    fc = RegularGridInterpolator((r, cos), Mc, bounds_error=False, fill_value=np.nan)
    fm = RegularGridInterpolator((r, cos), Mm, bounds_error=False, fill_value=np.nan)

    n = 500
    g = np.linspace(-R_IN, R_IN, n)
    X, Z = np.meshgrid(g, g)
    RS = np.sqrt(X ** 2 + Z ** 2)
    COSG = np.divide(Z, RS, out=np.ones_like(Z), where=RS > 1e-9)
    inside = RS <= r.max()
    pts = np.stack([np.clip(RS, r.min(), r.max()), np.clip(COSG, cos.min(), cos.max())], axis=-1)
    charge = fc(pts); matched = fm(pts)
    ratio = charge / matched
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
