"""One fixed PMT: charge map (left) + its arrival-time distribution (right) vs source position.

Fix a single PMT at the top pole (0, 0, R_out) — the same single-sensor view as
studies/two_boundary_single_sensor.py.

LEFT  : charge that ONE PMT collects vs source position in the (x, z) plane (turbo heatmap,
        remapped from the azimuthal response field), with the chosen source positions marked.
RIGHT : that same ONE PMT's photon arrival-time distribution (1 ns bins, expected/mean
        waveform — every slot deposits its continuous weight, no shot noise), PEAK-NORMALIZED
        per source. Sources span the z-axis top→bottom (near the PMT → far across the
        detector), so the direct-light peak walks evenly later as the time-of-flight grows.

Detector: nested JUNO (LS inner R=17.5 m, water R=19.5 m).
Out: studies/out/single_sensor_time_distribution.png
Run (GPU):  python studies/single_sensor_time_distribution.py
"""
import os, json
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
from lucid.detector_params import DetectorParams
from lucid.sources import isotropic_source
from lucid.simulation import setup_event_simulator
from lucid.geometry import generate_detector

GEOM = "config/JUNO_nested_labls_geom_config.json"
PHYS = "config/JUNO_nested_labls_physics_config.json"
R_IN, R_OUT = 17.5, 19.5
WL, INT, NR, K = 430.0, 5e6, 5_000_000, 16        # N = 5M rays
WINDOW, BIN = 220.0, 1.0
# Source (x, z) positions spread around the plane (not just the vertical axis): top / upper-
# right / left / lower-left / lower-right — i.e. a range of distances AND angles to the PMT.
SRC_POS = [(2.0, 15.0), (9.0, 9.0), (-13.0, 3.0), (11.0, -5.0), (-6.0, -12.0)]
OUTDIR = os.path.join(os.path.dirname(__file__), "out")


def dp():
    return DetectorParams.from_flat(scatter_length=50.0, wall_reflection_rate=0.2,
                                    sensor_reflection_rate=0.2, absorption_length=50.0,
                                    qe=0.065, qe_corrections=jnp.ones(10000))


def _charge_map():
    """Charge that the top-pole PMT collects vs source (x, z), remapped from the saved
    azimuthal response field g(r_s, cosγ) (see two_boundary_sensor_response.py)."""
    d = json.load(open(os.path.join(OUTDIR, "two_boundary_sensor_response.json")))
    r = np.array(d["r"]); cos = np.array(d["cos"]); M = np.array(d["contrast"])
    f = RegularGridInterpolator((r, cos), M, bounds_error=False, fill_value=np.nan)
    n = 420
    g = np.linspace(-R_IN, R_IN, n)
    X, Z = np.meshgrid(g, g)
    RS = np.sqrt(X**2 + Z**2)
    COSG = np.divide(Z, RS, out=np.ones_like(Z), where=RS > 1e-9)
    inside = RS <= r.max()
    pts = np.stack([np.clip(RS, r.min(), r.max()), np.clip(COSG, cos.min(), cos.max())], axis=-1)
    charge = f(pts); charge[~inside] = np.nan
    return X, Z, charge


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("backend:", jax.default_backend())

    # The fixed PMT: the sensor nearest the top pole (0, 0, R_out).
    sens = np.asarray(generate_detector(GEOM).all_points)
    top_idx = int(np.argmax(sens[:, 2]))
    print(f"fixed PMT idx={top_idx} at {np.round(sens[top_idx], 2)}")

    sim = setup_event_simulator(
        GEOM, NR, temperature=None, K=K, is_calibration=True, detector_type='nested_sphere',
        wavelength_mode=True, physics_config=PHYS, use_expected_value=True, hit_mode='waveform_expected',
        waveform_config={'window_ns': WINDOW, 'bin_width_ns': BIN, 'smear_time': True, 'tts_sigma_ns': 1.0})

    d = dp()
    n_bins = int(round(WINDOW / BIN))
    t = (np.arange(n_bins) + 0.5) * BIN
    dists = []
    for x, z in SRC_POS:
        src = isotropic_source(position=[float(x), 0.0, float(z)], intensity=INT, wavelength=WL)
        wf = np.asarray(sim(src, d, jax.random.PRNGKey(0))[0])[top_idx]   # ONE PMT's 1 ns waveform
        dists.append(wf)
        pk = t[wf.argmax()] if wf.max() > 0 else np.nan
        print(f"  source (x,z)=({x:5.1f},{z:6.1f}) m : PMT charge {wf.sum():.1f},  arrival peak {pk:.0f} ns")

    _plot(t, dists, *_charge_map())
    print(f"\nWrote {OUTDIR}/single_sensor_time_distribution.png")


def _plot(t, dists, X, Z, charge):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    # Distinct categorical colors (stand out on the turbo charge map; match the right curves).
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628'][:len(SRC_POS)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.0), gridspec_kw={'width_ratios': [1, 1.4]})

    # ── Left: charge that the fixed PMT collects vs source position (turbo heatmap) ──
    th = np.linspace(0, 2 * np.pi, 400)
    vmax = np.nanmax(charge)
    cm = axL.pcolormesh(X, Z, charge, cmap="turbo", shading="auto",
                        norm=LogNorm(vmin=vmax * 1e-2, vmax=vmax))
    axL.plot(R_OUT * np.cos(th), R_OUT * np.sin(th), 'k-', lw=1.5)
    axL.plot(R_IN * np.cos(th), R_IN * np.sin(th), 'k--', lw=1.0)
    axL.plot(0, R_OUT, marker='v', ms=15, color='yellow', mec='k', mew=1.2, zorder=6)
    axL.text(0, R_OUT + 0.8, "fixed PMT", ha='center', fontsize=9)
    for (x, z), c in zip(SRC_POS, colors):
        axL.plot(x, z, 'o', color=c, ms=12, mec='white', mew=2.0, zorder=5)
    axL.set_aspect('equal'); axL.set_xlim(-R_OUT - 1, R_OUT + 1); axL.set_ylim(-R_OUT - 1, R_OUT + 1.6)
    axL.set_xlabel("source x (m)"); axL.set_ylabel("source z (m)")
    axL.set_title("Charge in the fixed PMT vs. source position")
    fig.colorbar(cm, ax=axL, fraction=0.046, label="charge in that PMT (log)")

    # ── Right: that one PMT's 1 ns arrival-time distribution per source, PEAK-NORMALIZED ──
    # (charge spans ~50× across positions; normalize so every shape is visible.)
    for (x, z), dist, c in zip(SRC_POS, dists, colors):
        dist = np.asarray(dist)
        norm = dist / dist.max() if dist.max() > 0 else dist
        axR.step(t, norm, where='mid', color=c, lw=1.6, label=f"({x:+.0f}, {z:+.0f}) m")
    axR.set_xlabel("photon arrival time at the PMT (ns)  [1 ns bins]")
    axR.set_ylabel("expected charge per 1 ns bin  (peak-normalized)")
    axR.set_title("That PMT's arrival-time distribution (each normalized to its peak)")
    axR.set_xlim(0, np.max([t[np.asarray(d) > 1e-6].max() for d in dists if np.any(np.asarray(d) > 1e-6)]) * 1.05)
    axR.set_ylim(0, 1.08)
    axR.legend(title="source (x, z)", fontsize=9)
    axR.grid(alpha=0.25)

    fig.suptitle("One fixed PMT — charge map (left) + arrival-time distribution (right) vs source position "
                 "(nested JUNO, expected 1 ns waveform)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "single_sensor_time_distribution.png"), dpi=130)


if __name__ == "__main__":
    main()
