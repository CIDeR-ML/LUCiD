"""Expected (mean) photon arrival-time distribution vs source position — 1 ns binning.

Runs the simulator in EXPECTED-VALUE waveform mode (use_expected_value=True +
hit_mode='waveform_expected'): every propagation slot deposits its CONTINUOUS weight·QE into
the (sensor, 1 ns time-bin) it lands in — no Bernoulli coin drops a photon and there is no
shot noise, so the result is the smooth DiCE MEAN waveform. (The companion 'waveform' mode is
the sampled integer-photon version with shot noise.)

For each source position we sum the per-sensor 1 ns waveforms over all sensors → the detector's
total (mean) arrival-time distribution. As the source moves off-centre the distribution shifts
and broadens: a centred source illuminates every sensor at nearly the same time-of-flight (one
narrow late peak); an off-centre source lights the near wall early and the far wall late (an
early, broad, long-tailed distribution) — the timing information reconstruction uses.

Detector: nested JUNO (LS inner R=17.5 m, water R=19.5 m).
Out: studies/out/photon_arrival_time_distribution.png
Run (GPU):  python studies/photon_arrival_time_distribution.py
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from lucid.detector_params import DetectorParams
from lucid.sources import isotropic_source
from lucid.simulation import setup_event_simulator

GEOM = "config/JUNO_nested_labls_geom_config.json"
PHYS = "config/JUNO_nested_labls_physics_config.json"
R_IN, R_OUT = 17.5, 19.5
R_STAR = R_IN * np.sin(np.arcsin(1.33 / 1.48))     # TIR threshold radius ≈ 15.73 m
WL, INT, NR, K = 430.0, 5e6, 300_000, 16
WINDOW, BIN = 300.0, 1.0                            # 1 ns bins (expected mode is already smooth)
SRC_R = [0.0, 8.0, 14.0, 16.5]                     # source radius along +z (last is past r*: TIR regime)
OUTDIR = os.path.join(os.path.dirname(__file__), "out")


def dp():
    return DetectorParams.from_flat(scatter_length=50.0, wall_reflection_rate=0.2,
                                    sensor_reflection_rate=0.2, absorption_length=50.0,
                                    qe=0.065, qe_corrections=jnp.ones(10000))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("backend:", jax.default_backend())
    sim = setup_event_simulator(
        GEOM, NR, temperature=None, K=K, is_calibration=True, detector_type='nested_sphere',
        wavelength_mode=True, physics_config=PHYS, use_expected_value=True, hit_mode='waveform_expected',
        waveform_config={'window_ns': WINDOW, 'bin_width_ns': BIN, 'smear_time': True, 'tts_sigma_ns': 1.0})

    d = dp()
    n_bins = int(round(WINDOW / BIN))
    t = (np.arange(n_bins) + 0.5) * BIN
    dists = []
    for r in SRC_R:
        src = isotropic_source(position=[0.0, 0.0, float(r)], intensity=INT, wavelength=WL)
        wf = np.asarray(sim(src, d, jax.random.PRNGKey(0))[0]).sum(axis=0)   # expected mean over all PMTs
        dists.append(wf)
        print(f"  r_s={r:5.1f} m : charge {wf.sum():.0f},  peak {t[wf.argmax()]:.0f} ns,  "
              f"FWHM-ish span {np.ptp(t[wf > 0.5*wf.max()]):.0f} ns")

    _plot(t, dists)
    print(f"\nWrote {OUTDIR}/photon_arrival_time_distribution.png")


def _plot(t, dists):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (len(SRC_R) - 1)) for i in range(len(SRC_R))]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={'width_ratios': [1, 1.5]})

    # ── Left: detector cross-section + source positions ──
    th = np.linspace(0, 2 * np.pi, 400)
    axL.plot(R_OUT * np.cos(th), R_OUT * np.sin(th), 'k-', lw=1.6, label='PMT sphere (19.5 m)')
    axL.plot(R_IN * np.cos(th), R_IN * np.sin(th), 'k--', lw=1.0, label='LS / water interface (17.5 m)')
    for r, c in zip(SRC_R, colors):
        axL.plot(0, r, 'o', color=c, ms=12, mec='k', mew=1.0, zorder=5)
        axL.annotate(f"{r:.1f} m", (0, r), textcoords="offset points", xytext=(8, 0), fontsize=9)
    axL.set_aspect('equal'); axL.set_xlim(-R_OUT - 1, R_OUT + 1); axL.set_ylim(-R_OUT - 1, R_OUT + 1)
    axL.set_xlabel("x (m)"); axL.set_ylabel("z (m)")
    axL.set_title("Source positions (on the +z axis)")
    axL.legend(loc='lower center', fontsize=8)

    # ── Right: 1 ns photon arrival-time distribution per source ──
    for r, dist, c in zip(SRC_R, dists, colors):
        axR.step(t, dist, where='mid', color=c, lw=1.5, label=f"$r_s$ = {r:.1f} m")
    axR.set_xlabel("photon arrival time (ns)  [1 ns bins]")
    axR.set_ylabel("expected charge per 1 ns bin  (summed over all PMTs)")
    axR.set_title("Expected (mean) arrival-time distribution")
    axR.set_xlim(0, np.max([t[d > 1e-6].max() for d in dists]) * 1.02)
    axR.legend(title="source radius", fontsize=9)
    axR.grid(alpha=0.25)

    fig.suptitle("Expected waveform mode (DiCE mean, 1 ns): arrival-time distribution vs source position "
                 "— nested JUNO", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "photon_arrival_time_distribution.png"), dpi=130)


if __name__ == "__main__":
    main()
