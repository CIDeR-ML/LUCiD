"""Show the two-medium interface is working: engine output vs independent analytics.

(A) Fresnel + TIR: the engine SAMPLES transmit/reflect ~ Bernoulli(T) at the interface.
    Sampling the engine's interface kernel across incidence angles and counting the transmit
    fraction must reproduce the analytic unpolarised Fresnel transmission T(θ)=1-R(θ), and
    must drop to 0 at the critical angle θc=64° (total internal reflection).
(B) Snell: the engine's refracted-ray angle vs incidence must satisfy n_LS sinθ_i = n_W sinθ_t,
    and have no real solution beyond θc.
(C) Full forward: the detector-level interface effect (total charge contrast/matched, no
    absorption) must be ≈1 below the TIR threshold radius r* = R_in sinθc = 15.73 m (only
    weak Fresnel) and drop sharply above it as TIR removes light — i.e. the microscopic θc
    shows up as the correct geometric threshold in the full pipeline.

Run:  python studies/two_boundary_validation.py
Out:  studies/out/two_boundary_validation.png
"""
import os
import numpy as np
import jax, jax.numpy as jnp
from lucid.simulation.photon_step import _interface_refract_reflect
from lucid.simulation.reflection import fresnel_rr
from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source

N_LS, N_W, R_IN, R_OUT = 1.48, 1.33, 17.5, 19.5
THETA_C = np.degrees(np.arcsin(N_W / N_LS))
R_STAR = R_IN * (N_W / N_LS)
OUTDIR = os.path.join(os.path.dirname(__file__), "out")
CON = "config/JUNO_nested_labls_geom_config.json"
MAT = "config/JUNO_nested_labls_matched_geom_config.json"
PHYS = "config/JUNO_nested_labls_physics_config.json"


def engine_interface(theta_deg, n_samp=20000):
    """Sample the engine's interface kernel at a fixed LS->water incidence angle.
    Returns (empirical transmit fraction, mean refracted angle of transmitted rays)."""
    th = np.radians(theta_deg)
    d = jnp.array([np.cos(th), np.sin(th), 0.0])   # outward-going at incidence θ to radial x̂
    radial = jnp.array([1.0, 0.0, 0.0])
    us = jnp.linspace(1e-4, 1 - 1e-4, n_samp)
    f = jax.vmap(lambda u: _interface_refract_reflect(d, radial, jnp.array(0), N_LS, N_W, u))
    new_dir, new_mid, score, transmit = f(us)
    transmit = np.asarray(transmit)
    T_emp = transmit.mean()
    # refracted angle (transmitted rays only): angle of new_dir to radial
    nd = np.asarray(new_dir)
    ang = np.degrees(np.arccos(np.clip(np.abs(nd @ np.array([1.0, 0, 0])), 0, 1)))
    th_t = ang[transmit].mean() if transmit.any() else np.nan
    return T_emp, th_t


def forward_threshold():
    d = DetectorParams.from_flat(scatter_length=1e8, mie_scatter_length=1e8, absorption_length=1e8,
                                 wall_reflection_rate=0.0, sensor_reflection_rate=0.0, qe=0.065,
                                 qe_corrections=jnp.ones(10000))
    sc = setup_event_simulator(CON, 300000, temperature=None, K=24, is_calibration=True,
                               detector_type='nested_sphere', wavelength_mode=False, physics_config=PHYS)
    sm = setup_event_simulator(MAT, 300000, temperature=None, K=24, is_calibration=True,
                               detector_type='nested_sphere', wavelength_mode=False, physics_config=PHYS)
    def tot(s, r):
        src = isotropic_source(position=[0, 0, float(r)], intensity=5e7, wavelength=420.0)
        return float(np.mean([np.asarray(s(src, d, jax.random.PRNGKey(11 + b))[0]).sum() for b in range(4)]))
    radii = np.concatenate([np.linspace(0, 14, 8), np.linspace(14.5, 17.4, 9)])
    ratio = [tot(sc, r) / tot(sm, r) for r in radii]
    return radii, np.array(ratio)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    thetas = np.linspace(0, 89, 46)
    T_emp = np.array([engine_interface(t)[0] for t in thetas])
    th_t = np.array([engine_interface(t)[1] for t in thetas])
    # analytic
    ci = np.cos(np.radians(thetas))
    R_an = np.array([float(fresnel_rr(jnp.array(c), N_LS, N_W)[0]) for c in ci])
    T_an = 1 - R_an
    # Snell θ_t exists only for θ_i <= θc; past θc there is NO refracted ray (TIR) → NaN
    sin_t = N_LS / N_W * np.sin(np.radians(thetas))
    snell = np.where(sin_t <= 1.0, np.degrees(np.arcsin(np.clip(sin_t, 0, 1))), np.nan)
    print(f"θc={THETA_C:.1f}°, r*={R_STAR:.2f} m")
    print("forward threshold scan...")
    radii, fwd = forward_threshold()
    _plot(thetas, T_emp, T_an, th_t, snell, radii, fwd)
    print(f"Wrote {OUTDIR}/two_boundary_validation.png")


def _plot(th, T_emp, T_an, th_t, snell, radii, fwd):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.8))
    # A: Fresnel + TIR
    a.plot(th, T_an, "k-", lw=2, label="analytic Fresnel  T=1−R")
    a.plot(th, T_emp, "o", color="C2", ms=4, label="engine (sampled transmit fraction)")
    a.axvline(THETA_C, color="C3", ls="--", lw=1.2); a.text(THETA_C - 1, 0.5, f"θc={THETA_C:.0f}° (TIR)", color="C3", rotation=90, va="center", fontsize=9)
    a.set_xlabel("incidence angle at interface (deg)"); a.set_ylabel("transmission  T")
    a.set_title("(A) Fresnel + TIR  (LS→water)\nengine sampling vs analytic"); a.legend(fontsize=8); a.grid(alpha=0.3)
    # B: Snell
    b.plot(th, snell, "k-", lw=2, label=r"analytic Snell  $n_{LS}\sin\theta_i=n_W\sin\theta_t$")
    b.plot(th, th_t, "o", color="C0", ms=4, label="engine refracted angle")
    b.axvline(THETA_C, color="C3", ls="--", lw=1.2)
    b.text(THETA_C + 1, 20, f"θc={THETA_C:.0f}°\n(no refracted ray beyond)", color="C3", fontsize=8)
    b.set_xlabel("incidence angle θ_i (deg)"); b.set_ylabel("refracted angle θ_t (deg)")
    b.set_xlim(0, 90); b.set_ylim(0, 95)
    b.set_title("(B) Snell refraction\nrefracted ray bends; terminates at θc (TIR)"); b.legend(fontsize=8); b.grid(alpha=0.3)
    # C: forward threshold
    c.axhline(1.0, color="k", ls=":", lw=1)
    c.axvspan(R_STAR, 17.5, color="C3", alpha=0.08)
    c.axvline(R_STAR, color="C3", ls="--", lw=1.4); c.text(R_STAR - 0.3, 0.62, f"TIR threshold\nr*={R_STAR:.2f} m", color="C3", ha="right", fontsize=9)
    c.plot(radii, fwd, "o-", color="C4", ms=4)
    c.set_xlabel("source radius (m)"); c.set_ylabel("interface charge ratio  contrast/matched")
    c.set_title("(C) Full forward: TIR threshold emerges\n≈1 below r* (Fresnel only), drops above (TIR)"); c.grid(alpha=0.3)
    c.set_ylim(0.45, 1.05)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "two_boundary_validation.png"), dpi=130)


if __name__ == "__main__":
    main()
