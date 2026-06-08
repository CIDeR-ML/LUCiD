"""Compare Method A (cherenkov) vs Method B (cherenkov_qe) for shotgun-like
simulations, in two propagator modes:

  • expected_value=True  → continuous QE weight, no Bernoulli
  • expected_value=False → Bernoulli QE (MC, what shotgun uses)

For each setup and each origin position, draws `n_trials` independent runs
at fixed `n_photons`, computes per-sensor mean and std across trials, and
reports:

  • total-detected-charge mean and std (global pulse-count picture)
  • per-sensor mean ratio B/A (should be ~1 for unbiasedness)
  • per-sensor std ratio B/A on illuminated sensors
  • "equivalent photons": N × var_A/var_B — Method B with N photons matches
    Method A with that many photons.

Usage:
    python -m scripts.compare_method_ab \
        --detector config/WCTE_geom_config.json \
        --physics-config config/SK_physics_config.json \
        --n-photons 30000 --n-trials 15
"""
import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import load_physics_config
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source


def _percent(x):
    return f"{x * 100:+.2f}%"


def _summarize(label, A, B, n_photons):
    """A, B: shape (n_trials, num_sensors) charges. Prints a summary."""
    # Totals across sensors.
    tot_A = A.sum(axis=1)
    tot_B = B.sum(axis=1)
    mean_tot_A = float(tot_A.mean()); std_tot_A = float(tot_A.std())
    mean_tot_B = float(tot_B.mean()); std_tot_B = float(tot_B.std())

    # Per-sensor stats.
    mean_A = A.mean(axis=0); mean_B = B.mean(axis=0)
    std_A = A.std(axis=0) + 1e-30
    std_B = B.std(axis=0) + 1e-30
    # Illuminated sensors = those carrying at least 1% of the peak mean in A.
    peak = float(mean_A.max()) + 1e-30
    illum = mean_A > 0.01 * peak

    mean_ratio = mean_B[illum] / (mean_A[illum] + 1e-30)
    std_ratio = std_B[illum] / std_A[illum]
    var_ratio_A_over_B = (std_A[illum] / std_B[illum]) ** 2   # Neff = N * var_A/var_B
    eq_photons = n_photons * var_ratio_A_over_B

    print(f"\n── {label} ──")
    print(f"  total detected: A = {mean_tot_A:9.3f} ± {std_tot_A:7.3f}   "
          f"B = {mean_tot_B:9.3f} ± {std_tot_B:7.3f}")
    print(f"    mean ratio B/A          = {mean_tot_B/mean_tot_A:6.4f}  "
          f"(should be ~1.0 — unbiasedness)")
    print(f"    std ratio B/A (totals)  = {std_tot_B/std_tot_A:6.4f}")
    print(f"  per-sensor (illuminated, n={int(illum.sum())}):")
    print(f"    mean ratio median       = {float(jnp.median(mean_ratio)):6.4f}")
    print(f"    std  ratio median       = {float(jnp.median(std_ratio)):6.4f}")
    print(f"    var(A)/var(B) median    = {float(jnp.median(var_ratio_A_over_B)):6.4f}")
    print(f"    eq photons for B ≈ A    = {float(jnp.median(eq_photons)):9.0f}   "
          f"(Method B with N={n_photons} matches A with this many)")


def _build_sim(geom, dp, phys, n_photons, sampling, use_expected_value):
    # Force `hit_mode='aggregated'` so both modes return (charges, times) shape
    # (num_sensors,). Varying only use_expected_value isolates the propagator.
    return setup_event_simulator(
        geom,
        n_photons=n_photons,
        temperature=None if not use_expected_value else 0.2,
        K=4,
        is_calibration=True,
        default_detector_params=dp,
        physics_config=phys,
        wavelength_mode=True,
        wavelength_sampling=sampling,
        use_expected_value=use_expected_value,
        hit_mode='aggregated',
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--detector', required=True)
    p.add_argument('--physics-config', required=True)
    p.add_argument('--n-photons', type=int, default=30000)
    p.add_argument('--n-trials', type=int, default=15)
    p.add_argument('--K', type=int, default=4)
    args = p.parse_args()

    det = generate_detector(args.detector)
    dp_full, _, _ = load_physics_config(args.physics_config)
    # Downsize qe_corrections to this geometry's sensor count.
    from lucid.detector_params import DetectorParams
    N_sensors = len(det.all_points)
    dp = DetectorParams.from_flat(
        num_sensors=N_sensors,
        scatter_length=dp_full.scattering.scatter_length,
        mie_scatter_length=dp_full.scattering.mie_scatter_length,
        g=dp_full.scattering.g,
        wall_reflection_rate=dp_full.reflection.wall_reflection_rate,
        sensor_reflection_rate=dp_full.reflection.sensor_reflection_rate,
        absorption_length=dp_full.absorption.absorption_length,
        qe=dp_full.response.qe,
        qe_corrections=jnp.ones(N_sensors),
    )

    # A few source positions to vary geometry-dependent variance.
    H = float(det.H) if hasattr(det, 'H') else 3.0
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.0, H / 4.0]),
    ]

    for use_ev, tag in [(True, "expected_value=True"), (False, "use_expected_value=False (Bernoulli MC)")]:
        print("\n" + "=" * 72)
        print(f"Propagator mode: {tag}")
        print("=" * 72)

        t0 = time.time()
        sim_A = _build_sim(args.detector, dp, args.physics_config, args.n_photons,
                           'cherenkov', use_ev)
        sim_B = _build_sim(args.detector, dp, args.physics_config, args.n_photons,
                           'cherenkov_qe', use_ev)
        print(f"  [setup {time.time()-t0:.1f}s]")

        for pos in positions:
            src = isotropic_source(position=jnp.asarray(pos, dtype=jnp.float32),
                                   intensity=1.0)
            keys = jax.random.split(jax.random.PRNGKey(123), args.n_trials)
            t0 = time.time()
            A = jnp.stack([sim_A(src, k)[0] for k in keys])
            B = jnp.stack([sim_B(src, k)[0] for k in keys])
            dt = time.time() - t0
            _summarize(f"origin = {tuple(pos.tolist())}  [{dt:.1f}s]",
                       A, B, args.n_photons)


if __name__ == '__main__':
    main()
