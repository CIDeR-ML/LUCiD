"""Compare per-photon emission distributions: SIREN vs PhotonSim Geant4.

Both are evaluated in the source frame (muon at origin, pointing along ẑ),
so we can compare directly without dealing with rotations.

For each photon we compute:
    distance along track   = ray_origin_z      (mm)
    angle from track       = arccos(direction_z) (rad)

SIREN photons are weighted by `total_norm · weights / Nphot` (the per-photon
intensity used downstream by the propagation pipeline).
PhotonSim photons each get weight 1.

Plots:
  - 1D angle histogram (SIREN-weighted vs PhotonSim, density-normalized)
  - 1D distance histogram (same)
  - 2D (distance, angle) heatmap, SIREN side and PhotonSim side
  - Ratio (SIREN/PhotonSim) heatmap to highlight where the +1.5%
    "propagation efficiency" excess lives (positive = SIREN over-represents)
"""
from __future__ import annotations
import sys
sys.path.append('..')

import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from parameter_scans_1D_gaussian import DATA_FILE
from lucid.utils import unpack_photonsim_params
from lucid.siren.training.inference import SIRENPredictor
from lucid.siren.core import create_photonsim_siren_grid
from lucid.sources.siren_rays import photonsim_differentiable_get_rays
from lucid.generate import read_photon_data_from_photonsim


N_PHSIM_ENTRIES = 20      # aggregate over many entries for clean stats
SIREN_NPHOT     = 1_000_000
ENERGY          = 1050.0
ANGLE_BINS      = 80
DIST_BINS       = 80
DIST_MAX_MM     = 6000.0
ANGLE_MAX_RAD   = np.pi
OUT = Path('figures/multi_event_bias/distribution_compare.png')


def main():
    params = unpack_photonsim_params('muon', 'water')
    a, b, c = params['tot_n_photons_normalization']
    sa, sb, sc = params['num_seeds']
    predictor = SIRENPredictor(params['siren_model_path'])
    grid_data = create_photonsim_siren_grid(predictor)
    model_params = predictor.params
    total_norm = a * ENERGY**b + c
    print(f'SIREN total_norm at E={ENERGY:.0f}: {total_norm:.1f}')

    # --- SIREN: source frame: track at origin, direction = ẑ -----
    track_origin = jnp.array([0., 0., 0.])
    track_direction = jnp.array([0., 0., 1.])
    key = jax.random.PRNGKey(42)
    ray_vec, ray_orig, photon_weights = photonsim_differentiable_get_rays(
        track_origin, track_direction, ENERGY, SIREN_NPHOT,
        grid_data, model_params, key, sa, sb, sc)
    ray_vec = np.asarray(ray_vec)
    ray_orig = np.asarray(ray_orig)
    photon_weights = np.asarray(photon_weights)

    siren_dist_mm = ray_orig[:, 2] * 1000.0       # m → mm
    siren_cos = np.clip(ray_vec[:, 2], -1.0, 1.0)
    siren_angle = np.arccos(siren_cos)
    siren_intensity = total_norm * photon_weights / SIREN_NPHOT
    print(f'SIREN: total emitted intensity = {siren_intensity.sum():.1f}, '
          f'mean angle = {(siren_angle * siren_intensity).sum() / siren_intensity.sum():.3f} rad, '
          f'mean dist  = {(siren_dist_mm * siren_intensity).sum() / siren_intensity.sum():.1f} mm')

    # --- PhotonSim: aggregate across N_PHSIM_ENTRIES ---
    phsim_dist_mm_list, phsim_angle_list = [], []
    phsim_N_total = 0
    for entry in range(N_PHSIM_ENTRIES):
        pd = read_photon_data_from_photonsim(DATA_FILE, entry)
        po = np.asarray(pd['photon_origins'])      # cm
        pdir = np.asarray(pd['photon_directions'])  # unit vec
        n = len(po)
        phsim_N_total += n
        dz_mm = po[:, 2] * 10.0                   # cm → mm
        cos_th = np.clip(pdir[:, 2], -1.0, 1.0)
        angle = np.arccos(cos_th)
        phsim_dist_mm_list.append(dz_mm)
        phsim_angle_list.append(angle)
    phsim_dist_mm = np.concatenate(phsim_dist_mm_list)
    phsim_angle = np.concatenate(phsim_angle_list)
    print(f'PhotonSim: total photons = {phsim_N_total} (mean per entry '
          f'{phsim_N_total / N_PHSIM_ENTRIES:.0f}), mean angle = '
          f'{phsim_angle.mean():.3f} rad, mean dist = {phsim_dist_mm.mean():.1f} mm')

    # --- Plots ---
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1D angle
    ax = axes[0, 0]
    bins_ang = np.linspace(0, ANGLE_MAX_RAD, ANGLE_BINS + 1)
    h_siren, _ = np.histogram(siren_angle, bins=bins_ang,
                                weights=siren_intensity, density=True)
    h_phsim, _ = np.histogram(phsim_angle, bins=bins_ang, density=True)
    centers = 0.5 * (bins_ang[1:] + bins_ang[:-1])
    ax.plot(centers, h_siren, '-', color='C0', lw=1.5,
             label='SIREN (intensity-weighted)')
    ax.plot(centers, h_phsim, '-', color='C3', lw=1.5,
             label='PhotonSim (Geant4)')
    ax.axvline(np.arccos(1.0/1.33), color='k', linestyle=':', linewidth=1.0,
                label=f'Cherenkov θ_C = {np.arccos(1/1.33):.3f}')
    ax.set_xlabel('angle from track [rad]')
    ax.set_ylabel('density')
    ax.set_title('Emission angle distribution')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 1D distance
    ax = axes[0, 1]
    bins_dist = np.linspace(0, DIST_MAX_MM, DIST_BINS + 1)
    h_siren_d, _ = np.histogram(siren_dist_mm, bins=bins_dist,
                                  weights=siren_intensity, density=True)
    h_phsim_d, _ = np.histogram(phsim_dist_mm, bins=bins_dist, density=True)
    centers_d = 0.5 * (bins_dist[1:] + bins_dist[:-1])
    ax.plot(centers_d, h_siren_d, '-', color='C0', lw=1.5,
             label='SIREN (intensity-weighted)')
    ax.plot(centers_d, h_phsim_d, '-', color='C3', lw=1.5,
             label='PhotonSim (Geant4)')
    ax.set_xlabel('distance along track [mm]')
    ax.set_ylabel('density')
    ax.set_title('Emission distance distribution')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Ratio of densities (1D angle)
    ax = axes[0, 2]
    safe = h_phsim > 1e-10
    ratio = np.where(safe, h_siren / np.maximum(h_phsim, 1e-12), np.nan)
    ax.plot(centers, ratio, color='C2', lw=1.5)
    ax.axhline(1.0, color='k', linestyle=':', linewidth=1.0)
    ax.set_xlabel('angle from track [rad]')
    ax.set_ylabel('SIREN / PhotonSim density')
    ax.set_title('Angle distribution ratio')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 2.5)

    # 2D SIREN
    ax = axes[1, 0]
    H_siren, x_edges, y_edges = np.histogram2d(
        siren_dist_mm, siren_angle, bins=[bins_dist, bins_ang],
        weights=siren_intensity)
    H_siren_density = H_siren / max(H_siren.sum(), 1e-30)
    im = ax.imshow(H_siren_density.T, origin='lower', aspect='auto',
                    extent=[0, DIST_MAX_MM, 0, ANGLE_MAX_RAD],
                    cmap='Blues')
    ax.set_xlabel('distance [mm]')
    ax.set_ylabel('angle [rad]')
    ax.set_title('SIREN (intensity-weighted, normalized)')
    plt.colorbar(im, ax=ax, label='density')

    # 2D PhotonSim
    ax = axes[1, 1]
    H_phsim, _, _ = np.histogram2d(
        phsim_dist_mm, phsim_angle, bins=[bins_dist, bins_ang])
    H_phsim_density = H_phsim / max(H_phsim.sum(), 1e-30)
    im = ax.imshow(H_phsim_density.T, origin='lower', aspect='auto',
                    extent=[0, DIST_MAX_MM, 0, ANGLE_MAX_RAD],
                    cmap='Reds')
    ax.set_xlabel('distance [mm]')
    ax.set_ylabel('angle [rad]')
    ax.set_title('PhotonSim Geant4 (normalized)')
    plt.colorbar(im, ax=ax, label='density')

    # 2D ratio (SIREN / PhotonSim)
    ax = axes[1, 2]
    valid = H_phsim_density > 1e-7
    ratio_2d = np.where(valid,
                          H_siren_density / np.maximum(H_phsim_density, 1e-12),
                          np.nan)
    log_ratio = np.log10(np.where(valid, ratio_2d, np.nan))
    im = ax.imshow(log_ratio.T, origin='lower', aspect='auto',
                    extent=[0, DIST_MAX_MM, 0, ANGLE_MAX_RAD],
                    cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('distance [mm]')
    ax.set_ylabel('angle [rad]')
    ax.set_title(r'log₁₀(SIREN / PhotonSim) — red = SIREN excess')
    plt.colorbar(im, ax=ax, label='log₁₀(ratio)')

    fig.suptitle(
        f'Photon emission distribution: SIREN vs PhotonSim @ E={ENERGY:.0f} MeV  •  '
        f'SIREN total = {siren_intensity.sum():.0f}, '
        f'PhotonSim mean N = {phsim_N_total / N_PHSIM_ENTRIES:.0f}',
        fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'\nsaved: {OUT}')

    # Tabular summary of quantile differences
    print()
    print('Quantile comparison:')
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    siren_q_ang = np.percentile(np.repeat(siren_angle,
                                              np.maximum(np.round(siren_intensity * 1000).astype(int), 1)),
                                              [100*q for q in qs])
    phsim_q_ang = np.percentile(phsim_angle, [100*q for q in qs])
    siren_q_d = np.percentile(np.repeat(siren_dist_mm,
                                              np.maximum(np.round(siren_intensity * 1000).astype(int), 1)),
                                              [100*q for q in qs])
    phsim_q_d = np.percentile(phsim_dist_mm, [100*q for q in qs])
    print(f'  angle quantiles (rad)')
    print(f'    {"q":>5} {"SIREN":>10} {"PhotonSim":>10} {"diff":>10}')
    for q, s, p in zip(qs, siren_q_ang, phsim_q_ang):
        print(f'    {q:>5.2f} {s:>10.4f} {p:>10.4f} {s - p:>+10.4f}')
    print(f'  distance quantiles (mm)')
    print(f'    {"q":>5} {"SIREN":>10} {"PhotonSim":>10} {"diff":>10}')
    for q, s, p in zip(qs, siren_q_d, phsim_q_d):
        print(f'    {q:>5.2f} {s:>10.1f} {p:>10.1f} {s - p:>+10.1f}')


if __name__ == '__main__':
    main()
