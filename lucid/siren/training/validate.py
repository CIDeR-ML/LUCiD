#!/usr/bin/env python3
"""The acceptance checks of a trained SIREN emitter.

    python -m lucid.siren.training.validate energy   [--material M] [--particle P] [--kind {cherenkov,dedx}] [--energies E1,E2,..] [--threshold T] [--output DIR]
    python -m lucid.siren.training.validate integral --h5-path T [--material M] [--particle P] [--energies E1,E2,.. | START,END,N] [--nphot N] [--output DIR]
    python -m lucid.siren.training.validate rays     [--material M] [--particle P] [--energies ...] [--nphot N] [--output DIR]
    python -m lucid.siren.training.validate all      [--h5-path T] [--material M] [--particle P] [--kind ...] [--output DIR]

``energy`` draws the emitter's density over (opening angle or dE/dx, distance) at a few energies
(the paper's ``fig_siren_profile``); ``integral`` compares the shape's phase-space integral with
N_photons(E) and the table sum (it needs the photon lookup table the emitter was trained on,
``--h5-path``; no table is distributed); ``rays`` histograms the generated rays. ``integral`` and
``rays`` are Cherenkov only, and ``all`` of a Cherenkov emitter needs ``--h5-path``.
``--material`` names the emitter set (``data/<material>/``).
"""

import os
from pathlib import Path
import argparse
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Import PhotonSim training modules
from lucid.siren.training.dataset import PhotonSimDataset

# Import tools
from lucid.sources.siren_rays import latin_hypercube_2d, make_cherenkov_surrogate_fn
from lucid.siren.training.monitor import setup_matplotlib_for_notebook
from lucid.utils import base_dir_path
from lucid.siren.emitter import density, load_emitter

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

class PhotonSimValidator:
    """Main validation class for PhotonSim SIREN model."""

    def __init__(self, material='water', particle='muon', kind='cherenkov', h5_path=None):
        """Initialize validator with material, particle type, and optional paths.

        Args:
            material: the emitter set (``data/<material>/``), e.g. 'water'
            particle: 'muon' or 'electron'
            kind: 'cherenkov' or 'dedx'; the integral and ray studies are Cherenkov only
            h5_path: the photon lookup table, for the integral study only
        """
        # Store material and particle
        self.material = material
        self.particle = particle
        self.kind = kind
        
        # The lookup table is only needed by integral_analysis (the "integral" subcommand); every
        # other study evaluates the SIREN model alone, so it is loaded lazily via the ``dataset``
        # property instead of being required here.
        self._h5_path = h5_path
        self._dataset = None

        print(f"🎯 PhotonSim SIREN Validation")
        print(f"  Material: {material}")
        print(f"  Particle: {particle}")
        print(f"  Kind: {kind}")
        self.emitter = load_emitter(particle, material, kind)
        self.model_params = self.emitter.params

        # Training ranges; axis2 is the opening angle (rad) or dE/dx (keV/mm)
        (self.energy_min, self.axis2_min, self.distance_min) = self.emitter.lo
        (self.energy_max, self.axis2_max, self.distance_max) = self.emitter.hi
        print(f"Training ranges - Energy: {self.energy_min}-{self.energy_max} MeV, "
              f"{kind} axis: {self.axis2_min:.4g}-{self.axis2_max:.4g}, "
              f"s/s_max: {self.distance_min}-{self.distance_max}")

        # The track-mode ray generator (Cherenkov only).
        self.ray_fn = make_cherenkov_surrogate_fn(self.emitter) if kind == 'cherenkov' else None

        # Standard simulation parameters
        self.origin = jnp.array([0.5, 0.0, -0.5])
        self.direction = jnp.array([1.0, -1.0, 0.2])
        self.key = random.PRNGKey(0)

        print("✅ PhotonSim validator initialized successfully")
    
    @property
    def dataset(self):
        """Lazily-loaded PhotonSim lookup table.

        Only ``integral_analysis`` consumes the table; loading is deferred to
        first access so the model-only studies (energy, rays)
        run without it. Raises a clear error here if the table is missing.
        """
        if self._dataset is None:
            if self._h5_path is None:
                raise ValueError("the integral study needs the photon lookup table the emitter "
                                 "was trained on: pass --h5-path (h5_path=)")
            h5_path = Path(self._h5_path)
            if not h5_path.exists():
                raise FileNotFoundError(f"photon lookup table not found: {h5_path}")
            print(f"Loading dataset from: {h5_path}")
            self._dataset = PhotonSimDataset(h5_path)
        return self._dataset

    def evaluate_photonsim_grid(self, energy, axis2_bins, distance_bins):
        """The emitter's density on the (axis2, s/s_max) grid at ``energy``."""
        axis2_mesh, distance_mesh = jnp.meshgrid(jnp.asarray(axis2_bins), jnp.asarray(distance_bins),
                                                 indexing='ij')
        w = density(self.emitter, self.model_params, energy, axis2_mesh.ravel(), distance_mesh.ravel())
        return np.asarray(w).reshape(len(axis2_bins), len(distance_bins))
    
    def energy_study(self, energies=None, threshold=4, vmax=None, output_dir=None,
                     fmt='png'):
        """
        Perform energy comparison analysis at fixed threshold.

        Args:
            energies: List of energies to analyze (default: [500, 1000, 1500])
            threshold: Fixed threshold value (used as vmin for colorbar)
            vmax: Maximum value for colorbar (optional, auto-determined if None)
            output_dir: Directory to save results
        """
        print(f"\n=== Energy Study Analysis ===")
        print(f"Threshold: {threshold}")

        if energies is None:
            energies = [500, 1000, 1500]

        # Create analysis grid. The axis2 axis is fixed; the distance axis is
        # built per-energy below in physical metres (its span depends on s_max(E)).
        n_angle_bins = 250
        n_distance_bins = 250
        angle_bins = np.linspace(self.axis2_min, self.axis2_max, n_angle_bins)
        if self.kind == 'cherenkov':      # the panel's y axis
            y_label, y_extent = 'Angle (degrees)', np.degrees([self.axis2_max, self.axis2_min])
        else:
            y_label, y_extent = 'dE/dx (keV/mm)', [self.axis2_max, self.axis2_min]

        print(f"Grid: {n_angle_bins}×{n_distance_bins} points")
        print(f"Energies: {energies} MeV")

        # Analyze different energies
        results = {
            'energies': energies,
            'threshold': threshold,
            'statistics': {}
        }

        # First pass: evaluate SIREN over the whole panel for each energy and
        # collect masked values (also used to determine vmax if not provided).
        # The distance axis runs 0 → ceil(s_max(E)) metres so the heatmap fills
        # the rounded axis. Physical distance is converted to s/s_max and clipped
        # to the trained range — beyond s_max(E) (the track end) the SIREN is not
        # valid and the already sub-threshold tail floors out, so no garbage
        # extrapolation leaks into the panel.
        all_masked_values = []
        all_axis_max_m = []
        for energy in energies[:4]:  # Only process first 4 energies
            s_max_m = self.emitter.s_max_m(energy)
            axis_max_m = float(np.ceil(s_max_m))
            phys_distance_m = np.linspace(0.0, axis_max_m, n_distance_bins)
            s_over_smax = np.clip(phys_distance_m / s_max_m,
                                  self.distance_min, self.distance_max)
            reco_value = self.evaluate_photonsim_grid(energy, angle_bins, s_over_smax)
            masked_values = jnp.where(reco_value > threshold, reco_value, threshold)
            all_masked_values.append(masked_values)
            all_axis_max_m.append(axis_max_m)

        # Determine vmax if not provided. Use the 99.99th percentile across all
        # panels (shared colour scale) rather than the absolute max, so a few
        # bright outliers don't compress the dynamic range of the rest.
        if vmax is None:
            pooled = np.concatenate(
                [np.asarray(mv).ravel() for mv in all_masked_values])
            vmax = float(np.percentile(pooled, 99.99))

        # Create visualization - determine layout
        n_energies = len(energies)
        if n_energies <= 4:
            fig, axes = plt.subplots(1, n_energies, figsize=(6, 2.5), constrained_layout=True)
            if n_energies == 1:
                axes = [axes]
        else:
            # Use 2x2 layout for 4 energies
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
            axes = axes.ravel()

        # Create plots
        images = []
        for i, energy in enumerate(energies):
            if i >= 4:  # Only plot first 4 energies
                break

            masked_values = all_masked_values[i]

            # Distance axis is physical metres, 0 → ceil(s_max(E)) (see first pass).
            distance_min_m = 0.0
            distance_max_m = all_axis_max_m[i]

            # Calculate statistics
            valid_count = np.sum(masked_values > threshold)
            total_weight = np.sum(masked_values)

            results['statistics'][energy] = {
                'valid_count': int(valid_count),
                'total_weight': float(total_weight),
                'fraction_valid': float(valid_count / masked_values.size)
            }

            # Plot with common normalization
            im = axes[i].imshow(masked_values, norm=LogNorm(vmin=threshold, vmax=vmax),
                               aspect='auto',
                               extent=[distance_min_m, distance_max_m, *y_extent])
            images.append(im)

            # X-axis label on all plots
            axes[i].set_xlabel('Distance (m)')

            # Round the axis limit up to a clean integer metre (see first pass)
            # and show exactly three ticks: 0, the midpoint, and the max. ``%g``
            # drops trailing zeros (3 not 3.0) while keeping a .5 midpoint.
            axes[i].set_xlim(0.0, distance_max_m)
            xticks = np.linspace(0.0, distance_max_m, 3)
            axes[i].set_xticks(xticks)
            axes[i].set_xticklabels([f'{t:g}' for t in xticks])

            # Y-axis: only show label and ticks on leftmost plots
            if n_energies <= 4:
                # For 1x3 layout, only first plot gets y-axis
                if i == 0:
                    axes[i].set_ylabel(y_label)
                else:
                    axes[i].set_yticklabels([])
            else:
                # For 2x2 layout, left column gets y-axis
                if i % 2 == 0:
                    axes[i].set_ylabel(y_label)
                else:
                    axes[i].set_yticklabels([])

            # Add energy label in bottom right corner
            axes[i].text(0.95, 0.05, f'{int(energy)} MeV', transform=axes[i].transAxes,
                        fontsize=12, ha='right', va='bottom',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            print(f"Energy {energy} MeV: {valid_count:,} valid points ({valid_count/masked_values.size:.3%})")

        # Add common colorbar on rightmost subplot
        if n_energies <= 4:
            # For horizontal layout, add colorbar to rightmost plot
            fig.colorbar(images[-1], ax=axes[-1], label='Intensity (a.u.)')
        else:
            # For 2x2 layout, add colorbar to top-right plot
            fig.colorbar(images[1], ax=axes[1], label='Intensity (a.u.)')

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = f"{output_dir}/energy_study_threshold_{threshold}.{fmt}"
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Energy study results saved to: {out_path}")

        plt.show()

        return results

    def integral_analysis(self, energies=None, nphot=1000000, output_dir=None):
        """Diagnostic: SIREN's phase-space integral vs the stored N_photons(E).

        For each energy, draw one area-uniform LHS set over (angle, s/s_max),
        evaluate SIREN once, and form a Monte-Carlo estimate of the total
        photons/event:  mean(weights) * N_HIST_BINS. Compare it against
          * the emitter's N_photons(E) fit (emitter.n_photons), and
          * the raw table sum (dataset.get_total_counts_for_energy).

        SIREN supplies only the *shape* during inference; this checks that the
        shape also *integrates* consistently with the absolute normalization.
        A flat ratio ~ 1 is good; an energy-dependent ratio flags drift.
        """
        print("\n=== SIREN integral vs N_photons(E) diagnostic ===")

        if energies is None:
            energies = np.linspace(max(self.energy_min, 100.0),
                                   min(self.energy_max, 10000.0), 40)
        energies = np.asarray(energies, dtype=float)
        print(f"Analyzing {len(energies)} energies, "
              f"{energies[0]:.0f}..{energies[-1]:.0f} MeV (nphot={nphot:,})")

        # Training histogram resolution: mean(weight) * N_BINS estimates the
        # total photons/event over the (angle, s/s_max) domain.
        n_hist_bins = 500 * 500

        siren_mc, stored, real = [], [], []
        key = self.key
        for energy in energies:
            key, sub = random.split(key)
            # one area-uniform LHS draw over the whole (angle, s/s_max) domain
            axis2, r = latin_hypercube_2d(sub, int(nphot), self.axis2_min, self.axis2_max,
                                          self.distance_min, self.distance_max)
            w = density(self.emitter, self.model_params, float(energy), axis2, r)
            siren_mc.append(float(jnp.mean(w)) * n_hist_bins)
            stored.append(float(self.emitter.n_photons(float(energy))))
            real.append(float(self.dataset.get_total_counts_for_energy(energy)))

        siren_mc = np.array(siren_mc)
        stored = np.array(stored)
        real = np.array(real)
        ratio = siren_mc / np.where(stored > 0, stored, np.nan)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(energies, real, 'k.', label='table sum (real)', ms=5)
        axes[0].plot(energies, stored, 'b-', label='N_photons(E) stored fit', lw=2)
        axes[0].plot(energies, siren_mc, 'r--', label='SIREN MC integral', lw=2)
        axes[0].set_xlabel('Energy (MeV)')
        axes[0].set_ylabel('Total photons / event')
        axes[0].set_title('SIREN integral vs N_photons(E)')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(energies, ratio, 'r.-', ms=5)
        axes[1].axhline(1.0, color='k', ls=':', lw=1)
        axes[1].set_xlabel('Energy (MeV)')
        axes[1].set_ylabel('SIREN MC integral / stored')
        axes[1].set_title('Ratio (flat ~1 = consistent)')
        axes[1].grid(True, alpha=0.3)
        fig.suptitle('SIREN Integral Diagnostic', fontsize=14)
        fig.tight_layout()

        finite = ratio[np.isfinite(ratio)]
        if finite.size:
            print(f"  ratio (SIREN MC / stored): mean={finite.mean():.4f}  "
                  f"min={finite.min():.4f}  max={finite.max():.4f}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(f"{output_dir}/integral_analysis.png",
                        dpi=150, bbox_inches='tight')
            print(f"  saved {output_dir}/integral_analysis.png")

        plt.show()

        return {
            'energies': energies.tolist(),
            'siren_mc_integral': siren_mc.tolist(),
            'stored_n_photons': stored.tolist(),
            'table_sum_real': real.tolist(),
            'ratio_siren_over_stored': ratio.tolist(),
            'nphot': int(nphot),
        }

    def calculate_opening_angles(self, ray_vectors, direction):
        """Calculate opening angles between ray vectors and reference direction."""
        direction_norm = direction / jnp.linalg.norm(direction)
        ray_vectors_norm = ray_vectors / jnp.linalg.norm(ray_vectors, axis=1)[:, None]
        cos_theta = jnp.dot(ray_vectors_norm, direction_norm)
        angles = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))
        return angles
    
    def rays_validation(self, energies=None, nphot=1000000, output_dir=None):
        """
        Perform ray generation validation.
        
        Args:
            energies: List of energies to analyze
            nphot: Number of photons for ray generation
            output_dir: Directory to save results
        """
        print(f"\n=== Ray Generation Validation ===")
        
        if energies is None:
            energies = np.linspace(200, 2000, 19)
        energies = np.asarray(energies, dtype=float)
        
        print(f"Analyzing {len(energies)} energies from {energies[0]} to {energies[-1]} MeV")
        print(f"N-photons: {nphot:,}")
        
        # Create visualization grid
        n_energies = len(energies)
        cols = 5
        rows = (n_energies + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        results = {
            'energies': energies.tolist(),
            'nphot': nphot,
            'statistics': {}
        }
        
        for i, energy in enumerate(energies):
            row = i // cols
            col = i % cols
            
            print(f"  Processing energy {energy:.0f} MeV ({i+1}/{n_energies})")
            
            # Generate rays (LHS sampling, one SIREN eval; intensities are
            # already normalised so sum(intensities) == N_photons(energy)).
            ray_vectors, ray_origins, photon_intensities = self.ray_fn(
                self.origin, self.direction, energy, int(nphot),
                self.model_params, self.key)

            # Calculate ranges and angles
            ranges = jnp.linalg.norm(ray_origins - self.origin, axis=1)
            angles = self.calculate_opening_angles(ray_vectors, self.direction)

            # Calculate statistics
            total_weight = float(jnp.sum(photon_intensities))

            results['statistics'][energy] = {
                'total_intensity': total_weight,
                'mean_range': float(jnp.mean(ranges)),
                'mean_angle': float(jnp.mean(angles))
            }

            # Create 2D histogram
            axes[row, col].hist2d(
                ranges, angles,
                weights=photon_intensities.squeeze(),
                bins=[500, 500],
                cmap='viridis',
                norm=LogNorm(),
                range=[[0, 10], [0, 3.14]]
            )
            
            axes[row, col].set_ylabel('Angle (radians)')
            axes[row, col].set_xlabel('Distance to Origin (m)')
            axes[row, col].set_title(f'Energy: {energy:.0f} MeV')
        
        # Hide unused subplots
        for i in range(n_energies, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        fig.suptitle('Ray Generation Validation - PhotonSim SIREN', fontsize=16)
        fig.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(f"{output_dir}/rays_validation.png", dpi=150, bbox_inches='tight')
            print(f"Ray validation results saved to: {output_dir}/rays_validation.png")
        
        plt.show()
        
        return results
    
    def run_all_validations(self, output_dir):
        """The energy study, and for a Cherenkov emitter the integral and ray studies."""
        results = {'energy_study': self.energy_study(output_dir=output_dir)}
        if self.kind == 'cherenkov':
            results['integral_analysis'] = self.integral_analysis(output_dir=output_dir)
            results['rays_validation'] = self.rays_validation(output_dir=output_dir)
        print(f"Results saved to: {output_dir}")
        return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='PhotonSim SIREN Validation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Add global material/particle arguments
    parser.add_argument('--material', type=str, default='water',
                        help='Material type (default: water)')
    parser.add_argument('--particle', type=str, default='muon',
                        help='Particle type (default: muon)')
    parser.add_argument('--kind', choices=('cherenkov', 'dedx'), default='cherenkov',
                        help='Emitter kind (default: cherenkov); integral and rays need cherenkov')
    parser.add_argument('--notebook-mode', action='store_true',
                        help='Force notebook mode for plot display')
    
    subparsers = parser.add_subparsers(dest='command', help='Validation command')
    
    # Energy study
    energy_parser = subparsers.add_parser('energy', help='Run energy comparison analysis at fixed threshold')
    energy_parser.add_argument('--material', type=str, default=argparse.SUPPRESS, help='Material type (overrides global setting)')
    energy_parser.add_argument('--particle', type=str, default=argparse.SUPPRESS, help='Particle type (overrides global setting)')
    energy_parser.add_argument('--kind', choices=('cherenkov', 'dedx'), default=argparse.SUPPRESS, help='Emitter kind (overrides global setting)')
    energy_parser.add_argument('--energies', type=str, default='500,1000,1500', help='Comma-separated energies (MeV)')
    energy_parser.add_argument('--threshold', type=float, default=4, help='Fixed threshold value')
    energy_parser.add_argument('--vmax', type=float, help='Maximum value for colorbar (optional)')
    energy_parser.add_argument('--format', type=str, default='png',
                               choices=['png', 'pdf', 'svg'],
                               help='Output figure format (default: png)')
    energy_parser.add_argument('--output', type=str, help='Output directory')
    energy_parser.add_argument('--save', action='store_true', help='Save results to output directory')
    energy_parser.add_argument('--notebook-mode', action='store_true', default=argparse.SUPPRESS, help='Force notebook mode for plot display')

    # Integral analysis
    integral_parser = subparsers.add_parser('integral', help='Run n-photon integral analysis')
    integral_parser.add_argument('--material', type=str, default=argparse.SUPPRESS, help='Material type (overrides global setting)')
    integral_parser.add_argument('--particle', type=str, default=argparse.SUPPRESS, help='Particle type (overrides global setting)')
    integral_parser.add_argument('--h5-path', type=Path, required=True,
                                 help='the photon lookup table the emitter was trained on')
    integral_parser.add_argument('--energies', type=str, help='Comma-separated energies or range (e.g., 100,2000,100)')
    integral_parser.add_argument('--nphot', type=int, default=1000000, help='Number of photons')
    integral_parser.add_argument('--output', type=str, help='Output directory')
    integral_parser.add_argument('--save', action='store_true', help='Save results to output directory')
    integral_parser.add_argument('--notebook-mode', action='store_true', default=argparse.SUPPRESS, help='Force notebook mode for plot display')
    
    # Ray validation
    rays_parser = subparsers.add_parser('rays', help='Run ray generation validation')
    rays_parser.add_argument('--material', type=str, default=argparse.SUPPRESS, help='Material type (overrides global setting)')
    rays_parser.add_argument('--particle', type=str, default=argparse.SUPPRESS, help='Particle type (overrides global setting)')
    rays_parser.add_argument('--energies', type=str, help='Comma-separated energies or range (e.g., 200,2000,20)')
    rays_parser.add_argument('--nphot', type=int, default=1000000, help='Number of photons')
    rays_parser.add_argument('--output', type=str, help='Output directory')
    rays_parser.add_argument('--save', action='store_true', help='Save results to output directory')
    rays_parser.add_argument('--notebook-mode', action='store_true', default=argparse.SUPPRESS, help='Force notebook mode for plot display')
    
    # All validations
    all_parser = subparsers.add_parser('all', help='Run all validations')
    all_parser.add_argument('--material', type=str, default=argparse.SUPPRESS, help='Material type (overrides global setting)')
    all_parser.add_argument('--particle', type=str, default=argparse.SUPPRESS, help='Particle type (overrides global setting)')
    all_parser.add_argument('--kind', choices=('cherenkov', 'dedx'), default=argparse.SUPPRESS, help='Emitter kind (overrides global setting)')
    all_parser.add_argument('--h5-path', type=Path, default=None,
                            help='the photon lookup table (needed for a Cherenkov emitter)')
    all_parser.add_argument('--output', type=str, help='Output directory')
    all_parser.add_argument('--save', action='store_true', help='Save results to output directory')
    all_parser.add_argument('--notebook-mode', action='store_true', default=argparse.SUPPRESS, help='Force notebook mode for plot display')
    
    args = parser.parse_args()
    
    if args.notebook_mode:
        setup_matplotlib_for_notebook()
    
    if args.command is None:
        parser.print_help()
        return
    
    # A subcommand's --material/--particle/--kind/--notebook-mode override the global ones only when given
    # (default=SUPPRESS); otherwise the global values and defaults stand.
    material = args.material
    particle = args.particle
    
    # Initialize validator with material and particle
    print(f"🎯 Validation Configuration:")
    print(f"  Material: {material}")
    print(f"  Particle: {particle}")
    
    h5_path = getattr(args, 'h5_path', None)
    if args.command == 'all' and args.kind == 'cherenkov' and h5_path is None:
        parser.error("all runs the integral study of a Cherenkov emitter, which needs --h5-path "
                     "(the photon lookup table it was trained on)")
    validator = PhotonSimValidator(material=material, particle=particle, kind=args.kind,
                                   h5_path=h5_path)
    
    # Parse energy ranges
    def parse_energies(energy_str):
        if ',' in energy_str:
            parts = energy_str.split(',')
            if len(parts) == 3:  # start,end,num format
                start, end, num = map(float, parts)
                return np.linspace(start, end, int(num))
            else:  # explicit list
                return [float(x) for x in parts]
        else:
            return [float(energy_str)]
    
    # Determine output directory
    def get_output_dir(args):
        if args.output:
            return args.output
        elif args.save:
            # Use material/particle structure
            return Path(base_dir_path()) / 'output' / 'siren' / validator.material / validator.particle
        else:
            return None
    
    # Execute command
    if args.command == 'energy':
        energies = [float(x) for x in args.energies.split(',')]
        output_dir = get_output_dir(args)
        validator.energy_study(energies=energies, threshold=args.threshold, vmax=args.vmax,
                               output_dir=output_dir, fmt=args.format)

    elif args.command == 'integral':
        energies = None
        if args.energies:
            energies = parse_energies(args.energies)
        output_dir = get_output_dir(args)
        validator.integral_analysis(energies=energies, nphot=args.nphot, output_dir=output_dir)
    
    elif args.command == 'rays':
        energies = None
        if args.energies:
            energies = parse_energies(args.energies)
        output_dir = get_output_dir(args)
        validator.rays_validation(energies=energies, nphot=args.nphot, output_dir=output_dir)
    
    elif args.command == 'all':
        validator.run_all_validations(output_dir=get_output_dir(args) or
                                      Path(base_dir_path()) / 'output' / 'siren' / material / particle)


if __name__ == '__main__':
    main()