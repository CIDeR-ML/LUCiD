#!/usr/bin/env python3
"""
PhotonSim SIREN Validation Suite

This module consolidates validation functionality from three notebooks:
1. photonsim_cut_off_study.ipynb - Cut-off threshold analysis
2. photonsim_n_photon_integral.ipynb - N-photon integral analysis  
3. photonsim_rays_validation.ipynb - Ray generation validation

Supports different particle types and materials with automatic path detection.

Usage:
    python validate.py cutoff [--material MATERIAL] [--particle PARTICLE] [--energy ENERGY] [--thresholds THRESHOLDS] [--output OUTPUT]
    python validate.py energy [--material MATERIAL] [--particle PARTICLE] [--energies ENERGIES] [--threshold THRESHOLD] [--output OUTPUT]
    python validate.py valid-points [--material MATERIAL] [--particle PARTICLE] [--energies ENERGIES] [--thresholds THRESHOLDS] [--output OUTPUT]
    python validate.py integral [--material MATERIAL] [--particle PARTICLE] [--energies ENERGIES] [--nphot NPHOT] [--output OUTPUT]
    python validate.py rays [--material MATERIAL] [--particle PARTICLE] [--energies ENERGIES] [--nphot NPHOT] [--output OUTPUT]
    python validate.py all [--material MATERIAL] [--particle PARTICLE] [--output OUTPUT]

Examples:
    # Default water/muon
    python validate.py cutoff --energy 500 --thresholds 1,2,4,8

    # Energy comparison at fixed threshold
    python validate.py energy --energies 500,1000,1500 --threshold 4

    # Valid points vs energy analysis
    python validate.py valid-points --energies 200,2000,20 --thresholds 1,2,4,8

    # Specific material/particle
    python validate.py integral --material ice --particle electron --energies 200,500,800

    # All validations for water/muon
    python validate.py all

    # All validations for custom material/particle
    python validate.py all --material water --particle muon --output validation_results/
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.optimize import curve_fit

# Add parent directories to path
script_dir = Path(__file__).parent
tools_dir = script_dir.parent
project_root = tools_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tools_dir))

# Import PhotonSim training modules
from tools.siren.training.inference import SIRENPredictor
from tools.siren.training.dataset import PhotonSimDataset

# Import tools
from tools.siren.core import SIREN
from tools.siren.core import create_photonsim_siren_grid
from tools.generate import generate_random_cone_vectors, normalize, photonsim_differentiable_get_rays
from tools.utils import base_dir_path, setup_matplotlib_for_notebook

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

class PhotonSimValidator:
    """Main validation class for PhotonSim SIREN model."""
    
    def __init__(self, material='water', particle='muon', model_path=None, h5_path=None):
        """Initialize validator with material, particle type, and optional paths.
        
        Args:
            material: Material type (e.g., 'water', 'ice')
            particle: Particle type (e.g., 'muon', 'electron')
            model_path: Optional explicit model path
            h5_path: Optional explicit h5 file path
        """
        # Store material and particle
        self.material = material
        self.particle = particle
        
        # Get base directory path
        base_dir = base_dir_path()
        
        # Default model path in data directory structure
        if model_path is None:
            model_path = Path(base_dir) / 'data' / material / particle / 'siren_training' / 'trained_model' / 'photonsim_siren'
        
        # Default h5 path with material/particle structure
        if h5_path is None:
            h5_path = Path(base_dir) / 'data' / material / particle / 'photon_lookup_table.h5'
            if not h5_path.exists():
                raise FileNotFoundError(
                    f"HDF5 lookup table not found at {h5_path}\n"
                    f"Please ensure the PhotonSim table exists for {material}/{particle}"
                )
        
        print(f"🎯 PhotonSim SIREN Validation")
        print(f"  Material: {material}")
        print(f"  Particle: {particle}")
        print(f"Loading model from: {model_path}")
        self.photonsim_predictor = SIRENPredictor(model_path)
        self.model_params = self.photonsim_predictor.params
        
        print(f"Loading dataset from: {h5_path}")
        self.dataset = PhotonSimDataset(h5_path)
        
        # Get training ranges
        self.dataset_info = self.photonsim_predictor.dataset_info
        self.energy_min, self.energy_max = self.dataset_info['energy_range']
        self.angle_min, self.angle_max = self.dataset_info['angle_range']
        self.distance_min, self.distance_max = self.dataset_info['distance_range']
        
        print(f"Training ranges - Energy: {self.energy_min}-{self.energy_max} MeV, "
              f"Angle: {np.degrees(self.angle_min):.1f}°-{np.degrees(self.angle_max):.1f}°, "
              f"Distance: {self.distance_min}-{self.distance_max} mm")
        
        # Create table data for ray generation
        self.table_data = create_photonsim_siren_grid(self.photonsim_predictor, 250)

        # Load num_seeds parameters from configuration
        from tools.utils import unpack_photonsim_params
        photonsim_params = unpack_photonsim_params(particle, material)
        self.num_seeds_a, self.num_seeds_b, self.num_seeds_c = photonsim_params['num_seeds']

        print(f"Loaded num_seeds parameters: a={self.num_seeds_a:.6f}, b={self.num_seeds_b:.6f}, c={self.num_seeds_c:.2f}")

        # Standard simulation parameters
        self.origin = jnp.array([0.5, 0.0, -0.5])
        self.direction = jnp.array([1.0, -1.0, 0.2])
        self.key = random.PRNGKey(0)

        print("✅ PhotonSim validator initialized successfully")
    
    def evaluate_photonsim_grid(self, energy, angle_bins, distance_bins):
        """Evaluate PhotonSim model on angle/distance grid for given energy."""
        angle_mesh, distance_mesh = jnp.meshgrid(angle_bins, distance_bins, indexing='ij')
        
        # Create evaluation grid for PhotonSim: [energy, angle, distance]
        evaluation_grid = jnp.stack([
            jnp.full_like(angle_mesh, energy).ravel(),
            angle_mesh.ravel(),
            distance_mesh.ravel(),
        ], axis=1)
        
        # Get predictions
        photon_weights = self.photonsim_predictor.predict_batch(evaluation_grid)
        return np.array(photon_weights).reshape(len(angle_bins), len(distance_bins))
    
    def energy_study(self, energies=None, threshold=4, vmax=None, output_dir=None):
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

        # Create analysis grid
        n_angle_bins = 250
        n_distance_bins = 250
        angle_bins = np.linspace(self.angle_min, self.angle_max, n_angle_bins)
        distance_bins = np.linspace(self.distance_min, self.distance_max, n_distance_bins)

        print(f"Grid: {n_angle_bins}×{n_distance_bins} points")
        print(f"Energies: {energies} MeV")

        # Analyze different energies
        results = {
            'energies': energies,
            'threshold': threshold,
            'statistics': {}
        }

        # First pass: collect all masked values to determine vmax if not provided
        all_masked_values = []
        for energy in energies[:4]:  # Only process first 4 energies
            reco_value = self.evaluate_photonsim_grid(energy, angle_bins, distance_bins)
            masked_values = jnp.where(reco_value > threshold, reco_value, 0)
            all_masked_values.append(masked_values)

        # Determine vmax if not provided
        if vmax is None:
            vmax = max(np.max(mv) for mv in all_masked_values)

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

        # Convert distance range to meters
        distance_min_m = 0#self.distance_min / 1000.0
        distance_max_m = 10#self.distance_max / 1000.0

        # Create plots
        images = []
        for i, energy in enumerate(energies):
            if i >= 4:  # Only plot first 4 energies
                break

            masked_values = all_masked_values[i]

            # Calculate statistics
            valid_count = np.sum(masked_values > 0)
            total_weight = np.sum(masked_values)

            results['statistics'][energy] = {
                'valid_count': int(valid_count),
                'total_weight': float(total_weight),
                'fraction_valid': float(valid_count / masked_values.size)
            }

            # Plot with common normalization
            im = axes[i].imshow(masked_values, norm=LogNorm(vmin=threshold, vmax=vmax),
                               aspect='auto',
                               extent=[distance_min_m, distance_max_m,
                                      np.degrees(self.angle_max), np.degrees(self.angle_min)])
            images.append(im)

            # X-axis label on all plots
            axes[i].set_xlabel('Distance (m)')

            # Y-axis: only show label and ticks on leftmost plots
            if n_energies <= 4:
                # For 1x3 layout, only first plot gets y-axis
                if i == 0:
                    axes[i].set_ylabel('Angle (degrees)')
                else:
                    axes[i].set_yticklabels([])
            else:
                # For 2x2 layout, left column gets y-axis
                if i % 2 == 0:
                    axes[i].set_ylabel('Angle (degrees)')
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
            fig.savefig(f"{output_dir}/energy_study_threshold_{threshold}.png", dpi=150, bbox_inches='tight')
            print(f"Energy study results saved to: {output_dir}/energy_study_threshold_{threshold}.png")

        plt.show()

        return results

    def cutoff_study(self, energy=600, thresholds=None, output_dir=None):
        """
        Perform cut-off threshold analysis.

        Args:
            energy: Analysis energy in MeV
            thresholds: List of cut-off thresholds to analyze
            output_dir: Directory to save results
        """
        print(f"\n=== Cut-off Study Analysis ===")
        print(f"Energy: {energy} MeV")

        if thresholds is None:
            thresholds = [1, 2, 4, 8]

        # Create analysis grid
        n_angle_bins = 250
        n_distance_bins = 250
        angle_bins = np.linspace(self.angle_min, self.angle_max, n_angle_bins)
        distance_bins = np.linspace(self.distance_min, self.distance_max, n_distance_bins)

        print(f"Grid: {n_angle_bins}×{n_distance_bins} points")
        print(f"Thresholds: {thresholds}")

        # Evaluate model at given energy
        reco_value = self.evaluate_photonsim_grid(energy, angle_bins, distance_bins)

        # Analyze different thresholds
        results = {
            'energy': energy,
            'thresholds': thresholds,
            'statistics': {}
        }

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        axes = axes.ravel()

        for i, threshold in enumerate(thresholds):
            if i >= 4:  # Only plot first 4 thresholds
                break

            # Apply threshold
            masked_values = jnp.where(reco_value > threshold, reco_value, 0)

            # Calculate statistics
            valid_count = np.sum(masked_values > 0)
            total_weight = np.sum(masked_values)

            results['statistics'][threshold] = {
                'valid_count': int(valid_count),
                'total_weight': float(total_weight),
                'fraction_valid': float(valid_count / masked_values.size)
            }

            # Plot
            im = axes[i].imshow(masked_values, norm=LogNorm(vmin=threshold), aspect='auto',
                               extent=[self.distance_min, self.distance_max,
                                      np.degrees(self.angle_max), np.degrees(self.angle_min)])
            axes[i].set_xlabel('Distance (mm)')
            axes[i].set_ylabel('Angle (degrees)')
            axes[i].set_title(f'Threshold: {threshold}\nValid points: {valid_count:,}', fontsize=13)
            fig.colorbar(im, ax=axes[i], label='Photon Density')

            print(f"Threshold {threshold}: {valid_count:,} valid points ({valid_count/masked_values.size:.3%})")

        fig.suptitle(f'Cut-off Study - Energy: {energy} MeV', fontsize=16)
        fig.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(f"{output_dir}/cutoff_study_energy_{energy}.png", dpi=150, bbox_inches='tight')
            print(f"Cutoff study results saved to: {output_dir}/cutoff_study_energy_{energy}.png")

        plt.show()

        return results
    
    def integral_analysis(self, energies=None, nphot=1000000, output_dir=None):
        """
        Perform n-photon integral analysis.
        
        Args:
            energies: List of energies to analyze
            nphot: Number of photons for analysis
            output_dir: Directory to save results
        """
        print(f"\n=== N-Photon Integral Analysis ===")
        
        if energies is None:
            energies = np.linspace(100, 1900, 50)
        
        print(f"Analyzing {len(energies)} energies from {energies[0]} to {energies[-1]} MeV")
        print(f"N-photons: {nphot:,}")
        
        tot_real_photons = []
        tot_pred_photons = []
        
        for energy in energies:
            # Get real photon count from dataset
            real_count = self.dataset.get_total_counts_for_energy(energy)
            tot_real_photons.append(real_count)
            
            # Get predicted photon count using configured num_seeds parameters
            _, _, photon_weights = photonsim_differentiable_get_rays(
                self.origin, self.direction, energy, nphot, self.table_data, self.model_params, self.key,
                self.num_seeds_a, self.num_seeds_b, self.num_seeds_c
            )
            pred_count = np.sum(photon_weights)
            tot_pred_photons.append(pred_count)
            
            if len(tot_real_photons) % 10 == 0:
                print(f"  Processed {len(tot_real_photons)}/{len(energies)} energies")
        
        # Calculate ratio and fit
        y_data = np.array(tot_real_photons) / (np.array(tot_pred_photons) / nphot)

        # Filter data for energies above Cherenkov threshold
        fit_mask = np.array(energies) >= 250 # assume it is a muon
        energies_fit = np.array(energies)[fit_mask]
        y_data_fit = y_data[fit_mask]

        print(f"Using {len(energies_fit)}/{len(energies)} data points for power law fit (energies ≥ 250 MeV)")

        # Define power law function: y = a * x^b + c
        def power_law(x, a, b, c):
            return a * np.power(x, b) + c

        # Power law fit on filtered data
        try:
            popt, pcov = curve_fit(power_law, energies_fit, y_data_fit, p0=[1000.0, 0.3, -2000.])
            a_fit, b_fit, c_fit = popt

            # Calculate R-squared
            residuals = y_data_fit - power_law(energies_fit, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data_fit - np.mean(y_data_fit))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Calculate standard errors
            perr = np.sqrt(np.diag(pcov))
            a_err, b_err, c_err = perr

            # Generate fit line
            line = power_law(np.array(energies), *popt)

            print(f"\nPower Law Fit Results:")
            print(f"  Equation: y = {a_fit:.6f} * x^{b_fit:.6f} + {c_fit:.6f}")
            print(f"  Parameters: a = {a_fit:.6f} ± {a_err:.6f}")
            print(f"              b = {b_fit:.6f} ± {b_err:.6f}")
            print(f"              c = {c_fit:.6f} ± {c_err:.6f}")
            print(f"  R-squared: {r_squared:.6f}")

        except Exception as e:
            print(f"Warning: Power law fit failed: {e}")
            print(f"Falling back to linear fit...")
            slope, intercept, r_value, p_value, std_err = stats.linregress(energies_fit, y_data_fit)
            line = slope * energies + intercept
            a_fit, b_fit, c_fit = slope, 1.0, intercept
            a_err, b_err, c_err = std_err, 0.0, 0.0
            r_squared = r_value**2
            print(f"  Linear fallback: y = {slope:.6f}x + {intercept:.6f}")
            print(f"  R-squared: {r_squared:.4f}")

        # Apply correction function
        def corr_function(x):
            return power_law(x, a_fit, b_fit, c_fit)

        y_corrected = np.array(tot_real_photons) / (corr_function(energies) * np.array(tot_pred_photons) / nphot)

        # Fit corrected data (also filtered for energies >= 200 MeV)
        y_corrected_fit = y_corrected[fit_mask]
        try:
            popt_corr, pcov_corr = curve_fit(power_law, energies_fit, y_corrected_fit, p0=[1.0, 1.0, 0.0])
            a_corr, b_corr, c_corr = popt_corr

            # Calculate R-squared for corrected fit
            residuals_corr = y_corrected_fit - power_law(energies_fit, *popt_corr)
            ss_res_corr = np.sum(residuals_corr**2)
            ss_tot_corr = np.sum((y_corrected_fit - np.mean(y_corrected_fit))**2)
            r_squared_corr = 1 - (ss_res_corr / ss_tot_corr)

            # Calculate standard errors for corrected fit
            perr_corr = np.sqrt(np.diag(pcov_corr))
            a_err_corr, b_err_corr, c_err_corr = perr_corr

            # Generate fit line
            line_corr = power_law(np.array(energies), *popt_corr)

        except Exception as e:
            print(f"Warning: Power law fit for corrected data failed: {e}")
            slope_corr, intercept_corr, r_value_corr, _, std_err_corr = stats.linregress(energies_fit, y_corrected_fit)
            line_corr = slope_corr * energies + intercept_corr
            a_corr, b_corr, c_corr = slope_corr, 1.0, intercept_corr
            a_err_corr, b_err_corr, c_err_corr = std_err_corr, 0.0, 0.0
            r_squared_corr = r_value_corr**2
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Original data
        axes[0].plot(energies, y_data, 'bo', label='All Data', markersize=3, alpha=0.6)
        axes[0].plot(energies_fit, y_data_fit, 'ro', label='Fit Data (≥200 MeV)', markersize=3)
        axes[0].plot(energies, line, 'r-', label=f'Power law fit (R²={r_squared:.4f})', linewidth=2)
        axes[0].set_xlabel('Energy (MeV)')
        axes[0].set_ylabel('Ratio')
        axes[0].set_title('Original Data vs Power Law Fit')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Add fit equation to plot
        fit_text = f'y = {a_fit:.4f}·x$^{{{b_fit:.4f}}}$ + {c_fit:.2f}'
        axes[0].text(0.05, 0.95, fit_text, transform=axes[0].transAxes,
                    fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Corrected data
        axes[1].plot(energies, y_corrected, 'go', label='All Corrected Data', markersize=3, alpha=0.6)
        axes[1].plot(energies_fit, y_corrected_fit, 'mo', label='Fit Data (≥200 MeV)', markersize=3)
        axes[1].plot(energies, line_corr, 'r-', label=f'Power law fit (R²={r_squared_corr:.4f})', linewidth=2)
        axes[1].set_xlabel('Energy (MeV)')
        axes[1].set_ylabel('Corrected Ratio')
        axes[1].set_title('Corrected Data vs Power Law Fit')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(500,1500)
        axes[1].set_ylim(0.95,1.05)

        # Add corrected fit equation to plot
        fit_text_corr = f'y = {a_corr:.4f}·x$^{{{b_corr:.4f}}}$ + {c_corr:.2f}'
        axes[1].text(0.05, 0.95, fit_text_corr, transform=axes[1].transAxes,
                    fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('N-Photon Integral Analysis (Power Law Fit)', fontsize=16)
        fig.tight_layout()
        
        # Save results
        results = {
            'energies': energies.tolist(),
            'real_photons': tot_real_photons,
            'pred_photons': tot_pred_photons,
            'nphot': nphot,
            'fit_type': 'power_law',
            'fit_equation': 'y = a * x^b + c',
            'fit_filter': {
                'min_energy': 200,
                'energies_used': energies_fit.tolist(),
                'data_points_used': len(energies_fit),
                'total_data_points': len(energies)
            },
            'original_fit': {
                'a': a_fit,
                'b': b_fit,
                'c': c_fit,
                'a_err': a_err,
                'b_err': b_err,
                'c_err': c_err,
                'r_squared': r_squared,
                'equation': f'y = {a_fit:.6f} * x^{b_fit:.6f} + {c_fit:.6f}'
            },
            'corrected_fit': {
                'a': a_corr,
                'b': b_corr,
                'c': c_corr,
                'a_err': a_err_corr,
                'b_err': b_err_corr,
                'c_err': c_err_corr,
                'r_squared': r_squared_corr,
                'equation': f'y = {a_corr:.6f} * x^{b_corr:.6f} + {c_corr:.6f}'
            }
        }
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY OF POWER LAW FITS (y = a * x^b + c)")
        print("="*70)
        print("\nOriginal Fit:")
        print(f"  Equation: y = {a_fit:.6f} * x^{b_fit:.6f} + {c_fit:.6f}")
        print(f"  R-squared: {r_squared:.6f}")
        print(f"  Parameters:")
        print(f"    a = {a_fit:.6f} ± {a_err:.6f}")
        print(f"    b = {b_fit:.6f} ± {b_err:.6f}")
        print(f"    c = {c_fit:.6f} ± {c_err:.6f}")

        print("\nCorrected Fit:")
        print(f"  Equation: y = {a_corr:.6f} * x^{b_corr:.6f} + {c_corr:.6f}")
        print(f"  R-squared: {r_squared_corr:.6f}")
        print(f"  Parameters:")
        print(f"    a = {a_corr:.6f} ± {a_err_corr:.6f}")
        print(f"    b = {b_corr:.6f} ± {b_err_corr:.6f}")
        print(f"    c = {c_corr:.6f} ± {c_err_corr:.6f}")
        print("="*70)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(f"{output_dir}/integral_analysis.png", dpi=150, bbox_inches='tight')
            print(f"\nIntegral analysis results saved to: {output_dir}/integral_analysis.png")

        plt.show()

        return results
    
    def calculate_opening_angles(self, ray_vectors, direction):
        """Calculate opening angles between ray vectors and reference direction."""
        direction_norm = direction / jnp.linalg.norm(direction)
        ray_vectors_norm = ray_vectors / jnp.linalg.norm(ray_vectors, axis=1)[:, None]
        cos_theta = jnp.dot(ray_vectors_norm, direction_norm)
        angles = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))
        return angles
    
    def valid_points_vs_energy(self, energies=None, thresholds=None, output_dir=None):
        """
        Analyze valid points vs energy for multiple thresholds with linear fits.
        
        Args:
            energies: List of energies to analyze
            thresholds: List of thresholds to test
            output_dir: Directory to save results
        """
        print(f"\n=== Valid Points vs Energy Analysis ===")
        
        if energies is None:
            energies = np.linspace(200, 2000, 200)
        
        if thresholds is None:
            thresholds = [1, 2, 4, 8]
        
        print(f"Analyzing {len(energies)} energies from {energies[0]} to {energies[-1]} MeV")
        print(f"Thresholds: {thresholds}")
        
        # Create analysis grid
        n_angle_bins = 250
        n_distance_bins = 250
        angle_bins = np.linspace(self.angle_min, self.angle_max, n_angle_bins)
        distance_bins = np.linspace(self.distance_min, self.distance_max, n_distance_bins)
        
        # Store results for each threshold
        threshold_results = {}
        
        # Define power law function: y = a * x^b + c
        def power_law(x, a, b, c):
            return a * np.power(x, b) + c

        for threshold in thresholds:
            valid_counts = []

            print(f"\nProcessing threshold {threshold}:")
            for i, energy in enumerate(energies):
                # Evaluate model at given energy
                reco_value = self.evaluate_photonsim_grid(energy, angle_bins, distance_bins)

                # Apply threshold and count valid points
                masked_values = jnp.where(reco_value > threshold, reco_value, 0)
                valid_count = np.sum(masked_values > 0)
                valid_counts.append(valid_count)

                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(energies)} energies")

            # Perform power law fit: y = a * x^b + c
            # Initial guess: a=1, b=1 (linear), c=0
            try:
                popt, pcov = curve_fit(power_law, energies, valid_counts, p0=[1.0, 1.0, 0.0])
                a_fit, b_fit, c_fit = popt

                # Calculate R-squared
                residuals = np.array(valid_counts) - power_law(energies, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((np.array(valid_counts) - np.mean(valid_counts))**2)
                r_squared = 1 - (ss_res / ss_tot)

                # Calculate standard errors from covariance matrix
                perr = np.sqrt(np.diag(pcov))
                a_err, b_err, c_err = perr

                # Generate fit line
                fit_line = power_law(np.array(energies), *popt)

                threshold_results[threshold] = {
                    'valid_counts': valid_counts,
                    'a': a_fit,
                    'b': b_fit,
                    'c': c_fit,
                    'a_err': a_err,
                    'b_err': b_err,
                    'c_err': c_err,
                    'r_squared': r_squared,
                    'fit_line': fit_line
                }

                print(f"  Power Law Fit Results for threshold {threshold}:")
                print(f"    Equation: y = {a_fit:.4f} * x^{b_fit:.4f} + {c_fit:.2f}")
                print(f"    Parameters: a = {a_fit:.4f} ± {a_err:.4f}")
                print(f"                b = {b_fit:.4f} ± {b_err:.4f}")
                print(f"                c = {c_fit:.2f} ± {c_err:.2f}")
                print(f"    R-squared: {r_squared:.6f}")

            except Exception as e:
                print(f"  Warning: Power law fit failed for threshold {threshold}: {e}")
                print(f"  Falling back to linear fit...")
                # Fallback to linear fit
                slope, intercept, r_value, p_value, std_err = stats.linregress(energies, valid_counts)
                fit_line = slope * np.array(energies) + intercept

                threshold_results[threshold] = {
                    'valid_counts': valid_counts,
                    'a': slope,
                    'b': 1.0,
                    'c': intercept,
                    'a_err': std_err,
                    'b_err': 0.0,
                    'c_err': 0.0,
                    'r_squared': r_value**2,
                    'fit_line': fit_line,
                    'fallback_linear': True
                }
                print(f"    Linear fallback: y = {slope:.2f}x + {intercept:.2f}")
                print(f"    R-squared: {r_value**2:.4f}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        colors = ['blue', 'green', 'red', 'purple']

        for idx, threshold in enumerate(thresholds[:4]):  # Plot up to 4 thresholds
            results = threshold_results[threshold]

            # Plot data and fit
            axes[idx].scatter(energies, results['valid_counts'],
                            color=colors[idx], alpha=0.6, s=30, label='Data')
            axes[idx].plot(energies, results['fit_line'],
                         color=colors[idx], linestyle='--', linewidth=2,
                         label=f'Fit: y = {results["a"]:.4f}x^{results["b"]:.4f} + {results["c"]:.2f}')

            axes[idx].set_xlabel('Energy (MeV)', fontsize=11)
            axes[idx].set_ylabel('Valid Points', fontsize=11)
            axes[idx].set_title(f'Threshold = {threshold}\nR² = {results["r_squared"]:.6f}', fontsize=12)
            axes[idx].legend(loc='upper left', fontsize=9)
            axes[idx].grid(True, alpha=0.3)

            # Add text with fit parameters
            text_str = (f'a = {results["a"]:.4f} ± {results["a_err"]:.4f}\n'
                       f'b = {results["b"]:.4f} ± {results["b_err"]:.4f}\n'
                       f'c = {results["c"]:.2f} ± {results["c_err"]:.2f}')
            axes[idx].text(0.95, 0.05, text_str, transform=axes[idx].transAxes,
                         fontsize=9, ha='right', va='bottom',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Valid Points vs Energy for Different Thresholds', fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        # Print summary of all fits
        print("\n" + "="*70)
        print("SUMMARY OF POWER LAW FITS (y = a * x^b + c)")
        print("="*70)
        for threshold in thresholds:
            results = threshold_results[threshold]
            print(f"\nThreshold {threshold}:")
            if 'fallback_linear' in results and results['fallback_linear']:
                print(f"  ⚠️  Linear fallback used (power law fit failed)")
                print(f"  Equation: y = {results['a']:.2f}x + {results['c']:.2f}")
            else:
                print(f"  Equation: y = {results['a']:.6f} * x^{results['b']:.6f} + {results['c']:.2f}")
            print(f"  R-squared: {results['r_squared']:.6f}")
            print(f"  Parameters:")
            print(f"    a = {results['a']:.6f} ± {results['a_err']:.6f}")
            print(f"    b = {results['b']:.6f} ± {results['b_err']:.6f}")
            print(f"    c = {results['c']:.2f} ± {results['c_err']:.2f}")
            print(f"  Valid points range: {min(results['valid_counts']):,} - {max(results['valid_counts']):,}")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(f"{output_dir}/valid_points_vs_energy.png", dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_dir}/valid_points_vs_energy.png")
        
        plt.show()
        
        return {
            'energies': energies.tolist(),
            'thresholds': thresholds,
            'fit_type': 'power_law',
            'fit_equation': 'y = a * x^b + c',
            'results': {
                threshold: {
                    'valid_counts': threshold_results[threshold]['valid_counts'],
                    'a': threshold_results[threshold]['a'],
                    'b': threshold_results[threshold]['b'],
                    'c': threshold_results[threshold]['c'],
                    'a_err': threshold_results[threshold]['a_err'],
                    'b_err': threshold_results[threshold]['b_err'],
                    'c_err': threshold_results[threshold]['c_err'],
                    'r_squared': threshold_results[threshold]['r_squared'],
                    'fallback_linear': threshold_results[threshold].get('fallback_linear', False)
                }
                for threshold in thresholds
            }
        }
    
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
            
            # Generate rays using configured num_seeds parameters
            ray_vectors, ray_origins, photon_weights = photonsim_differentiable_get_rays(
                self.origin, self.direction, energy, nphot, self.table_data, self.model_params, self.key,
                self.num_seeds_a, self.num_seeds_b, self.num_seeds_c
            )
            
            # Calculate ranges and angles
            ranges = jnp.linalg.norm(ray_origins - self.origin, axis=1)
            angles = self.calculate_opening_angles(ray_vectors, self.direction)
            
            # Calculate statistics
            total_weight = float(jnp.sum(photon_weights))
            
            results['statistics'][energy] = {
                'total_weight': total_weight,
                'mean_range': float(jnp.mean(ranges)),
                'mean_angle': float(jnp.mean(angles))
            }
            
            # Create 2D histogram
            h = axes[row, col].hist2d(
                ranges, angles,
                weights=photon_weights.squeeze(),
                bins=[500, 500],
                cmap='viridis',
                norm=LogNorm(vmin=3.5), # our num_seeds fit should be cutting at ~4
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
    
    def run_all_validations(self, output_dir=None):
        """Run all validation studies."""
        print("\n🚀 Running all PhotonSim SIREN validations...")
        
        # Set default output directory with material/particle structure if not provided
        if output_dir is None:
            output_dir = Path(base_dir_path()) / 'data' / self.material / self.particle / 'siren_training' / 'validation'
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to: {output_dir}")
        
        # Run cut-off study
        cutoff_results = self.cutoff_study(output_dir=output_dir)
        
        # Run valid points vs energy analysis
        valid_points_results = self.valid_points_vs_energy(output_dir=output_dir)
        
        # Run integral analysis
        integral_results = self.integral_analysis(output_dir=output_dir)
        
        # Run ray validation
        rays_results = self.rays_validation(output_dir=output_dir)
        
        print(f"\n✅ All validations completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        return {
            'cutoff_study': cutoff_results,
            'valid_points_vs_energy': valid_points_results,
            'integral_analysis': integral_results,
            'rays_validation': rays_results
        }


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
    parser.add_argument('--notebook-mode', action='store_true',
                        help='Force notebook mode for plot display')
    
    subparsers = parser.add_subparsers(dest='command', help='Validation command')
    
    # Cut-off study
    cutoff_parser = subparsers.add_parser('cutoff', help='Run cut-off threshold analysis')
    cutoff_parser.add_argument('--material', type=str, help='Material type (overrides global setting)')
    cutoff_parser.add_argument('--particle', type=str, help='Particle type (overrides global setting)')
    cutoff_parser.add_argument('--energy', type=float, default=500, help='Analysis energy (MeV)')
    cutoff_parser.add_argument('--thresholds', type=str, default='1,2,4,8', help='Comma-separated thresholds')
    cutoff_parser.add_argument('--output', type=str, help='Output directory')
    cutoff_parser.add_argument('--save', action='store_true', help='Save results to output directory')
    cutoff_parser.add_argument('--notebook-mode', action='store_true', help='Force notebook mode for plot display')

    # Energy study
    energy_parser = subparsers.add_parser('energy', help='Run energy comparison analysis at fixed threshold')
    energy_parser.add_argument('--material', type=str, help='Material type (overrides global setting)')
    energy_parser.add_argument('--particle', type=str, help='Particle type (overrides global setting)')
    energy_parser.add_argument('--energies', type=str, default='500,1000,1500', help='Comma-separated energies (MeV)')
    energy_parser.add_argument('--threshold', type=float, default=4, help='Fixed threshold value')
    energy_parser.add_argument('--vmax', type=float, help='Maximum value for colorbar (optional)')
    energy_parser.add_argument('--output', type=str, help='Output directory')
    energy_parser.add_argument('--save', action='store_true', help='Save results to output directory')
    energy_parser.add_argument('--notebook-mode', action='store_true', help='Force notebook mode for plot display')

    # Valid points vs energy analysis
    valid_points_parser = subparsers.add_parser('valid-points', help='Run valid points vs energy analysis')
    valid_points_parser.add_argument('--material', type=str, help='Material type (overrides global setting)')
    valid_points_parser.add_argument('--particle', type=str, help='Particle type (overrides global setting)')
    valid_points_parser.add_argument('--energies', type=str, help='Comma-separated energies or range (e.g., 200,2000,20)')
    valid_points_parser.add_argument('--thresholds', type=str, default='1,2,4,8', help='Comma-separated thresholds')
    valid_points_parser.add_argument('--output', type=str, help='Output directory')
    valid_points_parser.add_argument('--save', action='store_true', help='Save results to output directory')
    valid_points_parser.add_argument('--notebook-mode', action='store_true', help='Force notebook mode for plot display')
    
    # Integral analysis
    integral_parser = subparsers.add_parser('integral', help='Run n-photon integral analysis')
    integral_parser.add_argument('--material', type=str, help='Material type (overrides global setting)')
    integral_parser.add_argument('--particle', type=str, help='Particle type (overrides global setting)')
    integral_parser.add_argument('--energies', type=str, help='Comma-separated energies or range (e.g., 100,2000,100)')
    integral_parser.add_argument('--nphot', type=int, default=1000000, help='Number of photons')
    integral_parser.add_argument('--output', type=str, help='Output directory')
    integral_parser.add_argument('--save', action='store_true', help='Save results to output directory')
    integral_parser.add_argument('--notebook-mode', action='store_true', help='Force notebook mode for plot display')
    
    # Ray validation
    rays_parser = subparsers.add_parser('rays', help='Run ray generation validation')
    rays_parser.add_argument('--material', type=str, help='Material type (overrides global setting)')
    rays_parser.add_argument('--particle', type=str, help='Particle type (overrides global setting)')
    rays_parser.add_argument('--energies', type=str, help='Comma-separated energies or range (e.g., 200,2000,20)')
    rays_parser.add_argument('--nphot', type=int, default=1000000, help='Number of photons')
    rays_parser.add_argument('--output', type=str, help='Output directory')
    rays_parser.add_argument('--save', action='store_true', help='Save results to output directory')
    rays_parser.add_argument('--notebook-mode', action='store_true', help='Force notebook mode for plot display')
    
    # All validations
    all_parser = subparsers.add_parser('all', help='Run all validations')
    all_parser.add_argument('--material', type=str, help='Material type (overrides global setting)')
    all_parser.add_argument('--particle', type=str, help='Particle type (overrides global setting)')
    all_parser.add_argument('--output', type=str, help='Output directory')
    all_parser.add_argument('--save', action='store_true', help='Save results to output directory')
    all_parser.add_argument('--notebook-mode', action='store_true', help='Force notebook mode for plot display')
    
    args = parser.parse_args()
    
    # Configure matplotlib for notebook display if needed
    setup_matplotlib_for_notebook(force_notebook_mode=getattr(args, 'notebook_mode', False))
    
    if args.command is None:
        parser.print_help()
        return
    
    # Get material and particle (defaults will be used if not specified)
    material = args.material or 'water'
    particle = args.particle or 'muon'
    
    # Initialize validator with material and particle
    print(f"🎯 Validation Configuration:")
    print(f"  Material: {material}")
    print(f"  Particle: {particle}")
    
    validator = PhotonSimValidator(material=material, particle=particle)
    
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
        elif hasattr(args, 'save') and args.save:
            # Use material/particle structure
            return Path(base_dir_path()) / 'output' / 'siren' / validator.material / validator.particle
        else:
            return None
    
    # Execute command
    if args.command == 'cutoff':
        thresholds = [float(x) for x in args.thresholds.split(',')]
        output_dir = get_output_dir(args)
        validator.cutoff_study(energy=args.energy, thresholds=thresholds, output_dir=output_dir)

    elif args.command == 'energy':
        energies = [float(x) for x in args.energies.split(',')]
        output_dir = get_output_dir(args)
        vmax = args.vmax if hasattr(args, 'vmax') else None
        validator.energy_study(energies=energies, threshold=args.threshold, vmax=vmax, output_dir=output_dir)

    elif args.command == 'valid-points':
        energies = None
        if args.energies:
            energies = parse_energies(args.energies)
        thresholds = [float(x) for x in args.thresholds.split(',')]
        output_dir = get_output_dir(args)
        validator.valid_points_vs_energy(energies=energies, thresholds=thresholds, output_dir=output_dir)
    
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
        output_dir = get_output_dir(args)
        validator.run_all_validations(output_dir=output_dir)


if __name__ == '__main__':
    main()