#!/usr/bin/env python3
"""
PhotonSim SIREN Validation Suite

This module consolidates validation functionality from three notebooks:
1. photonsim_cut_off_study.ipynb - Cut-off threshold analysis
2. photonsim_n_photon_integral.ipynb - N-photon integral analysis  
3. photonsim_rays_validation.ipynb - Ray generation validation

Usage:
    python validation.py cutoff [--energy ENERGY] [--thresholds THRESHOLDS] [--output OUTPUT]
    python validation.py integral [--energies ENERGIES] [--nphot NPHOT] [--output OUTPUT]
    python validation.py rays [--energies ENERGIES] [--nphot NPHOT] [--output OUTPUT]
    python validation.py all [--output OUTPUT]

Examples:
    python validation.py cutoff --energy 500 --thresholds 1,2,4,8
    python validation.py integral --energies 200,500,800 --nphot 1000000
    python validation.py rays --energies 200,400,600,800,1000 --nphot 1000000
    python validation.py all --output validation_results/
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

# Add parent directory to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / 'tools'))

# Add training modules
training_path = script_dir / 'training'
sys.path.append(str(training_path))

# Import PhotonSim training modules
from training.inference import SIRENPredictor
from training.dataset import PhotonSimDataset

# Import tools
from tools.siren import SIREN
from tools.simulation import create_photonsim_siren_grid
from tools.generate import generate_random_cone_vectors, normalize, photonsim_differentiable_get_rays
from tools.utils import base_dir_path

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

class PhotonSimValidator:
    """Main validation class for PhotonSim SIREN model."""
    
    def __init__(self, model_path=None, h5_path=None):
        """Initialize validator with model and dataset paths."""
        # Get base directory path
        base_dir = base_dir_path()
        
        # Default model path
        if model_path is None:
            model_path = Path(base_dir) / 'notebooks/output/photonsim_siren_training/trained_model/photonsim_siren'
        
        # Default h5 path
        if h5_path is None:
            h5_path = Path(base_dir) / 'data/water/muon/photon_lookup_table.h5'
        
        print(f"Loading PhotonSim SIREN model from: {model_path}")
        self.photonsim_predictor = SIRENPredictor(model_path)
        self.model_params = self.photonsim_predictor.params
        
        print(f"Loading PhotonSim dataset from: {h5_path}")
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
        self.table_data = create_photonsim_siren_grid(self.photonsim_predictor, 500)
        
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
    
    def cutoff_study(self, energy=500, thresholds=None, output_dir=None):
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
        n_angle_bins = 500
        n_distance_bins = 500
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
            energies = np.linspace(100, 1000, 20)
        
        print(f"Analyzing {len(energies)} energies from {energies[0]} to {energies[-1]} MeV")
        print(f"N-photons: {nphot:,}")
        
        tot_real_photons = []
        tot_pred_photons = []
        
        for energy in energies:
            # Get real photon count from dataset
            real_count = self.dataset.get_total_counts_for_energy(energy)
            tot_real_photons.append(real_count)
            
            # Get predicted photon count
            _, _, photon_weights = photonsim_differentiable_get_rays(
                self.origin, self.direction, energy, nphot, self.table_data, self.model_params, self.key
            )
            pred_count = np.sum(photon_weights)
            tot_pred_photons.append(pred_count)
            
            if len(tot_real_photons) % 10 == 0:
                print(f"  Processed {len(tot_real_photons)}/{len(energies)} energies")
        
        # Calculate ratio and fit
        y_data = np.array(tot_real_photons) / (np.array(tot_pred_photons) / nphot)
        
        # Filter data for energies above 200 MeV for linear fit
        fit_mask = np.array(energies) >= 200
        energies_fit = np.array(energies)[fit_mask]
        y_data_fit = y_data[fit_mask]
        
        print(f"Using {len(energies_fit)}/{len(energies)} data points for linear fit (energies ≥ 200 MeV)")
        
        # Linear fit on filtered data
        slope, intercept, r_value, p_value, std_err = stats.linregress(energies_fit, y_data_fit)
        line = slope * energies + intercept
        
        # Print results
        print(f"\nLinear Fit Results:")
        print(f"  Slope: {slope:.6f} ± {std_err:.6f}")
        print(f"  Intercept: {intercept:.6f}")
        print(f"  R-squared: {r_value**2:.4f}")
        print(f"  Equation: y = {slope:.6f}x + {intercept:.6f}")
        
        # Apply correction function
        def corr_function(x):
            return slope * x + intercept
        
        y_corrected = np.array(tot_real_photons) / (corr_function(energies) * np.array(tot_pred_photons) / nphot)
        
        # Fit corrected data (also filtered for energies >= 200 MeV)
        y_corrected_fit = y_corrected[fit_mask]
        slope_corr, intercept_corr, r_value_corr, _, _ = stats.linregress(energies_fit, y_corrected_fit)
        line_corr = slope_corr * energies + intercept_corr
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Original data
        axes[0].plot(energies, y_data, 'bo', label='All Data', markersize=3, alpha=0.6)
        axes[0].plot(energies_fit, y_data_fit, 'ro', label='Fit Data (≥200 MeV)', markersize=3)
        axes[0].plot(energies, line, 'r-', label=f'Linear fit (R²={r_value**2:.3f})')
        axes[0].set_xlabel('Energy (MeV)')
        axes[0].set_ylabel('Ratio')
        axes[0].set_title('Original Data vs Linear Fit')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Corrected data
        axes[1].plot(energies, y_corrected, 'go', label='All Corrected Data', markersize=3, alpha=0.6)
        axes[1].plot(energies_fit, y_corrected_fit, 'mo', label='Fit Data (≥200 MeV)', markersize=3)
        axes[1].plot(energies, line_corr, 'r-', label=f'Linear fit (R²={r_value_corr**2:.3f})')
        axes[1].set_xlabel('Energy (MeV)')
        axes[1].set_ylabel('Corrected Ratio')
        axes[1].set_title('Corrected Data vs Linear Fit')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('N-Photon Integral Analysis', fontsize=16)
        fig.tight_layout()
        
        # Save results
        results = {
            'energies': energies.tolist(),
            'real_photons': tot_real_photons,
            'pred_photons': tot_pred_photons,
            'nphot': nphot,
            'fit_filter': {
                'min_energy': 200,
                'energies_used': energies_fit.tolist(),
                'data_points_used': len(energies_fit),
                'total_data_points': len(energies)
            },
            'original_fit': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'equation': f'y = {slope:.6f}x + {intercept:.6f}'
            },
            'corrected_fit': {
                'slope': slope_corr,
                'intercept': intercept_corr,
                'r_squared': r_value_corr**2,
                'equation': f'y = {slope_corr:.6f}x + {intercept_corr:.6f}'
            }
        }
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(f"{output_dir}/integral_analysis.png", dpi=150, bbox_inches='tight')
            print(f"Integral analysis results saved to: {output_dir}/integral_analysis.png")
        
        plt.show()
        
        return results
    
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
            energies = np.linspace(200, 1000, 20)
        
        print(f"Analyzing {len(energies)} energies from {energies[0]} to {energies[-1]} MeV")
        print(f"N-photons: {nphot:,}")
        
        # Create visualization grid
        n_energies = len(energies)
        cols = 5
        rows = (n_energies + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 2*rows))
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
            
            # Generate rays
            ray_vectors, ray_origins, photon_weights = photonsim_differentiable_get_rays(
                self.origin, self.direction, energy, nphot, self.table_data, self.model_params, self.key
            )
            
            # Calculate ranges and angles
            ranges = jnp.linalg.norm(ray_origins - self.origin, axis=1)
            angles = self.calculate_opening_angles(ray_vectors, self.direction)
            
            # Calculate statistics
            num_seeds = jnp.int32(energy * 11.136 - 720.3)
            total_weight = float(jnp.sum(photon_weights))
            
            results['statistics'][energy] = {
                'num_seeds': int(num_seeds),
                'total_weight': total_weight,
                'mean_range': float(jnp.mean(ranges)),
                'mean_angle': float(jnp.mean(angles))
            }
            
            # Create 2D histogram
            h = axes[row, col].hist2d(
                ranges, angles,
                weights=photon_weights.squeeze(),
                bins=[100, 100],
                cmap='gnuplot',
                norm=LogNorm(vmin=1),
                range=[[0, 6], [0, 3.14]]
            )
            
            axes[row, col].set_ylabel('Angle (radians)')
            axes[row, col].set_xlabel('Distance to Origin (m)')
            axes[row, col].set_title(f'Energy: {energy:.0f} MeV\nSeeds: {num_seeds:,}')
        
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
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = Path(base_dir_path()) / 'output/siren'
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to: {output_dir}")
        
        # Run cut-off study
        cutoff_results = self.cutoff_study(output_dir=output_dir)
        
        # Run integral analysis
        integral_results = self.integral_analysis(output_dir=output_dir)
        
        # Run ray validation
        rays_results = self.rays_validation(output_dir=output_dir)
        
        print(f"\n✅ All validations completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        return {
            'cutoff_study': cutoff_results,
            'integral_analysis': integral_results,
            'rays_validation': rays_results
        }


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='PhotonSim SIREN Validation Suite')
    
    subparsers = parser.add_subparsers(dest='command', help='Validation command')
    
    # Cut-off study
    cutoff_parser = subparsers.add_parser('cutoff', help='Run cut-off threshold analysis')
    cutoff_parser.add_argument('--energy', type=float, default=500, help='Analysis energy (MeV)')
    cutoff_parser.add_argument('--thresholds', type=str, default='1,2,4,8', help='Comma-separated thresholds')
    cutoff_parser.add_argument('--output', type=str, help='Output directory (default: output/siren)')
    cutoff_parser.add_argument('--save', action='store_true', help='Save results to output/siren directory')
    
    # Integral analysis
    integral_parser = subparsers.add_parser('integral', help='Run n-photon integral analysis')
    integral_parser.add_argument('--energies', type=str, help='Comma-separated energies or range (e.g., 100,1000,100)')
    integral_parser.add_argument('--nphot', type=int, default=1000000, help='Number of photons')
    integral_parser.add_argument('--output', type=str, help='Output directory (default: output/siren)')
    integral_parser.add_argument('--save', action='store_true', help='Save results to output/siren directory')
    
    # Ray validation
    rays_parser = subparsers.add_parser('rays', help='Run ray generation validation')
    rays_parser.add_argument('--energies', type=str, help='Comma-separated energies or range (e.g., 200,1000,20)')
    rays_parser.add_argument('--nphot', type=int, default=1000000, help='Number of photons')
    rays_parser.add_argument('--output', type=str, help='Output directory (default: output/siren)')
    rays_parser.add_argument('--save', action='store_true', help='Save results to output/siren directory')
    
    # All validations
    all_parser = subparsers.add_parser('all', help='Run all validations')
    all_parser.add_argument('--output', type=str, help='Output directory (default: output/siren)')
    all_parser.add_argument('--save', action='store_true', help='Save results to output/siren directory')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize validator
    validator = PhotonSimValidator()
    
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
            return Path(base_dir_path()) / 'output/siren'
        else:
            return None
    
    # Execute command
    if args.command == 'cutoff':
        thresholds = [float(x) for x in args.thresholds.split(',')]
        output_dir = get_output_dir(args)
        validator.cutoff_study(energy=args.energy, thresholds=thresholds, output_dir=output_dir)
    
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