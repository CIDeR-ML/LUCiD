#!/usr/bin/env python3
"""
dEdx SIREN Validation Script

Visualize trained dEdx SIREN model predictions as 2D plots.
Shows dE/dx vs distance distributions at various energies.

Usage:
    python validate_dedx.py [OPTIONS]

    # Default water/muon
    python validate_dedx.py

    # Specific energies
    python validate_dedx.py --energies 300,600,1000,1500

    # With threshold
    python validate_dedx.py --threshold 0.01

    # Save to output directory
    python validate_dedx.py --output validation_results/
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Add parent directories to path
script_dir = Path(__file__).parent
tools_dir = script_dir.parent
project_root = tools_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tools_dir))

# Import modules
from tools.siren.training.inference import SIRENPredictor
from tools.siren.training.dataset import PhotonSimDataset
from tools.utils import base_dir_path, setup_matplotlib_for_notebook

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12


class DedxValidator:
    """Validation class for dEdx SIREN model."""

    def __init__(self, material='water', particle='muon', model_path=None, h5_path=None):
        """
        Initialize validator with material, particle type, and optional paths.

        Args:
            material: Material type (e.g., 'water', 'ice')
            particle: Particle type (e.g., 'muon', 'electron')
            model_path: Optional explicit model path
            h5_path: Optional explicit h5 file path
        """
        self.material = material
        self.particle = particle

        # Get base directory path
        base_dir = base_dir_path()

        # Default model path for dEdx in data directory structure
        if model_path is None:
            model_path = Path(base_dir) / 'data' / material / particle / 'dedx_siren_training' / 'trained_model' / 'dedx_siren'

        # Default h5 path for dEdx
        if h5_path is None:
            h5_path = Path(base_dir) / 'data' / material / particle / 'dedx_lookup_table.h5'
            if not h5_path.exists():
                raise FileNotFoundError(
                    f"HDF5 dEdx lookup table not found at {h5_path}\n"
                    f"Please ensure the dEdx table exists for {material}/{particle}"
                )

        print(f"dEdx SIREN Validation")
        print(f"  Material: {material}")
        print(f"  Particle: {particle}")
        print(f"Loading model from: {model_path}")

        self.predictor = SIRENPredictor(model_path)
        self.model_params = self.predictor.params

        print(f"Loading dataset from: {h5_path}")
        self.dataset = PhotonSimDataset(h5_path)

        # Get training ranges from predictor metadata
        self.dataset_info = self.predictor.dataset_info
        self.energy_min, self.energy_max = self.dataset_info['energy_range']

        # dEdx range instead of angle range
        if 'dedx_range' in self.dataset_info:
            self.dedx_min, self.dedx_max = self.dataset_info['dedx_range']
        else:
            # Fallback to dataset ranges
            self.dedx_min, self.dedx_max = self.dataset.dedx_range

        self.distance_min, self.distance_max = self.dataset_info['distance_range']

        print(f"Training ranges:")
        print(f"  Energy: {self.energy_min}-{self.energy_max} MeV")
        print(f"  dE/dx: {self.dedx_min:.1f}-{self.dedx_max:.1f} keV/mm")
        print(f"  Distance: {self.distance_min}-{self.distance_max} mm")
        print("dEdx validator initialized successfully")

    def evaluate_dedx_grid(self, energy, dedx_bins, distance_bins):
        """
        Evaluate dEdx model on dedx/distance grid for given energy.

        Args:
            energy: Energy value in MeV
            dedx_bins: Array of dE/dx bin centers
            distance_bins: Array of distance bin centers

        Returns:
            2D array of predictions
        """
        dedx_mesh, distance_mesh = np.meshgrid(dedx_bins, distance_bins, indexing='ij')

        # Create evaluation grid: [energy, dedx, distance]
        evaluation_grid = np.stack([
            np.full_like(dedx_mesh, energy).ravel(),
            dedx_mesh.ravel(),
            distance_mesh.ravel(),
        ], axis=1)

        # Get predictions
        predictions = self.predictor.predict_batch(evaluation_grid)
        return np.array(predictions).reshape(len(dedx_bins), len(distance_bins))

    def plot_2d_slices(self, energies=None, threshold=None, vmax=None, output_dir=None):
        """
        Plot 2D dE/dx vs distance slices at multiple energies.

        Args:
            energies: List of energies to plot (default: [300, 600, 1000, 1500])
            threshold: Minimum value threshold for display (optional)
            vmax: Maximum value for colorbar (optional, auto-determined if None)
            output_dir: Directory to save results
        """
        print(f"\n=== 2D dE/dx vs Distance Slices ===")

        if energies is None:
            energies = [300, 600, 1000, 1500]

        # Create analysis grid
        n_dedx_bins = 250
        n_distance_bins = 250
        dedx_bins = np.linspace(self.dedx_min, self.dedx_max, n_dedx_bins)
        distance_bins = np.linspace(self.distance_min, self.distance_max, n_distance_bins)

        print(f"Grid: {n_dedx_bins} x {n_distance_bins} points")
        print(f"Energies: {energies} MeV")

        # Collect predictions for all energies
        all_predictions = []
        for energy in energies:
            print(f"  Processing {energy} MeV...")
            predictions = self.evaluate_dedx_grid(energy, dedx_bins, distance_bins)
            all_predictions.append(predictions)

        # Determine vmax if not provided
        if vmax is None:
            vmax = max(np.max(p) for p in all_predictions)

        # Determine vmin (threshold or auto)
        if threshold is not None:
            vmin = threshold
        else:
            # Use small positive value for log scale
            vmin = 1e-3

        # Create visualization
        n_energies = len(energies)
        if n_energies <= 4:
            fig, axes = plt.subplots(1, n_energies, figsize=(4 * n_energies, 4), constrained_layout=True)
            if n_energies == 1:
                axes = [axes]
        else:
            cols = 4
            rows = (n_energies + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), constrained_layout=True)
            axes = axes.ravel()

        # Convert distance to meters for display
        distance_min_m = self.distance_min / 1000.0
        distance_max_m = self.distance_max / 1000.0

        images = []
        for i, (energy, predictions) in enumerate(zip(energies, all_predictions)):
            ax = axes[i]

            # Apply threshold if provided
            if threshold is not None:
                masked_predictions = np.where(predictions > threshold, predictions, np.nan)
            else:
                masked_predictions = predictions

            # Plot with log scale
            im = ax.imshow(
                masked_predictions,
                norm=LogNorm(vmin=vmin, vmax=vmax),
                aspect='auto',
                origin='lower',
                extent=[distance_min_m, distance_max_m, self.dedx_min, self.dedx_max],
                cmap='viridis'
            )
            images.append(im)

            ax.set_xlabel('Distance (m)')
            if i == 0:
                ax.set_ylabel('dE/dx (keV/mm)')

            # Add energy label
            ax.text(0.95, 0.95, f'{int(energy)} MeV', transform=ax.transAxes,
                    fontsize=12, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Calculate statistics
            valid_count = np.sum(predictions > (threshold if threshold else 0))
            total_weight = np.sum(predictions)
            print(f"  {energy} MeV: {valid_count:,} valid points, total weight: {total_weight:.2e}")

        # Hide unused axes
        for i in range(len(energies), len(axes)):
            axes[i].set_visible(False)

        # Add colorbar
        cbar = fig.colorbar(images[-1], ax=axes[:len(energies)], label='Intensity (entries/event)', shrink=0.8)

        fig.suptitle(f'dE/dx SIREN Model - {self.material}/{self.particle}', fontsize=14)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = f"{output_dir}/dedx_2d_slices.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")

        plt.show()

        return {'energies': energies, 'predictions': all_predictions}

    def plot_energy_comparison(self, energy, n_bins=250, output_dir=None):
        """
        Compare model predictions with original lookup table data for a single energy.

        Args:
            energy: Energy value in MeV
            n_bins: Number of bins for the grid
            output_dir: Directory to save results
        """
        print(f"\n=== Model vs Table Comparison at {energy} MeV ===")

        # Create analysis grid
        dedx_bins = np.linspace(self.dedx_min, self.dedx_max, n_bins)
        distance_bins = np.linspace(self.distance_min, self.distance_max, n_bins)

        # Get model predictions
        print("  Evaluating model...")
        model_predictions = self.evaluate_dedx_grid(energy, dedx_bins, distance_bins)

        # Get table data from dataset
        print("  Loading table data...")
        # Find the closest energy index in the dataset
        energy_centers = self.dataset.metadata.get('energy_centers', None)
        if energy_centers is None:
            # Get from the normalized bounds
            inputs = self.dataset.data['inputs']
            unique_energies = np.unique(inputs[:, 0])
            energy_idx = np.argmin(np.abs(unique_energies - energy))
            closest_energy = unique_energies[energy_idx]
        else:
            energy_idx = np.argmin(np.abs(np.array(energy_centers) - energy))
            closest_energy = energy_centers[energy_idx]

        print(f"  Closest energy in table: {closest_energy:.1f} MeV")

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # Convert distance to meters
        distance_min_m = self.distance_min / 1000.0
        distance_max_m = self.distance_max / 1000.0

        # Determine common color scale
        vmin = 1e-3
        vmax = np.nanmax(model_predictions)

        # Plot model predictions
        im1 = axes[0].imshow(
            model_predictions,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            aspect='auto',
            origin='lower',
            extent=[distance_min_m, distance_max_m, self.dedx_min, self.dedx_max],
            cmap='viridis'
        )
        axes[0].set_xlabel('Distance (m)')
        axes[0].set_ylabel('dE/dx (keV/mm)')
        axes[0].set_title(f'SIREN Model ({energy} MeV)')
        fig.colorbar(im1, ax=axes[0], label='Intensity')

        # Plot 1D projections
        # dE/dx projection (sum over distance)
        dedx_projection = np.sum(model_predictions, axis=1)
        axes[1].plot(dedx_bins, dedx_projection, 'b-', linewidth=2)
        axes[1].set_xlabel('dE/dx (keV/mm)')
        axes[1].set_ylabel('Sum over distance')
        axes[1].set_title('dE/dx Projection')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 200)  # Focus on lower dE/dx range

        # Distance projection (sum over dE/dx)
        distance_projection = np.sum(model_predictions, axis=0)
        axes[2].plot(distance_bins / 1000, distance_projection, 'r-', linewidth=2)
        axes[2].set_xlabel('Distance (m)')
        axes[2].set_ylabel('Sum over dE/dx')
        axes[2].set_title('Distance Projection')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'dE/dx SIREN Analysis - {energy} MeV', fontsize=14)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = f"{output_dir}/dedx_comparison_{int(energy)}MeV.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")

        plt.show()

        return {'predictions': model_predictions, 'dedx_projection': dedx_projection, 'distance_projection': distance_projection}

    def plot_dedx_vs_energy(self, distances=None, dedx_values=None, output_dir=None):
        """
        Plot model predictions as a function of energy for fixed distances and dE/dx values.

        Args:
            distances: List of distances (mm) to plot (default: [1000, 3000, 5000, 7000])
            dedx_values: List of dE/dx values (keV/mm) to use (default: [5, 10, 20, 50])
            output_dir: Directory to save results
        """
        print(f"\n=== dE/dx Response vs Energy ===")

        if distances is None:
            distances = [1000, 3000, 5000, 7000]
        if dedx_values is None:
            dedx_values = [5, 10, 20, 50]

        # Energy grid
        energies = np.linspace(self.energy_min, self.energy_max, 200)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        # Plot 1: Fixed dE/dx, varying distance
        ax = axes[0]
        dedx_fixed = 10.0  # keV/mm
        for distance in distances:
            inputs = np.column_stack([energies, np.full_like(energies, dedx_fixed), np.full_like(energies, distance)])
            predictions = self.predictor.predict_batch(inputs)
            ax.plot(energies, predictions, label=f'd = {distance/1000:.1f} m', linewidth=2)

        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'dE/dx = {dedx_fixed} keV/mm')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Fixed distance, varying dE/dx
        ax = axes[1]
        distance_fixed = 3000  # mm
        for dedx in dedx_values:
            inputs = np.column_stack([energies, np.full_like(energies, dedx), np.full_like(energies, distance_fixed)])
            predictions = self.predictor.predict_batch(inputs)
            ax.plot(energies, predictions, label=f'dE/dx = {dedx} keV/mm', linewidth=2)

        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Distance = {distance_fixed/1000:.1f} m')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'dE/dx SIREN Response - {self.material}/{self.particle}', fontsize=14)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = f"{output_dir}/dedx_vs_energy.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")

        plt.show()

    def run_all_validations(self, output_dir=None):
        """Run all validation plots."""
        print("\nRunning all dEdx SIREN validations...")

        # Set default output directory
        if output_dir is None:
            output_dir = Path(base_dir_path()) / 'data' / self.material / self.particle / 'dedx_siren_training' / 'validation'

        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to: {output_dir}")

        # 2D slices at multiple energies
        self.plot_2d_slices(output_dir=output_dir)

        # Comparison at sample energy
        self.plot_energy_comparison(energy=1000, output_dir=output_dir)

        # Energy response plots
        self.plot_dedx_vs_energy(output_dir=output_dir)

        print(f"\nAll validations completed!")
        print(f"Results saved to: {output_dir}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='dEdx SIREN Validation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Material/particle arguments
    parser.add_argument('--material', type=str, default='water',
                        help='Material type (default: water)')
    parser.add_argument('--particle', type=str, default='muon',
                        help='Particle type (default: muon)')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Validation command')

    # 2D slices command
    slices_parser = subparsers.add_parser('slices', help='Plot 2D dE/dx vs distance slices')
    slices_parser.add_argument('--energies', type=str, default='300,600,1000,1500',
                               help='Comma-separated energies (MeV)')
    slices_parser.add_argument('--threshold', type=float, help='Minimum value threshold')
    slices_parser.add_argument('--output', type=str, help='Output directory')
    slices_parser.add_argument('--notebook-mode', action='store_true', help='Force notebook mode')

    # Comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare model with table at single energy')
    compare_parser.add_argument('--energy', type=float, default=1000, help='Energy (MeV)')
    compare_parser.add_argument('--output', type=str, help='Output directory')
    compare_parser.add_argument('--notebook-mode', action='store_true', help='Force notebook mode')

    # Energy response command
    response_parser = subparsers.add_parser('response', help='Plot energy response curves')
    response_parser.add_argument('--distances', type=str, default='1000,3000,5000,7000',
                                 help='Comma-separated distances (mm)')
    response_parser.add_argument('--dedx-values', type=str, default='5,10,20,50',
                                 help='Comma-separated dE/dx values (keV/mm)')
    response_parser.add_argument('--output', type=str, help='Output directory')
    response_parser.add_argument('--notebook-mode', action='store_true', help='Force notebook mode')

    # All validations
    all_parser = subparsers.add_parser('all', help='Run all validations')
    all_parser.add_argument('--output', type=str, help='Output directory')
    all_parser.add_argument('--notebook-mode', action='store_true', help='Force notebook mode')

    args = parser.parse_args()

    # Configure matplotlib
    setup_matplotlib_for_notebook(force_notebook_mode=getattr(args, 'notebook_mode', False))

    # Default command if none specified
    if args.command is None:
        args.command = 'slices'

    # Initialize validator
    print(f"Validation Configuration:")
    print(f"  Material: {args.material}")
    print(f"  Particle: {args.particle}")

    try:
        validator = DedxValidator(material=args.material, particle=args.particle)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Created the dEdx lookup table (dedx_lookup_table.h5)")
        print("  2. Trained the dEdx SIREN model (python train.py --data-type dedx)")
        return 1

    # Execute command
    if args.command == 'slices':
        energies = [float(x) for x in args.energies.split(',')]
        validator.plot_2d_slices(
            energies=energies,
            threshold=getattr(args, 'threshold', None),
            output_dir=getattr(args, 'output', None)
        )

    elif args.command == 'compare':
        validator.plot_energy_comparison(
            energy=args.energy,
            output_dir=getattr(args, 'output', None)
        )

    elif args.command == 'response':
        distances = [float(x) for x in args.distances.split(',')]
        dedx_values = [float(x) for x in args.dedx_values.split(',')]
        validator.plot_dedx_vs_energy(
            distances=distances,
            dedx_values=dedx_values,
            output_dir=getattr(args, 'output', None)
        )

    elif args.command == 'all':
        validator.run_all_validations(output_dir=getattr(args, 'output', None))

    return 0


if __name__ == '__main__':
    sys.exit(main())
