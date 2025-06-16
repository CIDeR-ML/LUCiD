#!/usr/bin/env python3
"""
Create training dataset for JAXSiren from the density-normalized 3D lookup table.
Samples points from the 3D space and creates input-output pairs for neural network training.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import argparse
from tqdm import tqdm

class SirenDatasetCreator:
    """Create training dataset for SIREN from 3D photon density table."""
    
    def __init__(self, table_path):
        """
        Initialize dataset creator.
        
        Parameters:
        -----------
        table_path : str or Path
            Path to the HDF5 lookup table file (photon_lookup_table.h5)
        """
        self.table_path = Path(table_path)
        self.load_table()
        
    def load_table(self):
        """Load the 3D density table and metadata from HDF5 file."""
        print(f"Loading 3D density table from {self.table_path}...")
        
        if not self.table_path.exists():
            raise FileNotFoundError(f"HDF5 lookup table not found: {self.table_path}")
        
        with h5py.File(self.table_path, 'r') as f:
            # Load density table
            self.density_table = f['data/photon_table_density'][:]
            
            # Load coordinate arrays
            self.energy_values = f['coordinates/energy_values'][:]
            self.angle_centers = f['coordinates/angle_centers'][:]
            self.distance_centers = f['coordinates/distance_centers'][:]
            self.angle_edges = f['coordinates/angle_edges'][:]
            self.distance_edges = f['coordinates/distance_edges'][:]
            
            # Load metadata attributes
            meta = f['metadata']
            self.energy_min = meta.attrs['energy_min']
            self.energy_max = meta.attrs['energy_max']
            self.density_units = meta.attrs['density_units'].decode() if isinstance(meta.attrs['density_units'], bytes) else meta.attrs['density_units']
        
        print(f"Table loaded: {self.density_table.shape}")
        print(f"Energy range: {self.energy_values[0]}-{self.energy_values[-1]} MeV")
        print(f"Non-zero density bins: {np.count_nonzero(self.density_table):,} / {self.density_table.size:,}")
        
    def create_dataset(self, n_samples=1000000, sampling_strategy='importance', 
                      normalize_inputs=True, normalize_outputs=False,
                      log_transform_output=True):
        """
        Create training dataset by sampling from the 3D table.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        sampling_strategy : str
            'uniform' - uniform sampling in 3D space
            'importance' - sample proportional to photon density
            'mixed' - combination of uniform and importance sampling
        normalize_inputs : bool
            Whether to normalize inputs to [-1, 1]
        normalize_outputs : bool
            Whether to normalize outputs
        log_transform_output : bool
            Whether to apply log transform to outputs
        
        Returns:
        --------
        dataset : dict
            Dictionary containing training data and metadata
        """
        print(f"\nCreating SIREN dataset with {n_samples:,} samples...")
        print(f"Sampling strategy: {sampling_strategy}")
        
        if sampling_strategy == 'uniform':
            inputs, outputs = self._uniform_sampling(n_samples)
        elif sampling_strategy == 'importance':
            inputs, outputs = self._importance_sampling(n_samples)
        elif sampling_strategy == 'mixed':
            # 80% importance, 20% uniform for better coverage of data-rich regions
            n_uniform = int(n_samples * 0.2)  # 20% uniform
            n_importance = n_samples - n_uniform  # 80% importance
            
            inputs1, outputs1 = self._uniform_sampling(n_uniform)
            inputs2, outputs2 = self._importance_sampling(n_importance)
            
            inputs = np.vstack([inputs1, inputs2])
            outputs = np.hstack([outputs1, outputs2])
            
            print(f"Mixed sampling: {n_uniform:,} uniform + {n_importance:,} importance")
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        # Apply transformations
        if log_transform_output:
            # Add small epsilon to avoid log(0)
            outputs = np.log10(outputs + 1e-10)
            print(f"Applied log10 transform to outputs")
        
        # Store original ranges for denormalization
        input_ranges = {
            'energy': (self.energy_values.min(), self.energy_values.max()),
            'angle': (self.angle_edges.min(), self.angle_edges.max()),
            'distance': (self.distance_edges.min(), self.distance_edges.max())
        }
        
        # Normalize inputs to [-1, 1] if requested
        if normalize_inputs:
            inputs_norm = np.zeros_like(inputs)
            
            # Energy: [min, max] -> [-1, 1]
            e_min, e_max = input_ranges['energy']
            inputs_norm[:, 0] = 2 * (inputs[:, 0] - e_min) / (e_max - e_min) - 1
            
            # Angle: [0, pi] -> [-1, 1]
            a_min, a_max = input_ranges['angle']
            inputs_norm[:, 1] = 2 * (inputs[:, 1] - a_min) / (a_max - a_min) - 1
            
            # Distance: [0, max] -> [-1, 1]
            d_min, d_max = input_ranges['distance']
            inputs_norm[:, 2] = 2 * (inputs[:, 2] - d_min) / (d_max - d_min) - 1
            
            inputs = inputs_norm
            print(f"Normalized inputs to [-1, 1]")
        
        # Normalize outputs if requested
        output_stats = {
            'min': outputs.min(),
            'max': outputs.max(),
            'mean': outputs.mean(),
            'std': outputs.std()
        }
        
        if normalize_outputs:
            # Standardize outputs (zero mean, unit variance)
            outputs = (outputs - output_stats['mean']) / output_stats['std']
            print(f"Normalized outputs (standardization)")
        
        # Shuffle the dataset
        indices = np.random.permutation(len(inputs))
        inputs = inputs[indices]
        outputs = outputs[indices]
        
        # Create dataset dictionary
        dataset = {
            'inputs': inputs.astype(np.float32),
            'outputs': outputs.astype(np.float32),
            'n_samples': n_samples,
            'sampling_strategy': sampling_strategy,
            'input_dim': 3,
            'output_dim': 1,
            'input_names': ['energy_mev', 'angle_rad', 'distance_mm'],
            'output_name': 'photon_density',
            'input_ranges': input_ranges,
            'output_stats': output_stats,
            'normalize_inputs': normalize_inputs,
            'normalize_outputs': normalize_outputs,
            'log_transform_output': log_transform_output,
            'density_units': 'photons/(event·sr·mm)' if not log_transform_output else 'log10(photons/(event·sr·mm))'
        }
        
        return dataset
    
    def _uniform_sampling(self, n_samples):
        """Uniform sampling in 3D space."""
        print("  Performing uniform sampling...")
        
        # Random sampling in each dimension
        energy_indices = np.random.randint(0, len(self.energy_values), n_samples)
        angle_indices = np.random.randint(0, len(self.angle_centers), n_samples)
        distance_indices = np.random.randint(0, len(self.distance_centers), n_samples)
        
        # Get actual values
        energies = self.energy_values[energy_indices]
        angles = self.angle_centers[angle_indices]
        distances = self.distance_centers[distance_indices]
        
        # Get density values
        densities = self.density_table[energy_indices, angle_indices, distance_indices]
        
        # Create input array
        inputs = np.column_stack([energies, angles, distances])
        
        return inputs, densities
    
    def _importance_sampling(self, n_samples):
        """Importance sampling proportional to photon density."""
        print("  Performing importance sampling...")
        
        # Flatten the density table
        flat_density = self.density_table.flatten()
        
        # Create probability distribution (avoid zero probabilities)
        probabilities = flat_density + 1e-10
        probabilities = probabilities / probabilities.sum()
        
        # Sample indices according to probability
        flat_indices = np.random.choice(len(flat_density), size=n_samples, p=probabilities)
        
        # Convert flat indices back to 3D indices
        energy_indices, angle_indices, distance_indices = np.unravel_index(
            flat_indices, self.density_table.shape)
        
        # Get actual values
        energies = self.energy_values[energy_indices]
        angles = self.angle_centers[angle_indices]
        distances = self.distance_centers[distance_indices]
        
        # Get density values
        densities = self.density_table[energy_indices, angle_indices, distance_indices]
        
        # Create input array
        inputs = np.column_stack([energies, angles, distances])
        
        return inputs, densities
    
    def save_dataset(self, dataset, output_path):
        """Save dataset to numpy files."""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save arrays
        np.save(output_path / "train_inputs.npy", dataset['inputs'])
        np.save(output_path / "train_outputs.npy", dataset['outputs'])
        
        # Save metadata
        metadata = {k: v for k, v in dataset.items() if k not in ['inputs', 'outputs']}
        np.savez(output_path / "dataset_metadata.npz", **metadata)
        
        print(f"\nDataset saved to {output_path}")
        print(f"  - train_inputs.npy: {dataset['inputs'].shape}")
        print(f"  - train_outputs.npy: {dataset['outputs'].shape}")
        print(f"  - dataset_metadata.npz: Dataset configuration and statistics")
    
    def visualize_dataset(self, dataset, n_samples=10000, output_path=None):
        """Visualize the training dataset."""
        print("\nVisualizing dataset...")
        
        # Sample subset for visualization
        if len(dataset['inputs']) > n_samples:
            indices = np.random.choice(len(dataset['inputs']), n_samples, replace=False)
            inputs = dataset['inputs'][indices]
            outputs = dataset['outputs'][indices]
        else:
            inputs = dataset['inputs']
            outputs = dataset['outputs']
        
        # Denormalize inputs for visualization
        if dataset['normalize_inputs']:
            inputs_denorm = np.zeros_like(inputs)
            # Energy
            e_min, e_max = dataset['input_ranges']['energy']
            inputs_denorm[:, 0] = (inputs[:, 0] + 1) * (e_max - e_min) / 2 + e_min
            # Angle
            a_min, a_max = dataset['input_ranges']['angle']
            inputs_denorm[:, 1] = (inputs[:, 1] + 1) * (a_max - a_min) / 2 + a_min
            # Distance
            d_min, d_max = dataset['input_ranges']['distance']
            inputs_denorm[:, 2] = (inputs[:, 2] + 1) * (d_max - d_min) / 2 + d_min
            
            inputs = inputs_denorm
        
        # Convert angle to degrees for visualization
        angles_deg = np.degrees(inputs[:, 1])
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 3D scatter plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        scatter = ax1.scatter(inputs[:, 0], angles_deg, inputs[:, 2], 
                            c=outputs, cmap='viridis', s=1, alpha=0.5)
        ax1.set_xlabel('Energy (MeV)')
        ax1.set_ylabel('Angle (degrees)')
        ax1.set_zlabel('Distance (mm)')
        ax1.set_title(f'Dataset Sample (n={len(inputs)})')
        plt.colorbar(scatter, ax=ax1, label='Output')
        
        # 2. Energy distribution
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(inputs[:, 0], bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Energy (MeV)')
        ax2.set_ylabel('Count')
        ax2.set_title('Energy Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Angle distribution
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.hist(angles_deg, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Angle (degrees)')
        ax3.set_ylabel('Count')
        ax3.set_title('Angle Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Distance distribution
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.hist(inputs[:, 2], bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Distance (mm)')
        ax4.set_ylabel('Count')
        ax4.set_title('Distance Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Output distribution
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(outputs, bins=100, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Output Value')
        ax5.set_ylabel('Count')
        title = 'Output Distribution'
        if dataset['log_transform_output']:
            title += ' (log10 scale)'
        ax5.set_title(title)
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistics
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        stats_text = f"""Dataset Statistics:
        
Total samples: {dataset['n_samples']:,}
Sampling: {dataset['sampling_strategy']}

Input ranges:
  Energy: {dataset['input_ranges']['energy'][0]}-{dataset['input_ranges']['energy'][1]} MeV
  Angle: 0-{np.degrees(dataset['input_ranges']['angle'][1]):.1f}°
  Distance: 0-{dataset['input_ranges']['distance'][1]:.0f} mm

Output statistics:
  Min: {dataset['output_stats']['min']:.6f}
  Max: {dataset['output_stats']['max']:.6f}
  Mean: {dataset['output_stats']['mean']:.6f}
  Std: {dataset['output_stats']['std']:.6f}

Transformations:
  Input norm: {dataset['normalize_inputs']}
  Output norm: {dataset['normalize_outputs']}
  Log output: {dataset['log_transform_output']}"""
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=10, family='monospace', verticalalignment='top')
        
        plt.suptitle('SIREN Training Dataset Visualization', fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Create training dataset for JAXSiren from 3D density table')
    parser.add_argument('--table-path', '-t',
                       default='../PhotonSim/output/3d_lookup_table_density/photon_lookup_table.h5',
                       help='Path to HDF5 photon lookup table file')
    parser.add_argument('--output', '-o',
                       default='output/siren_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--n-samples', '-n', type=int, default=1000000,
                       help='Number of samples to generate')
    parser.add_argument('--sampling', '-s', 
                       choices=['uniform', 'importance', 'mixed'],
                       default='mixed',
                       help='Sampling strategy')
    parser.add_argument('--no-normalize-inputs', action='store_true',
                       help='Do not normalize inputs to [-1, 1]')
    parser.add_argument('--normalize-outputs', action='store_true',
                       help='Normalize outputs (standardization)')
    parser.add_argument('--no-log-transform', action='store_true',
                       help='Do not apply log transform to outputs')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of dataset')
    
    args = parser.parse_args()
    
    # Create dataset creator
    creator = SirenDatasetCreator(args.table_path)
    
    # Create dataset
    dataset = creator.create_dataset(
        n_samples=args.n_samples,
        sampling_strategy=args.sampling,
        normalize_inputs=not args.no_normalize_inputs,
        normalize_outputs=args.normalize_outputs,
        log_transform_output=not args.no_log_transform
    )
    
    # Save dataset
    creator.save_dataset(dataset, args.output)
    
    # Visualize if requested
    if args.visualize:
        viz_path = Path(args.output) / "dataset_visualization.png"
        creator.visualize_dataset(dataset, output_path=viz_path)
    
    print("\nDataset creation complete!")


if __name__ == "__main__":
    main()