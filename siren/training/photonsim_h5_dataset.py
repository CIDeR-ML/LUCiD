"""
PhotonSim HDF5 Dataset Loader for JAX SIREN Training

This module loads PhotonSim HDF5 lookup tables and pre-sampled datasets
for training JAX-based SIREN networks.
"""

import numpy as np
import h5py
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Tuple, Dict, Optional, Iterator, Union
import logging

logger = logging.getLogger(__name__)

class PhotonSimH5Dataset:
    """
    Dataset class for loading PhotonSim HDF5 lookup tables and pre-sampled datasets.
    
    This class can work with:
    1. HDF5 lookup tables (photon_lookup_table.h5) from PhotonSim
    2. Pre-sampled datasets created by create_siren_dataset.py
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 16384,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize PhotonSim HDF5 dataset.
        
        Args:
            data_path: Path to HDF5 lookup table OR directory containing sampled dataset
            batch_size: Number of samples per batch
            val_split: Fraction of data to use for validation
            seed: Random seed for train/val split
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        
        # Determine data type and load accordingly
        if self.data_path.is_file() and self.data_path.suffix == '.h5':
            self._load_h5_lookup_table()
        elif self.data_path.is_dir():
            self._load_sampled_dataset()
        else:
            raise ValueError(f"Invalid data path: {data_path}. Must be HDF5 file or directory with sampled data.")
        
        # Create train/validation split
        self._create_split()
        
        logger.info(f"Dataset initialized with {len(self.train_inputs)} train, {len(self.val_inputs)} validation samples")
    
    def _load_h5_lookup_table(self):
        """Load PhotonSim HDF5 lookup table and create dataset."""
        logger.info(f"Loading PhotonSim HDF5 lookup table from {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # Load density table and coordinates
            density_table = f['data/photon_table_density'][:]
            energy_values = f['coordinates/energy_values'][:]
            angle_centers = f['coordinates/angle_centers'][:]
            distance_centers = f['coordinates/distance_centers'][:]
            
            # Load metadata
            meta = f['metadata']
            self.density_units = meta.attrs['density_units'].decode() if isinstance(meta.attrs['density_units'], bytes) else meta.attrs['density_units']
            self.energy_min = meta.attrs['energy_min']
            self.energy_max = meta.attrs['energy_max']
        
        # Create coordinate grids
        energy_grid, angle_grid, distance_grid = np.meshgrid(
            energy_values, angle_centers, distance_centers, indexing='ij'
        )
        
        # Flatten and filter
        energy_flat = energy_grid.flatten()
        angle_flat = angle_grid.flatten()
        distance_flat = distance_grid.flatten()
        density_flat = density_table.flatten()
        
        # Filter out zero density entries
        valid_mask = density_flat > 0
        
        logger.info(f"Filtering: {valid_mask.sum():,} / {len(valid_mask):,} samples "
                   f"({100*valid_mask.sum()/len(valid_mask):.1f}%)")
        
        # Create inputs and outputs
        self.inputs = np.column_stack([
            energy_flat[valid_mask],
            angle_flat[valid_mask], 
            distance_flat[valid_mask]
        ]).astype(np.float32)
        
        self.outputs = density_flat[valid_mask].astype(np.float32)
        
        # Set up metadata for compatibility
        self.input_ranges = {
            'energy': (energy_values.min(), energy_values.max()),
            'angle': (angle_centers.min(), angle_centers.max()),
            'distance': (distance_centers.min(), distance_centers.max())
        }
        
        self.output_stats = {
            'min': float(self.outputs.min()),
            'max': float(self.outputs.max()),
            'mean': float(self.outputs.mean()),
            'std': float(self.outputs.std())
        }
        
        self.normalize_inputs = True
        self.normalize_outputs = False
        self.log_transform_output = True
        self.n_samples = len(self.inputs)
        
        # Apply log transform to outputs
        self.outputs = np.log10(self.outputs + 1e-10)
        
        # Normalize inputs to [-1, 1]
        self._normalize_inputs()
        
        logger.info(f"Loaded H5 lookup table: {self.n_samples:,} valid samples")
    
    def _load_sampled_dataset(self):
        """Load pre-sampled dataset from numpy files."""
        logger.info(f"Loading pre-sampled dataset from {self.data_path}")
        
        # Load inputs and outputs
        inputs_file = self.data_path / "train_inputs.npy"
        outputs_file = self.data_path / "train_outputs.npy"
        metadata_file = self.data_path / "dataset_metadata.npz"
        
        if not all(f.exists() for f in [inputs_file, outputs_file, metadata_file]):
            raise FileNotFoundError(f"Missing dataset files in {self.data_path}")
        
        self.inputs = np.load(inputs_file).astype(np.float32)
        self.outputs = np.load(outputs_file).astype(np.float32)
        metadata = np.load(metadata_file, allow_pickle=True)
        
        # Extract metadata
        self.n_samples = int(metadata['n_samples'])
        self.input_ranges = metadata['input_ranges'].item()
        self.output_stats = metadata['output_stats'].item()
        self.normalize_inputs = bool(metadata['normalize_inputs'])
        self.normalize_outputs = bool(metadata['normalize_outputs'])
        self.log_transform_output = bool(metadata['log_transform_output'])
        self.density_units = str(metadata['density_units'])
        
        logger.info(f"Loaded sampled dataset: {len(self.inputs):,} samples")
    
    def _normalize_inputs(self):
        """Normalize inputs to [-1, 1] range."""
        normalized = self.inputs.copy()
        
        # Energy: [min_energy, max_energy] -> [-1, 1]
        e_min, e_max = self.input_ranges['energy']
        normalized[:, 0] = 2 * (self.inputs[:, 0] - e_min) / (e_max - e_min) - 1
        
        # Angle: [min_angle, max_angle] -> [-1, 1]
        a_min, a_max = self.input_ranges['angle']
        normalized[:, 1] = 2 * (self.inputs[:, 1] - a_min) / (a_max - a_min) - 1
        
        # Distance: [min_distance, max_distance] -> [-1, 1]
        d_min, d_max = self.input_ranges['distance']
        normalized[:, 2] = 2 * (self.inputs[:, 2] - d_min) / (d_max - d_min) - 1
        
        self.inputs = normalized
    
    def _create_split(self):
        """Create train/validation split."""
        np.random.seed(self.seed)
        
        n_val = int(self.val_split * len(self.inputs))
        n_train = len(self.inputs) - n_val
        
        # Random permutation for splitting
        indices = np.random.permutation(len(self.inputs))
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        self.train_inputs = self.inputs[train_indices]
        self.train_outputs = self.outputs[train_indices]
        self.val_inputs = self.inputs[val_indices]
        self.val_outputs = self.outputs[val_indices]
        
        logger.info(f"Split: {len(self.train_inputs)} train, {len(self.val_inputs)} validation")
    
    def get_batch_iterator(self, shuffle=True) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Get batch iterator for training.
        
        Args:
            shuffle: Whether to shuffle the training data
            
        Yields:
            Tuples of (batch_inputs, batch_outputs)
        """
        n_train = len(self.train_inputs)
        
        if shuffle:
            indices = np.random.permutation(n_train)
        else:
            indices = np.arange(n_train)
        
        for i in range(0, n_train, self.batch_size):
            end_idx = min(i + self.batch_size, n_train)
            batch_indices = indices[i:end_idx]
            
            batch_inputs = jnp.array(self.train_inputs[batch_indices])
            batch_outputs = jnp.array(self.train_outputs[batch_indices])
            
            yield batch_inputs, batch_outputs
    
    def get_validation_data(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get validation data.
        
        Returns:
            Tuple of (val_inputs, val_outputs)
        """
        return jnp.array(self.val_inputs), jnp.array(self.val_outputs)
    
    def get_sample_batch(self, n_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get a sample batch for evaluation.
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            Tuple of (sample_inputs, sample_outputs)
        """
        if n_samples >= len(self.val_inputs):
            return jnp.array(self.val_inputs), jnp.array(self.val_outputs)
        else:
            indices = np.random.choice(len(self.val_inputs), n_samples, replace=False)
            return jnp.array(self.val_inputs[indices]), jnp.array(self.val_outputs[indices])
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'n_samples': self.n_samples,
            'n_train': len(self.train_inputs),
            'n_validation': len(self.val_inputs),
            'batch_size': self.batch_size,
            'input_shape': self.inputs.shape,
            'output_shape': self.outputs.shape,
            'input_ranges': self.input_ranges,
            'output_stats': self.output_stats,
            'normalize_inputs': self.normalize_inputs,
            'normalize_outputs': self.normalize_outputs,
            'log_transform_output': self.log_transform_output,
            'density_units': self.density_units,
            'val_split': self.val_split,
        }
    
    def get_sample_for_visualization(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a sample of the dataset for visualization (in numpy format).
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            Tuple of (sample_inputs, sample_outputs) as numpy arrays
        """
        if n_samples >= len(self.inputs):
            return self.inputs, self.outputs
        else:
            indices = np.random.choice(len(self.inputs), n_samples, replace=False)
            return self.inputs[indices], self.outputs[indices]


def load_photonsim_h5_dataset(
    data_path: str,
    **kwargs
) -> PhotonSimH5Dataset:
    """
    Convenience function to create a PhotonSim HDF5 dataset.
    
    Args:
        data_path: Path to HDF5 lookup table or sampled dataset directory
        **kwargs: Additional arguments for PhotonSimH5Dataset
    
    Returns:
        PhotonSimH5Dataset instance
    """
    return PhotonSimH5Dataset(data_path, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default path assuming script is run from diffCherenkov root
        data_path = "../PhotonSim/output/3d_lookup_table_density/photon_lookup_table.h5"
    
    print(f"Loading PhotonSim dataset from: {data_path}")
    
    # Create dataset
    dataset = load_photonsim_h5_dataset(
        data_path,
        batch_size=1024,
    )
    
    # Print statistics
    stats = dataset.get_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        if key not in ['input_ranges', 'output_stats']:
            print(f"  {key}: {value}")
    
    print("\nInput ranges:")
    for dim, range_val in stats['input_ranges'].items():
        print(f"  {dim}: {range_val}")
    
    print("\nOutput statistics:")
    for stat, value in stats['output_stats'].items():
        print(f"  {stat}: {value:.6f}")
    
    # Test batch iterator
    print(f"\nTesting batch iterator...")
    batch_count = 0
    for inputs, outputs in dataset.get_batch_iterator():
        print(f"  Batch {batch_count}: inputs {inputs.shape}, outputs {outputs.shape}")
        batch_count += 1
        if batch_count >= 3:  # Only test first few batches
            break
    
    # Test validation data
    val_inputs, val_outputs = dataset.get_validation_data()
    print(f"\nValidation data: inputs {val_inputs.shape}, outputs {val_outputs.shape}")
    
    print("\nDataset testing complete!")