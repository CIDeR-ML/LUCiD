"""
PhotonSim Dataset Loader for JAX SIREN Training

This module provides functionality to load PhotonSim 3D lookup tables
and prepare them for training JAX-based SIREN networks, bypassing the
need for CProfSiren.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Tuple, Dict, Optional, Iterator
import logging

logger = logging.getLogger(__name__)

class PhotonSimDataset:
    """
    Dataset class for loading and preprocessing PhotonSim 3D lookup tables
    for SIREN training.
    """
    
    def __init__(
        self,
        table_path: str,
        batch_size: int = 1024*16,  # Smaller than CProfSiren's 16^5 for memory efficiency
        normalize_inputs: bool = True,
        normalize_output: bool = True,
        min_photon_count: float = 1.0,  # Minimum photon count to include
        max_distance: float = 5000.0,   # Max distance in mm
        max_angle: float = 0.2,         # Max angle in radians
    ):
        """
        Initialize PhotonSim dataset.
        
        Args:
            table_path: Path to directory containing photon_table_3d.npy and table_metadata.npz
            batch_size: Number of samples per batch
            normalize_inputs: Whether to normalize inputs to [-1, 1]
            normalize_output: Whether to normalize output
            min_photon_count: Minimum photon count to include in training
            max_distance: Maximum distance to include (mm)
            max_angle: Maximum angle to include (radians)
        """
        self.table_path = Path(table_path)
        self.batch_size = batch_size
        self.normalize_inputs = normalize_inputs
        self.normalize_output = normalize_output
        self.min_photon_count = min_photon_count
        self.max_distance = max_distance
        self.max_angle = max_angle
        
        # Load data
        self._load_data()
        
        # Prepare training dataset
        self._prepare_dataset()
    
    def _load_data(self):
        """Load the 3D photon table and metadata."""
        logger.info(f"Loading PhotonSim data from {self.table_path}")
        
        # Load 3D table
        table_file = self.table_path / "photon_table_3d.npy"
        if not table_file.exists():
            raise FileNotFoundError(f"Table file not found: {table_file}")
        
        self.photon_table = np.load(table_file)
        logger.info(f"Loaded photon table with shape: {self.photon_table.shape}")
        
        # Load metadata
        metadata_file = self.table_path / "table_metadata.npz"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        self.metadata = np.load(metadata_file)
        
        # Extract axis information
        self.energy_values = self.metadata['energy_values']      # MeV
        self.angle_centers = self.metadata['angle_centers']      # radians
        self.distance_centers = self.metadata['distance_centers'] # mm
        
        logger.info(f"Energy range: {self.energy_values[0]}-{self.energy_values[-1]} MeV")
        logger.info(f"Angle range: {self.angle_centers[0]:.3f}-{self.angle_centers[-1]:.3f} rad")
        logger.info(f"Distance range: {self.distance_centers[0]:.1f}-{self.distance_centers[-1]:.1f} mm")
        
        # Store original ranges for normalization
        self.energy_range = (self.energy_values.min(), self.energy_values.max())
        self.angle_range = (0.0, self.max_angle)  # Focus on small angles
        self.distance_range = (0.0, self.max_distance)  # Focus on main region
    
    def _prepare_dataset(self):
        """Prepare the dataset by flattening the 3D table and filtering."""
        logger.info("Preparing dataset...")
        
        # Create coordinate grids
        energy_grid, angle_grid, distance_grid = np.meshgrid(
            self.energy_values,
            self.angle_centers,
            self.distance_centers,
            indexing='ij'
        )
        
        # Flatten everything
        energy_flat = energy_grid.flatten()
        angle_flat = angle_grid.flatten()
        distance_flat = distance_grid.flatten()
        counts_flat = self.photon_table.flatten()
        
        # Apply filters
        valid_mask = (
            (counts_flat >= self.min_photon_count) &
            (angle_flat <= self.max_angle) &
            (distance_flat <= self.max_distance) &
            (counts_flat > 0)  # Remove zero entries
        )
        
        logger.info(f"Filtering: {valid_mask.sum():,} / {len(valid_mask):,} samples "
                   f"({100*valid_mask.sum()/len(valid_mask):.1f}%)")
        
        # Apply mask
        self.inputs = np.column_stack([
            energy_flat[valid_mask],
            angle_flat[valid_mask],
            distance_flat[valid_mask]
        ])
        self.outputs = counts_flat[valid_mask]
        
        # Normalize inputs to [-1, 1] if requested
        if self.normalize_inputs:
            self.inputs_normalized = self._normalize_inputs(self.inputs)
        else:
            self.inputs_normalized = self.inputs.copy()
        
        # Normalize outputs if requested
        if self.normalize_output:
            self.output_scale = np.max(self.outputs)
            self.outputs_normalized = self.outputs / self.output_scale
        else:
            self.output_scale = 1.0
            self.outputs_normalized = self.outputs.copy()
        
        # Convert to JAX arrays
        self.inputs_jax = jnp.array(self.inputs_normalized, dtype=jnp.float32)
        self.outputs_jax = jnp.array(self.outputs_normalized, dtype=jnp.float32)
        
        self.n_samples = len(self.inputs_jax)
        logger.info(f"Dataset prepared: {self.n_samples:,} training samples")
        logger.info(f"Input range: [{self.inputs_normalized.min():.3f}, {self.inputs_normalized.max():.3f}]")
        logger.info(f"Output range: [{self.outputs_normalized.min():.3f}, {self.outputs_normalized.max():.3f}]")
        
        # Store normalization parameters for later use
        self.normalization_params = {
            'energy_range': self.energy_range,
            'angle_range': self.angle_range,
            'distance_range': self.distance_range,
            'output_scale': self.output_scale,
            'normalize_inputs': self.normalize_inputs,
            'normalize_output': self.normalize_output
        }
    
    def _normalize_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize inputs to [-1, 1] range."""
        normalized = inputs.copy()
        
        # Energy: [min_energy, max_energy] -> [-1, 1]
        normalized[:, 0] = 2 * (inputs[:, 0] - self.energy_range[0]) / (self.energy_range[1] - self.energy_range[0]) - 1
        
        # Angle: [0, max_angle] -> [-1, 1]
        normalized[:, 1] = 2 * (inputs[:, 1] - self.angle_range[0]) / (self.angle_range[1] - self.angle_range[0]) - 1
        
        # Distance: [0, max_distance] -> [-1, 1]
        normalized[:, 2] = 2 * (inputs[:, 2] - self.distance_range[0]) / (self.distance_range[1] - self.distance_range[0]) - 1
        
        return normalized
    
    def denormalize_inputs(self, normalized_inputs: jnp.ndarray) -> jnp.ndarray:
        """Convert normalized inputs back to physical units."""
        if not self.normalize_inputs:
            return normalized_inputs
        
        denormalized = jnp.zeros_like(normalized_inputs)
        
        # Energy: [-1, 1] -> [min_energy, max_energy]
        denormalized = denormalized.at[:, 0].set(
            (normalized_inputs[:, 0] + 1) / 2 * (self.energy_range[1] - self.energy_range[0]) + self.energy_range[0]
        )
        
        # Angle: [-1, 1] -> [0, max_angle]
        denormalized = denormalized.at[:, 1].set(
            (normalized_inputs[:, 1] + 1) / 2 * (self.angle_range[1] - self.angle_range[0]) + self.angle_range[0]
        )
        
        # Distance: [-1, 1] -> [0, max_distance]
        denormalized = denormalized.at[:, 2].set(
            (normalized_inputs[:, 2] + 1) / 2 * (self.distance_range[1] - self.distance_range[0]) + self.distance_range[0]
        )
        
        return denormalized
    
    def denormalize_output(self, normalized_output: jnp.ndarray) -> jnp.ndarray:
        """Convert normalized output back to photon counts."""
        if not self.normalize_output:
            return normalized_output
        return normalized_output * self.output_scale
    
    def get_batch_iterator(self, shuffle: bool = True) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get an iterator over batches of training data."""
        n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size
        
        if shuffle:
            key = jax.random.PRNGKey(42)
            indices = jax.random.permutation(key, self.n_samples)
        else:
            indices = jnp.arange(self.n_samples)
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.n_samples)
            
            batch_indices = indices[start_idx:end_idx]
            batch_inputs = self.inputs_jax[batch_indices]
            batch_outputs = self.outputs_jax[batch_indices]
            
            yield batch_inputs, batch_outputs
    
    def get_validation_data(self, validation_fraction: float = 0.1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get a validation dataset."""
        n_val = int(self.n_samples * validation_fraction)
        
        # Use last samples for validation
        val_inputs = self.inputs_jax[-n_val:]
        val_outputs = self.outputs_jax[-n_val:]
        
        return val_inputs, val_outputs
    
    def get_sample_batch(self, n_samples: int = 1000) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get a small sample batch for testing."""
        key = jax.random.PRNGKey(123)
        indices = jax.random.choice(key, self.n_samples, (n_samples,), replace=False)
        
        return self.inputs_jax[indices], self.outputs_jax[indices]
    
    def save_normalization_params(self, path: str):
        """Save normalization parameters for later use in inference."""
        np.savez(path, **self.normalization_params)
        logger.info(f"Saved normalization parameters to {path}")
    
    @classmethod
    def load_normalization_params(cls, path: str) -> Dict:
        """Load normalization parameters."""
        return dict(np.load(path))
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'n_samples': self.n_samples,
            'n_energies': len(self.energy_values),
            'n_angles': len(self.angle_centers),
            'n_distances': len(self.distance_centers),
            'total_photons': float(self.outputs.sum()),
            'mean_photons': float(self.outputs.mean()),
            'std_photons': float(self.outputs.std()),
            'max_photons': float(self.outputs.max()),
            'energy_range': self.energy_range,
            'angle_range': self.angle_range,
            'distance_range': self.distance_range,
            'sparsity': 100 * self.n_samples / self.photon_table.size,
        }


def create_photonsim_dataset(
    table_path: str,
    **kwargs
) -> PhotonSimDataset:
    """
    Convenience function to create a PhotonSim dataset.
    
    Args:
        table_path: Path to PhotonSim 3D table directory
        **kwargs: Additional arguments for PhotonSimDataset
    
    Returns:
        PhotonSimDataset instance
    """
    return PhotonSimDataset(table_path, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        table_path = sys.argv[1]
    else:
        # Default path assuming script is run from diffCherenkov root
        table_path = "../PhotonSim/3d_lookup_table"
    
    print(f"Loading PhotonSim dataset from: {table_path}")
    
    # Create dataset
    dataset = create_photonsim_dataset(
        table_path,
        batch_size=1024,
        min_photon_count=10.0,  # Focus on regions with decent statistics
        max_distance=3000.0,    # Focus on main region
        max_angle=0.1           # Focus on forward Cherenkov cone
    )
    
    # Print statistics
    stats = dataset.get_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
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