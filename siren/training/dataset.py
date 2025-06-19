"""
PhotonSim Dataset Module for SIREN Training

This module provides dataset classes for loading and managing PhotonSim data,
including both HDF5 lookup tables and pre-sampled datasets.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import h5py
import numpy as np
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class PhotonSimDataset:
    """
    Base class for PhotonSim datasets that can load from either
    HDF5 lookup tables or pre-sampled dataset directories.
    """
    
    def __init__(self, data_path: Union[str, Path], val_split: float = 0.1):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to HDF5 file or dataset directory
            val_split: Fraction of data to use for validation
        """
        self.data_path = Path(data_path)
        self.val_split = val_split
        self.data_type = None
        self.data = {}
        self.normalized_bounds = {}
        
        # Load data based on type
        if self.data_path.is_file() and self.data_path.suffix == '.h5':
            self._load_h5_lookup_table()
        elif self.data_path.is_dir():
            self._load_sampled_dataset()
        else:
            raise ValueError(f"Invalid data path: {data_path}")
            
    def _load_h5_lookup_table(self):
        """Load data from HDF5 lookup table and prepare for training."""
        logger.info(f"Loading HDF5 lookup table from {self.data_path}")
        self.data_type = 'h5_lookup'
        
        with h5py.File(self.data_path, 'r') as f:
            # Load average table (photons per event)
            average_table = f['data/photon_table_average'][:]
            
            # Load coordinates
            energy_centers = f['coordinates/energy_centers'][:]
            angle_centers = f['coordinates/angle_centers'][:]
            distance_centers = f['coordinates/distance_centers'][:]
            
            # Get metadata
            metadata = dict(f['metadata'].attrs)
            
        # Create coordinate grids
        E, A, D = np.meshgrid(energy_centers, angle_centers, distance_centers, indexing='ij')
        
        # Flatten for training
        self.data['inputs'] = np.stack([
            E.flatten(),
            A.flatten(), 
            D.flatten()
        ], axis=-1).astype(np.float32)
        
        self.data['targets'] = average_table.flatten()[:, np.newaxis].astype(np.float32)
        
        # Store metadata
        self.metadata = metadata
        self.energy_range = (energy_centers.min(), energy_centers.max())
        self.angle_range = (angle_centers.min(), angle_centers.max())
        self.distance_range = (distance_centers.min(), distance_centers.max())
        
        logger.info(f"Loaded {len(self.data['inputs']):,} data points from lookup table")
        logger.info(f"Energy range: {self.energy_range[0]:.0f}-{self.energy_range[1]:.0f} MeV")
        logger.info(f"Angle range: {np.degrees(self.angle_range[0]):.1f}-{np.degrees(self.angle_range[1]):.1f} degrees")
        logger.info(f"Distance range: {self.distance_range[0]:.0f}-{self.distance_range[1]:.0f} mm")
        logger.info(f"Table type: {metadata.get('normalization', 'unknown')} ({metadata.get('average_units', 'unknown units')})") 
        
        # Normalize inputs and prepare bounds
        self._normalize_data()
        
    def _load_sampled_dataset(self):
        """Load pre-sampled dataset from directory."""
        logger.info(f"Loading sampled dataset from {self.data_path}")
        self.data_type = 'sampled'
        
        # Load data arrays
        inputs_path = self.data_path / 'inputs.npy'
        targets_path = self.data_path / 'targets.npy'
        
        if not inputs_path.exists() or not targets_path.exists():
            raise FileNotFoundError(f"Dataset files not found in {self.data_path}")
            
        self.data['inputs'] = np.load(inputs_path).astype(np.float32)
        self.data['targets'] = np.load(targets_path).astype(np.float32)
        
        # Load metadata if available
        metadata_path = self.data_path / 'metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
        # Load normalization bounds
        bounds_path = self.data_path / 'normalization_bounds.npz'
        if bounds_path.exists():
            bounds = np.load(bounds_path)
            self.normalized_bounds = {
                'input_min': bounds['input_min'],
                'input_max': bounds['input_max'],
                'target_min': bounds['target_min'],
                'target_max': bounds['target_max']
            }
            
        logger.info(f"Loaded {len(self.data['inputs']):,} data points from sampled dataset")
        
    def _normalize_data(self):
        """Normalize input and target data."""
        # Normalize inputs to [-1, 1]
        self.normalized_bounds['input_min'] = self.data['inputs'].min(axis=0)
        self.normalized_bounds['input_max'] = self.data['inputs'].max(axis=0)
        
        self.data['inputs_normalized'] = 2 * (
            (self.data['inputs'] - self.normalized_bounds['input_min']) / 
            (self.normalized_bounds['input_max'] - self.normalized_bounds['input_min'])
        ) - 1
        
        # Log-normalize targets for better training stability
        self.data['targets_log'] = np.log10(self.data['targets'] + 1e-10)
        self.normalized_bounds['target_min'] = self.data['targets_log'].min()
        self.normalized_bounds['target_max'] = self.data['targets_log'].max()
        
        # Normalize log targets to [0, 1] range for training
        self.data['targets_log_normalized'] = (
            (self.data['targets_log'] - self.normalized_bounds['target_min']) / 
            (self.normalized_bounds['target_max'] - self.normalized_bounds['target_min'])
        )
        
        # Filter data but keep 0.2% of zero values for better training coverage
        mask = self.data['targets'][:, 0] > 1e-10
        zero_mask = ~mask
        
        # Randomly select 0.2% of zero values to keep
        zero_indices = np.where(zero_mask)[0]
        if len(zero_indices) > 0:
            n_zeros_to_keep = max(1, int(len(zero_indices) * 0.002))
            rng = np.random.RandomState(42)
            zeros_to_keep = rng.choice(zero_indices, size=n_zeros_to_keep, replace=False)
            
            # Create final mask: all non-zeros + 0.2% of zeros
            final_mask = mask.copy()
            final_mask[zeros_to_keep] = True
            
            logger.info(f"Filtered data: keeping {mask.sum():,} non-zero values + {len(zeros_to_keep):,} zero values ({len(zeros_to_keep)/len(zero_indices)*100:.1f}% of zeros)")
            logger.info(f"Total data points: {final_mask.sum():,} / {len(final_mask):,} ({final_mask.sum()/len(final_mask)*100:.1f}%)")
            
            # Apply the filter
            self.data['inputs'] = self.data['inputs'][final_mask]
            self.data['targets'] = self.data['targets'][final_mask]
            self.data['inputs_normalized'] = self.data['inputs_normalized'][final_mask]
            self.data['targets_log'] = self.data['targets_log'][final_mask]
            self.data['targets_log_normalized'] = self.data['targets_log_normalized'][final_mask]
        
        # Create train/val split
        n_samples = len(self.data['inputs'])
        n_val = int(n_samples * self.val_split)
        
        # Random shuffle for split
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_samples)
        
        self.val_indices = indices[:n_val]
        self.train_indices = indices[n_val:]
        
        logger.info(f"Train samples: {len(self.train_indices):,}")
        logger.info(f"Validation samples: {len(self.val_indices):,}")
        
    def get_sample_input(self) -> jnp.ndarray:
        """Get a sample input for model initialization."""
        return jnp.array(self.data['inputs_normalized'][:1])
        
    def get_batch(
        self, 
        batch_size: int, 
        rng: jax.random.PRNGKey,
        split: str = 'train',
        normalized: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get a random batch of data.
        
        Args:
            batch_size: Number of samples in batch
            rng: JAX random key
            split: 'train' or 'val'
            normalized: Whether to return normalized data
            
        Returns:
            Tuple of (inputs, targets) arrays
        """
        # Select indices based on split
        if split == 'train':
            indices = self.train_indices
        else:
            indices = self.val_indices
            
        # Random sampling
        batch_indices = jax.random.choice(rng, indices, shape=(batch_size,))
        
        # Get data with consistent normalization
        if normalized:
            inputs = self.data['inputs_normalized'][batch_indices]
            # Always use log-normalized targets for consistency
            targets = self.data['targets_log_normalized'][batch_indices]
        else:
            inputs = self.data['inputs'][batch_indices]
            targets = self.data['targets'][batch_indices]
            
        return jnp.array(inputs), jnp.array(targets)
        
    def get_full_data(
        self, 
        split: str = 'train',
        normalized: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get full dataset for a given split.
        
        Args:
            split: 'train', 'val', or 'all'
            normalized: Whether to return normalized data
            
        Returns:
            Tuple of (inputs, targets) arrays
        """
        if split == 'train':
            indices = self.train_indices
        elif split == 'val':
            indices = self.val_indices
        else:
            indices = np.arange(len(self.data['inputs']))
            
        if normalized:
            inputs = self.data['inputs_normalized'][indices]
            # Use log-normalized targets for consistency with get_batch
            targets = self.data['targets_log_normalized'][indices]
        else:
            inputs = self.data['inputs'][indices]
            targets = self.data['targets'][indices]
            
        return inputs, targets
        
    def denormalize_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Convert normalized inputs back to original scale."""
        inputs_01 = (inputs + 1) / 2  # From [-1, 1] to [0, 1]
        return (
            inputs_01 * (self.normalized_bounds['input_max'] - self.normalized_bounds['input_min']) +
            self.normalized_bounds['input_min']
        )
        
    def denormalize_targets(self, targets_log: np.ndarray) -> np.ndarray:
        """Convert log-normalized targets back to original scale."""
        return 10 ** targets_log - 1e-10
        
    def denormalize_targets_from_normalized(self, targets_normalized: np.ndarray) -> np.ndarray:
        """Convert normalized log targets [0,1] back to original scale."""
        # First denormalize from [0,1] to log scale
        targets_log = (
            targets_normalized * (self.normalized_bounds['target_max'] - self.normalized_bounds['target_min']) +
            self.normalized_bounds['target_min']
        )
        # Then convert from log to linear scale
        return 10 ** targets_log - 1e-10
        
    @property
    def has_validation(self) -> bool:
        """Check if dataset has validation split."""
        return len(self.val_indices) > 0
        
    def save_sampled_dataset(self, output_dir: Path, n_samples: int = 1000000):
        """
        Save a sampled version of the dataset for faster loading.
        
        Args:
            output_dir: Directory to save dataset
            n_samples: Number of samples to save
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Sample data if needed
        if n_samples < len(self.data['inputs']):
            rng = np.random.RandomState(42)
            sample_indices = rng.choice(len(self.data['inputs']), n_samples, replace=False)
            inputs = self.data['inputs'][sample_indices]
            targets = self.data['targets'][sample_indices]
        else:
            inputs = self.data['inputs']
            targets = self.data['targets']
            
        # Save arrays
        np.save(output_dir / 'inputs.npy', inputs)
        np.save(output_dir / 'targets.npy', targets)
        
        # Save normalization bounds
        np.savez(
            output_dir / 'normalization_bounds.npz',
            input_min=self.normalized_bounds['input_min'],
            input_max=self.normalized_bounds['input_max'],
            target_min=self.normalized_bounds.get('target_min', 0),
            target_max=self.normalized_bounds.get('target_max', 1)
        )
        
        # Save metadata
        import json
        metadata = {
            'n_samples': len(inputs),
            'data_type': self.data_type,
            'source': str(self.data_path),
            **self.metadata
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved sampled dataset to {output_dir}")
        
    def get_total_counts_for_energy(self, energy: float) -> float:
        """
        Get total table counts for a given energy.
        
        Args:
            energy: Energy value in MeV
            
        Returns:
            Total sum of all table values for the closest energy bin
        """
        if self.data_type != 'h5_lookup':
            raise ValueError("This method only works with H5 lookup tables")
            
        # Load the necessary data if not already cached
        if not hasattr(self, '_cached_lookup_data'):
            with h5py.File(self.data_path, 'r') as f:
                self._cached_lookup_data = {
                    'average_table': f['data/photon_table_average'][:],
                    'energy_centers': f['coordinates/energy_centers'][:]
                }
                
        # Find the closest energy index
        energy_centers = self._cached_lookup_data['energy_centers']
        energy_idx = np.argmin(np.abs(energy_centers - energy))
        
        # Sum all values for this energy slice
        average_table = self._cached_lookup_data['average_table']
        total_counts = np.sum(average_table[energy_idx, :, :])
        
        logger.info(f"Energy {energy:.1f} MeV (closest bin: {energy_centers[energy_idx]:.1f} MeV) - Total average counts: {total_counts:.2e}")
        
        return total_counts