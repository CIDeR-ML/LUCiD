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
            # Load density table
            density_table = f['data/photon_table_density'][:]
            
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
        
        self.data['targets'] = density_table.flatten()[:, np.newaxis].astype(np.float32)
        
        # Remove zero or very small values (optional, for better training)
        mask = self.data['targets'][:, 0] > 1e-10
        self.data['inputs'] = self.data['inputs'][mask]
        self.data['targets'] = self.data['targets'][mask]
        
        # Store metadata
        self.metadata = metadata
        self.energy_range = (energy_centers.min(), energy_centers.max())
        self.angle_range = (angle_centers.min(), angle_centers.max())
        self.distance_range = (distance_centers.min(), distance_centers.max())
        
        logger.info(f"Loaded {len(self.data['inputs']):,} data points from lookup table")
        logger.info(f"Energy range: {self.energy_range[0]:.0f}-{self.energy_range[1]:.0f} MeV")
        logger.info(f"Angle range: {np.degrees(self.angle_range[0]):.1f}-{np.degrees(self.angle_range[1]):.1f} degrees")
        logger.info(f"Distance range: {self.distance_range[0]:.0f}-{self.distance_range[1]:.0f} mm")
        
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
        
        # Get data
        if normalized:
            inputs = self.data['inputs_normalized'][batch_indices]
            targets = self.data['targets_log'][batch_indices]
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
            targets = self.data['targets_log'][indices]
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
        

class PhotonSimTableDataset(PhotonSimDataset):
    """
    Specialized dataset for PhotonSim lookup tables with additional
    features for importance sampling and adaptive batch generation.
    """
    
    def __init__(
        self, 
        data_path: Union[str, Path], 
        val_split: float = 0.1,
        importance_sampling: bool = True,
        log_transform_targets: bool = True
    ):
        """
        Initialize table dataset with advanced features.
        
        Args:
            data_path: Path to HDF5 lookup table
            val_split: Validation split fraction
            importance_sampling: Whether to use importance sampling
            log_transform_targets: Whether to log-transform targets
        """
        self.importance_sampling = importance_sampling
        self.log_transform_targets = log_transform_targets
        
        super().__init__(data_path, val_split)
        
        if importance_sampling:
            self._compute_sampling_weights()
            
    def _compute_sampling_weights(self):
        """Compute importance sampling weights based on target values."""
        # Use target values as sampling weights (higher values = higher probability)
        weights = self.data['targets'][:, 0]
        
        # Apply sqrt to reduce extreme weighting
        weights = np.sqrt(weights)
        
        # Normalize
        self.sampling_weights = weights / weights.sum()
        
        logger.info("Computed importance sampling weights")
        
    def get_batch(
        self, 
        batch_size: int, 
        rng: jax.random.PRNGKey,
        split: str = 'train',
        normalized: bool = True,
        use_importance: Optional[bool] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get batch with optional importance sampling.
        
        Args:
            batch_size: Batch size
            rng: Random key
            split: Data split
            normalized: Use normalized data
            use_importance: Override importance sampling setting
            
        Returns:
            Batch of (inputs, targets)
        """
        use_importance = use_importance if use_importance is not None else self.importance_sampling
        
        if use_importance and hasattr(self, 'sampling_weights'):
            # Importance sampling
            indices = self.train_indices if split == 'train' else self.val_indices
            weights = self.sampling_weights[indices]
            weights = weights / weights.sum()
            
            batch_indices = np.random.choice(
                indices, 
                size=batch_size, 
                p=weights
            )
        else:
            # Regular sampling
            return super().get_batch(batch_size, rng, split, normalized)
            
        # Get data
        if normalized:
            inputs = self.data['inputs_normalized'][batch_indices]
            targets = self.data['targets_log'][batch_indices]
        else:
            inputs = self.data['inputs'][batch_indices]
            targets = self.data['targets'][batch_indices]
            
        return jnp.array(inputs), jnp.array(targets)