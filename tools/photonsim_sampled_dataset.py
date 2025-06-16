"""
PhotonSim Sampled Dataset Loader for JAX SIREN Training

This module loads pre-sampled PhotonSim datasets (created by PhotonSim tools)
instead of loading the full 3D lookup table.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Tuple, Dict, Optional, Iterator
import logging

logger = logging.getLogger(__name__)

class PhotonSimSampledDataset:
    """
    Dataset class for loading pre-sampled PhotonSim datasets for SIREN training.
    
    This class is designed to work with datasets created by PhotonSim's
    create_siren_dataset.py tool, which provides pre-sampled and normalized
    training data.
    """
    
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 16384,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize PhotonSim sampled dataset.
        
        Args:
            dataset_path: Path to directory containing train_inputs.npy, train_outputs.npy, 
                         and dataset_metadata.npz
            batch_size: Number of samples per batch
            val_split: Fraction of data to use for validation
            seed: Random seed for train/val split
        """
        self.dataset_path = Path(dataset_path)
        self.table_path = self.dataset_path  # For compatibility with training script
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        
        # Load data
        self._load_data()
        
        # Create train/validation split
        self._create_split()
        
        logger.info(f"Dataset initialized with {len(self.train_inputs)} train, {len(self.val_inputs)} validation samples")
    
    def _load_data(self):
        """Load the pre-sampled dataset."""
        logger.info(f"Loading PhotonSim sampled dataset from {self.dataset_path}")
        
        # Load inputs and outputs
        inputs_file = self.dataset_path / "train_inputs.npy"
        outputs_file = self.dataset_path / "train_outputs.npy"
        metadata_file = self.dataset_path / "dataset_metadata.npz"
        
        if not inputs_file.exists():
            raise FileNotFoundError(f"Inputs file not found: {inputs_file}")
        if not outputs_file.exists():
            raise FileNotFoundError(f"Outputs file not found: {outputs_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        self.inputs = np.load(inputs_file).astype(np.float32)
        self.outputs = np.load(outputs_file).astype(np.float32)
        self.metadata = np.load(metadata_file, allow_pickle=True)
        
        logger.info(f"Loaded dataset with {len(self.inputs)} samples")
        logger.info(f"Input shape: {self.inputs.shape}")
        logger.info(f"Output shape: {self.outputs.shape}")
        
        # Extract metadata
        self.n_samples = int(self.metadata['n_samples'])
        self.input_ranges = self.metadata['input_ranges'].item()
        self.output_stats = self.metadata['output_stats'].item()
        self.normalize_inputs = bool(self.metadata['normalize_inputs'])
        self.normalize_output = bool(self.metadata['normalize_outputs'])
        self.log_transform_output = bool(self.metadata['log_transform_output'])
        self.density_units = str(self.metadata['density_units'])
        
        logger.info(f"Energy range: {self.input_ranges['energy']} MeV")
        logger.info(f"Angle range: {self.input_ranges['angle']} rad")
        logger.info(f"Distance range: {self.input_ranges['distance']} mm")
        logger.info(f"Output units: {self.density_units}")
        logger.info(f"Log transformed: {self.log_transform_output}")
        
        # Set up normalization parameters for compatibility with SIREN trainer
        self.normalization_params = {
            'normalize_inputs': self.normalize_inputs,
            'normalize_output': self.normalize_output,
            'energy_range': self.input_ranges['energy'],
            'angle_range': self.input_ranges['angle'],
            'distance_range': self.input_ranges['distance'],
            'output_scale': 1.0,  # Outputs are already in the desired scale
            'log_transform_output': self.log_transform_output,
            'output_stats': self.output_stats
        }
        
        # Add compatibility attributes expected by training script
        self.min_photon_count = 1.0  # Not used for sampled data
        self.max_distance = self.input_ranges['distance'][1]
        self.max_angle = self.input_ranges['angle'][1]
    
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
            'normalize_output': self.normalize_output,
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