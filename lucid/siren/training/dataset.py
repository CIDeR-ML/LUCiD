"""
PhotonSim Dataset Module for SIREN Training

This module provides dataset classes for loading and managing PhotonSim data,
including both HDF5 lookup tables and pre-sampled datasets.
"""
from __future__ import annotations

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
    
    def __init__(self, data_path: Union[str, Path], val_split: float = 0.1,
                 zero_threshold: float = 1e-2,
                 zero_keep_frac: float = 0.002,
                 energy_balance: str = 'none',
                 target_importance: float = 0.0):
        """
        Initialize dataset.

        Args:
            data_path: Path to HDF5 file or dataset directory
            val_split: Fraction of data to use for validation
            zero_threshold: Targets below this value are treated as "zero".
                Used both as the log offset in ``log10(targets + zero_threshold)``
                and as the filter cutoff. Samples with ``target > zero_threshold``
                are kept in full; a fraction of the rest is kept as anchors
                for the low-density tail. Must be > 0. Default 1e-2.
            zero_keep_frac: Fraction of below-threshold samples to keep as
                training anchors. 0 drops all of them, 1 keeps everything.
                Default 0.002 (0.2%).
            energy_balance: How to weight samples across the energy grid.
                'none' (default) — uniform across raw samples (low-E
                over-represented because the grid has more low-E points).
                'uniform' — each energy gets equal probability per batch.
                'log_uniform' — uniform in log(E) (each decade weighted equally).
            target_importance: Mixture coefficient β ∈ [0, 1] for
                importance-sampling on the target value within each energy.
                Within each energy the per-sample weight is
                ``(1-β)·uniform + β·target_normalised``. β=0 (default) leaves
                sampling uniform across (angle, s/s_max) bins; β=1 weights
                samples by their target value (so the sharp Cherenkov ring at
                high E gets many more samples than the surrounding empty
                space). Combines cleanly with ``energy_balance``: the two
                knobs act on orthogonal axes (energy vs. within-energy shape).
        """
        if zero_threshold <= 0:
            raise ValueError(
                f"zero_threshold must be > 0 (got {zero_threshold!r}); "
                f"it acts as the log offset and a value of 0 makes "
                f"log10(target) blow up for the empty bins."
            )
        if not 0.0 <= zero_keep_frac <= 1.0:
            raise ValueError(
                f"zero_keep_frac must be in [0, 1] (got {zero_keep_frac!r})."
            )
        if energy_balance not in ('none', 'uniform', 'log_uniform'):
            raise ValueError(
                f"energy_balance must be one of 'none' / 'uniform' / "
                f"'log_uniform' (got {energy_balance!r})."
            )
        if not 0.0 <= target_importance <= 1.0:
            raise ValueError(
                f"target_importance must be in [0, 1] (got {target_importance!r})."
            )
        self.data_path = Path(data_path)
        self.val_split = val_split
        self.zero_threshold = float(zero_threshold)
        self.zero_keep_frac = float(zero_keep_frac)
        self.energy_balance = energy_balance
        self.target_importance = float(target_importance)
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
            # Auto-detect data type based on available datasets
            if 'data/dedx_table_average' in f:
                # This is a dEdx lookup table
                self.table_type = 'dedx'
                average_table = f['data/dedx_table_average'][:]
                energy_centers = f['coordinates/energy_centers'][:]
                dedx_centers = f['coordinates/dedx_centers'][:]
                distance_centers = f['coordinates/distance_centers'][:]
                second_dim_centers = dedx_centers
                second_dim_name = 'dedx'
            elif 'data/photon_table_average' in f:
                # This is a photon lookup table
                self.table_type = 'photon'
                average_table = f['data/photon_table_average'][:]
                energy_centers = f['coordinates/energy_centers'][:]
                angle_centers = f['coordinates/angle_centers'][:]
                distance_centers = f['coordinates/distance_centers'][:]
                second_dim_centers = angle_centers
                second_dim_name = 'angle'
            else:
                raise ValueError(f"Unknown lookup table format in {self.data_path}")

            # Get metadata
            metadata = dict(f['metadata'].attrs)

        # Create coordinate grids
        E, X, D = np.meshgrid(energy_centers, second_dim_centers, distance_centers, indexing='ij')

        # Flatten for training
        self.data['inputs'] = np.stack([
            E.flatten(),
            X.flatten(),
            D.flatten()
        ], axis=-1).astype(np.float32)

        self.data['targets'] = average_table.flatten()[:, np.newaxis].astype(np.float32)

        # Store metadata
        self.metadata = metadata
        self.energy_range = (energy_centers.min(), energy_centers.max())
        self.distance_range = (distance_centers.min(), distance_centers.max())

        # Distance axis label: new tables use s/s_max ∈ [0,1] and tag it via
        # metadata.attrs['distance_axis']. Older tables had absolute s in mm.
        dist_axis = metadata.get('distance_axis', b'')
        if isinstance(dist_axis, bytes):
            dist_axis = dist_axis.decode()
        if dist_axis == 's_over_smax':
            dist_label = f"s/s_max range: {self.distance_range[0]:.3f}-{self.distance_range[1]:.3f}"
        else:
            dist_label = f"Distance range: {self.distance_range[0]:.0f}-{self.distance_range[1]:.0f} mm"

        # Store second dimension range with appropriate name
        if self.table_type == 'dedx':
            self.dedx_range = (second_dim_centers.min(), second_dim_centers.max())
            self.angle_range = None  # Not applicable
            logger.info(f"Loaded {len(self.data['inputs']):,} data points from dE/dx lookup table")
            logger.info(f"Energy range: {self.energy_range[0]:.0f}-{self.energy_range[1]:.0f} MeV")
            logger.info(f"dE/dx range: {self.dedx_range[0]:.1f}-{self.dedx_range[1]:.1f} keV/mm")
            logger.info(dist_label)
        else:
            self.angle_range = (second_dim_centers.min(), second_dim_centers.max())
            self.dedx_range = None  # Not applicable
            logger.info(f"Loaded {len(self.data['inputs']):,} data points from photon lookup table")
            logger.info(f"Energy range: {self.energy_range[0]:.0f}-{self.energy_range[1]:.0f} MeV")
            logger.info(f"Angle range: {np.degrees(self.angle_range[0]):.1f}-{np.degrees(self.angle_range[1]):.1f} degrees")
            logger.info(dist_label)

        logger.info(f"Table type: {self.table_type} - {metadata.get('normalization', 'unknown')} ({metadata.get('average_units', 'unknown units')})")

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
        
        # Log-normalize targets: offset avoids log(0) while staying small relative to typical target values
        self.data['targets_log'] = np.log10(self.data['targets'] + self.zero_threshold)
        self.normalized_bounds['target_min'] = self.data['targets_log'].min()
        self.normalized_bounds['target_max'] = self.data['targets_log'].max()
        
        # Normalize log targets to [0, 1] range for training
        self.data['targets_log_normalized'] = (
            (self.data['targets_log'] - self.normalized_bounds['target_min']) / 
            (self.normalized_bounds['target_max'] - self.normalized_bounds['target_min'])
        )
        
        # Filter data but keep a fraction of zero values as low-density anchors.
        # "Zero" is defined as target <= self.zero_threshold (which is also the
        # log offset above — the two are paired, since anything below the
        # offset is indistinguishable after the log transform).
        mask = self.data['targets'][:, 0] > self.zero_threshold
        zero_mask = ~mask

        zero_indices = np.where(zero_mask)[0]
        if len(zero_indices) > 0:
            # 0.0 → drop all zeros; 1.0 → keep everything; otherwise random
            # subsample. ceil so a tiny non-zero fraction still keeps at least 1.
            if self.zero_keep_frac >= 1.0:
                zeros_to_keep = zero_indices
            elif self.zero_keep_frac <= 0.0:
                zeros_to_keep = np.array([], dtype=zero_indices.dtype)
            else:
                n_zeros_to_keep = max(1,
                    int(np.ceil(len(zero_indices) * self.zero_keep_frac)))
                rng = np.random.RandomState(42)
                zeros_to_keep = rng.choice(zero_indices,
                                           size=n_zeros_to_keep, replace=False)

            final_mask = mask.copy()
            final_mask[zeros_to_keep] = True

            kept_frac_pct = (len(zeros_to_keep) / len(zero_indices) * 100
                             if len(zero_indices) else 0.0)
            logger.info(f"Filtered data: keeping {mask.sum():,} non-zero values + {len(zeros_to_keep):,} zero values ({kept_frac_pct:.2f}% of zeros)")
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

        # Build per-sample sampling weights from two orthogonal knobs:
        #   - energy_balance ('none' / 'uniform' / 'log_uniform'): controls
        #     the relative weight of each energy point in the grid.
        #   - target_importance (β ∈ [0, 1]): within each energy, mixes
        #     uniform sampling with target-weighted importance sampling.
        # The two are decoupled by per-energy-normalising the target
        # component before combining with the energy component.
        self.train_weights = None
        self.val_weights = None

        need_weights = (self.energy_balance != 'none'
                        or self.target_importance > 0.0)
        if need_weights:
            energies = self.data['inputs'][:, 0]
            targets  = self.data['targets'][:, 0].astype(np.float64)
            unique_e, inv = np.unique(energies, return_inverse=True)
            counts = np.bincount(inv).astype(np.float64)

            # --- Energy-level weight (one value per energy) -----------------
            if self.energy_balance == 'uniform':
                e_weight = 1.0 / counts
            elif self.energy_balance == 'log_uniform':
                log_e = np.log(unique_e.astype(np.float64))
                widths = np.empty_like(log_e)
                widths[1:-1] = 0.5 * (log_e[2:] - log_e[:-2])
                widths[0]    = log_e[1]  - log_e[0]
                widths[-1]   = log_e[-1] - log_e[-2]
                e_weight = widths / counts
            else:  # 'none' but target_importance > 0: keep row-uniform energy mass
                e_weight = np.ones_like(counts)

            # --- Per-energy shape: (1-β) · uniform + β · target ------------
            beta = self.target_importance
            if beta > 0.0:
                # Per-energy normalisation of target weights and of the
                # uniform component decouples the within-energy "shape" knob
                # from the across-energy "energy_balance" knob.
                target_sum_per_e = np.bincount(inv,
                                                weights=np.maximum(targets, 0.0))
                # Per-sample target component, normalised so sum-over-samples
                # at each energy = 1.
                with np.errstate(divide='ignore', invalid='ignore'):
                    per_sample_target = np.where(
                        target_sum_per_e[inv] > 0,
                        np.maximum(targets, 0.0) / target_sum_per_e[inv],
                        0.0,
                    )
                # Per-sample uniform component, also normalised per energy.
                per_sample_uniform = 1.0 / counts[inv]
                shape_w = (1.0 - beta) * per_sample_uniform + beta * per_sample_target
            else:
                shape_w = 1.0 / counts[inv]   # uniform-within-energy

            # Combine: energy-level mass × within-energy shape.
            per_sample = e_weight[inv] * counts[inv] * shape_w
            # (e_weight[inv] * counts[inv]) is just the desired energy
            # mass spread evenly per sample at that energy; shape_w then
            # redistributes within the energy according to the mixture.

            self.train_weights = per_sample[self.train_indices]
            self.val_weights   = per_sample[self.val_indices]
            self.train_weights = self.train_weights / self.train_weights.sum()
            self.val_weights   = self.val_weights   / self.val_weights.sum()

            logger.info(
                "Sampling weights: energy_balance='%s', target_importance=%.3g "
                "across %d unique energies (%.0f–%.0f MeV).",
                self.energy_balance, self.target_importance,
                len(unique_e), unique_e[0], unique_e[-1],
            )

        # Precompute device-side caches so get_batch doesn't host→device
        # transfer the 50M-element indices/weights arrays every step. For
        # weighted sampling we also precompute the cumulative-sum once and
        # roll our own inverse-CDF lookup: jax.random.choice(..., p=p) would
        # recompute cumsum(p) over all ~50M samples on EVERY call (no JIT
        # caching across calls since get_batch is invoked outside a jit
        # boundary), which dominates per-step cost.
        self._train_indices_jnp = jnp.asarray(self.train_indices, dtype=jnp.int32)
        self._val_indices_jnp   = jnp.asarray(self.val_indices,   dtype=jnp.int32)
        if self.train_weights is not None:
            self._train_cdf_jnp = jnp.cumsum(
                jnp.asarray(self.train_weights, dtype=jnp.float32))
            self._val_cdf_jnp   = jnp.cumsum(
                jnp.asarray(self.val_weights,   dtype=jnp.float32))
        else:
            self._train_cdf_jnp = None
            self._val_cdf_jnp   = None

    def get_sample_input(self) -> jax.Array:
        """Get a sample input for model initialization."""
        return jnp.array(self.data['inputs_normalized'][:1])
        
    def get_batch(
        self, 
        batch_size: int, 
        rng: jax.random.PRNGKey,
        split: str = 'train',
        normalized: bool = True
    ) -> Tuple[jax.Array, jax.Array]:
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
        # Select indices based on split. Indices and CDFs are device-cached
        # so we don't pay the ~200 MB host→device transfer every batch.
        if split == 'train':
            indices_jnp = self._train_indices_jnp
            cdf_jnp     = self._train_cdf_jnp
        else:
            indices_jnp = self._val_indices_jnp
            cdf_jnp     = self._val_cdf_jnp

        if cdf_jnp is None:
            batch_indices = jax.random.choice(rng, indices_jnp,
                                              shape=(batch_size,))
        else:
            # Weighted inverse-CDF lookup with a precomputed cumulative sum.
            # Equivalent to jax.random.choice(p=p, replace=True) but without
            # recomputing cumsum(p) over the full sample array on every call.
            u = jax.random.uniform(rng, (batch_size,)) * cdf_jnp[-1]
            local = jnp.searchsorted(cdf_jnp, u)
            local = jnp.clip(local, 0, indices_jnp.shape[0] - 1)
            batch_indices = indices_jnp[local]
        
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
        return 10 ** targets_log - self.zero_threshold

    def denormalize_targets_from_normalized(self, targets_normalized: np.ndarray) -> np.ndarray:
        """Convert normalized log targets [0,1] back to original scale."""
        # First denormalize from [0,1] to log scale
        targets_log = (
            targets_normalized * (self.normalized_bounds['target_max'] - self.normalized_bounds['target_min']) +
            self.normalized_bounds['target_min']
        )
        # Then convert from log to linear scale
        return 10 ** targets_log - self.zero_threshold
        
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