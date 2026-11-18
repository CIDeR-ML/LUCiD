"""
PhotonSim Dataset Module for SIREN Training

This module loads a PhotonSim HDF5 lookup table (as ``lucid-build-photon-table`` /
``lucid-build-dedx-table`` write it) and serves normalized training batches.
:func:`table_curves` reads the table's fitted s_max(E) and N_photons(E) back as ``emitter.json``
blocks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union
import h5py
import numpy as np
import jax
import jax.numpy as jnp

from lucid.siren.emitter import (SMAX_PARAMS, decode_target, encode_target, normalize_inputs,
                                 target_range)

logger = logging.getLogger(__name__)


def _attr(v):
    """An h5 attribute as a plain Python value."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def table_curves(attrs: dict, kind: str):
    """``(s_max_mm, n_photons)``: a table's fitted curves as ``emitter.json`` blocks.

    ``n_photons`` is the Cherenkov photon-count fit, None for a dE/dx table.
    """
    attrs = {k: _attr(v) for k, v in attrs.items()}
    smax = {k[len('smax_'):]: v for k, v in attrs.items() if k.startswith('smax_')}
    form = smax.pop('form')
    s_max_mm = {'form': form, 'params': {p: float(smax.pop(p)) for p in SMAX_PARAMS[form]}, **smax}
    if kind == 'dedx':
        return s_max_mm, None
    return s_max_mm, {k[len('nphot_'):]: v for k, v in attrs.items() if k.startswith('nphot_')}


class PhotonSimDataset:
    """
    A PhotonSim photon or dE/dx lookup table, normalized and split for training.
    """
    
    def __init__(self, data_path: Union[str, Path], val_split: float = 0.1,
                 zero_threshold: float = 1e-2,
                 zero_keep_frac: float = 0.002,
                 energy_balance: str = 'none',
                 target_importance: float = 0.0):
        """
        Initialize dataset.

        Args:
            data_path: Path to the HDF5 lookup table
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
        self.data = {}
        self.normalized_bounds = {}
        
        if not (self.data_path.is_file() and self.data_path.suffix == '.h5'):
            raise ValueError(f"Not an .h5 lookup table: {data_path}")
        self._load_h5_lookup_table()
            
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
            elif 'data/photon_table_average' in f:
                # This is a photon lookup table
                self.table_type = 'photon'
                average_table = f['data/photon_table_average'][:]
                energy_centers = f['coordinates/energy_centers'][:]
                angle_centers = f['coordinates/angle_centers'][:]
                distance_centers = f['coordinates/distance_centers'][:]
                second_dim_centers = angle_centers
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
        self.kind = 'dedx' if self.table_type == 'dedx' else 'cherenkov'      # the emitter kind
        if self.kind == 'cherenkov':
            # The emitter's N_photons(E) is the table's fit; check it before any training, not
            # when the finished run is written.
            form = _attr(metadata.get('nphot_form', b''))
            if form != 'logpoly':
                raise ValueError(
                    f"{self.data_path}: its N_photons(E) fit has form {form!r}, but an emitter needs "
                    "'logpoly'; the table predates the current builder. Rebuild it with "
                    "lucid-build-photon-table (stage 2).")
        self.energy_range = (energy_centers.min(), energy_centers.max())
        self.distance_range = (distance_centers.min(), distance_centers.max())
        # The input domain (bin-centre ranges): the emitter's normalisation bounds.
        self.domain = tuple((float(a.min()), float(a.max()))
                            for a in (energy_centers, second_dim_centers, distance_centers))

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
        
    def _normalize_data(self):
        """Normalize input and target data."""
        # Normalize inputs to [-1, 1], with the emitter's own normalisation (the domain bounds)
        self.normalized_bounds['input_min'] = np.array([lo for lo, _ in self.domain])
        self.normalized_bounds['input_max'] = np.array([hi for _, hi in self.domain])
        self.data['inputs_normalized'] = self.normalize(self.data['inputs'])
        
        # The training target, encode_target (lucid.siren.emitter): log10(t + offset) scaled to
        # [0, 1] over the table; the offset avoids log(0) and is small against typical targets.
        self.data['targets_log'] = np.log10(self.data['targets'] + self.zero_threshold)
        lo, hi = target_range(self.data['targets'], self.zero_threshold)
        self.normalized_bounds['target_min'], self.normalized_bounds['target_max'] = lo, hi
        self.data['targets_log_normalized'] = encode_target(self.data['targets'], lo, hi,
                                                            self.zero_threshold)
        
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

    def normalize(self, inputs: np.ndarray) -> np.ndarray:
        """Physical ``(n, 3)`` inputs ``[E, axis2, s/s_max]`` to the net's ``[-1, 1]`` inputs."""
        (e0, e1), (a0, a1), (r0, r1) = self.domain
        return np.asarray(normalize_inputs(jnp.asarray(inputs, jnp.float32), e0, e1, a0, a1, r0, r1))

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
        
    def denormalize_targets_from_normalized(self, targets_normalized: np.ndarray) -> np.ndarray:
        """Net outputs back to targets, with inference's own :func:`decode_target`."""
        return np.asarray(decode_target(jnp.asarray(targets_normalized),
                                        self.normalized_bounds['target_min'],
                                        self.normalized_bounds['target_max'], self.zero_threshold))
        
    @property
    def has_validation(self) -> bool:
        """Check if dataset has validation split."""
        return len(self.val_indices) > 0
        
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