"""
Inference module for trained SIREN models.

This module provides functionality to load and use trained SIREN models
for photon density predictions with proper normalization handling.
"""
from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze

logger = logging.getLogger(__name__)


def repo_nphot_path(model_path):
    """Where the shipped ``nphot.json`` for a trained Cherenkov model lives, or None.

    The model sits at ``data/<material>/<particle>/siren_training/trained_model/<stem>``, so the
    file, when there is one, is ``data/<material>/<particle>/nphot.json``. Two candidates, both
    LEXICAL -- no absolute symlink is ever followed, so the answer does not depend on how the data
    was installed and never reads outside the repository:

    1. THE PATH AS GIVEN. With the default install ``siren_training`` is a real directory; with
       ``download_data.sh --store-dir`` it is an ABSOLUTE link into the store, which holds no
       ``nphot.json``. Resolving it would find the file on the first install and miss it on the
       second.
    2. ONE RELATIVE HOP. wbls and ice link their ``siren_training`` to
       ``../../water/<particle>/siren_training`` on every install, because they run water's model
       for now. Following that relative link, and only that, reaches water's file -- the
       coefficients belong to the model, not to the directory the caller asked through.

    CHERENKOV MODELS ONLY. The file belongs to the model reached through ``siren_training``. The
    dE/dx model beside it (``dedx_siren_training``) shares the particle directory but not the
    curve, and its context never reads nphot; keying on the directory alone handed it the
    Cherenkov coefficients, which the mismatch check then refused, failing every dE/dx load.

    The caller still checks the file against the model's own legacy a/b/c, so a wrong pairing
    raises instead of being applied.
    """
    stem = Path(os.path.abspath(model_path))
    try:
        training = stem.parents[1]
    except IndexError:
        return None
    if training.name != 'siren_training':
        return None
    candidate = training.parent / 'nphot.json'
    if candidate.is_file():
        return candidate
    if training.is_symlink():
        target = os.readlink(training)
        if not os.path.isabs(target):
            hop = Path(os.path.normpath(training.parent / target))
            candidate = hop.parent / 'nphot.json'
            if hop.name == 'siren_training' and candidate.is_file():
                return candidate
    return None


class SIRENPredictor:
    """
    Load and use trained SIREN models for inference.
    
    This class handles:
    - Loading model weights and metadata
    - Input normalization
    - Model inference
    - Output denormalization (if applicable)
    
    Example usage:
    ```python
    predictor = SIRENPredictor('path/to/model_dir/siren_model')
    
    # Single prediction
    energy = 500  # MeV
    angle = np.radians(45)  # radians
    distance = 2000  # mm
    density = predictor.predict(energy, angle, distance)
    
    # Batch prediction
    inputs = np.array([[500, np.radians(45), 2000],
                       [600, np.radians(30), 3000]])
    densities = predictor.predict_batch(inputs)
    ```
    """
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize the predictor by loading model and metadata.
        
        Args:
            model_path: Path to model files (without extension).
                        Expects {model_path}_weights.npz and {model_path}_metadata.json
        """
        self.model_path = Path(model_path)
        
        # Load metadata
        metadata_path = f"{self.model_path}_metadata.json"
        if not Path(metadata_path).exists():
            # Try with .json extension if not found
            metadata_path = f"{self.model_path}.json"
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self._apply_repo_nphot()
        
        # Load model weights
        weights_path = f"{self.model_path}_weights.npz"
        if not Path(weights_path).exists():
            # Try with .npz extension if not found
            weights_path = f"{self.model_path}.npz"
            
        weights_data = np.load(weights_path, allow_pickle=True)
        
        # Reconstruct the nested parameter structure
        def reconstruct_nested_params(flat_params):
            """Reconstruct nested parameter structure from flattened keys."""
            nested = {}
            
            for key, value in flat_params.items():
                if key.startswith('params_'):
                    # Remove 'params_' prefix
                    remaining = key[7:]  # Remove 'params_'
                    
                    # Split and reconstruct: SineLayer_0_Dense_0_kernel -> SineLayer_0/Dense_0/kernel
                    parts = remaining.split('_')
                    
                    # Reconstruct layer names: SineLayer_0, Dense_0, etc.
                    if len(parts) >= 3:
                        # Pattern: SineLayer_X_Dense_Y_paramname
                        layer_part = f"{parts[0]}_{parts[1]}"  # SineLayer_0
                        if len(parts) >= 5:
                            dense_part = f"{parts[2]}_{parts[3]}"  # Dense_0  
                            param_name = parts[4]  # kernel/bias
                        else:
                            dense_part = parts[2]  # fallback
                            param_name = parts[3] if len(parts) > 3 else parts[-1]
                        
                        # Create nested structure
                        if layer_part not in nested:
                            nested[layer_part] = {}
                        if dense_part not in nested[layer_part]:
                            nested[layer_part][dense_part] = {}
                        nested[layer_part][dense_part][param_name] = jnp.array(value)
                    else:
                        # Fallback for unexpected patterns
                        current = nested
                        for part in parts[:-1]:
                            if part not in current:
                                current[part] = {}
                            current = current[part]
                        current[parts[-1]] = jnp.array(value)
                else:
                    # Handle keys that don't start with 'params_'
                    nested[key] = jnp.array(value)
            
            return nested
        
        # Check if we have flattened parameters (any key with underscores indicating nested structure)
        has_flattened = any('_' in key and not key.startswith('.') for key in weights_data.keys())
        
        if has_flattened:
            # Flattened format: reconstruct the nested structure
            nested_params = reconstruct_nested_params(weights_data)
            
            # Wrap in 'params' key as expected by Flax
            self.params = freeze({'params': nested_params})
            
        elif 'params' in weights_data and len(weights_data.keys()) == 1:
            # Old format: single 'params' key containing nested structure
            try:
                params_data = weights_data['params']
                
                # Check if it's a string array (corrupted save)
                if isinstance(params_data, np.ndarray) and params_data.dtype.kind in ['U', 'S']:
                    logger.error(f"Corrupted model file: contains string array instead of parameters")
                    logger.error(f"The model needs to be re-saved with the fixed trainer")
                    raise ValueError(f"Corrupted parameters: got string array instead of numeric data. Please re-save the model.")
                
                if isinstance(params_data, np.ndarray) and params_data.dtype == 'O':
                    params_numpy = params_data.item()
                else:
                    params_numpy = params_data
                
                # Convert to JAX arrays
                self.params = freeze(jax.tree.map(jnp.array, params_numpy))
                
            except Exception as e:
                logger.error(f"Failed to load old format: {e}")
                raise
                
        else:
            # Direct format: parameters saved as individual keys
            params_dict = {k: jnp.array(v) for k, v in weights_data.items()}
            self.params = freeze(params_dict)
        
        # Initialize model
        self._init_model()
        
        # Extract normalization info
        self.input_norm = self.metadata['input_normalization']
        self.input_min = np.array(self.input_norm['input_min'])
        self.input_max = np.array(self.input_norm['input_max'])
        
        # Extract dataset info
        self.dataset_info = self.metadata['dataset_info']
        
        logger.info(f"Loaded SIREN model from {self.model_path}")
        logger.info(f"Model config: {self.metadata['model_config']}")
        logger.info(f"Energy range: {self.dataset_info['energy_range']} MeV")

        # Handle both photon (angle) and dEdx table types
        table_type = self.dataset_info.get('table_type', 'photon')
        if table_type == 'dedx' and self.dataset_info.get('dedx_range'):
            logger.info(f"dE/dx range: {self.dataset_info['dedx_range']} keV/mm")
        elif self.dataset_info.get('angle_range'):
            logger.info(f"Angle range: {np.degrees(self.dataset_info['angle_range'])} degrees")

        logger.info(f"Distance range: {self.dataset_info['distance_range']} mm")
        
    def _apply_repo_nphot(self):
        """Overlay `data/<material>/<particle>/nphot.json` onto the `nphot` metadata block.

        WHY THIS EXISTS. The trained models are fetched with `scripts/download_data.sh`, and
        `data/*/*/siren_training/` is gitignored, so the model's own metadata is NOT in version
        control. The log-log polynomial coefficients lived only there -- meaning a fresh clone
        selected the legacy power law, which misses its own training table by rms 6.5% for the
        muon and 10.5% for the electron, and by +0.9% across the 400-1800 MeV reconstruction
        band. Shipping the coefficients in the repo is what makes the fix reach anyone who did
        not happen to have the same local files.

        THE REPO FILE IS THE ONLY COPY, deliberately -- the same arrangement as `t0.json`, whose
        coefficients the model metadata also does not carry. The upstream `nphot` block holds
        only `form`, `a`, `b`, `c`, `r_squared` and the fit range, and the coefficients were
        previously hand-added to the FETCHED file. That is worse than duplication: it changes the
        artefact's size, and `download_data.sh` decides whether to re-fetch by comparing size
        against the remote, then resumes with `curl -C -` from an offset past the remote file's
        end. So an edited metadata file breaks the downloader for that model.

        Found from the MODEL PATH, not from a particle name -- see `repo_nphot_path`, including why
        it looks where the caller asked before following symlinks. Nothing has to be told which
        particle this is.

        REFUSES A MISMATCHED PAIR. These coefficients were fitted to ONE table. If the downloaded
        model is later replaced by one fitted to a different table, applying them would silently
        rescale every reconstructed energy -- the failure would look like physics, not like a
        stale file. So the shipped file records the legacy a/b/c it was found beside, and this
        raises rather than overriding if the model no longer matches.
        """
        nphot = self.metadata.get('nphot')
        if not isinstance(nphot, dict):
            return
        override_path = repo_nphot_path(self.model_path)
        if override_path is None:
            # No shipped polynomial for this model. Leave the block alone -- the legacy power law
            # is what it has always used -- but record WHICH model, so the warning raised
            # downstream can name it. With six (material, particle) bundles and a polynomial
            # fitted for water only, "some model fell back" is not an actionable message.
            nphot.setdefault('_origin', str(self.model_path))
            return
        with open(override_path, 'r') as f:
            shipped = json.load(f)

        expect = shipped.get('for_model_legacy')
        if expect:
            got = {k: nphot.get(k) for k in ('a', 'b', 'c')}
            if any(expect[k] != got[k] for k in ('a', 'b', 'c')):
                raise ValueError(
                    f"{override_path} was fitted alongside a different model: it records legacy "
                    f"a/b/c = {expect}, this model's metadata carries {got}. Applying it would "
                    f"silently rescale every reconstructed energy. Re-fit the polynomial against "
                    f"this model's table, or delete the file to fall back to its power law."
                )
        merged = {**nphot, **{k: v for k, v in shipped.items()
                              if k not in ('for_model_legacy', 'source')}}
        merged['_origin'] = str(override_path)
        self.metadata['nphot'] = merged

    def _init_model(self):
        """Initialize the SIREN model architecture."""
        # Import SIREN model
        try:
            from ..core import SIREN
        except ImportError:
            # Fallback to absolute import
            from lucid.siren.core import SIREN
        
        config = self.metadata['model_config']
        self.model = SIREN(
            hidden_features=config['hidden_features'],
            hidden_layers=config['hidden_layers'],
            out_features=config['out_features'],
            w0=config['w0']
        )
        
    def normalize_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Normalize inputs to [-1, 1] range expected by SIREN.
        
        Args:
            inputs: Raw inputs with shape (..., 3) containing [energy, angle, distance]
        
        Returns:
            Normalized inputs in [-1, 1] range
        """
        # Linear normalization to [-1, 1]
        normalized = 2 * ((inputs - self.input_min) / (self.input_max - self.input_min)) - 1
        return normalized
    
    def predict(self, energy: float, angle: float, distance: float) -> float:
        """
        Predict photon density for a single point.
        
        Args:
            energy: Energy in MeV
            angle: Angle in radians
            distance: Distance in mm
        
        Returns:
            Photon density in photons/mm^2
        """
        inputs = np.array([[energy, angle, distance]])
        result = self.predict_batch(inputs)
        
        # Handle both scalar and array results
        if np.isscalar(result):
            return float(result)
        else:
            return float(result[0])
    
    def predict_batch(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict photon density for multiple points.
        
        Args:
            inputs: Array of shape (n_points, 3) with columns [energy, angle, distance]
                   Units: energy in MeV, angle in radians, distance in mm
        
        Returns:
            Array of photon densities in photons/mm^2
        """
        # Normalize inputs
        inputs_norm = self.normalize_inputs(inputs)
        
        # Convert to JAX array
        inputs_jax = jnp.array(inputs_norm)
        
        # Run model - wrap params in the expected format
        # The model expects {'params': param_dict} structure
        if hasattr(self.params, 'keys') and 'params' not in self.params:
            # Wrap in the expected 'params' structure
            model_params = {'params': self.params}
        else:
            # Use as-is if already has 'params' key
            model_params = self.params
            
        output = self.model.apply(model_params, inputs_jax)
        
        # Handle tuple output from SIREN
        if isinstance(output, tuple):
            output = output[0]
        
        # Convert back to numpy
        predictions = np.array(output).squeeze()
        
        # Check if we need to denormalize the output
        if 'target_normalization' in self.metadata:
            target_norm = self.metadata['target_normalization']
            if target_norm['scheme'] == 'linear_normalized_to_01':
                # Denormalize from [0, 1] back to original linear scale
                linear_min = target_norm['linear_min']
                linear_max = target_norm['linear_max']
                predictions = predictions * (linear_max - linear_min) + linear_min
            elif target_norm['scheme'] == 'log_normalized_to_01':
                # Handle log-scale denormalization
                log_min = target_norm['log_min']
                log_max = target_norm['log_max']
                log_predictions = predictions * (log_max - log_min) + log_min
                # Inverse of log10(x + 1e-2) in dataset.py; the 1e-10 is a guard against float noise
                predictions = 10 ** log_predictions - 1e-10
        
        return predictions
    
    def create_grid(self, 
                    energies: np.ndarray,
                    angles: np.ndarray, 
                    distances: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create a 3D grid of predictions for visualization.
        
        Args:
            energies: 1D array of energies in MeV
            angles: 1D array of angles in radians
            distances: 1D array of distances in mm
        
        Returns:
            Tuple of (predictions_grid, (E_mesh, A_mesh, D_mesh))
            where predictions_grid has shape (len(energies), len(angles), len(distances))
        """
        # Create meshgrid
        E, A, D = np.meshgrid(energies, angles, distances, indexing='ij')
        
        # Flatten for batch prediction
        inputs = np.stack([E.ravel(), A.ravel(), D.ravel()], axis=-1)
        
        # Predict
        predictions = self.predict_batch(inputs)
        
        # Reshape back to grid
        predictions_grid = predictions.reshape(E.shape)
        
        return predictions_grid, (E, A, D)
    
    def create_2d_slice(self,
                        fixed_param: str,
                        fixed_value: float,
                        param1_values: np.ndarray,
                        param2_values: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Create a 2D slice by fixing one parameter.
        
        Args:
            fixed_param: Which parameter to fix ('energy', 'angle', or 'distance')
            fixed_value: Value of the fixed parameter
            param1_values: Values for first varying parameter
            param2_values: Values for second varying parameter
        
        Returns:
            Tuple of (predictions_2d, (mesh1, mesh2))
        """
        # Create 2D meshgrid
        mesh1, mesh2 = np.meshgrid(param1_values, param2_values, indexing='ij')
        
        # Create input array based on which parameter is fixed
        if fixed_param == 'energy':
            inputs = np.stack([
                np.full_like(mesh1.ravel(), fixed_value),
                mesh1.ravel(),
                mesh2.ravel()
            ], axis=-1)
        elif fixed_param == 'angle':
            inputs = np.stack([
                mesh1.ravel(),
                np.full_like(mesh1.ravel(), fixed_value),
                mesh2.ravel()
            ], axis=-1)
        elif fixed_param == 'distance':
            inputs = np.stack([
                mesh1.ravel(),
                mesh2.ravel(),
                np.full_like(mesh1.ravel(), fixed_value)
            ], axis=-1)
        else:
            raise ValueError(f"Invalid fixed_param: {fixed_param}")
        
        # Predict
        predictions = self.predict_batch(inputs)
        
        # Reshape to 2D
        predictions_2d = predictions.reshape(mesh1.shape)
        
        return predictions_2d, (mesh1, mesh2)
    
    def get_info(self) -> Dict:
        """Get model information and metadata."""
        return {
            'model_config': self.metadata['model_config'],
            'dataset_info': self.dataset_info,
            'training_info': self.metadata.get('training_info', {}),
            'normalization_info': self.input_norm
        }