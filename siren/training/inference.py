"""
Inference module for trained SIREN models.

This module provides functionality to load and use trained SIREN models
for photon density predictions with proper normalization handling.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze

logger = logging.getLogger(__name__)


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
                self.params = freeze(jax.tree_map(jnp.array, params_numpy))
                
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
        logger.info(f"Angle range: {np.degrees(self.dataset_info['angle_range'])} degrees")
        logger.info(f"Distance range: {self.dataset_info['distance_range']} mm")
        
    def _init_model(self):
        """Initialize the SIREN model architecture."""
        # Import SIREN model
        try:
            from tools.siren import SIREN
        except ImportError:
            # Fallback to local definition
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from tools.siren import SIREN
        
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