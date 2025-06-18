import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import numpy as np
import torch
from typing import Sequence, Callable, Any
from flax.core.frozen_dict import freeze

class SineLayer(nn.Module):
    features: int
    is_first: bool = False
    omega_0: float = 30.0
    
    @nn.compact
    def __call__(self, inputs):
        input_dim = inputs.shape[-1]
        
        # Initialize weights following SIREN paper
        if self.is_first:
            weight_init = nn.initializers.uniform(scale=1/input_dim)
        else:
            scale = np.sqrt(6/input_dim) / self.omega_0
            weight_init = nn.initializers.uniform(scale=scale)
            
        x = nn.Dense(
            features=self.features,
            kernel_init=weight_init,
            bias_init=nn.initializers.uniform(scale=1)
        )(inputs)
        
        return jnp.sin(self.omega_0 * x)

class SIREN(nn.Module):
    hidden_features: int
    hidden_layers: int
    out_features: int
    outermost_linear: bool = False
    first_omega_0: float = 30.0
    hidden_omega_0: float = 30.0
    w0: float = 30.0  # Alternative parameter name for compatibility
    
    def setup(self):
        # Setup method - no attribute modification needed
        pass
    
    @nn.compact
    def __call__(self, inputs):
        # Use w0 parameter directly, falling back to separate omega_0 parameters
        first_omega = self.w0 if self.w0 != 30.0 else self.first_omega_0
        hidden_omega = self.w0 if self.w0 != 30.0 else self.hidden_omega_0
        
        x = SineLayer(
            features=self.hidden_features,
            is_first=True,
            omega_0=first_omega,
            name='SineLayer_0'
        )(inputs)
        
        for i in range(self.hidden_layers):
            x = SineLayer(
                features=self.hidden_features,
                is_first=False,
                omega_0=hidden_omega,
                name=f'SineLayer_{i+1}'
            )(x)
            
        if self.outermost_linear:
            scale = np.sqrt(6/self.hidden_features) / hidden_omega
            init = nn.initializers.uniform(scale=scale)
            x = nn.Dense(
                features=self.out_features,
                kernel_init=init,
                bias_init=nn.initializers.uniform(scale=1),
                name='Dense_0'
            )(x)
        else:
            x = SineLayer(
                features=self.out_features,
                is_first=False,
                omega_0=hidden_omega,
                name='SineLayer_final'
            )(x)
        
        # Always square the output for compatibility with trained models
        x = x * x
            
        return x, inputs

def torch_to_jax(tensor):
    """Convert a PyTorch tensor to JAX array, handling CUDA tensors."""
    return jnp.array(tensor.cpu().numpy())

def convert_pytorch_to_jax(pytorch_state_dict: dict, jax_model: SIREN):
    """Convert PyTorch SIREN weights to JAX/Flax format"""
    params = {}
    
    # First sine layer (SineLayer_0)
    params['SineLayer_0'] = {
        'Dense_0': {
            'kernel': torch_to_jax(pytorch_state_dict['net.0.linear.weight'].T),
            'bias': torch_to_jax(pytorch_state_dict['net.0.linear.bias'])
        }
    }
    
    # Hidden layers (SineLayer_1 through SineLayer_3)
    for i in range(1, 4):
        params[f'SineLayer_{i}'] = {
            'Dense_0': {
                'kernel': torch_to_jax(pytorch_state_dict[f'net.{i}.linear.weight'].T),
                'bias': torch_to_jax(pytorch_state_dict[f'net.{i}.linear.bias'])
            }
        }
    
    # Final dense layer
    params['Dense_0'] = {
        'kernel': torch_to_jax(pytorch_state_dict['net.4.weight'].T),
        'bias': torch_to_jax(pytorch_state_dict['net.4.bias'])
    }
    
    return freeze({'params': params})

def load_siren_jax(pytorch_weights_path: str):
    """
    Load PyTorch SIREN weights and create equivalent JAX model. Works with both CPU and GPU-saved weights.
    
    Args:
        pytorch_weights_path: Path to saved PyTorch weights
    
    Returns:
        Tuple of (jax_model, jax_params)
    """
    # Load PyTorch weights with automatic device placement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pytorch_state = torch.load(pytorch_weights_path, map_location=device, weights_only=True)
    
    # Initialize JAX model with same architecture
    jax_model = SIREN(
        hidden_features=256,
        hidden_layers=3,
        out_features=1,
        outermost_linear=True
    )
    
    # Convert weights
    jax_params = convert_pytorch_to_jax(pytorch_state, jax_model)
    
    return jax_model, jax_params


def load_photonsim_siren(model_path: str):
    """
    Load PhotonSim-trained SIREN model (native JAX format).
    
    Args:
        model_path: Path to saved PhotonSim SIREN model (.npz file)
    
    Returns:
        Tuple of (jax_model, jax_params, normalization_params, metadata)
    """
    # Load model data
    data = np.load(model_path, allow_pickle=True)
    
    # Extract model configuration
    model_config = data['model_config'].item()
    
    # Create SIREN model with PhotonSim configuration
    jax_model = SIREN(**model_config)
    
    # Extract parameters (convert back to JAX format)
    jax_params = freeze({'params': jax.tree.map(lambda x: jnp.array(x), data['params'].item())})
    
    # Extract normalization parameters
    normalization_params = data['normalization_params'].item()
    
    # Extract metadata
    metadata = {
        'dataset_stats': data['dataset_stats'].item(),
        'final_step': int(data['final_step']),
        'final_train_loss': float(data['final_train_loss']) if data['final_train_loss'] is not None else None,
        'final_val_loss': float(data['final_val_loss']) if data['final_val_loss'] is not None else None,
    }
    
    return jax_model, jax_params, normalization_params, metadata


def normalize_photonsim_inputs(inputs: jnp.ndarray, normalization_params: dict) -> jnp.ndarray:
    """
    Normalize inputs according to PhotonSim training normalization.
    
    Args:
        inputs: Input array [N, 3] with [energy, angle, distance]
        normalization_params: Normalization parameters from training
    
    Returns:
        Normalized inputs in [-1, 1] range
    """
    if not normalization_params['normalize_inputs']:
        return inputs
    
    energy_range = normalization_params['energy_range']
    angle_range = normalization_params['angle_range']
    distance_range = normalization_params['distance_range']
    
    normalized = jnp.zeros_like(inputs)
    
    # Energy: [min_energy, max_energy] -> [-1, 1]
    normalized = normalized.at[:, 0].set(
        2 * (inputs[:, 0] - energy_range[0]) / (energy_range[1] - energy_range[0]) - 1
    )
    
    # Angle: [0, max_angle] -> [-1, 1]
    normalized = normalized.at[:, 1].set(
        2 * (inputs[:, 1] - angle_range[0]) / (angle_range[1] - angle_range[0]) - 1
    )
    
    # Distance: [0, max_distance] -> [-1, 1]
    normalized = normalized.at[:, 2].set(
        2 * (inputs[:, 2] - distance_range[0]) / (distance_range[1] - distance_range[0]) - 1
    )
    
    return normalized


def denormalize_photonsim_output(normalized_output: jnp.ndarray, normalization_params: dict) -> jnp.ndarray:
    """
    Denormalize output according to PhotonSim training normalization.
    
    Args:
        normalized_output: Normalized output from model
        normalization_params: Normalization parameters from training
    
    Returns:
        Denormalized output (photon counts)
    """
    if not normalization_params['normalize_output']:
        return normalized_output
    
    output_scale = normalization_params['output_scale']
    return normalized_output * output_scale


class PhotonSimSIREN:
    """
    Wrapper class for PhotonSim-trained SIREN models.
    
    This class provides a convenient interface for using PhotonSim-trained
    SIREN models with automatic normalization and denormalization.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize PhotonSim SIREN model.
        
        Args:
            model_path: Path to saved PhotonSim SIREN model
        """
        self.model, self.params, self.normalization_params, self.metadata = load_photonsim_siren(model_path)
        self.model_path = model_path
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the model on inputs.
        
        Args:
            inputs: Input array [N, 3] with [energy, angle, distance]
        
        Returns:
            Photon counts [N,]
        """
        # Normalize inputs
        normalized_inputs = normalize_photonsim_inputs(inputs, self.normalization_params)
        
        # Forward pass
        output, _ = self.model.apply(self.params, normalized_inputs)
        
        # Ensure correct shape
        if output.ndim > 1:
            output = output.squeeze()
        
        # Denormalize output
        denormalized_output = denormalize_photonsim_output(output, self.normalization_params)
        
        return denormalized_output
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            'model_path': self.model_path,
            'model_config': {
                'hidden_features': self.model.hidden_features,
                'hidden_layers': self.model.hidden_layers,
                'out_features': self.model.out_features,
                'w0': self.model.w0,
            },
            'normalization_params': self.normalization_params,
            'metadata': self.metadata,
        }
    
    def save_for_inference(self, output_path: str):
        """
        Save model in a format optimized for inference.
        
        Args:
            output_path: Path to save inference-ready model
        """
        inference_data = {
            'params': self.params,
            'model_config': {
                'hidden_features': self.model.hidden_features,
                'hidden_layers': self.model.hidden_layers,
                'out_features': self.model.out_features,
                'w0': self.model.w0,
            },
            'normalization_params': self.normalization_params,
        }
        
        # Convert JAX arrays to numpy for saving
        inference_data_np = jax.tree.map(lambda x: np.array(x) if hasattr(x, 'shape') else x, inference_data)
        
        np.savez(output_path, **inference_data_np)
        print(f"Saved inference model to {output_path}")


# Backward compatibility function
def load_siren_model(model_path: str, model_type: str = "auto"):
    """
    Load SIREN model with automatic type detection.
    
    Args:
        model_path: Path to model file
        model_type: "pytorch", "photonsim", or "auto" for automatic detection
    
    Returns:
        Appropriate model object
    """
    if model_type == "auto":
        # Detect model type based on file extension and contents
        if model_path.endswith('.pkl'):
            model_type = "pytorch"
        elif model_path.endswith('.npz'):
            # Check if it contains PhotonSim-specific keys
            try:
                data = np.load(model_path, allow_pickle=True)
                if 'model_config' in data and 'normalization_params' in data:
                    model_type = "photonsim"
                else:
                    model_type = "pytorch"
            except:
                model_type = "pytorch"
        else:
            model_type = "pytorch"
    
    if model_type == "photonsim":
        return PhotonSimSIREN(model_path)
    else:
        return load_siren_jax(model_path)
