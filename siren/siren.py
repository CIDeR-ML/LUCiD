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
    
    @nn.compact
    def __call__(self, inputs):
        x = SineLayer(
            features=self.hidden_features,
            is_first=True,
            omega_0=self.first_omega_0,
            name='SineLayer_0'
        )(inputs)
        
        for i in range(self.hidden_layers):
            x = SineLayer(
                features=self.hidden_features,
                is_first=False,
                omega_0=self.hidden_omega_0,
                name=f'SineLayer_{i+1}'
            )(x)
            
        if self.outermost_linear:
            scale = np.sqrt(6/self.hidden_features) / self.hidden_omega_0
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
                omega_0=self.hidden_omega_0,
                name='SineLayer_final'
            )(x)
            
        return x * x, inputs

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
