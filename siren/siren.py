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

def convert_pytorch_to_jax(pytorch_state_dict: dict, jax_model: SIREN, input_shape: Sequence[int]):
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

# def load_siren_jax(pytorch_weights_path: str, input_shape: Sequence[int]):
#     """
#     Load PyTorch SIREN weights and create equivalent JAX model
    
#     Args:
#         pytorch_weights_path: Path to saved PyTorch weights
#         input_shape: Shape of input tensor (batch_size, features)
    
#     Returns:
#         Tuple of (jax_model, jax_params)
#     """
#     # Load PyTorch weights
#     pytorch_state = torch.load(pytorch_weights_path, weights_only=True)
    
#     # Initialize JAX model with same architecture
#     jax_model = SIREN(
#         hidden_features=256,
#         hidden_layers=3,
#         out_features=1,
#         outermost_linear=True
#     )
    
#     # Convert weights
#     jax_params = convert_pytorch_to_jax(pytorch_state, jax_model, input_shape)
    
#     return jax_model, jax_params

def load_siren_jax(pytorch_weights_path: str, input_shape: Sequence[int]):
    """
    Load PyTorch SIREN weights and create equivalent JAX model. Works with both CPU and GPU-saved weights.
    
    Args:
        pytorch_weights_path: Path to saved PyTorch weights
        input_shape: Shape of input tensor (batch_size, features)
    
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
    jax_params = convert_pytorch_to_jax(pytorch_state, jax_model, input_shape)
    
    return jax_model, jax_params

    



# import jax
# import jax.numpy as jnp
# from jax import random
# import flax.linen as nn
# import numpy as np
# import torch
# from typing import Sequence, Callable, Any

# class SineLayer(nn.Module):
#     features: int
#     is_first: bool = False
#     omega_0: float = 30.0
    
#     @nn.compact
#     def __call__(self, inputs):
#         input_dim = inputs.shape[-1]
        
#         # Initialize weights following SIREN paper
#         if self.is_first:
#             weight_init = nn.initializers.uniform(scale=1/input_dim)
#         else:
#             scale = np.sqrt(6/input_dim) / self.omega_0
#             weight_init = nn.initializers.uniform(scale=scale)
            
#         x = nn.Dense(
#             features=self.features,
#             kernel_init=weight_init,
#             bias_init=nn.initializers.uniform(scale=1)
#         )(inputs)
        
#         return jnp.sin(self.omega_0 * x)

# class SIREN(nn.Module):
#     hidden_features: int
#     hidden_layers: int
#     out_features: int
#     outermost_linear: bool = False
#     first_omega_0: float = 30.0
#     hidden_omega_0: float = 30.0
    
#     @nn.compact
#     def __call__(self, inputs):
#         x = SineLayer(
#             features=self.hidden_features,
#             is_first=True,
#             omega_0=self.first_omega_0
#         )(inputs)
        
#         for _ in range(self.hidden_layers):
#             x = SineLayer(
#                 features=self.hidden_features,
#                 is_first=False,
#                 omega_0=self.hidden_omega_0
#             )(x)
            
#         if self.outermost_linear:
#             # Initialize the final layer following SIREN paper
#             scale = np.sqrt(6/self.hidden_features) / self.hidden_omega_0
#             init = nn.initializers.uniform(scale=scale)
#             x = nn.Dense(
#                 features=self.out_features,
#                 kernel_init=init,
#                 bias_init=nn.initializers.uniform(scale=1)
#             )(x)
#         else:
#             x = SineLayer(
#                 features=self.out_features,
#                 is_first=False,
#                 omega_0=self.hidden_omega_0
#             )(x)
            
#         return x * x, inputs

# def convert_pytorch_to_jax(pytorch_state_dict: dict, jax_model: SIREN, input_shape: Sequence[int]):
#     """
#     Convert PyTorch SIREN weights to JAX/Flax format
    
#     Args:
#         pytorch_state_dict: State dict from PyTorch SIREN model
#         jax_model: Initialized JAX SIREN model
#         input_shape: Shape of input tensor (batch_size, *feature_dims)
    
#     Returns:
#         JAX model parameters dictionary
#     """
#     # Initialize JAX model to get parameter structure
#     key = random.PRNGKey(0)
#     variables = jax_model.init(key, jnp.ones(input_shape))
#     params = variables['params']
    
#     # Create new params dict with converted weights
#     new_params = {}
#     layer_idx = 0
    
#     # Helper to convert torch tensor to jax array
#     def torch_to_jax(t):
#         return jnp.array(t.cpu().numpy())
    
#     for name, param in pytorch_state_dict.items():
#         if 'net' not in name:
#             continue
            
#         # Extract layer number and parameter type
#         parts = name.split('.')
#         layer_num = int(parts[1])
#         param_type = parts[-1]
        
#         if layer_num > layer_idx:
#             layer_idx = layer_num
            
#         # Map to JAX parameter names
#         if 'linear.weight' in name:
#             param = param.T  # Transpose weight matrices
#             if layer_idx == len(params) - 1 and 'Dense' in params[f'Dense_{layer_idx}']:
#                 new_params[f'Dense_{layer_idx}'] = {'kernel': torch_to_jax(param)}
#             else:
#                 new_params[f'SineLayer_{layer_idx}'] = {'Dense_0': {'kernel': torch_to_jax(param)}}
#         elif 'linear.bias' in name:
#             if layer_idx == len(params) - 1 and 'Dense' in params[f'Dense_{layer_idx}']:
#                 new_params[f'Dense_{layer_idx}']['bias'] = torch_to_jax(param)
#             else:
#                 new_params[f'SineLayer_{layer_idx}']['Dense_0']['bias'] = torch_to_jax(param)
    
#     return {'params': new_params}

# # Example usage:
# def load_siren_jax(pytorch_weights_path: str, input_shape: Sequence[int]):
#     """
#     Load PyTorch SIREN weights and create equivalent JAX model
    
#     Args:
#         pytorch_weights_path: Path to saved PyTorch weights
#         input_shape: Shape of input tensor (batch_size, *feature_dims)
    
#     Returns:
#         Initialized JAX SIREN model with converted weights
#     """
#     # Load PyTorch weights
#     pytorch_state = torch.load(pytorch_weights_path)
    
#     # Initialize JAX model with same architecture
#     jax_model = SIREN(
#         hidden_features=256,
#         hidden_layers=3,
#         out_features=1,
#         outermost_linear=True
#     )
    
#     # Convert weights
#     jax_params = convert_pytorch_to_jax(pytorch_state, jax_model, input_shape)
    
#     return jax_model, jax_params