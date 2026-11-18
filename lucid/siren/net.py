"""The SIREN network (sinusoidal representation network) every emitter evaluates."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

import numpy as np

__all__ = ['SineLayer', 'SIREN']

def _sym_uniform(scale):
    """Symmetric uniform initializer U(-scale, scale) — the SIREN-paper init.

    flax's ``nn.initializers.uniform(scale)`` is ONE-SIDED, U(0, scale). Using it for a
    from-scratch JAX ``model.init`` gives all-positive first-layer weights, degenerate
    sinusoidal features, and training that plateaus (~70% residual even on a linear ramp).
    Trained Cherenkov/dedx models load every weight and bias, so this only affects (and fixed)
    from-scratch JAX initialisation.
    """
    return lambda key, shape, dtype=jnp.float32: jax.random.uniform(key, shape, dtype, -scale, scale)


class SineLayer(nn.Module):
    features: int
    is_first: bool = False
    omega_0: float = 30.0

    @nn.compact
    def __call__(self, inputs):
        input_dim = inputs.shape[-1]

        # Initialize weights following the SIREN paper (symmetric U(-b, b); see _sym_uniform).
        if self.is_first:
            weight_init = _sym_uniform(1/input_dim)
        else:
            scale = np.sqrt(6/input_dim) / self.omega_0
            weight_init = _sym_uniform(scale)

        x = nn.Dense(
            features=self.features,
            kernel_init=weight_init,
            bias_init=_sym_uniform(np.pi)
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
    square_output: bool = True  # see the square step below; False = general/linear SIREN

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
            init = _sym_uniform(scale)
            x = nn.Dense(
                features=self.out_features,
                kernel_init=init,
                bias_init=_sym_uniform(np.pi),
                name='Dense_0'
            )(x)
        else:
            x = SineLayer(
                features=self.out_features,
                is_first=False,
                omega_0=hidden_omega,
                name='SineLayer_final'
            )(x)

        # Square the output to enforce NON-NEGATIVITY for the Cherenkov/dedx photon-density
        # surrogate (the trained models predict a (log-)density >= 0). This is the default for
        # backward compatibility with every loaded model. Set square_output=False for a
        # general / LINEAR SIREN — e.g. the spatial ABSORPTION FIELD, whose deviation is SIGNED
        # (field = 1 +/- dev), which a squared output cannot represent.
        if self.square_output:
            x = x * x

        return x
