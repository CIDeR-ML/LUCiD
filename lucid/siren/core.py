from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import numpy as np
from typing import Sequence, Callable, Any, NamedTuple
from flax.core.frozen_dict import freeze

__all__ = [
    'SineLayer', 'SIREN',
    'PhotonSimContext', 'build_photonsim_context',
    'make_smax_fn', 'make_power_law_fn',
    'torch_to_jax', 'convert_pytorch_to_jax', 'load_siren_jax',
]

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
    import torch  # noqa: F811 — lazy import, torch is an optional dependency
    return jnp.array(tensor.cpu().numpy())

def convert_pytorch_to_jax(pytorch_state_dict: dict, jax_model: SIREN):
    """Convert PyTorch SIREN weights to JAX/Flax format.

    Args:
        pytorch_state_dict: PyTorch state_dict from a trained SIREN model.
        jax_model: Target JAX SIREN model instance (used for architecture reference).

    Returns:
        FrozenDict of JAX/Flax parameters matching the model's expected structure.
    """
    params = {}

    params['SineLayer_0'] = {
        'Dense_0': {
            'kernel': torch_to_jax(pytorch_state_dict['net.0.linear.weight'].T),
            'bias': torch_to_jax(pytorch_state_dict['net.0.linear.bias'])
        }
    }

    for i in range(1, 4):
        params[f'SineLayer_{i}'] = {
            'Dense_0': {
                'kernel': torch_to_jax(pytorch_state_dict[f'net.{i}.linear.weight'].T),
                'bias': torch_to_jax(pytorch_state_dict[f'net.{i}.linear.bias'])
            }
        }

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
    import torch  # lazy import — torch is an optional dependency
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pytorch_state = torch.load(pytorch_weights_path, map_location=device, weights_only=True)

    jax_model = SIREN(
        hidden_features=256,
        hidden_layers=3,
        out_features=1,
        outermost_linear=True
    )

    jax_params = convert_pytorch_to_jax(pytorch_state, jax_model)

    return jax_model, jax_params


# --- PhotonSim track-mode inference context ---------------------------------


class PhotonSimContext(NamedTuple):
    """Everything `photonsim_differentiable_get_rays` needs, resolved once at
    model-load time.

    A plain Python container — it holds a Flax module + Python closures +
    floats, so it is never traced. The ray-function factory
    (`make_photonsim_ray_fn`) closes over it; `model_params` (a real pytree) is
    passed separately as a traced argument.
    """
    model:         SIREN
    energy_min:    float
    energy_max:    float
    angle_min:     float
    angle_max:     float
    smax_dist_min: float          # s/s_max range the model was trained on
    smax_dist_max: float          #   (dataset_info['distance_range'])
    log_min:       float          # target_normalization log range
    log_max:       float
    s_max_fn:      Callable        # s_max(E_mev) -> mm
    n_photons_fn:  Callable        # N_photons(E_mev) -> total photons/event
    grid_bins:     int             # first-pass grid resolution (per axis)
    threshold:     float           # seed threshold, fraction of per-energy grid max


def make_smax_fn(smax: dict) -> Callable:
    """Build a jittable ``s_max(E_mev) -> mm`` closure from a trained-model
    ``smax`` metadata block.

    The ``form`` is fixed per model, so it is dispatched in Python here; the
    returned closure is fully traceable (the ``piecewise`` branch uses
    ``jnp.where``). Mirrors ``_eval_smax`` in
    ``PhotonSim/tools/smax/analyze_smax.py`` and ``build_tables.py``.
    """
    form = smax['form']
    p = smax['params']
    if form == 'A*E^B':
        A, B = float(p['A']), float(p['B'])
        return lambda E: A * E ** B
    if form == 'smooth_two_power':
        a, b1, b2, E0 = (float(p[k]) for k in ('a', 'b1', 'b2', 'E0'))
        return lambda E: a * E ** b1 / (1.0 + (E / E0) ** (b1 - b2))
    if form == 'piecewise':
        ej = float(p['e_join_mev'])
        a, b1, b2, E0 = (float(p[k]) for k in ('a', 'b1', 'b2', 'E0'))
        ah, bh1, bh2, Eh0 = (float(p[k])
                             for k in ('a_hi', 'b1_hi', 'b2_hi', 'E0_hi'))

        def _piecewise(E):
            low = a * E ** b1 / (1.0 + (E / E0) ** (b1 - b2))
            high = ah * E ** bh1 / (1.0 + (E / Eh0) ** (bh1 - bh2))
            return jnp.where(E < ej, low, high)

        return _piecewise
    raise ValueError(f"unknown smax form: {form!r}")


def make_power_law_fn(nphot: dict) -> Callable:
    """Build a jittable ``N_photons(E_mev)`` closure from a trained-model
    ``nphot`` metadata block (form ``'A*E^B+C'``).

    Clamped at 0 — the ``a*E^b+c`` fit can dip slightly negative below the
    Cherenkov threshold.
    """
    a, b, c = float(nphot['a']), float(nphot['b']), float(nphot['c'])
    return lambda E: jnp.maximum(a * E ** b + c, 0.0)


# Defaults for the ray-sampling knobs when `siren_params.json` omits them.
_DEFAULT_RAY_SAMPLING = {"grid_bins": 250, "threshold": 0.05}


def build_photonsim_context(photonsim_predictor,
                            ray_sampling: dict = None) -> PhotonSimContext:
    """Resolve a ``SIRENPredictor`` into a :class:`PhotonSimContext`.

    Reads the dataset ranges, the SIREN architecture (``metadata['model_config']``
    — no longer hardcoded), the target-normalization log range, and builds the
    ``s_max(E)`` / ``N_photons(E)`` closures from the ``smax`` / ``nphot``
    metadata blocks.

    ``ray_sampling`` is the ``ray_sampling`` block from ``siren_params.json``
    (``{'grid_bins', 'threshold'}``); missing keys fall back to
    ``_DEFAULT_RAY_SAMPLING``. It drives the importance-sampling ray generator:
    a first-pass ``grid_bins x grid_bins`` SIREN evaluation, then seeding only
    in bins above ``threshold * max`` of that grid.
    """
    rs = {**_DEFAULT_RAY_SAMPLING, **(ray_sampling or {})}
    meta = photonsim_predictor.metadata
    dataset_info = photonsim_predictor.dataset_info
    energy_min, energy_max = dataset_info['energy_range']
    angle_min, angle_max = dataset_info['angle_range']        # radians
    dist_min, dist_max = dataset_info['distance_range']       # s/s_max ∈ [0,1]

    target_norm = meta['target_normalization']
    if target_norm['scheme'] != 'log_normalized_to_01':
        raise ValueError(
            f"Expected target normalization scheme 'log_normalized_to_01', "
            f"but got '{target_norm['scheme']}'")

    if 'smax' not in meta:
        raise ValueError(
            "trained-model metadata has no 'smax' block — this model predates "
            "s/s_max support. Retrain or re-sync the model metadata.")
    if 'nphot' not in meta:
        raise ValueError(
            "trained-model metadata has no 'nphot' block — rebuild the h5 with "
            "the N_photons fit and re-sync the model metadata.")

    model = SIREN(**meta['model_config'])

    return PhotonSimContext(
        model=model,
        energy_min=float(energy_min), energy_max=float(energy_max),
        angle_min=float(angle_min), angle_max=float(angle_max),
        smax_dist_min=float(dist_min), smax_dist_max=float(dist_max),
        log_min=float(target_norm['log_min']),
        log_max=float(target_norm['log_max']),
        s_max_fn=make_smax_fn(meta['smax']),
        n_photons_fn=make_power_law_fn(meta['nphot']),
        grid_bins=int(rs['grid_bins']),
        threshold=float(rs['threshold']),
    )
