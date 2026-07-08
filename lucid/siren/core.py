from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import numpy as np
from typing import Sequence, Callable, Any, NamedTuple, Optional
from flax.core.frozen_dict import freeze

__all__ = [
    'SineLayer', 'SIREN',
    'SirenContext', 'build_cherenkov_context', 'build_dedx_context',
    'make_smax_fn', 'make_power_law_fn',
    'torch_to_jax', 'convert_pytorch_to_jax', 'load_siren_jax',
]

def _sym_uniform(scale):
    """Symmetric uniform initializer U(-scale, scale) — the SIREN-paper init.

    flax's ``nn.initializers.uniform(scale)`` is ONE-SIDED, U(0, scale). Using it for a
    from-scratch JAX ``model.init`` gives all-positive first-layer weights, degenerate
    sinusoidal features, and training that plateaus (~70% residual even on a linear ramp).
    ``load_siren_jax`` overwrites every weight/bias with the PyTorch-trained values, so this
    change is byte-identical for the trained Cherenkov/dedx models — it only fixes the
    (previously broken) from-scratch JAX initialisation.
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


# --- SIREN surrogate inference context ---------------------------------


class SirenContext(NamedTuple):
    """Inference-time inputs shared between the Cherenkov and dE/dx SIREN
    surrogates.

    Each SIREN is a 3D scalar field over ``(energy, axis2, s/s_max)``. The
    2nd axis is opening angle for Cherenkov, dE/dx for the energy-loss
    surrogate; ``axis2_min/axis2_max`` carry whatever range applies. All other
    fields play identical roles for both surrogates. ``n_photons_fn`` is the
    Cherenkov absolute-count normalization and is left ``None`` for dE/dx
    contexts (scintillation gets its absolute scale from the medium's
    light-yield parameters instead).

    A plain Python container — it holds a Flax module + Python closures +
    floats, so it is never traced. The ray-function factories
    (``make_cherenkov_surrogate_fn`` / ``make_scintillation_surrogate_fn``)
    close over it; ``model_params`` (a real pytree) is passed separately as
    a traced argument.
    """
    model:         SIREN
    energy_min:    float
    energy_max:    float
    axis2_min:     float          # opening angle (Cherenkov) or dE/dx (dedx)
    axis2_max:     float
    smax_dist_min: float          # s/s_max range the model was trained on
    smax_dist_max: float          #   (dataset_info['distance_range'])
    log_min:       float          # target_normalization log range
    log_max:       float
    s_max_fn:      Callable        # s_max(E_mev) -> mm
    n_photons_fn:  Optional[Callable]   # N_photons(E_mev) -> total photons/event (Cherenkov only)
    grid_bins:     int             # first-pass grid resolution (per axis)
    threshold:     float           # seed threshold, fraction of per-energy grid max
    seed_mode:     str = 'importance'  # 'uniform' (legacy: uniform over >=threshold bins) |
    #   'importance' (bins ∝ grid density; keeps the sub-threshold tail, threshold unused)


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
_DEFAULT_RAY_SAMPLING = {"grid_bins": 250, "threshold": 0.05, "seed_mode": "importance"}


def _build_siren_context(predictor, ray_sampling: dict | None,
                         axis2_key: str, require_nphot: bool) -> SirenContext:
    """Shared SIREN-context resolver — used by both Cherenkov and dE/dx
    surrogates.

    ``axis2_key`` is the ``dataset_info`` key that carries the 2nd-axis range
    (``'angle_range'`` for Cherenkov, ``'dedx_range'`` for dE/dx). The
    ``nphot`` block is required only for Cherenkov contexts; dE/dx contexts
    return ``n_photons_fn=None``.
    """
    rs = {**_DEFAULT_RAY_SAMPLING, **(ray_sampling or {})}
    meta = predictor.metadata
    dataset_info = predictor.dataset_info
    energy_min, energy_max = dataset_info['energy_range']
    axis2_min, axis2_max = dataset_info[axis2_key]
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
    if require_nphot and 'nphot' not in meta:
        raise ValueError(
            "trained-model metadata has no 'nphot' block — rebuild the h5 with "
            "the N_photons fit and re-sync the model metadata.")

    model = SIREN(**meta['model_config'])

    return SirenContext(
        model=model,
        energy_min=float(energy_min), energy_max=float(energy_max),
        axis2_min=float(axis2_min), axis2_max=float(axis2_max),
        smax_dist_min=float(dist_min), smax_dist_max=float(dist_max),
        log_min=float(target_norm['log_min']),
        log_max=float(target_norm['log_max']),
        s_max_fn=make_smax_fn(meta['smax']),
        n_photons_fn=make_power_law_fn(meta['nphot']) if require_nphot else None,
        grid_bins=int(rs['grid_bins']),
        threshold=float(rs['threshold']),
        seed_mode=str(rs['seed_mode']),
    )


def build_cherenkov_context(predictor,
                            ray_sampling: dict | None = None) -> SirenContext:
    """Resolve a Cherenkov ``SIRENPredictor`` into a :class:`SirenContext`.

    The 2nd axis is the opening angle (radians). ``n_photons_fn`` is the
    Cherenkov power-law fit from the ``nphot`` metadata block.

    ``ray_sampling`` is the ``ray_sampling`` block from ``siren_params.json``
    (``{'grid_bins', 'threshold'}``); missing keys fall back to
    ``_DEFAULT_RAY_SAMPLING``.
    """
    return _build_siren_context(predictor, ray_sampling,
                                axis2_key='angle_range', require_nphot=True)


def build_dedx_context(predictor,
                       ray_sampling: dict | None = None) -> SirenContext:
    """Resolve a dE/dx ``SIRENPredictor`` into a :class:`SirenContext`.

    The 2nd axis is dE/dx (keV/mm). ``n_photons_fn`` is ``None`` — the
    scintillation surrogate gets its absolute photon count from the medium's
    light-yield parameters (S, kB, C), not from a stored curve.
    """
    return _build_siren_context(predictor, ray_sampling,
                                axis2_key='dedx_range', require_nphot=False)
