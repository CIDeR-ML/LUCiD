"""Gradient analysis sweep utilities for LUCiD."""

import warnings
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# SweepParam
# ---------------------------------------------------------------------------

@dataclass
class SweepParam:
    """Specification for a single parameter sweep axis."""

    name: str                          # Display name: 'X', 'Energy', 'Scatter Length'
    field: str                         # NamedTuple field: 'position', 'energy', ...
    half_width: float                  # Sweep range: center +/- half_width
    index: Optional[int] = None        # Sub-index for array fields (0,1,2 for position)
    center: Optional[float] = None     # Auto-filled from base_params if None
    num_points: int = 41               # Default scan resolution
    unit: str = ""                     # For axis labels: 'm', 'MeV', 'rad'
    min_val: Optional[float] = None    # Lower clamp
    max_val: Optional[float] = None    # Upper clamp
    grad_scale: float = 1.0            # Gradient scaling for streamline viz (plot time)

    def resolve(self, base_params):
        """Return copy with center filled from base_params if needed."""
        if self.center is not None:
            return self
        return replace(self, center=get_param_value(base_params, self))

    @property
    def values(self):
        """linspace(center-hw, center+hw, N) with min/max clipping."""
        lo = self.center - self.half_width
        hi = self.center + self.half_width
        if self.min_val is not None:
            lo = max(lo, self.min_val)
        if self.max_val is not None:
            hi = min(hi, self.max_val)
        return np.linspace(lo, hi, self.num_points)

    @property
    def label(self):
        """Formatted axis label."""
        if self.name == 'theta':
            return r'$\theta$ (rad)'
        if self.name == 'phi':
            return r'$\phi$ (rad)'
        return f'{self.name} ({self.unit})' if self.unit else self.name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_leaf(params, field):
    """Locate a (possibly nested) leaf ``field``.

    Returns ``(parent, attr)`` where ``getattr(parent, attr)`` is the leaf.
    Supports both flat NamedTuples (``ParticleParams``) and DetectorParams'
    physics-nested sub-tuples — ``field`` keys on the LEAF name (e.g.
    ``'scatter_length'``, ``'gain'``).
    """
    if field in getattr(type(params), "_fields", ()):
        return params, field
    for attr in getattr(type(params), "_fields", ()):
        child = getattr(params, attr)
        if field in getattr(type(child), "_fields", ()):
            return child, field
    raise AttributeError(f"{type(params).__name__} has no leaf field {field!r}")


def get_param_value(params, sp):
    """Extract scalar value for a SweepParam from a (possibly nested) NamedTuple."""
    parent, attr = _find_leaf(params, sp.field)
    val = getattr(parent, attr)
    return float(val[sp.index]) if sp.index is not None else float(val)


def set_param_value(params, sp, value):
    """Return params with one (possibly nested) leaf field replaced."""
    parent, attr = _find_leaf(params, sp.field)
    arr = getattr(parent, attr)
    idx = sp.index if sp.index is not None else ()
    new_parent = parent._replace(**{attr: arr.at[idx].set(value)})
    if parent is params:
        return new_parent  # flat case: leaf lived directly on params
    # Nested: find which top-level attr holds `parent` and replace it.
    for top_attr in type(params)._fields:
        if getattr(params, top_attr) is parent:
            return params._replace(**{top_attr: new_parent})
    raise AttributeError(f"could not locate sub-tuple for field {sp.field!r}")


def get_grad_component(grads, sp):
    """Extract the gradient component for a (possibly nested) SweepParam."""
    parent, attr = _find_leaf(grads, sp.field)
    val = getattr(parent, attr)
    return float(val[sp.index]) if sp.index is not None else float(val)


def numerical_gradient(losses, values):
    """2-point central difference."""
    grad = np.zeros(len(values))
    grad[1:-1] = (losses[2:] - losses[:-2]) / (values[2:] - values[:-2])
    grad[0] = (losses[1] - losses[0]) / (values[1] - values[0])
    grad[-1] = (losses[-1] - losses[-2]) / (values[-1] - values[-2])
    return grad


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SweepResult1D:
    param: SweepParam            # Resolved (center filled)
    values: np.ndarray           # (N,)
    losses: np.ndarray           # (N,)
    gradients: np.ndarray        # (N,) raw analytical gradients
    numerical_grads: np.ndarray  # (N,) 2-point central difference


@dataclass
class SweepResult2D:
    param_x: SweepParam
    param_y: SweepParam
    x_values: np.ndarray         # (Nx,)
    y_values: np.ndarray         # (Ny,)
    losses: np.ndarray           # (Nx, Ny)
    grad_x: np.ndarray           # (Nx, Ny) raw
    grad_y: np.ndarray           # (Nx, Ny) raw


# ---------------------------------------------------------------------------
# Sweep functions
# ---------------------------------------------------------------------------

def sweep_1d(
    loss_and_grad_fn,
    base_params,
    sweep_params: List[SweepParam],
) -> Dict[str, SweepResult1D]:
    """Perform 1D parameter sweeps.

    Args:
        loss_and_grad_fn: fn(params) -> (scalar_loss, grads_pytree)
        base_params: NamedTuple with true/center parameter values
        sweep_params: list of SweepParam specifications

    Returns:
        Dict[str, SweepResult1D] keyed by param name
    """
    results = {}
    for sp in sweep_params:
        sp = sp.resolve(base_params)
        vals = sp.values
        n = len(vals)
        losses = np.zeros(n)
        gradients = np.zeros(n)
        warned = False

        for i, v in enumerate(tqdm(vals, desc=sp.name)):
            params = set_param_value(base_params, sp, float(v))
            loss, grads = loss_and_grad_fn(params)
            l_val = float(loss)
            g_val = get_grad_component(grads, sp)

            if not warned and (not np.isfinite(l_val) or not np.isfinite(g_val)):
                warnings.warn(f"NaN/Inf detected in sweep for '{sp.name}'")
                warned = True

            losses[i] = l_val if np.isfinite(l_val) else np.nan
            gradients[i] = g_val if np.isfinite(g_val) else np.nan

        numerical_grads = numerical_gradient(losses, vals)

        results[sp.name] = SweepResult1D(
            param=sp,
            values=vals,
            losses=losses,
            gradients=gradients,
            numerical_grads=numerical_grads,
        )

    return results


def sweep_2d(
    loss_and_grad_fn,
    base_params,
    param_x: SweepParam,
    param_y: SweepParam,
    num_points: Optional[int] = None,
) -> SweepResult2D:
    """Perform 2D parameter sweep.

    Args:
        loss_and_grad_fn: fn(params) -> (scalar_loss, grads_pytree)
        base_params: NamedTuple with true/center parameter values
        param_x: SweepParam for x-axis
        param_y: SweepParam for y-axis
        num_points: optional override for both params' num_points

    Returns:
        SweepResult2D
    """
    px = param_x.resolve(base_params)
    py = param_y.resolve(base_params)

    if num_points is not None:
        px = replace(px, num_points=num_points)
        py = replace(py, num_points=num_points)

    x_vals = px.values
    y_vals = py.values
    nx, ny = len(x_vals), len(y_vals)

    losses = np.zeros((nx, ny))
    grad_x_arr = np.zeros((nx, ny))
    grad_y_arr = np.zeros((nx, ny))
    warned = False

    for i, vx in enumerate(tqdm(x_vals, desc=f"{px.name} x {py.name}")):
        for j, vy in enumerate(y_vals):
            params = set_param_value(base_params, px, float(vx))
            params = set_param_value(params, py, float(vy))
            loss, grads = loss_and_grad_fn(params)
            l_val = float(loss)
            gx = get_grad_component(grads, px)
            gy = get_grad_component(grads, py)

            if not warned and (
                not np.isfinite(l_val)
                or not np.isfinite(gx)
                or not np.isfinite(gy)
            ):
                warnings.warn(
                    f"NaN/Inf in 2D sweep for '{px.name}' x '{py.name}'"
                )
                warned = True

            losses[i, j] = l_val if np.isfinite(l_val) else np.nan
            grad_x_arr[i, j] = gx if np.isfinite(gx) else np.nan
            grad_y_arr[i, j] = gy if np.isfinite(gy) else np.nan

    return SweepResult2D(
        param_x=px,
        param_y=py,
        x_values=x_vals,
        y_values=y_vals,
        losses=losses,
        grad_x=grad_x_arr,
        grad_y=grad_y_arr,
    )


# ---------------------------------------------------------------------------
# Pre-built sweep specs
# ---------------------------------------------------------------------------

RECO_SWEEPS = [
    SweepParam('X',      'position', half_width=2.0,   index=0, unit='m',   grad_scale=0.05),
    SweepParam('Y',      'position', half_width=2.0,   index=1, unit='m',   grad_scale=0.05),
    SweepParam('Z',      'position', half_width=2.0,   index=2, unit='m',   grad_scale=0.05),
    SweepParam('t0',     't0',       half_width=5.0,   unit='ns'),
    SweepParam('Energy', 'energy',   half_width=100.0, unit='MeV', min_val=1.0, grad_scale=1000.0),
    SweepParam('theta',  'theta',    half_width=0.5,   unit='rad', grad_scale=0.005),
    SweepParam('phi',    'phi',      half_width=0.5,   unit='rad', grad_scale=0.005),
]


# ---------------------------------------------------------------------------
# Zero-crossing helper
# ---------------------------------------------------------------------------

def find_zero_crossing(values, gradients):
    """Find the first zero crossing of gradients via linear interpolation.

    Returns
    -------
    zero_val : float or np.nan
    """
    for i in range(len(gradients) - 1):
        g1, g2 = gradients[i], gradients[i + 1]
        if g1 == 0:
            return float(values[i])
        if g1 * g2 < 0:
            x1, x2 = values[i], values[i + 1]
            return float(x1 - g1 * (x2 - x1) / (g2 - g1))
    return np.nan

CALIB_SWEEPS = [
    SweepParam('Scatter Length',    'scatter_length',         half_width=20.0,  unit='m', min_val=0.001, grad_scale=100.0),
    SweepParam('Wall Reflection',   'wall_reflection_rate',   half_width=0.15,  min_val=0.0, max_val=1.0, grad_scale=0.1),
    SweepParam('Sensor Reflection', 'sensor_reflection_rate', half_width=0.08,  min_val=0.0, max_val=1.0, grad_scale=0.1),
    SweepParam('Absorption Length', 'absorption_length',      half_width=80.0,  unit='m', min_val=0.001, grad_scale=100.0),
]
