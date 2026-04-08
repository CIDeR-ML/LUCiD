# Gradient Sweep Abstraction for LUCiD

## Context

Across ~92 notebooks, gradient sweep analysis has ~100-180 lines of copy-pasted boilerplate:
if/elif chains for parameter access in nested tuples, identical sweep loops, identical plotting.
The goal: reduce to ~10 lines while staying generic across reconstruction and calibration use cases.

## Design Philosophy

**No physics-specific convenience layer.** The reco/calib distinction is artificial at the sweep level -
both are just `loss_fn(params) -> scalar` with a set of parameters to sweep. Loss closures are 3 lines;
wrapping them in factories adds indirection without real savings. Instead:

1. **One powerful sweep engine** that works with any JAX-differentiable function
2. **Pre-built sweep specs as plain data** (lists of `SweepParam`) - reusable, composable, no factory functions
3. **`center` is optional** - auto-filled from `base_params` at sweep time, so specs are reusable across events

## Files to Create

```
LUCiD/tools/gradient_analysis/
  __init__.py     - Exports + docstring usage examples
  sweep.py        - SweepParam, SweepResult1D, SweepResult2D, sweep_1d, sweep_2d,
                    path helpers, numerical_gradient, pre-built sweep specs
  plotting.py     - plot_sweep_1d, plot_sweep_1d_multi, plot_sweep_2d
```

## Key Design: Path-Based Parameter Addressing

A `path` tuple addresses any element in a nested param tuple, eliminating all if/elif chains:

| Parameter   | Path     | Get              | Set                          | Grad Extract  |
|-------------|----------|------------------|------------------------------|---------------|
| Energy      | `(0,)`   | `params[0]`      | Replace `params[0]`          | `grad[0]`     |
| Position X  | `(1, 0)` | `params[1][0]`   | `params[1].at[0].set(val)`   | `grad[1][0]`  |
| Phi         | `(2, 1)` | `params[2][1]`   | `params[2].at[1].set(val)`   | `grad[2][1]`  |

```python
def get_param_value(params, path):    # Follow path to extract scalar
def set_param_value(params, path, v): # len(path)==1: replace tuple element; len>=2: .at[].set()
def get_grad_component(grad, path):   # Same indexing on grad pytree
```

## `sweep.py` - Core Engine

### `SweepParam` dataclass
```python
@dataclass
class SweepParam:
    name: str
    path: Tuple[int, ...]
    half_width: float
    center: Optional[float] = None    # If None, auto-filled from base_params in sweep_1d/2d
    num_points: int = 101
    unit: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def resolve(self, base_params):
        """Return a copy with center filled from base_params if needed."""

    @property
    def values(self) -> np.ndarray:
        """linspace(center-hw, center+hw, N) with min/max clipping. Requires center to be set."""

    @property
    def label(self) -> str:
        """'Energy (MeV)' formatted axis label."""
```

Making `center` optional is the key design choice. It separates *what* to sweep from *where* to center:
```python
# Define once, reuse across events:
RECO_SWEEPS = [
    SweepParam('Energy',    (0,),    half_width=200.0, unit='MeV', min_val=1.0),
    SweepParam('Position X',(1, 0),  half_width=1.5,   unit='m'),
    SweepParam('Theta',     (2, 0),  half_width=0.5,   unit='rad', min_val=0.01, max_val=3.13),
    SweepParam('Phi',       (2, 1),  half_width=0.5,   unit='rad'),
]
results_event1 = sweep_1d(loss_fn, event1_params, RECO_SWEEPS)
results_event2 = sweep_1d(loss_fn, event2_params, RECO_SWEEPS)  # Same specs, different center
```

### Pre-built sweep specs (plain data, not functions)
```python
# Standard reco params: (energy, position[3], direction[2])
RECO_SWEEPS = [
    SweepParam('Energy',    (0,),    half_width=200.0, unit='MeV', min_val=1.0),
    SweepParam('Position X',(1, 0),  half_width=1.5,   unit='m'),
    SweepParam('Position Y',(1, 1),  half_width=1.5,   unit='m'),
    SweepParam('Position Z',(1, 2),  half_width=1.5,   unit='m'),
    SweepParam('Theta',     (2, 0),  half_width=0.5,   unit='rad', min_val=0.01, max_val=3.13),
    SweepParam('Phi',       (2, 1),  half_width=0.5,   unit='rad'),
]

# Standard 5-param detector calibration
CALIB_SWEEPS = [
    SweepParam('Scatter Length',     (0,), half_width=2.0,   unit='m',  min_val=0.1),
    SweepParam('Wall Reflection',    (1,), half_width=0.15,  min_val=0.0, max_val=1.0),
    SweepParam('Sensor Reflection',  (2,), half_width=0.15,  min_val=0.0, max_val=1.0),
    SweepParam('Absorption Length',  (3,), half_width=2.0,   unit='m',  min_val=0.1),
    SweepParam('Tau GS',            (4,), half_width=0.005,  min_val=1e-6),
]
```

Users customize by slicing, replacing, or creating their own:
```python
# Subset
my_sweeps = [RECO_SWEEPS[0], RECO_SWEEPS[1], RECO_SWEEPS[4]]

# Override half_width
from dataclasses import replace
my_energy = replace(RECO_SWEEPS[0], half_width=500.0)

# Fully custom
qe_42 = SweepParam('QE Sensor 42', path=(6, 42), half_width=0.3, min_val=0.01)
```

### `SweepResult1D` and `SweepResult2D` dataclasses
```python
@dataclass
class SweepResult1D:
    param: SweepParam           # Resolved (with center filled in)
    values: np.ndarray          # (N,)
    losses: np.ndarray          # (N,)
    gradients: np.ndarray       # (N,) analytical from jax.grad
    numerical_grads: np.ndarray # (N,) auto-computed from losses via 5-point central diff

@dataclass
class SweepResult2D:
    param_x: SweepParam
    param_y: SweepParam
    x_values: np.ndarray        # (Nx,)
    y_values: np.ndarray        # (Ny,)
    losses: np.ndarray          # (Nx, Ny)
    grad_x: np.ndarray          # (Nx, Ny)
    grad_y: np.ndarray          # (Nx, Ny)
```

### `sweep_1d(loss_fn, base_params, sweep_params, jit_compile=True, show_progress=True)`
- Resolves centers from `base_params` for any SweepParam where `center is None`
- Wraps `loss_fn` with `jax.value_and_grad` + optional `jax.jit`
- Loops over each SweepParam's values: `set_param_value` -> `loss_and_grad_fn` -> `get_grad_component`
- NaN/Inf handling: store NaN, catch exceptions
- Returns `Dict[str, SweepResult1D]`

### `sweep_2d(loss_fn, base_params, param_x, param_y, ...)`
- Same but double loop over grid
- Validates `param_x.path != param_y.path`
- Returns `SweepResult2D`

### `numerical_gradient(losses, values)`
5-point central difference, matching `temperature_gradient_analysis.py:31-53`.

## `plotting.py` - Visualization

Style: `serif` font, `usetex=False`, `font.size=10` (matching existing notebooks).
All functions return `(fig, axes)` for further customization.

### `plot_sweep_1d(results, title=None, show_numerical=False, save_path=None)`
- 2 rows (loss, gradient) x N columns (parameters)
- Vertical dashed line at true value, horizontal zero line for gradients
- Optional numerical gradient overlay

### `plot_sweep_1d_multi(results_dict, title=None, colors=None, save_path=None)`
- `results_dict`: `{'T=0.0': sweep_results, 'T=0.1': sweep_results, ...}`
- Overlays conditions with different colors
- Matches multi-temperature pattern in `temperature_gradient_analysis.py:278-348`

### `plot_sweep_2d(result, title=None, show_streamlines=True, show_true_marker=True, save_path=None)`
- Heatmap (`imshow`) + gradient streamlines (`streamplot`) + true value star marker
- Normalized streamlines (direction-only) by default

## Usage Examples

### Reconstruction sweep (~8 lines vs ~180):
```python
from tools.gradient_analysis import RECO_SWEEPS, sweep_1d, plot_sweep_1d

# User writes their own loss closure - 3 lines, fully flexible
def loss_fn(p_params):
    sim_data = simulator(p_params, detector_params, key)
    return compute_softmin_loss(det_pts, *true_data, *sim_data, tau=0.05)

results = sweep_1d(loss_fn, true_params, RECO_SWEEPS)
plot_sweep_1d(results)
```

### Calibration sweep:
```python
def loss_fn(d_params):
    sim_data = simulator(source_params, d_params, key)
    return compute_softmin_loss(det_pts, *true_data, *sim_data, tau=0.05)

results = sweep_1d(loss_fn, true_detector_params, CALIB_SWEEPS)
plot_sweep_1d(results, title='Calibration Gradient Analysis')
```

### Multi-temperature comparison:
```python
sweep_params = [RECO_SWEEPS[0], RECO_SWEEPS[1], RECO_SWEEPS[4]]  # Energy, Pos X, Theta
results_by_temp = {}
for temp in [0.0, 0.1, 0.5]:
    sim = setup_event_simulator(config, Nphot, temperature=temp)
    true_data = jax.lax.stop_gradient(sim(true_params, det_params, key))
    def loss_fn(p, _sim=sim, _td=true_data):
        return compute_softmin_loss(det_pts, *_td, *_sim(p, det_params, key), tau=0.05)
    results_by_temp[f'T={temp}'] = sweep_1d(loss_fn, true_params, sweep_params)
plot_sweep_1d_multi(results_by_temp, title='Temperature Comparison')
```

### 2D sweep:
```python
energy = SweepParam('Energy', (0,), half_width=200.0, num_points=51, unit='MeV', min_val=1.0)
theta = SweepParam('Theta', (2, 0), half_width=1.0, num_points=51, unit='rad')
result = sweep_2d(loss_fn, true_params, energy, theta)
plot_sweep_2d(result, title='Energy vs Theta')
```

## Verification

1. Test with simple quadratic: `loss(params) = (params[0] - 3)**2 + (params[1][0] - 1)**2`
   - Verify parabolic loss, linear gradients, numerical matches analytical
2. Test `set_param_value` preserves tuple structure for JAX tracing
3. End-to-end with LUCiD simulator in a notebook, compare against existing plots

## Implementation Order

1. `sweep.py` (self-contained, no LUCiD imports - only jax, numpy, dataclasses, tqdm)
2. `plotting.py` (depends on sweep.py dataclasses + matplotlib)
3. `__init__.py` (wire exports)
