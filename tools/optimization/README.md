# LUCiD Optimization Framework

This directory contains the unified optimization framework for LUCiD parameter reconstruction. The framework supports multiple optimization methods:

- **Numerical optimization** (adaptive search)
- **Gradient-based optimization** (using autodifferentiation)
- **Hybrid optimization** (combining numerical and gradient methods)

## Quick Start

### Basic Usage

```bash
# Numerical optimization (same as old adaptive_search.py)
python -m tools.optimization.optimize --mode numerical -i 40 -p 30 -n 10 -q

# Gradient-based optimization
python -m tools.optimization.optimize --mode gradient --n-gradient 100 --energy-lr 0.5

# Hybrid optimization (recommended)
python -m tools.optimization.optimize --mode hybrid -i 20 --n-gradient 50
```

### Advanced Usage

```bash
# Hybrid optimization with custom parameters
python -m tools.optimization.optimize --mode hybrid \
  --iterations 20 --population 50 \
  --n-gradient 100 --energy-lr 0.5 --spatial-lr 0.05 \
  --energy-scale 0.01 --position-scale 0.1 --direction-scale 0.1 \
  --patience 25 --events 10 --save-event-plots

# Test with different detector geometries
python -m tools.optimization.optimize --mode hybrid -d Sphere --events 5
python -m tools.optimization.optimize --mode hybrid -d Box --events 5
```

## Architecture

### Main Entry Point
- **`optimize.py`** - Master script with unified CLI interface

### Core Algorithm Module
- **`algorithms.py`** - All optimization algorithms (numerical + gradient-based)
  - Numerical: `adaptive_search`, `sample_around_point`
  - Gradient: `create_gradient_optimizer`, `gradient_step`, `gradient_optimization_with_patience`, `hybrid_optimization`

### Utility Modules (`utils/`)
- **`geometry.py`** - Cherenkov cone geometry calculations and surface intersections
- **`visualization.py`** - Plotting and visualization functions

### Legacy Files (for reference)
- **`../optimization_deprecated.py`** - Old gradient optimization implementation

## Optimization Modes

### 1. Numerical Optimization (`--mode numerical`)
- Uses adaptive search with population-based optimization
- Good for initial exploration and global optimization
- Parameters: `--iterations`, `--population`

### 2. Gradient-Based Optimization (`--mode gradient`)
- Uses autodifferentiation on the `compute_softmin_loss` function
- Fast convergence near optima
- Parameters: `--n-gradient`, `--energy-lr`, `--spatial-lr`, `--energy-scale`, `--position-scale`, `--direction-scale`, `--patience`

### 3. Hybrid Optimization (`--mode hybrid`, recommended)
- Combines numerical and gradient methods
- First runs numerical optimization to get close to optimum
- Then uses gradient optimization for final convergence
- Uses parameters from both modes

## Parameter Guidelines

### Gradient Optimization Parameters
- **Learning rates**: Start with `--energy-lr 1.0` and `--spatial-lr 0.1`
- **Gradient scales**: Use `--energy-scale 0.01`, `--position-scale 0.1`, `--direction-scale 0.1`
- **Patience**: Use `--patience 20` for learning rate reduction

### Numerical Optimization Parameters
- **Iterations**: Use `--iterations 20-40` for initial exploration
- **Population**: Use `--population 20-50` depending on complexity

## Migration from Old Scripts

### From `adaptive_search.py`:
```bash
# Old command
python -m tools.optimization.adaptive_search -i 40 -p 30 -n 10 -q --save-event-plots

# New command (identical functionality)
python -m tools.optimization.optimize --mode numerical -i 40 -p 30 -n 10 -q --save-event-plots
```

### From old `optimization.py`:
```bash
# Old gradient optimization is now available as:
python -m tools.optimization.optimize --mode gradient --n-gradient 100

# Or better yet, use hybrid:
python -m tools.optimization.optimize --mode hybrid --n-gradient 100
```

## Technical Details

### Gradient Computation
- Uses JAX's `jax.grad` on the `compute_softmin_loss` function
- Computes gradients for all parameters: energy, position, and direction angles
- Implements parameter-specific scaling to handle different gradient magnitudes

### Patience-Based Learning Rate Reduction
- Monitors loss improvement over iterations
- Reduces learning rates when loss plateaus
- Separate tracking for energy and spatial parameters

### Parameter Scaling
Following the pattern from the original optimization.py:
```python
scaled_energy_update = jax.tree.map(lambda x: x * energy_lr_multiplier, energy_update)
scaled_position_update = jax.tree.map(lambda x: x * spatial_lr_multiplier, position_update)
scaled_direction_update = jax.tree.map(lambda x: x * spatial_lr_multiplier, direction_update)
```

## Output

The optimization produces:
- **Progress information** during optimization
- **Final results** with position, direction, and energy errors
- **Summary statistics** for multiple events
- **Visualization plots** (if `--save-event-plots` is used)
- **Convergence plots** for multi-event runs

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating different optimization modes and parameter tuning strategies.