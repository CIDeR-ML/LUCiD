"""
Efficient Bayesian Optimization with subset selection for large datasets.
Uses GPJax for Gaussian Process modeling with smart point selection.
Works with any number of dimensions.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial
from jax.scipy.stats import norm
from scipy.stats import qmc
import gpjax as gpx


@partial(jit, static_argnums=(2, 3))
def select_subset_for_gp(X, y, M, exploit_fraction=0.25, key=None):
    """
    Select M points from N evaluated points for GP construction.
    Uses best points (exploitation) + random sampling (exploration).
    """
    N = X.shape[0]
    
    if key is None:
        key = jr.PRNGKey(0)
    
    # Calculate split between exploitation and exploration
    n_exploit = int(exploit_fraction * M)
    n_explore = M - n_exploit
    
    # Select best points (exploitation)
    sorted_indices = jnp.argsort(y)
    best_indices = sorted_indices[:n_exploit]
    
    # Select random points from remaining (exploration)
    remaining_pool = sorted_indices[n_exploit:]
    key, subkey = jr.split(key)
    n_remaining = N - n_exploit
    perm = jr.permutation(subkey, n_remaining)
    explore_indices = remaining_pool[perm[:n_explore]]
    
    # Combine exploitation and exploration indices
    selected_indices = jnp.concatenate([best_indices, explore_indices])
    
    return selected_indices


def setup_gp_model(n_dims, n_datapoints, lengthscale=0.2):
    """
    Setup GP model with fixed, sensible defaults for normalized data.
    
    For normalized data:
    - variance = 1.0 (since y has std=1 after normalization)
    - noise = 0.1 (10% of normalized std, good balance)
    """
    # Handle lengthscale - can be scalar or per-dimension
    if jnp.isscalar(lengthscale):
        lengthscale = jnp.ones(n_dims) * lengthscale
    
    # RBF kernel
    kernel = gpx.kernels.RBF(
        active_dims=list(range(n_dims)),
        lengthscale=lengthscale,
        variance=1.0  # Fixed for normalized data
    )
    
    # Zero mean function
    mean_function = gpx.mean_functions.Zero()
    
    # Create GP prior
    prior = gpx.gps.Prior(mean_function=mean_function, kernel=kernel)
    
    # Gaussian likelihood with fixed noise for normalized data
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=n_datapoints,
        obs_stddev=0.1  # Fixed 10% noise for normalized data
    )
    
    # Conjugate posterior
    posterior = prior * likelihood
    
    return posterior


def generate_sobol_candidates(n_candidates, n_dims, seed=42):
    """
    Generate well-distributed candidate points using Sobol sequences.
    """
    sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
    candidates = sampler.random(n_candidates)
    return jnp.array(candidates, dtype=jnp.float32)


@jit
def expected_improvement(mean, std, f_best):
    """
    Expected Improvement acquisition function.
    """
    # Ensure minimum std for numerical stability
    std = jnp.maximum(std, 1e-10)
    
    z = (f_best - mean) / std
    # Clip z to prevent numerical overflow
    z = jnp.clip(z, -10, 10)
    
    ei = (f_best - mean) * norm.cdf(z) + std * norm.pdf(z)
    ei = jnp.where(std > 1e-10, ei, 0.0)
    return ei


@jit  
def upper_confidence_bound(mean, std, beta=2.0):
    """
    Upper Confidence Bound acquisition function.
    Simple and robust.
    """
    return -(mean - beta * std)


def gp_predict_next_point(
        X_train,
        y_train,
        M=100,
        n_candidates=2000,
        acquisition='ucb',
        beta=2.0,
        lengthscale=0.2,
        exploit_fraction=0.25,
        seed=42
):
    """
    Simplified GP prediction for next point.
    
    Key simplifications:
    1. Always normalize y to mean=0, std=1
    2. Use fixed variance=1.0 and noise=0.1 for normalized data
    3. Simple fallback to random selection if GP fails
    
    Args:
        X_train: Training inputs (N, D) in [0, 1]
        y_train: Training outputs (N,) or (N, 1)
        M: Max points for GP subset (default: 100)
        n_candidates: Number of candidates to evaluate (default: 2000)
        acquisition: 'ei' or 'ucb' (default: 'ucb')
        beta: UCB exploration parameter (default: 2.0)
        lengthscale: RBF kernel lengthscale (default: 0.2)
        exploit_fraction: Fraction of best points in subset (default: 0.25)
        seed: Random seed (default: 42)
    
    Returns:
        next_point: Next point to evaluate (D,)
        acq_value: Acquisition value at that point
        selected_indices: Indices used for GP
    """
    # Ensure correct types
    X_train = jnp.asarray(X_train, dtype=jnp.float32)
    y_train = jnp.asarray(y_train, dtype=jnp.float32).squeeze()
    
    N, n_dims = X_train.shape
    
    # Select subset if N > M
    key = jr.PRNGKey(seed)
    if N > M:
        key, subkey = jr.split(key)
        selected_indices = select_subset_for_gp(
            X_train, y_train, M,
            exploit_fraction=exploit_fraction,
            key=subkey
        )
        X_subset = X_train[selected_indices]
        y_subset = y_train[selected_indices]
    else:
        X_subset = X_train
        y_subset = y_train
        selected_indices = jnp.arange(N)
    
    # Normalize y to mean=0, std=1
    y_mean = jnp.mean(y_subset)
    y_std = jnp.std(y_subset)
    
    # Handle zero variance case
    if y_std < 1e-6:
        y_std = 1.0  # Prevent division by zero
    
    y_normalized = (y_subset - y_mean) / y_std
    y_for_gp = y_normalized.reshape(-1, 1)
    
    # Setup GP with fixed parameters for normalized data
    actual_M = X_subset.shape[0]
    posterior = setup_gp_model(
        n_dims=n_dims,
        n_datapoints=actual_M,
        lengthscale=lengthscale
    )
    
    # Create training dataset
    train_data = gpx.Dataset(X=X_subset, y=y_for_gp)
    
    # Generate candidates
    candidates = generate_sobol_candidates(n_candidates, n_dims, seed)
    
    # Predict at candidates
    try:
        latent_dist = posterior.predict(candidates, train_data)
        predictive_dist = posterior.likelihood(latent_dist)
        
        # Get predictions in normalized space
        mean_norm = predictive_dist.mean.squeeze()
        var_norm = predictive_dist.variance.squeeze()
        
        # Ensure non-negative variance
        var_norm = jnp.maximum(var_norm, 0.0)
        std_norm = jnp.sqrt(var_norm)
        
        # Transform back to original scale
        mean = mean_norm * y_std + y_mean
        std = std_norm * y_std
        
        # Compute acquisition function
        if acquisition.lower() == 'ei':
            f_best = float(jnp.min(y_subset))
            acq_values = expected_improvement(mean, std, f_best)
        elif acquisition.lower() == 'ucb':
            acq_values = upper_confidence_bound(mean, std, beta)
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")
        
        # Check for NaN/Inf
        valid_mask = ~(jnp.isnan(acq_values) | jnp.isinf(acq_values))
        
        if jnp.sum(valid_mask) == 0:
            # Complete failure - random selection
            key, subkey = jr.split(key)
            best_idx = jr.choice(subkey, len(candidates))
            acq_value = 0.0  # Indicates fallback
        else:
            # Normal case - use best valid acquisition
            acq_values_safe = jnp.where(valid_mask, acq_values, -jnp.inf)
            best_idx = jnp.argmax(acq_values_safe)
            acq_value = float(acq_values_safe[best_idx])
            
            # Final safety check
            if jnp.isinf(acq_value) or jnp.isnan(acq_value):
                key, subkey = jr.split(key)
                best_idx = jr.choice(subkey, len(candidates))
                acq_value = 0.0
        
    except Exception:
        # GP failed - random selection
        key, subkey = jr.split(key)
        best_idx = jr.choice(subkey, len(candidates))
        acq_value = 0.0
    
    next_point = candidates[best_idx]
    
    # Ensure within bounds
    next_point = jnp.clip(next_point, 0.0, 1.0)
    
    return next_point, acq_value, selected_indices