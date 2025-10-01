"""
Loss functions for optimization in LUCiD.
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy import special
from jax.scipy.special import gammaln
from functools import partial

@jit
def energy_loss(simulated_counts, true_counts):
    """
    Core energy loss computation using intensity matching.
    
    Parameters:
    -----------
    simulated_counts : jnp.ndarray
        Simulated counts values
    true_counts : jnp.ndarray
        True counts values
        
    Returns:
    --------
    float
        Energy loss value
    """
    total_true_counts = jnp.sum(true_counts)
    total_sim_counts = jnp.sum(simulated_counts)
    eps = 1e-8

    return jnp.abs(jnp.log(total_sim_counts / (total_true_counts + eps)))

@jit
def counts_loss(true: jnp.ndarray, pred: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Compute the Poisson negative log-likelihood loss.

    Parameters
    ----------
    true : jnp.ndarray
        Ground truth values (non-negative).
    pred : jnp.ndarray
        Predicted values (non-negative).
    eps : float, optional
        Small constant to prevent log(0), by default 1e-8.

    Returns
    -------
    jnp.ndarray
        Poisson negative log-likelihood loss.
    """
    nll = pred - true * jnp.log(pred + eps) + gammaln(true + 1.0)
    normalized_nll = jnp.sum(nll) / (jnp.sum(true) + eps)
    return normalized_nll


@jit
def origin_time_loss(origin, detector_positions, true_times, true_q, t0, photosensor_radius=0.25, c_medium=(0.299792/1.33), percentile_threshold=85., scale_w=1e8):
    """Vertex time loss component"""
    distances = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
    expected_times = (distances - photosensor_radius)/ c_medium
    time_residuals = true_times - expected_times - t0
    
    threshold = jnp.percentile(true_q, percentile_threshold)
    w = jnp.where(true_q > threshold, true_q, 0.)
    
    neg_time_res = jnp.where(time_residuals < 0, jnp.abs(time_residuals), 0.0)
    pos_time_res = jnp.where(time_residuals > 0, jnp.abs(time_residuals), 0.0)
    
    neg_loss_per_detector = neg_time_res * w 
    pos_loss_per_detector = pos_time_res * w 
    
    total_loss = jnp.sum(neg_loss_per_detector) / (jnp.sum(w) + 1e-8) + jnp.sum(jnp.abs(pos_loss_per_detector * w)) / (jnp.sum(w) + 1e-8) / scale_w

    return total_loss



