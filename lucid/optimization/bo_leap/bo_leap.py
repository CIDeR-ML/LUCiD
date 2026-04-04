#!/usr/bin/env python3
"""
BO-LEAP: Bayesian Optimization with Local search via Evolution And gradients with Patience.

This algorithm combines:
1. Bayesian Optimization for global exploration
2. CMA-ES for population-based local search
3. Gradient descent for refined local optimization

The algorithm maintains structured history for detailed analysis and visualization.
"""

import jax
import jax.numpy as jnp
from functools import partial
from evosax.algorithms import CMA_ES
from .gp_predict import gp_predict_next_point
from tqdm import tqdm


def make_bounded_functions(loss_fn, lower_bounds, upper_bounds, clip_fn):
    """Create bounded loss and loss_and_grad functions with proper gradient scaling."""
    range_scale = upper_bounds - lower_bounds

    def denormalize(x_norm):
        """Transform from [0,1] to original space"""
        x_clipped = clip_fn(x_norm)
        return x_clipped * range_scale + lower_bounds

    def bounded_loss(x_norm):
        """Loss function in normalized [0,1] space"""
        x_original = denormalize(x_norm)
        return loss_fn(x_original)

    def bounded_loss_and_grad(x_norm):
        """Loss and gradient with proper scaling"""
        loss, grad = jax.value_and_grad(bounded_loss)(x_norm)
        return loss, grad

    def normalize(x):
        """Transform from original to [0,1] space"""
        return (x - lower_bounds) / range_scale

    return bounded_loss, bounded_loss_and_grad, normalize, denormalize


def make_cyclic_aware_clip(cyclic_mask):
    """Create a clip function that handles both regular and cyclic parameters."""

    def clip(x):
        # Regular parameters: clip to [0, 1]
        # Cyclic parameters: wrap using modulo
        return jnp.where(cyclic_mask, x % 1.0, jnp.clip(x, 0.0, 1.0))

    return clip


def clip_to_bounds(x):
    """Clip to [0,1] bounds."""
    return jnp.clip(x, 0.0, 1.0)


def bounded_gradient_step(x, grad, alpha, clip_fn):
    """Gradient step with clipping in [0,1] space."""
    x_proposed = x - alpha * grad
    return clip_fn(x_proposed)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def bounded_gradient_descent(s_init, loss_and_grad_fn, clip_fn, gradient_step_fn, J, alpha=0.1):
    """Gradient descent with early stopping."""

    def step_fn(carry, i):
        s, prev_loss, stopped = carry

        loss_new, grad = loss_and_grad_fn(s)
        s_new = gradient_step_fn(s, grad, alpha, clip_fn)

        should_continue = jnp.logical_and(
            ~stopped,
            jnp.logical_or(i < 3, loss_new < prev_loss - 1e-6)
        )

        s_next = jax.lax.cond(should_continue, lambda: s_new, lambda: s)

        return (s_next, loss_new, ~should_continue), (s, loss_new, should_continue)

    init_loss, _ = loss_and_grad_fn(s_init)
    _, (s_trajectory, s_losses, s_valid_mask) = jax.lax.scan(
        step_fn,
        (s_init, jnp.inf, False),
        jnp.arange(J)
    )

    s_trajectory = jnp.concatenate([s_init[None], s_trajectory])
    s_losses = jnp.concatenate([jnp.array([init_loss]), s_losses])
    s_valid_mask = jnp.concatenate([jnp.array([True]), s_valid_mask])

    return s_trajectory, s_losses, s_valid_mask


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def bo_leap_local_search(key, x_n, loss_fn, loss_and_grad_fn, clip_fn, gradient_step_fn,
                         alpha, K, J, LOCAL_STEPS):
    """
    BO-LEAP local search algorithm combining evolutionary strategies with gradient descent.

    This algorithm alternates between:
    1. CMA-ES population sampling and evaluation
    2. Gradient descent from the current mean

    Args:
        key: JAX random key for stochastic operations
        x_n: Initial point in normalized [0,1] space (shape: [n_dim])
        loss_fn: Loss function for population evaluation
        loss_and_grad_fn: Loss and gradient function for gradient descent
        clip_fn: Function to handle bounds/cyclic constraints
        gradient_step_fn: Function to perform gradient steps with clipping
        alpha: Step size for gradient descent
        K: Population size for CMA-ES
        J: Number of gradient descent steps per iteration
        LOCAL_STEPS: Number of outer iterations

    Returns:
        x_points: All population points evaluated (shape: [K*LOCAL_STEPS, n_dim])
        x_losses: Losses for all population points (shape: [K*LOCAL_STEPS])
        s_points: All gradient descent points (shape: [(J+1)*LOCAL_STEPS, n_dim])
        s_losses: Losses for gradient descent points (shape: [(J+1)*LOCAL_STEPS])
        s_masks_flat: Boolean mask for valid gradient points (shape: [(J+1)*LOCAL_STEPS])
        final_mean: Final mean of the CMA-ES distribution (shape: [n_dim])
    """

    es = CMA_ES(
        population_size=K,
        solution=x_n
    )

    params = es.default_params
    params = params.replace(std_init=1.0)

    key, init_key = jax.random.split(key)
    state = es.init(init_key, x_n, params)

    def local_iteration(carry, _):
        state, params, key = carry

        # Ask
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = es.ask(key_ask, state, params)
        population = clip_fn(population)

        # Evaluate using scan instead of vmap
        def eval_fn(_, x):
            return None, loss_fn(x)

        _, fitness = jax.lax.scan(eval_fn, None, population)

        # Tell
        state, metrics = es.tell(key_tell, population, fitness, state, params)

        # Gradient descent from best mean
        s_init = clip_fn(state.mean)

        s_trajectory, s_losses, s_valid_mask = bounded_gradient_descent(
            s_init, loss_and_grad_fn, clip_fn, gradient_step_fn, J, alpha
        )

        # Update mean with last valid point
        last_valid_idx = jnp.sum(s_valid_mask) - 1
        final_s = s_trajectory[last_valid_idx]
        mean_to_use = clip_fn(final_s)
        state = state.replace(mean=mean_to_use)

        return (state, params, key), (population, fitness, s_trajectory, s_losses, s_valid_mask)

    (final_state, _, _), (x_pops, x_fits, s_trajs, s_loss_trajs, s_masks) = jax.lax.scan(
        local_iteration,
        (state, params, key),
        None,
        length=LOCAL_STEPS
    )

    x_points = x_pops.reshape(-1, x_n.shape[0])
    x_losses = x_fits.reshape(-1)
    s_points = s_trajs.reshape(-1, x_n.shape[0])
    s_losses = s_loss_trajs.reshape(-1)
    s_masks_flat = s_masks.reshape(-1)

    return x_points, x_losses, s_points, s_losses, s_masks_flat, final_state.mean


def bo_leap_optimize(
        initial_guess,
        loss_fn,
        loss_and_grad_fn,
        clip_fn,
        gradient_step_fn,
        n_iterations=10,
        local_steps=20,
        K=10,  # CMA-ES population size
        J=10,  # Gradient descent steps
        M=100,
        alpha=0.1,
        key=jax.random.PRNGKey(0)
):
    """
    BO-LEAP optimization with structured history output.

    Args:
        initial_guess: Starting point in normalized [0,1] space
        loss_fn: Loss function for evaluation
        loss_and_grad_fn: Loss and gradient function
        clip_fn: Function to clip/wrap parameters to bounds
        gradient_step_fn: Function for gradient steps with clipping
        n_iterations: Number of BO-LEAP iterations
        local_steps: Steps for local search
        K: CMA-ES population size
        J: Number of gradient descent steps
        M: Max points for GP subset
        alpha: Step size for gradient descent
        key: JAX random key

    Returns:
        best_x: Best point found
        best_loss: Best loss value
        history: Dict with full optimization history including:
            - all_X: All points evaluated
            - all_y: All losses
            - n_evaluations: Total number of evaluations
            - iterations: List of dicts with detailed iteration structure
    """
    # Initialize storage
    all_X = []
    all_y = []
    iterations = []  # Structured iteration data
    best_x = None
    best_loss = jnp.inf
    total_evals = 0

    # Progress bar
    pbar = tqdm(range(n_iterations), desc="BO-LEAP")

    for iteration in pbar:
        # Get starting point for this iteration
        if iteration == 0:
            next_start = initial_guess
        else:
            X_train = jnp.array(all_X)
            y_train = jnp.array(all_y)

            key, subkey = jax.random.split(key)
            next_start, acq_value, selected_indices = gp_predict_next_point(
                X_train, y_train,
                M=M,
                n_candidates=2048,
                acquisition='ucb',
                beta=2.0,
                seed=int(subkey[0])
            )

        # Run local search
        key, subkey = jax.random.split(key)
        x_points, x_losses, s_points, s_losses, s_masks_flat, final_mean = bo_leap_local_search(
            subkey, next_start, loss_fn, loss_and_grad_fn,
            clip_fn, gradient_step_fn, alpha,
            K=K, J=J, LOCAL_STEPS=local_steps
        )

        # Collect valid gradient descent points
        valid_s_points = s_points[s_masks_flat]
        valid_s_losses = s_losses[s_masks_flat]

        # Record start index for this iteration
        iter_start_idx = len(all_X)

        # Combine all points from this iteration
        iter_X = jnp.concatenate([x_points, valid_s_points])
        iter_y = jnp.concatenate([x_losses, valid_s_losses])

        # Add to accumulated data
        all_X.extend(iter_X)
        all_y.extend(iter_y)

        # Record end index for this iteration
        iter_end_idx = len(all_X)

        # Store structured iteration data with evaluation order
        # Compute TRUE evaluation order accounting for interleaved local_steps
        # During execution: local_step_0: pop[K], grad[J+1], local_step_1: pop[K], grad[J+1], ...
        
        # Population indices - these are interleaved across local steps
        pop_eval_indices = []
        base_idx = iter_start_idx
        for ls in range(local_steps):
            # Each local step starts with K population evaluations
            for k in range(K):
                pop_eval_indices.append(base_idx)
                base_idx += 1
            # Then has gradient evaluations (we'll count valid ones)
            ls_grad_start = ls * (J + 1)
            ls_grad_end = (ls + 1) * (J + 1)
            n_valid_grads_this_step = jnp.sum(s_masks_flat[ls_grad_start:ls_grad_end])
            base_idx += int(n_valid_grads_this_step)
        pop_eval_indices = jnp.array(pop_eval_indices)
        
        # Gradient indices - also interleaved
        grad_eval_indices = []
        base_idx = iter_start_idx
        for ls in range(local_steps):
            # Skip past this local step's population
            base_idx += K
            # Now we're at gradient position for this local step
            ls_grad_start = ls * (J + 1)
            ls_grad_end = (ls + 1) * (J + 1)
            valid_mask_this_step = s_masks_flat[ls_grad_start:ls_grad_end]
            n_valid_this_step = jnp.sum(valid_mask_this_step)
            # Add indices for valid gradients in this local step
            for j in range(J + 1):
                if valid_mask_this_step[j]:
                    grad_eval_indices.append(base_idx)
                    base_idx += 1
        grad_eval_indices = jnp.array(grad_eval_indices)
        
        iteration_data = {
            'iteration': iteration,
            'x_points': x_points,
            'x_losses': x_losses,
            'x_eval_indices': pop_eval_indices,  # Evaluation order for population
            's_points': s_points,  # All gradient points (including invalid)
            's_losses': s_losses,  # All gradient losses
            's_masks': s_masks_flat,  # Valid mask
            'valid_s_points': valid_s_points,  # Only valid gradient points
            'valid_s_losses': valid_s_losses,  # Only valid gradient losses
            'valid_s_eval_indices': grad_eval_indices,  # Evaluation order for valid gradients
            'final_mean': final_mean,
            'start_idx': iter_start_idx,  # Index in all_X where this iteration starts
            'end_idx': iter_end_idx,  # Index in all_X where this iteration ends
            'n_population': len(x_points),
            'n_gradient_valid': len(valid_s_points),
            'n_gradient_total': len(s_points),
        }
        iterations.append(iteration_data)

        # Update tracking
        n_new_evals = len(iter_y)
        total_evals += n_new_evals

        # Update best point
        min_idx = jnp.argmin(iter_y)
        if iter_y[min_idx] < best_loss:
            best_loss = iter_y[min_idx]
            best_x = iter_X[min_idx]

        # Update progress bar
        pbar.set_postfix({
            'best_loss': f'{best_loss:.6f}',
            'total_evals': total_evals,
            'iter_evals': n_new_evals
        })

    # Return results with structured history
    history = {
        'all_X': jnp.array(all_X),
        'all_y': jnp.array(all_y),
        'n_evaluations': total_evals,
        'iterations': iterations,  # Structured iteration data
        'n_iterations': n_iterations,
        'K': K,
        'J': J,
        'local_steps': local_steps,
        'alpha': alpha
    }

    return best_x, best_loss, history


def setup_and_run_bo_leap(
        loss_fn_original,
        lower_bounds,
        upper_bounds,
        initial_guess=None,
        cyclic_mask=None,
        n_iterations=3,
        local_steps=20,
        K=10,
        J=10,
        M=100,
        alpha=0.1,
        key=jax.random.PRNGKey(0)
):
    """
    Setup bounded optimization and run BO-LEAP with structured history.

    Args:
        loss_fn_original: Original loss function in original space
        lower_bounds: Lower bounds for parameters
        upper_bounds: Upper bounds for parameters
        initial_guess: Starting point in original space (if None, uses center)
        cyclic_mask: Boolean array indicating which parameters are cyclic
        n_iterations: Number of BO-LEAP iterations
        local_steps: Steps for local search
        K: CMA-ES population size
        J: Number of max gradient descent steps
        M: Max points for GP subset
        alpha: Step size for gradient descent
        key: JAX random key

    Returns:
        best_x_original: Best point in original space
        best_loss: Best loss value
        history: Optimization history with structured iteration data
    """
    # Create clip function based on cyclic mask
    if cyclic_mask is not None:
        clip_fn = make_cyclic_aware_clip(cyclic_mask)
    else:
        clip_fn = clip_to_bounds

    # Create bounded loss functions and transformation functions
    bounded_loss, bounded_loss_and_grad, normalize, denormalize = make_bounded_functions(
        loss_fn_original, lower_bounds, upper_bounds, clip_fn
    )

    # Set initial guess (normalized to [0,1])
    if initial_guess is None:
        initial_guess = 0.5 * jnp.ones_like(lower_bounds)  # Center of [0,1] space
    else:
        initial_guess = normalize(initial_guess)

    # Run BO-LEAP optimization with structured history
    best_x_norm, best_loss, history = bo_leap_optimize(
        initial_guess=initial_guess,
        loss_fn=bounded_loss,
        loss_and_grad_fn=bounded_loss_and_grad,
        clip_fn=clip_fn,
        gradient_step_fn=bounded_gradient_step,
        n_iterations=n_iterations,
        local_steps=local_steps,
        K=K,
        J=J,
        M=M,
        alpha=alpha,
        key=key
    )

    # Transform best point back to original space
    best_x_original = denormalize(best_x_norm)

    # Also transform history points to original space
    history['all_X_original'] = jax.vmap(denormalize)(history['all_X'])
    
    # Transform iteration data points to original space
    for iter_data in history['iterations']:
        iter_data['x_points_original'] = jax.vmap(denormalize)(iter_data['x_points'])
        iter_data['valid_s_points_original'] = jax.vmap(denormalize)(iter_data['valid_s_points'])
        iter_data['s_points_original'] = jax.vmap(denormalize)(iter_data['s_points'])
        iter_data['final_mean_original'] = denormalize(iter_data['final_mean'])

    return best_x_original, best_loss, history


# Example usage:
if __name__ == "__main__":
    # Define your loss function in original space
    def my_loss(x):
        # Example: Rosenbrock function
        return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    # Set bounds
    n_dims = 10
    lower_bounds = -5.0 * jnp.ones(n_dims)
    upper_bounds = 5.0 * jnp.ones(n_dims)

    # Run optimization with structured history
    best_x, best_loss, history = setup_and_run_bo_leap(
        loss_fn_original=my_loss,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        n_iterations=5,
        local_steps=10,
        K=10,
        J=10,
        alpha=0.01,
        key=jax.random.PRNGKey(42)
    )

    print(f"Best loss: {best_loss:.6f}")
    print(f"Best point: {best_x}")
    print(f"Total evaluations: {history['n_evaluations']}")
    
    # Show structured iteration info
    print("\nIteration structure:")
    for iter_data in history['iterations']:
        print(f"  Iter {iter_data['iteration']}: "
              f"{iter_data['n_population']} population + "
              f"{iter_data['n_gradient_valid']}/{iter_data['n_gradient_total']} gradient points")