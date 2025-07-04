import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Optional
from jax import vmap, jit
from functools import partial
import os
import json
from tools.utils import base_dir_path

def gaussian_kernel(rho: float, theta: float, d: float, r: float, sigma: float) -> float:
    """2D Gaussian distribution centered at the origin with std sigma.

    Parameters
    ----------
    rho : float
        Radial coordinate for integration
    theta : float
        Angular coordinate for integration
    d : float
        Distance from center
    r : float
        Circle radius
    sigma : float
        Standard deviation of the Gaussian

    Returns
    -------
    float
        Value of the Gaussian kernel at the given point
    """
    dist_sq = d ** 2 + rho ** 2 - 2 * d * rho * jnp.cos(theta)
    return (rho / (2 * jnp.pi * sigma ** 2)) * jnp.exp(-dist_sq / (2 * sigma ** 2))


def lorentz_kernel(rho: float, theta: float, d: float, r: float, gamma: float) -> float:
    """2D Lorentzian distribution centered at the origin.

    Parameters
    ----------
    rho : float
        Radial coordinate for integration
    theta : float
        Angular coordinate for integration
    d : float
        Distance from center
    r : float
        Circle radius
    gamma : float
        Width parameter of the Lorentzian

    Returns
    -------
    float
        Value of the Lorentzian kernel at the given point
    """
    dist_sq = d ** 2 + rho ** 2 - 2 * d * rho * jnp.cos(theta)
    return (gamma / (2 * jnp.pi)) * (rho / (gamma ** 2 + dist_sq) ** (3 / 2))


def get_cache_filename(r: float, sigma: float) -> str:
    """Generate a unique filename for caching results.

    Parameters
    ----------
    r : float
        Circle radius
    sigma : float
        Width parameter

    Returns
    -------
    str
        Cache filename
    """
    return f"gaussian_overlap_r{r:.6f}_sigma{sigma:.6f}.json"


def save_overlap_values(r: float, sigma: float, d_values: jnp.ndarray, f_values: jnp.ndarray) -> None:
    """Save overlap values to a cache file.

    Parameters
    ----------
    r : float
        Circle radius
    sigma : float
        Width parameter
    d_values : jnp.ndarray
        Array of distance values
    f_values : jnp.ndarray
        Array of overlap probabilities
    """
    # Create cache directory if it doesn't exist
    os.makedirs(base_dir_path()+'/spatial_overlap_integrals/', exist_ok=True)

    # Convert to Python lists for JSON serialization
    cache_data = {
        'r': float(r),
        'sigma': float(sigma),
        'd_values': d_values.tolist(),
        'f_values': f_values.tolist()
    }

    filename = os.path.join(base_dir_path()+'/spatial_overlap_integrals/', get_cache_filename(r, sigma))
    with open(filename, 'w') as f:
        json.dump(cache_data, f)


def load_overlap_values(r: float, sigma: float) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Load overlap values from cache if they exist.

    Parameters
    ----------
    r : float
        Circle radius
    sigma : float
        Width parameter

    Returns
    -------
    Optional[Tuple[jnp.ndarray, jnp.ndarray]]
        Cached values if they exist, None otherwise
    """
    print(base_dir_path()+'/spatial_overlap_integrals/')
    filename = os.path.join(base_dir_path()+'/spatial_overlap_integrals/', get_cache_filename(r, sigma))

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            cache_data = json.load(f)

        # Convert back to jnp arrays
        d_values = jnp.array(cache_data['d_values'])
        f_values = jnp.array(cache_data['f_values'])
        return d_values, f_values

    return None


@partial(jax.jit, device=jax.devices('cpu')[0])
def integral_f_of_d(d: float, r: float, sigma: float,
                    theta_vals: jnp.ndarray, rho_vals: jnp.ndarray) -> float:
    """Computes the double integral of the Gaussian kernel over polar coordinates.

    Parameters
    ----------
    d : float
        Distance from center
    r : float
        Circle radius
    sigma : float
        Standard deviation of the Gaussian
    theta_vals : jnp.ndarray
        Array of theta values for angular integration
    rho_vals : jnp.ndarray
        Array of rho values for radial integration

    Returns
    -------
    float
        Value of the double integral
    """

    def integrand_theta(theta):
        def integrand_rho(rho_):
            return gaussian_kernel(rho_, theta, d, r, sigma)

        return vmap(integrand_rho)(rho_vals)

    integrand = vmap(integrand_theta)(theta_vals)
    integral_theta = jnp.trapezoid(integrand, theta_vals, axis=0)
    integral_rho = jnp.trapezoid(integral_theta, rho_vals)
    return integral_rho


def precompute_lookup(r: float,
                      sigma: float,
                      n_theta: int = 1000,
                      n_rho: int = 1000,
                      num_dense: int = 150,
                      num_sparse: int = 50,
                      d_max_factor: float = 10.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precomputes lookup tables for overlap probability calculation.

    Parameters
    ----------
    r : float
        Circle radius
    sigma : float
        Standard deviation of the Gaussian
    n_theta : int, optional
        Number of points for angular integration, by default 1000
    n_rho : int, optional
        Number of points for radial integration, by default 1000
    num_dense : int, optional
        Number of points in transition region, by default 150
    num_sparse : int, optional
        Number of points outside transition region, by default 50
    d_max_factor : float, optional
        Maximum distance as multiple of radius, by default 10.0

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Arrays of distances and overlap probabilities
    """
    theta_vals = jnp.linspace(0, 2 * jnp.pi, n_theta)
    rho_vals = jnp.linspace(0, r, n_rho)

    # Calculate transition region
    transition_start = max(0, r - 3 * sigma)  # Ensure we don't go below 0
    transition_end = r + 3 * sigma

    # Dense spacing in transition region
    d_dense = jnp.linspace(transition_start, transition_end, num_dense)

    # Sparse spacing before and after transition region
    if transition_start > 0:
        d_sparse_before = jnp.linspace(0, transition_start, num_sparse // 2)[:-1]
    else:
        d_sparse_before = jnp.array([])

    d_sparse_after = jnp.linspace(transition_end, d_max_factor * r, num_sparse // 2)[1:]

    # Combine all regions
    d_values = jnp.concatenate((d_sparse_before, d_dense, d_sparse_after))

    def f_of_d(d_):
        return integral_f_of_d(d_, r, sigma, theta_vals, rho_vals)

    f_values = vmap(f_of_d)(d_values)
    return d_values, f_values


def create_overlap_prob(sigma: Optional[float],
                        r: float,
                        n_theta: int = 2000,
                        n_rho: int = 2000,
                        num_dense: int = 150,
                        num_sparse: int = 50,
                        d_max_factor: float = 10.0,
                        use_cache: bool = True) -> Callable[[float], float]:
    """Creates a function that calculates overlap probability between detector and photon.

    Parameters
    ----------
    sigma : Optional[float]
        Width parameter (if None or < 0.02*r, uses step function)
    r : float
        Radius (must be >0)
    n_theta : int, optional
        Number of points for angular integration, by default 2000
    n_rho : int, optional
        Number of points for radial integration, by default 2000
    num_dense : int, optional
        Number of points in transition region, by default 150
    num_sparse : int, optional
        Number of points outside transition region, by default 50
    d_max_factor : float, optional
        Maximum distance as multiple of radius, by default 10.0
    use_cache : bool, optional
        Whether to use cached values if available, by default True

    Returns
    -------
    Callable[[float], float]
        Function that takes distance d and returns overlap probability

    Raises
    ------
    ValueError
        If r is not positive
    """
    if r <= 0:
        raise ValueError("r must be positive.")

    # Use step function if sigma is None or very small
    if sigma is None or sigma < 0.02 * r:
        def overlap_prob(d: float) -> float:
            return jnp.where(d < r, 1.0, 0.0)

        return overlap_prob

    # Try to load from cache first
    if use_cache:
        cached_values = load_overlap_values(r, sigma)
        if cached_values is not None:
            d_values, f_values = cached_values
        else:
            d_values, f_values = precompute_lookup(
                r, sigma, n_theta, n_rho, num_dense, num_sparse, d_max_factor
            )
            save_overlap_values(r, sigma, d_values, f_values)
    else:
        d_values, f_values = precompute_lookup(
            r, sigma, n_theta, n_rho, num_dense, num_sparse, d_max_factor
        )

    def overlap_prob(d: float) -> float:
        d_clamped = jnp.clip(d, d_values[0], d_values[-1])
        return jnp.interp(d_clamped, d_values, f_values)

    return overlap_prob


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Set up parameters
    r = 0.04  # radius
    gamma_values = [0.01 * r, 0.02 * r, 0.05 * r, 0.1 * r, 0.2 * r, 0.4 * r, 0.5 * r, 0.8 * r, 1.0 * r]
    gamma_labels = [f'{g / r:.2f}r' for g in gamma_values]

    # Create figure with professional styling
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color map for different lines
    colors = plt.cm.viridis(jnp.linspace(0, 1, len(gamma_values)))

    # Generate d values from 0 to 8r
    d_values = jnp.linspace(0, 3 * r, 200)

    # Store values for table
    table_data = []
    table_d_values = [0, 2 * r, 3 * r, 4 * r]

    # Calculate and plot for each gamma
    for gamma, label, color in zip(gamma_values, gamma_labels, colors):
        overlap_func = create_overlap_prob(gamma, r)
        overlap_vmap = vmap(overlap_func)
        overlaps = overlap_vmap(d_values)

        # Plot with custom styling
        ax.plot(d_values / r, overlaps, '-', linewidth=2, label=f'σ = {label}', color=color)

        # Collect values for table
        table_values = [overlap_func(d) for d in table_d_values]
        table_data.append([label] + [f'{v:.2e}' for v in table_values])

    # Customize plot
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Distance (d/r)', fontsize=12)
    ax.set_ylabel('Overlap Probability', fontsize=12)
    ax.set_title('Overlap Probability vs Distance for Different σ Values (r = 0.04)', fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # Add reference lines
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.3)

    # Set reasonable axis limits
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.1, 1.1)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig('output/overlap_probability.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

    # Print table without tabulate
    print("\nOverlap Probability Values at Selected Distances:")
    print("-" * 80)
    print(f"{'σ':>10} {'d = 0':>15} {'d = 2r':>15} {'d = 3r':>15} {'d = 4r':>15}")
    print("-" * 80)
    for row in table_data:
        print(f"{row[0]:>10} {row[1]:>15} {row[2]:>15} {row[3]:>15} {row[4]:>15}")
    print("-" * 80)