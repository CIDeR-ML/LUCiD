"""Cherenkov spectrum sampling.

Provides wavelength sampling from the Cherenkov radiation spectrum
``dN/dlambda ~ 1/lambda^2`` via exact inverse-CDF sampling.
"""
import jax
import jax.numpy as jnp


def sample_cherenkov_wavelengths(key, n_photons,
                                 lambda_min=300.0, lambda_max=700.0):
    """Sample wavelengths from the Cherenkov spectrum dN/dlambda ~ 1/lambda^2.

    Uses exact inverse-CDF sampling (no rejection step needed).

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for sampling.
    n_photons : int
        Number of wavelengths to sample.
    lambda_min : float
        Minimum wavelength in nm (default: 300, lower edge of LUCiD's water
        medium grid). Callers in ``setup_event_simulator`` pass tighter
        bounds derived from the loaded QE curve.
    lambda_max : float
        Maximum wavelength in nm (default: 700, red edge of visible).

    Returns
    -------
    jnp.ndarray
        Shape ``(n_photons,)`` with wavelengths in nm.
    """
    u = jax.random.uniform(key, shape=(n_photons,))
    inv_min = 1.0 / lambda_min
    inv_max = 1.0 / lambda_max
    return 1.0 / (inv_min - u * (inv_min - inv_max))
