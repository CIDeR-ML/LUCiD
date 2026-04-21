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


def build_qe_weighted_cherenkov_sampler(qe_fn, lambda_min, lambda_max,
                                        n_grid=500):
    """Build an inverse-CDF sampler for QE(λ) · 1/λ² and the scalar
    ``<QE>_C = ∫QE(λ)/λ² dλ / ∫1/λ² dλ`` over ``[lambda_min, lambda_max]``.

    Importance-sampling the wavelength by the QE curve turns per-photon
    QE weighting into a single scalar multiplier, which is variance-optimal
    in the expected-value propagator and preserves all expected observables
    in the Bernoulli propagator (under the large-N / density-estimate
    interpretation of N — fluctuations are smoother than real per-shot
    Binomial noise, so this is not the right sampler if you want a literal
    physical-shot realization).

    Parameters
    ----------
    qe_fn : callable
        ``qe_fn(wavelength_nm) -> qe_fraction``. Zero outside its support.
    lambda_min, lambda_max : float
        Sampling range in nm. Must match the medium grid range.
    n_grid : int
        Resolution of the tabulated CDF.

    Returns
    -------
    sample_fn : callable
        ``sample_fn(key, n) -> (n,)`` array of wavelengths in nm.
    mean_qe : float
        Scalar ``<QE>_C`` used downstream as the constant per-photon QE
        weight that replaces the λ-dependent curve.
    """
    lam = jnp.linspace(lambda_min, lambda_max, n_grid)
    pdf_qc = qe_fn(lam) / (lam ** 2)
    z_qc = jnp.trapezoid(pdf_qc, lam)
    z_c = jnp.trapezoid(1.0 / (lam ** 2), lam)
    if float(z_qc) <= 0.0:
        raise ValueError(
            "QE curve is zero everywhere on the sampling range "
            f"[{lambda_min}, {lambda_max}] nm — cannot build QE-weighted "
            "Cherenkov sampler. Check the QE curve overlaps this range.")
    cdf_qc = jnp.cumsum(pdf_qc)
    cdf_qc = cdf_qc / cdf_qc[-1]
    mean_qe = float(z_qc / z_c)

    def sample_fn(key, n):
        u = jax.random.uniform(key, shape=(n,))
        return jnp.interp(u, cdf_qc, lam)

    return sample_fn, mean_qe
