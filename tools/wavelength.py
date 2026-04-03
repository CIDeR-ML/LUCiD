"""
Wavelength-dependent optical properties for LUCiD.

This module is the canonical home for all wavelength-related physics:
  - Scattering lengths (Rayleigh symmetric, Mie asymmetric)
  - Absorption length
  - Quantum efficiency
  - Scatter direction sampling (Rayleigh and Mie phase functions)

Data sources:
  - SK calibration paper (arXiv:1307.0162), Eqs. 14-17, Table 3
  - SK PMT QE data (SK_QE.csv)
"""

import jax.numpy as jnp
import jax
import numpy as np
import os

from tools.utils import normalize, create_local_frame, solve_rayleigh_inverse_cdf


# ---------------------------------------------------------------------------
# QE interpolator — built once at module load
# ---------------------------------------------------------------------------

def _build_qe_interpolator():
    """Load SK_QE.csv and return a JAX-compatible interpolation function."""
    csv_path = os.path.join(os.path.dirname(__file__), "SK_QE.csv")
    data = np.loadtxt(csv_path, delimiter=",")
    wavelengths = data[:, 0]
    qe = data[:, 1]

    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    qe = qe[sort_idx] / 100.0  # percent -> fraction

    wl_knots = jnp.array(wavelengths)
    qe_knots = jnp.array(qe)
    wl_min = wl_knots[0]
    wl_max = wl_knots[-1]

    @jax.jit
    def get_qe(wavelength):
        """
        Interpolate QE at the given wavelength(s).

        Parameters
        ----------
        wavelength : float or jnp.ndarray
            Wavelength(s) in nm.

        Returns
        -------
        qe : same shape as input
            Quantum efficiency as a fraction (0 to 1).
            Returns 0 outside the tabulated range (~294-648 nm).
        """
        qe = jnp.interp(wavelength, wl_knots, qe_knots)
        in_range = (wavelength >= wl_min) & (wavelength <= wl_max)
        return jnp.where(in_range, qe, 0.0)

    return get_qe

_qe_interpolator = _build_qe_interpolator()


# ---------------------------------------------------------------------------
# Optical property lookups
# ---------------------------------------------------------------------------

def get_wavelength_dependent_scatter_absorption(wavelengths):
    """
    Compute wavelength-dependent optical properties using the SK empirical water model.
    From SK calibration paper (arXiv:1307.0162), Eqs. 14-17, Table 3.

    Parameters
    ----------
    wavelengths : float or jnp.ndarray
        Photon wavelengths in nm.

    Returns
    -------
    sym_scatter_length : jnp.ndarray
        Symmetric (Rayleigh) scattering length in meters: 1/alpha_sym
    asym_scatter_length : jnp.ndarray
        Asymmetric (forward Mie) scattering length in meters: 1/alpha_asym
    absorption_length : jnp.ndarray
        Absorption mean free path in meters: 1/alpha_abs
    """
    # --- Scattering ---
    # Symmetric scattering (Rayleigh + symmetric Mie), Eq. 16:
    #   alpha_sym(lambda) = P4/lambda^4 * (1 + P5/lambda^2)
    P4 = 8.51e7
    P5 = 1.14e5
    alpha_sym = P4 / (wavelengths ** 4) * (1.0 + P5 / (wavelengths ** 2))

    # Asymmetric scattering (forward Mie), Eq. 17:
    #   alpha_asym(lambda) = P6 * (1 + P7/lambda^4 * (lambda - P8)^2)
    P6 = 1.00e-4
    P7 = 4.62e6
    P8 = 392.0
    alpha_asym = P6 * (1.0 + P7 / (wavelengths ** 4) * (wavelengths - P8) ** 2)

    sym_scatter_length = 1.0 / alpha_sym
    asym_scatter_length = 1.0 / alpha_asym

    # --- Absorption ---
    # Eq. 14: alpha_abs(lambda) = P0 * P1 / lambda^4 + C(lambda)
    # Eq. 15: C = P0 * P2 * (lambda/500)^P3  for lambda <= 464 nm
    #         C = Pope & Fry (1997) data       for lambda >= 464 nm
    P0 = 0.624
    P1 = 2.96e7
    P2 = 3.24e-2
    P3 = 10.9

    # Pope & Fry (1997) pure water absorption coefficients (m^-1)
    pf_wavelengths = jnp.array([
        464.0, 470.0, 480.0, 490.0, 500.0, 510.0, 520.0, 530.0,
        540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 600.0, 610.0,
        620.0, 630.0, 640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0
    ])
    pf_absorption = jnp.array([
        0.0054, 0.0058, 0.0064, 0.0082, 0.0257, 0.0357, 0.0477, 0.0507,
        0.0558, 0.0638, 0.0708, 0.0799, 0.1080, 0.1570, 0.2440, 0.2890,
        0.3090, 0.3190, 0.3290, 0.3490, 0.4000, 0.4300, 0.4500, 0.5000, 0.6500
    ])

    C_powerlaw = P0 * P2 * (wavelengths / 500.0) ** P3
    C_popefry = jnp.interp(wavelengths, pf_wavelengths, pf_absorption)
    C = jnp.where(wavelengths <= 464.0, C_powerlaw, C_popefry)

    alpha_abs = P0 * P1 / (wavelengths ** 4) + C
    absorption_length = 1.0 / alpha_abs

    return sym_scatter_length, asym_scatter_length, absorption_length


def get_wavelength_dependent_qe(wavelengths):
    """
    Get the SK PMT quantum efficiency at given wavelength(s).

    Uses linear interpolation of the SK R3600 QE curve (SK_QE.csv).
    This is the net detection efficiency -- no separate absorbance
    normalization is needed since LUCiD applies QE as a single-stage
    multiplicative weight.

    Parameters
    ----------
    wavelengths : float or jnp.ndarray
        Photon wavelength(s) in nm.

    Returns
    -------
    qe : same shape as input
        Quantum efficiency as a fraction (0 to 1).
        Returns 0 outside ~294-648 nm.
    """
    return _qe_interpolator(wavelengths)


# ---------------------------------------------------------------------------
# Scatter direction functions
# ---------------------------------------------------------------------------

def compute_scatter_direction(incident_dir, rng_key):
    """
    Compute a new scattering direction using the Rayleigh phase function.

    P(cos_theta) proportional to (1 + cos^2(theta)), sampled via Cardano's
    analytical inverse CDF.

    Parameters
    ----------
    incident_dir : jnp.ndarray
        (3,) current photon direction (unit vector).
    rng_key : jax.random.PRNGKey
        JAX PRNG key.

    Returns
    -------
    new_dir : jnp.ndarray
        (3,) scattered direction (unit vector).
    """
    k1, k2 = jax.random.split(rng_key)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)

    cos_theta = solve_rayleigh_inverse_cdf(u1)
    sin_theta = jnp.sqrt(1 - cos_theta ** 2)
    phi = 2 * jnp.pi * u2
    local_dir = normalize(jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta,
    ]))
    frame = create_local_frame(incident_dir)
    return normalize(frame @ local_dir)


def hg_sample_cos_theta(u, g):
    """
    Sample cos(theta) from the Henyey-Greenstein phase function.

    P(cos_theta) = (1 - g^2) / (2 * (1 + g^2 - 2*g*cos_theta)^(3/2))

    Analytical inverse CDF:
      cos_theta = (1/(2g)) * [1 + g^2 - ((1-g^2)/(1 - g + 2*g*u))^2]

    Uses internal clamping of g to avoid division-by-zero at g=0 and
    denominator collapse at g=1, ensuring NaN-free gradients for all
    input values. Safe for use inside JAX's jit, grad, and vmap.

    Parameters
    ----------
    u : float
        Uniform random sample in [0, 1].
    g : float
        Asymmetry parameter in (-1, 1).
        g > 0: forward-peaked, g = 0: isotropic, g < 0: backward-peaked.

    Returns
    -------
    cos_theta : float
        Sampled cosine of the scattering angle, in [-1, 1].
    """
    # Clamp g to avoid division by zero (g=0) and denominator collapse (g=1)
    # in both forward pass and gradient computation
    g_safe = jnp.clip(g, 1e-4, 1.0 - 1e-4)

    term = (1.0 - g_safe**2) / (1.0 - g_safe + 2.0 * g_safe * u)
    cos_theta = (1.0 + g_safe**2 - term**2) / (2.0 * g_safe)

    return jnp.clip(cos_theta, -1.0, 1.0)


def compute_asym_scatter_direction(incident_dir, rng_key, g=0.95):
    """
    Compute a new scattering direction for asymmetric (Mie) scattering
    using the Henyey-Greenstein phase function.

    This replaces the simplified linear cos_theta model with the proper
    HG phase function used in Geant4's G4OpMieHG. The asymmetry parameter
    g is differentiable, enabling calibration via gradient descent.

    Parameters
    ----------
    incident_dir : jnp.ndarray
        (3,) current photon direction (unit vector).
    rng_key : jax.random.PRNGKey
        JAX PRNG key.
    g : float
        HG asymmetry parameter. Default 0.95 (strongly forward-peaked,
        consistent with SK water Mie scattering).

    Returns
    -------
    new_dir : jnp.ndarray
        (3,) scattered direction (unit vector).
    """
    k1, k2 = jax.random.split(rng_key)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)

    cos_theta = hg_sample_cos_theta(u1, g)
    sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta ** 2, 0.0))
    phi = 2 * jnp.pi * u2
    local_dir = normalize(jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta,
    ]))
    frame = create_local_frame(incident_dir)
    return normalize(frame @ local_dir)