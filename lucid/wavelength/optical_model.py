"""Per-photon optical model — the single, testable λ → optical-property seam.

``evaluate_optical_model`` maps per-photon wavelengths to the per-photon optical
lengths (Rayleigh scatter, Mie scatter, absorption) and QE weight that the photon
transport step consumes. It is a **pure** function of ``(DetectorParams, wavelengths,
MediumProperties)`` — all λ-sampling lives upstream (the source / simulator), so the
model only ever *evaluates* at the wavelengths it is handed.

Two regimes, one return type (:class:`OpticalArrays`):

- ``wavelengths is None`` — monochromatic / scalar mode: broadcast the (fittable)
  ``DetectorParams`` optical scalars to all photons. QE weight is ``None`` (the scalar
  QE enters in ``make_hits``).
- ``wavelengths`` an ``(n,)`` array — wavelength mode: evaluate the medium reference
  curves at each photon's λ (``length = 1 / coeff(λ)``) and, if a ``qe_fn`` is given,
  the per-photon QE weight ``qe_fn(λ)``.

The design target (see ``docs/WAVELENGTH_DESIGN.md``) is one decomposition
``prop(λ) = reference(λ) · deviation(λ; DetectorParams)`` where the reference is fixed
medium physics and the deviation is the fittable curve. This module is that seam; the
deviation layer is added on top of it without changing this contract (deviation ≡ 1
reproduces the medium reference exactly).
"""

from typing import NamedTuple, Optional

import jax.numpy as jnp


# Control wavelengths (nm) at which the fittable λ-deviation curves are anchored.
# These are the SK calibration-laser lines; the deviation field of every optical
# property is a value PER control wavelength, interpolated to each photon's λ.
# A FIXED grid (not a DetectorParams field) so the curve length and this grid
# agree by construction. ``dev ≡ 1`` reproduces the pure medium reference.
CONTROL_WAVELENGTHS_NM = (337.0, 375.0, 398.0, 405.0, 445.0)
N_CONTROL = len(CONTROL_WAVELENGTHS_NM)


def _deviation(wl, control_lambda, curve):
    """Per-photon multiplicative deviation = interp(λ, control_λ, curve).

    ``curve`` is the (n_ctrl,) DetectorParams leaf; an all-ones curve gives a
    flat deviation of 1 (pure reference). ``jnp.interp`` clamps λ outside the
    control grid to the endpoint values (flat extrapolation).
    """
    return jnp.interp(wl, jnp.asarray(control_lambda), curve)


class OpticalArrays(NamedTuple):
    """Per-photon optical properties consumed by the transport step.

    Fields
    ------
    scatter_len : jnp.ndarray   (n,) Rayleigh scattering length, m
    mie_len : jnp.ndarray       (n,) Mie scattering length, m
    abs_len : jnp.ndarray       (n,) absorption length, m
    qe : jnp.ndarray or None    (n,) per-photon QE weight, or None in scalar mode
    """
    scatter_len: jnp.ndarray
    mie_len: jnp.ndarray
    abs_len: jnp.ndarray
    qe: Optional[jnp.ndarray]


def evaluate_optical_model(detector_params, wavelengths, medium, n_photons,
                           *, qe_fn=None, control_lambda=CONTROL_WAVELENGTHS_NM):
    """Evaluate per-photon optical properties.

    Parameters
    ----------
    detector_params : DetectorParams
        Nested pytree; the scattering/absorption sub-tuples supply the scalar
        optical lengths used in monochromatic mode.
    wavelengths : jnp.ndarray or None
        Per-photon wavelengths (nm), shape ``(n_photons,)``. ``None`` selects the
        scalar (monochromatic) mode.
    medium : MediumProperties
        Medium reference curves. Must carry ``wavelength_grid`` and the
        ``*_coeff`` arrays when ``wavelengths`` is not ``None``.
    n_photons : int
        Number of photons (used to broadcast scalars in monochromatic mode).
    qe_fn : callable, optional
        ``qe_fn(λ) -> qe_fraction``. Applied per-photon in wavelength mode only.
    control_lambda : sequence of float
        Control wavelengths (nm) anchoring the DetectorParams λ-deviation curves.
        Defaults to :data:`CONTROL_WAVELENGTHS_NM`.

    Returns
    -------
    OpticalArrays

    Notes
    -----
    In wavelength mode every optical length is the fixed medium reference scaled
    by a fittable multiplicative deviation:
    ``length(λ) = (1 / coeff_ref(λ)) / interp(λ, control_λ, dev_curve)``. The
    deviation curves are DetectorParams leaves (``scattering.rayleigh_dev`` /
    ``scattering.mie_dev`` / ``absorption.abs_dev`` / ``response.qe_dev``); an
    all-ones curve gives ``dev ≡ 1`` and reproduces the pure medium reference
    exactly. Monochromatic mode (``wavelengths is None``) uses the scalar fields
    and ignores the deviation curves.
    """
    sp = detector_params.scattering
    ab = detector_params.absorption
    rp = detector_params.response

    if wavelengths is None:
        # Monochromatic / scalar mode: the fittable DetectorParams scalars,
        # broadcast to every photon. QE weight is None (scalar QE → make_hits).
        return OpticalArrays(
            scatter_len=jnp.full(n_photons, sp.scatter_length),
            mie_len=jnp.full(n_photons, sp.mie_scatter_length),
            abs_len=jnp.full(n_photons, ab.absorption_length),
            qe=None,
        )

    # Wavelength mode: clamp λ into the medium grid, then length = 1 / coeff(λ),
    # scaled by the per-property multiplicative deviation curve (dev ≡ 1 default).
    wl = jnp.clip(jnp.asarray(wavelengths),
                  medium.wavelength_grid[0], medium.wavelength_grid[-1])
    sc = jnp.interp(wl, medium.wavelength_grid, medium.scatter_coeff)
    asym = jnp.interp(wl, medium.wavelength_grid, medium.mie_scatter_coeff)
    ac = jnp.interp(wl, medium.wavelength_grid, medium.absorption_coeff)

    dev_r = _deviation(wl, control_lambda, sp.rayleigh_dev)
    dev_m = _deviation(wl, control_lambda, sp.mie_dev)
    dev_a = _deviation(wl, control_lambda, ab.abs_dev)

    qe = None
    if qe_fn is not None:
        qe = qe_fn(wl) * _deviation(wl, control_lambda, rp.qe_dev)

    return OpticalArrays(
        scatter_len=(1.0 / (sc + 1e-30)) / dev_r,
        mie_len=(1.0 / (asym + 1e-30)) / dev_m,
        abs_len=(1.0 / (ac + 1e-30)) / dev_a,
        qe=qe,
    )
