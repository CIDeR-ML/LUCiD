"""Medium optical properties (material physics, NOT detector hardware).

MediumProperties holds the physical properties of the detection medium
(water, ice): refractive index, wavelength-dependent scattering and
absorption coefficients. QE is a detector property and lives elsewhere.
"""
import json
import os
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


class MediumProperties(NamedTuple):
    """Optical properties of the detection medium.

    Scalar fields are always populated.  Wavelength-dependent arrays are
    ``None`` until :func:`medium_on_grid` evaluates ``model`` on a grid.
    """
    material: str
    refractive_index: float
    speed_of_light: float                              # m/ns in this medium

    # Wavelength-dependent arrays (None = monochromatic mode)
    wavelength_grid: Optional[jnp.ndarray] = None      # (N,) nm
    scatter_coeff: Optional[jnp.ndarray] = None         # (N,) symmetric Rayleigh 1/m
    absorption_coeff: Optional[jnp.ndarray] = None      # (N,) 1/m
    refractive_index_curve: Optional[jnp.ndarray] = None  # (N,) n(lambda)
    mie_scatter_coeff: Optional[jnp.ndarray] = None     # (N,) asymmetric Mie 1/m
    mie_asymmetry: Optional[float] = None               # HG g parameter

    # --- Emission dispatch (static, non-differentiable; the multi-material merge) ---
    # Differentiable scintillation values (S/kB/C/tau) live on DetectorParams; these are
    # the setup-time knobs the simulator reads to pick Cherenkov vs scintillation surrogates.
    emission_processes: tuple = ("cherenkov",)          # ("cherenkov",) | ("cherenkov","scintillation")
    scintillation_lambda_min: float = 340.0             # Moyal sampling window (nm)
    scintillation_lambda_max: float = 550.0
    cherenkov_fraction: float = 0.5                     # n_photons split when both processes run
    moyal_loc: Optional[np.float32] = None              # Moyal emission-spectrum shape (nm);
    moyal_scale: Optional[np.float32] = None            # scintillating media only

    model: Optional[dict] = None                        # the parsed material JSON


# ---------------------------------------------------------------------------
# QE curve loading
# ---------------------------------------------------------------------------

class QECurve:
    """A PMT QE curve, read once from a JSON file with ``wavelengths_nm`` and ``qe_percent``.

    Calling it gives the QE fraction (0 to 1) at the given wavelengths, linear between knots and
    0 outside the tabulated range. ``bounds`` is the (min, max) knot and ``peak`` the largest QE
    (the curve's maximum, since it interpolates linearly).
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        wavelengths = np.array(data["wavelengths_nm"])
        qe_pct = np.array(data["qe_percent"])
        self.bounds = (float(wavelengths.min()), float(wavelengths.max()))
        self.peak = float(np.max(data["qe_percent"])) / 100.0

        sort_idx = np.argsort(wavelengths)
        wl_knots = jnp.array(wavelengths[sort_idx])
        qe_knots = jnp.array(qe_pct[sort_idx] / 100.0)  # percent -> fraction
        wl_min = float(wl_knots[0])
        wl_max = float(wl_knots[-1])

        @jax.jit
        def get_qe(wavelength):
            qe = jnp.interp(wavelength, wl_knots, qe_knots)
            in_range = (wavelength >= wl_min) & (wavelength <= wl_max)
            return jnp.where(in_range, qe, 0.0)

        self._get_qe = get_qe

    def __call__(self, wavelength):
        return self._get_qe(wavelength)


def load_qe_curve(json_path) -> QECurve:
    """Read a PMT QE curve JSON once; see :class:`QECurve`."""
    return QECurve(json_path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Default material lookup dir: config/materials/<material>.json (multi-material merge).
_MATERIALS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config", "materials")


def _scintillation_kwargs_from(data: dict) -> dict:
    """Extract the STATIC scintillation knobs from a material JSON.

    The differentiable values (light yield S/kB/C, timing tau) live on DetectorParams and are
    sourced via ``load_physics_config``; this returns what the simulator needs at setup to pick
    surrogates, the Moyal window and the Moyal shape. A material is non-scintillating if it lacks
    a ``"scintillation"`` block (→ Cherenkov-only defaults).
    """
    scint = data.get("scintillation")
    if scint is None:
        return {"emission_processes": ("cherenkov",)}
    procs = tuple(scint.get("emission_processes", ("cherenkov", "scintillation")))
    spec = scint.get("spectrum", {})
    split = scint.get("photon_split", {"cherenkov_fraction": 0.5})
    if "moyal_loc" not in spec or "moyal_scale" not in spec:
        raise ValueError(
            f"material {data['material']!r} enables scintillation but has no "
            "scintillation.spectrum.moyal_loc / moyal_scale: the surrogate cannot build its "
            "λ inverse-CDF sampler.")
    return {
        "emission_processes": procs,
        "scintillation_lambda_min": float(spec.get("lambda_min", 340.0)),
        "scintillation_lambda_max": float(spec.get("lambda_max", 550.0)),
        "cherenkov_fraction": float(split["cherenkov_fraction"]),
        "moyal_loc": np.float32(spec["moyal_loc"]),
        "moyal_scale": np.float32(spec["moyal_scale"]),
    }


def load_medium(material: str = "water", medium_model_path: str = None) -> MediumProperties:
    """Read a material JSON once: the scalar :class:`MediumProperties`, carrying the parsed
    file as ``model`` for :func:`medium_on_grid`.

    ``medium_model_path`` is the file; without it, ``config/materials/<material>.json``. The
    record's ``material`` is the file's own ``material`` key.
    """
    if medium_model_path is None:
        medium_model_path = os.path.join(_MATERIALS_DIR, f"{material}.json")
        if not os.path.exists(medium_model_path):
            raise ValueError(f"No material config at {medium_model_path!r}. Add a "
                             f"config/materials/{material}.json or pass medium_model_path.")
    with open(medium_model_path) as f:
        data = json.load(f)
    n = data["refractive_index"]
    c_vac = data["speed_of_light_vacuum_m_per_ns"]
    return MediumProperties(
        material=data["material"],
        refractive_index=n,
        speed_of_light=c_vac / n,
        model=data,
        **_scintillation_kwargs_from(data),
    )


def medium_on_grid(medium: MediumProperties, wavelength_grid) -> MediumProperties:
    """``medium`` with its wavelength-dependent arrays evaluated on ``wavelength_grid`` (nm)."""
    data = medium.model
    n = data["refractive_index"]
    wl = jnp.asarray(wavelength_grid, dtype=jnp.float32)

    # Symmetric (Rayleigh) scattering coefficient
    sc = data["scattering"]["symmetric"]
    P4, P5 = sc["P4"], sc["P5"]
    alpha_sym = P4 / (wl ** 4) * (1.0 + P5 / (wl ** 2))

    # Asymmetric (Mie) scattering coefficient
    ac = data["scattering"]["asymmetric"]
    P6, P7, P8 = ac["P6"], ac["P7"], ac["P8"]
    alpha_asym = P6 * (1.0 + P7 / (wl ** 4) * (wl - P8) ** 2)
    g = ac["default_asymmetry_g"]

    # Absorption coefficient
    ab = data["absorption"]
    P0, P1, P2, P3 = ab["P0"], ab["P1"], ab["P2"], ab["P3"]
    pf_wl = jnp.array(ab["pope_fry_wavelengths_nm"], dtype=jnp.float32)
    pf_abs = jnp.array(ab["pope_fry_absorption_m_inv"], dtype=jnp.float32)

    # Eq.(15) power-law fit (blue) spliced onto the Pope&Fry data (red) at 464 nm.
    # The two pieces differ by ~12% at the seam, so a hard switch leaves a kink; blend
    # them with a C2 smootherstep over a +-12 nm window. (The blend WEIGHT is C2; the
    # result inherits the piecewise-linear knot kinks of the interpolated Pope&Fry
    # table itself — tiny, but not strictly C2.)
    C_powerlaw = P0 * P2 * (wl / 500.0) ** P3
    C_popefry = jnp.interp(wl, pf_wl, pf_abs)
    _bw = 12.0
    _t = jnp.clip((wl - (464.0 - _bw)) / (2.0 * _bw), 0.0, 1.0)
    _s = _t * _t * _t * (_t * (_t * 6.0 - 15.0) + 10.0)   # smootherstep (C2)
    C = (1.0 - _s) * C_powerlaw + _s * C_popefry
    alpha_abs = P0 * P1 / (wl ** 4) + C

    return medium._replace(
        wavelength_grid=wl,
        scatter_coeff=alpha_sym,
        absorption_coeff=alpha_abs,
        refractive_index_curve=jnp.full_like(wl, n),  # constant n for now
        mie_scatter_coeff=alpha_asym,
        mie_asymmetry=g,
    )


def make_medium(material: str = "water",
                wavelength_grid: Optional[jnp.ndarray] = None,
                medium_model_path: str = None) -> MediumProperties:
    """:func:`load_medium`, then :func:`medium_on_grid` when a ``wavelength_grid`` is given."""
    medium = load_medium(material, medium_model_path)
    return medium if wavelength_grid is None else medium_on_grid(medium, wavelength_grid)

