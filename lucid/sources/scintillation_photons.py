"""Scintillation segment → photon expansion (data-mode).

Sibling of the SIREN-based track-mode surrogate in :mod:`siren_rays`. For
each Geant4 step stored on the PhotonSim TTree, draw
``N_i ~ Poisson(S · edep_i / (1 + kB·d_i + C·d_i²))`` scintillation
photons with ``d_i = (edep_i / Δs_i)`` in keV/mm (Birks/Chou). Per photon:

- **origin**: uniform along ``[Start, End]`` of its parent segment;
- **direction**: isotropic (cos θ ~ U[-1, 1], φ ~ U[0, 2π]);
- **wavelength**: inverse-CDF sample of the medium's Moyal emission
  spectrum truncated to ``[lambda_min, lambda_max]``;
- **time**: ``segment_time + u·Δs / (β·c) + hypoexp(τ_rise, τ_fall)``,
  with the same ``u`` used for the origin so position and time within
  the step stay consistent.

Pure NumPy, no JAX dependence. Fully vectorized via ``np.repeat`` — no
Python loop over segments. The output dict matches the per-photon key
names that :func:`lucid.sources.root_reader._read_event_raw` produces
for Cherenkov photons, plus an ``photon_emission_process`` column
tagged as :data:`lucid.sources.v3_writer.EMISSION_PROCESS_SCINTILLATION`.

See ``.claude/plans/scintillation-data-mode.md`` for the wider design.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np

from lucid.sources.v3_writer import EMISSION_PROCESS_SCINTILLATION

__all__ = [
    "expand_segments_to_photons",
    "scintillation_medium_params",
]


def scintillation_medium_params(detector_params, medium) -> dict:
    """Assemble the ``medium_params`` dict for :func:`expand_segments_to_photons`
    from a simulator's ``DetectorParams`` + ``MediumProperties``.

    The scintillation scalars live on the nested ``DetectorParams.scintillation``
    sub-tuple; the spectrum truncation window comes from the medium. Mirrors the
    inline assembly in ``event_generation.py`` so the data-mode loaders and the
    production path stay in sync.
    """
    sc = detector_params.scintillation
    return {
        'S':           float(sc.S),
        'kB':          float(sc.kB),
        'C':           float(sc.C),
        'tau_rise':    float(sc.tau_rise),
        'tau_fall':    float(sc.tau_fall),
        'moyal_loc':   float(sc.moyal_loc),
        'moyal_scale': float(sc.moyal_scale),
        'lambda_min':  float(medium.scintillation_lambda_min),
        'lambda_max':  float(medium.scintillation_lambda_max),
    }


_C_MM_PER_NS = 299.792458    # speed of light in vacuum, mm/ns
_EPS_DELTA_S_MM = 1e-9       # segments shorter than this contribute zero photons


# ---------------------------------------------------------------------------
# Per-photon RNG primitives — NumPy ports of the JAX helpers in siren_rays.
# Pure-stochastic (no LHS) per the data-mode convention: per-direction /
# per-wavelength stratification offers no variance reduction when each ray
# already carries an independent Poisson + QE Bernoulli stochasticity.
# ---------------------------------------------------------------------------


def _moyal_pdf_np(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    """Moyal PDF — NumPy port of :func:`siren_rays._moyal_pdf`."""
    z = (x - loc) / scale
    return (1.0 / (scale * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * (z + np.exp(-z)))


@lru_cache(maxsize=8)
def _build_moyal_inverse_cdf_np(lambda_min: float, lambda_max: float,
                                 moyal_loc: float, moyal_scale: float,
                                 n_knots: int = 512):
    """Precompute the inverse CDF of the Moyal PDF truncated to ``[λmin, λmax]``.

    Cached on (window, params, n_knots) — repeated calls within an event /
    across events of the same dataset reuse the same arrays. Mirrors
    :func:`siren_rays._build_moyal_inverse_cdf`.
    """
    lambda_grid = np.linspace(lambda_min, lambda_max, n_knots, dtype=np.float64)
    pdf = _moyal_pdf_np(lambda_grid, moyal_loc, moyal_scale)
    bw = lambda_grid[1:] - lambda_grid[:-1]
    cdf = np.concatenate([np.zeros(1, dtype=np.float64),
                          np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * bw)])
    cdf /= cdf[-1]
    return cdf, lambda_grid


def _sample_isotropic_np(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample ``n`` isotropic unit vectors. ``cos θ ~ U[-1, 1]``, ``φ ~ U[0, 2π]``."""
    cos_theta = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
    return np.stack([sin_theta * np.cos(phi),
                     sin_theta * np.sin(phi),
                     cos_theta], axis=1).astype(np.float32)


def _sample_hypoexp_np(rng: np.random.Generator, n: int,
                        tau_rise: float, tau_fall: float) -> np.ndarray:
    """Sample ``n`` times from the rise+fall biexp PDF via the hypoexp sum.

    ``T = -τ_rise·log(U₁) − τ_fall·log(U₂)`` with ``U₁, U₂ ~ U[0, 1]``.
    The tiny-floor on the uniform draw keeps ``log(U)`` finite at ``U == 0``.
    """
    tiny = np.finfo(np.float32).tiny
    u1 = rng.uniform(tiny, 1.0, size=n)
    u2 = rng.uniform(tiny, 1.0, size=n)
    return (-float(tau_rise) * np.log(u1)
            + -float(tau_fall) * np.log(u2)).astype(np.float32)


def _sample_moyal_np(rng: np.random.Generator, n: int,
                      moyal_loc: float, moyal_scale: float,
                      lambda_min: float, lambda_max: float) -> np.ndarray:
    """Inverse-CDF Moyal wavelength sample on ``[lambda_min, lambda_max]``."""
    cdf, lambda_grid = _build_moyal_inverse_cdf_np(
        float(lambda_min), float(lambda_max),
        float(moyal_loc), float(moyal_scale))
    u = rng.uniform(0.0, 1.0, size=n)
    return np.interp(u, cdf, lambda_grid).astype(np.float32)


# ---------------------------------------------------------------------------
# Empty-result template
# ---------------------------------------------------------------------------


def _empty_photon_dict() -> dict:
    """Zero-row arrays of the correct dtypes for the no-photon-this-event path."""
    return {
        'photon_origins':           np.zeros((0, 3), dtype=np.float32),
        'photon_directions':        np.zeros((0, 3), dtype=np.float32),
        'photon_times':             np.zeros(0, dtype=np.float32),
        'photon_wavelengths':       np.zeros(0, dtype=np.float32),
        'photon_segment_index_raw': np.zeros(0, dtype=np.int64),
        'photon_emission_process':  np.zeros(0, dtype=np.int8),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_segments_to_photons(segments: Mapping,
                                medium_params: Mapping,
                                rng: np.random.Generator) -> dict:
    """Expand a per-event segment table into a per-photon array dict.

    Parameters
    ----------
    segments : Mapping
        Per-event raw segment table — same shape as the ``segments_raw``
        block of :func:`lucid.sources.root_reader._read_event_raw`. Required
        keys (1-D ndarray of length ``n_segments`` unless noted):

        - ``start_x_mm``, ``start_y_mm``, ``start_z_mm`` : float, mm
        - ``end_x_mm``,   ``end_y_mm``,   ``end_z_mm``   : float, mm
        - ``edep``        : float, MeV — per-step energy deposit
        - ``time``        : float, ns — pre-step time (G4 frame)
        - ``beta_start``  : float, unitless β at pre-step

    medium_params : Mapping
        Scintillation scalars (units per ``DetectorParams`` docstring):

        - ``S``           : ph / MeV
        - ``kB``          : mm / keV
        - ``C``           : (mm / keV)²  — set to 0 for Birks-only
        - ``tau_rise``    : ns
        - ``tau_fall``    : ns
        - ``moyal_loc``   : nm
        - ``moyal_scale`` : nm
        - ``lambda_min``  : nm — truncation window lower bound
        - ``lambda_max``  : nm — truncation window upper bound

    rng : np.random.Generator
        Per-event RNG. Phase 2 derives this from the per-event JAX master
        key so reruns at the same master_seed are byte-equivalent.

    Returns
    -------
    dict
        Per-photon arrays plus the new emission-process tag column.
        Keys match what :func:`root_reader._read_event_raw` returns for
        Cherenkov photons (so callers can splice the two streams together):

        - ``photon_origins``           : (N_total, 3) float32, meters
        - ``photon_directions``        : (N_total, 3) float32, unit vectors
        - ``photon_times``             : (N_total,)   float32, ns (G4 frame —
                                          caller adds t0 to move into the
                                          detector frame, same as Cherenkov)
        - ``photon_wavelengths``       : (N_total,)   float32, nm
        - ``photon_segment_index_raw`` : (N_total,)   int64, into the
                                          ``segments`` row space
        - ``photon_emission_process``  : (N_total,)   int8,
                                          ``EMISSION_PROCESS_SCINTILLATION``

        All length-0 when no segment produces a Poisson sample.

    Notes
    -----
    Segments with ``Δs ≤ ε``, ``edep ≤ 0``, or ``β ≤ 0`` are masked (their
    ``N̄`` is forced to 0 and Poisson always yields 0 photons) to avoid
    divisions by zero. These cases shouldn't occur on healthy PhotonSim
    output but the guard makes the function defensible against malformed
    fixtures.
    """
    sx = np.asarray(segments['start_x_mm'], dtype=np.float64)
    sy = np.asarray(segments['start_y_mm'], dtype=np.float64)
    sz = np.asarray(segments['start_z_mm'], dtype=np.float64)
    ex = np.asarray(segments['end_x_mm'],   dtype=np.float64)
    ey = np.asarray(segments['end_y_mm'],   dtype=np.float64)
    ez = np.asarray(segments['end_z_mm'],   dtype=np.float64)
    edep_MeV = np.asarray(segments['edep'],       dtype=np.float64)
    seg_time = np.asarray(segments['time'],       dtype=np.float64)
    beta     = np.asarray(segments['beta_start'], dtype=np.float64)
    n_seg = int(edep_MeV.shape[0])

    if n_seg == 0:
        return _empty_photon_dict()

    dx = ex - sx; dy = ey - sy; dz = ez - sz
    delta_s_mm = np.sqrt(dx * dx + dy * dy + dz * dz)

    S       = float(medium_params['S'])
    kB      = float(medium_params['kB'])
    C       = float(medium_params['C'])
    tau_r   = float(medium_params['tau_rise'])
    tau_f   = float(medium_params['tau_fall'])
    moy_loc = float(medium_params['moyal_loc'])
    moy_sca = float(medium_params['moyal_scale'])
    lam_lo  = float(medium_params['lambda_min'])
    lam_hi  = float(medium_params['lambda_max'])

    # Birks/Chou. d in keV/mm so the medium kB / C (defined in mm/keV and
    # (mm/keV)² per DetectorParams) match dimensionally:
    #     1 MeV/mm = 1000 keV/mm.
    # Masked-out rows use a safe denominator (1.0) so the np.where on N̄
    # never produces a NaN we'd have to scrub later.
    mask = ((delta_s_mm > _EPS_DELTA_S_MM)
            & (edep_MeV > 0.0)
            & (beta > 0.0))
    safe_ds = np.where(mask, delta_s_mm, 1.0)
    dedx_keVmm = np.where(mask, edep_MeV / safe_ds * 1000.0, 0.0)
    N_mean = np.where(
        mask,
        S * edep_MeV / (1.0 + kB * dedx_keVmm + C * dedx_keVmm ** 2),
        0.0,
    )

    N_per_seg = rng.poisson(N_mean).astype(np.int32)
    N_total = int(N_per_seg.sum())
    if N_total == 0:
        return _empty_photon_dict()

    seg_idx = np.repeat(np.arange(n_seg, dtype=np.int64), N_per_seg)

    # `u` is reused for origin AND in-step travel time so the photon is at
    # u·Δs from Start at travel time u·Δs/(β·c). Consistent by construction.
    u = rng.uniform(0.0, 1.0, size=N_total)

    start_mm = np.stack([sx, sy, sz], axis=1)
    end_mm   = np.stack([ex, ey, ez], axis=1)
    origins_mm = start_mm[seg_idx] + u[:, None] * (end_mm - start_mm)[seg_idx]
    photon_origins = (origins_mm / 1000.0).astype(np.float32)

    travel_ns = (u * delta_s_mm[seg_idx]) / (beta[seg_idx] * _C_MM_PER_NS)
    delay_ns = _sample_hypoexp_np(rng, N_total, tau_r, tau_f).astype(np.float64)
    photon_times = (seg_time[seg_idx] + travel_ns + delay_ns).astype(np.float32)

    photon_directions = _sample_isotropic_np(rng, N_total)
    photon_wavelengths = _sample_moyal_np(
        rng, N_total, moy_loc, moy_sca, lam_lo, lam_hi)

    photon_emission_process = np.full(
        N_total, EMISSION_PROCESS_SCINTILLATION, dtype=np.int8)

    return {
        'photon_origins':           photon_origins,
        'photon_directions':        photon_directions,
        'photon_times':             photon_times,
        'photon_wavelengths':       photon_wavelengths,
        'photon_segment_index_raw': seg_idx,
        'photon_emission_process':  photon_emission_process,
    }
