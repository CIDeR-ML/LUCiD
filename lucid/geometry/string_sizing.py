"""
JIT shape sizing for the string-telescope propagator.

Given the medium's optical properties and the array's geometric properties,
derive the compile-time-static shape constants that the JIT'd propagator
needs to allocate buffers for. Pure-Python; no JAX. Called once at
config-load, output threaded through to create_string_propagator.

Two independent truncations are sized here:

  Per-segment ray cutoff
    Each scatter iteration traces a single straight segment for at most
    L_max meters before truncating. Captures all but `eps_seg` of the
    photon's straight-line interaction probability per segment.

        N_lambda = -ln(eps_seg)
        L_max    = N_lambda * lambda_total

  Number of scatter iterations K_min
    Each iteration drops photon weight by p_scat = lambda_total / lambda_scat
    (probability the interaction was scatter, not absorption). The total
    surviving weight after K iterations is p_scat^K. Pick K so the residual
    weight is <= eps_K.

        K_min = ceil( ln(1/eps_K) / ln(1/p_scat) )

These differ by medium and are auto-derived. Empirical tuning of K is
expected; the function returns K_min as a starting estimate.

Buffer sizes follow from L_max + the spatial-hash cell size:

    max_cells_per_segment = ceil(L_max / cell_size) + 2
    max_str_per_segment   = max_cells_per_segment * max_str_per_cell
    max_dom_per_segment   = max_str_per_segment * n_dom_snap
"""

from typing import NamedTuple
import math


class StringSizingResult(NamedTuple):
    """Compile-time-static shape constants for the string propagator.

    Fields
    ------
    L_max : float                 per-segment ray-trace cutoff (m)
    N_lambda : float              -ln(eps_seg); L_max = N_lambda * lambda_total
    cell_size : float             2D spatial-hash cell edge length (m)
    max_cells_per_segment : int   max DDA cell visits per segment
    max_str_per_cell : int        spatial-hash slots per cell
    max_str_per_segment : int     max strings tested per segment
    n_dom_snap : int              DOMs returned per surviving string
    max_dom_per_segment : int     max DOM ray-sphere ops per segment
    K_min : int                   suggested scatter iteration count
    eps_seg : float               per-segment truncation tolerance
    eps_K : float                 K-iteration weight floor target
    p_scat : float                per-iteration scatter survival
    """
    L_max: float
    N_lambda: float
    cell_size: float
    max_cells_per_segment: int
    max_str_per_cell: int
    max_str_per_segment: int
    n_dom_snap: int
    max_dom_per_segment: int
    K_min: int
    eps_seg: float
    eps_K: float
    p_scat: float


def compute_p_scat(lambda_abs: float, lambda_scat: float) -> float:
    """Per-iteration scatter survival probability.

    p_scat = lambda_total / lambda_scat where 1/lambda_total = 1/lambda_abs + 1/lambda_scat.

    Parameters
    ----------
    lambda_abs : float    absorption length (m), > 0
    lambda_scat : float   scattering length (m), > 0

    Returns
    -------
    p_scat : float in (0, 1)
    """
    if lambda_abs <= 0 or lambda_scat <= 0:
        raise ValueError(
            f"absorption and scattering lengths must be positive; "
            f"got lambda_abs={lambda_abs}, lambda_scat={lambda_scat}"
        )
    inv_total = 1.0 / lambda_abs + 1.0 / lambda_scat
    lambda_total = 1.0 / inv_total
    return lambda_total / lambda_scat


def compute_lambda_total(lambda_abs: float, lambda_scat: float) -> float:
    """Total mean free path: 1/lambda_total = 1/lambda_abs + 1/lambda_scat."""
    return 1.0 / (1.0 / lambda_abs + 1.0 / lambda_scat)


def compute_K_min(p_scat: float, eps_K: float) -> int:
    """Minimum K iterations so that p_scat^K <= eps_K.

    Parameters
    ----------
    p_scat : float in (0, 1)
    eps_K : float in (0, 1)    target residual weight floor

    Returns
    -------
    K_min : int >= 1
    """
    if not (0.0 < p_scat < 1.0):
        raise ValueError(f"p_scat must be in (0, 1); got {p_scat}")
    if not (0.0 < eps_K < 1.0):
        raise ValueError(f"eps_K must be in (0, 1); got {eps_K}")
    return max(1, math.ceil(math.log(1.0 / eps_K) / math.log(1.0 / p_scat)))


def compute_L_max(lambda_total: float, eps_seg: float) -> tuple[float, float]:
    """Per-segment cutoff length and N_lambda multiplier.

    L_max = N_lambda * lambda_total where N_lambda = -ln(eps_seg).

    Returns
    -------
    L_max : float (m)
    N_lambda : float
    """
    if not (0.0 < eps_seg < 1.0):
        raise ValueError(f"eps_seg must be in (0, 1); got {eps_seg}")
    N_lambda = -math.log(eps_seg)
    return N_lambda * lambda_total, N_lambda


def size_string_propagator(
    lambda_abs: float,
    lambda_scat: float,
    dxy: float,
    *,
    eps_seg: float = 0.01,
    eps_K: float = 0.01,
    cell_size: float | None = None,
    max_str_per_cell: int = 2,
    n_dom_snap: int = 2,
    boundary_cell_slack: int = 2,
) -> StringSizingResult:
    """Derive JIT shape constants for the string propagator.

    Use the worst-case (shortest) lambda_abs and lambda_scat across the
    Cherenkov spectrum for buffer sizing. Per-photon runtime can use a
    less conservative L_max via per-wavelength clamping, but compiled
    shapes must cover the worst case.

    Parameters
    ----------
    lambda_abs : float            absorption length (m), worst case across wavelengths
    lambda_scat : float           scattering length (m), worst case across wavelengths
    dxy : float                   nominal inter-string spacing (m); used as default cell_size
    eps_seg : float, default 0.01 per-segment truncation tolerance (1% loses 1% of seg weight)
    eps_K : float, default 0.01   K-iteration weight floor (1% leftover photon weight)
    cell_size : float or None     2D hash cell edge; defaults to dxy
    max_str_per_cell : int        hash density bound (2 for regular, 4+ for clustered)
    n_dom_snap : int              DOMs per surviving string (2 for straight, 3-4 for swayed)
    boundary_cell_slack : int     extra DDA cells for boundary crossings (default 2)

    Returns
    -------
    StringSizingResult
    """
    p_scat = compute_p_scat(lambda_abs, lambda_scat)
    lambda_total = compute_lambda_total(lambda_abs, lambda_scat)

    L_max, N_lambda = compute_L_max(lambda_total, eps_seg)
    K_min = compute_K_min(p_scat, eps_K)

    if cell_size is None:
        cell_size = dxy
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive; got {cell_size}")
    if max_str_per_cell < 1:
        raise ValueError(f"max_str_per_cell must be >= 1; got {max_str_per_cell}")
    if n_dom_snap < 2:
        raise ValueError(f"n_dom_snap must be >= 2; got {n_dom_snap}")

    max_cells_per_segment = math.ceil(L_max / cell_size) + boundary_cell_slack
    max_str_per_segment = max_cells_per_segment * max_str_per_cell
    max_dom_per_segment = max_str_per_segment * n_dom_snap

    return StringSizingResult(
        L_max=L_max,
        N_lambda=N_lambda,
        cell_size=cell_size,
        max_cells_per_segment=max_cells_per_segment,
        max_str_per_cell=max_str_per_cell,
        max_str_per_segment=max_str_per_segment,
        n_dom_snap=n_dom_snap,
        max_dom_per_segment=max_dom_per_segment,
        K_min=K_min,
        eps_seg=eps_seg,
        eps_K=eps_K,
        p_scat=p_scat,
    )


def auto_n_dom_snap(string_curv_max: float, mean_dz: float) -> int:
    """Derive n_dom_snap from worst-case per-string curvature.

    n_dom_snap is the total number of DOMs returned per string that passes
    the distance test. Base is 2 (the bracket pair k_low, k_high around s*).
    For curved strings the fitted-axis s* can be off by up to
    ceil(string_curv_max / mean_dz) DOM positions, so we pad each side.

        extra_per_side = ceil(string_curv_max / mean_dz) if curv > 0 else 0
        n_dom_snap     = 2 + 2 * extra_per_side

    Examples (real telescopes)
        IceCube/HUNT/Baikal (curv~0):   n_dom_snap = 2
        ARCA typical (curv~0.7, dz~36): n_dom_snap = 4  (conservative)
        ORCA strong  (curv~5, dz~9):    n_dom_snap = 4

    Parameters
    ----------
    string_curv_max : float    max per-string perpendicular DOM-from-axis offset (m)
    mean_dz : float            mean DOM vertical spacing along string (m)

    Returns
    -------
    n_dom_snap : int >= 2
    """
    if mean_dz <= 0:
        raise ValueError(f"mean_dz must be positive; got {mean_dz}")
    if string_curv_max <= 0.0:
        return 2
    extra_per_side = math.ceil(string_curv_max / mean_dz)
    return 2 + 2 * extra_per_side
