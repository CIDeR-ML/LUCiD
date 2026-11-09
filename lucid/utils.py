"""Numeric and array helpers shared across lucid, plus the event-parameter conveniences.

One responsibility: small, stateless functions with no home of their own -- vector normalisation
and frames (``normalize``, ``create_local_frame``, ``rotation_onto``), the direction
<-> angles codec (``spherical_to_cartesian``, ``cartesian_to_spherical``), the charge and time
smearing models (``smear_charges_SK_like``, ``smear_charges_spe``, ``smear_times``), and the
sparse/full hit conversions, and the code version a run records (``code_version``). Loading a
SIREN emitter and its t0 fit is ``lucid.siren.emitter``.

It imports no IO library and re-exports nothing: the PhotonSim ROOT reader is
``lucid.sources.root_reader`` and the dataset writer and reader are ``lucid.dataset.writer`` and
``lucid.dataset.reader``.
"""
import functools
import os
import re
import subprocess

import jax
import jax.numpy as jnp


def base_dir_path():
    return os.path.dirname(os.path.abspath(__file__))+'/../'


# The container image (container/Dockerfile) installs LUCiD at IMAGE_LUCID_DIR without its .git/
# and writes the commit it was built from into IMAGE_VERSIONS (``key=value`` lines).
IMAGE_LUCID_DIR = '/opt/LUCiD'
IMAGE_VERSIONS = '/opt/VERSIONS'
_SHA = re.compile(r'[0-9a-f]{7,40}')


@functools.lru_cache(maxsize=None)
def code_version(checkout=None, image=(IMAGE_LUCID_DIR, IMAGE_VERSIONS)):
    """``(commit, dirty)`` of the LUCiD code that is running, for provenance.

    ``checkout`` is the source tree lucid was imported from (default: this package's parent).
    When it is the top of a git checkout, ``commit`` is its HEAD and ``dirty`` whether a tracked
    file differs from HEAD (``git status --porcelain --untracked-files=no``). When it is the
    image's install directory ``image[0]``, which has no ``.git/``, both come from the image's
    version record ``image[1]``: ``lucid_sha`` and ``lucid_dirty``. Otherwise ``commit`` is
    ``'unknown'``. ``dirty`` is ``None`` whenever it is not known (an image built before it was
    recorded, or a failed ``git status``); a missing or malformed SHA is ``'unknown'``, never a
    guess.
    """
    checkout = os.path.realpath(checkout or base_dir_path())

    def git(*args):
        return subprocess.check_output(['git', '-C', checkout, *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    try:
        top = git('rev-parse', '--show-toplevel')
    except (subprocess.CalledProcessError, OSError):
        top = None
    # a tree that is merely inside some other repository is not that repository's code
    if top is not None and os.path.realpath(top) == checkout:
        try:
            commit = git('rev-parse', 'HEAD')
        except (subprocess.CalledProcessError, OSError):
            return 'unknown', None
        try:
            dirty = bool(git('status', '--porcelain', '--untracked-files=no'))
        except (subprocess.CalledProcessError, OSError):
            dirty = None
        return commit, dirty

    image_dir, versions = image
    if checkout != os.path.realpath(image_dir):
        return 'unknown', None
    try:
        with open(versions) as f:
            record = dict(line.strip().split('=', 1) for line in f if '=' in line)
    except OSError:
        return 'unknown', None
    sha = record.get('lucid_sha', '')
    dirty = {'true': True, 'false': False}.get(record.get('lucid_dirty'))
    return (sha if _SHA.fullmatch(sha) else 'unknown'), dirty


def git_commit():
    """The commit of :func:`code_version`: ``'unknown'`` when it is not known."""
    return code_version()[0]

def spherical_to_cartesian(theta, phi, xp=jnp):
    """Unit direction ``[x, y, z]`` at polar angle ``theta`` (from +z) and azimuth ``phi`` (from +x
    towards +y), in radians; array arguments give a ``(3, ...)`` result.

    The inverse of :func:`cartesian_to_spherical`. ``xp`` is the array module: ``jax.numpy``, or
    ``numpy`` for float64 host code.
    """
    sin_theta = xp.sin(theta)
    cos_theta = xp.cos(theta)
    sin_phi = xp.sin(phi)
    cos_phi = xp.cos(phi)

    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta

    return xp.array([x, y, z])


def cartesian_to_spherical(direction, xp=jnp):
    """``(theta, phi)`` of a unit ``direction``: ``theta = arccos(z)`` in ``[0, pi]`` (``z``
    clipped to ``[-1, 1]``), ``phi = arctan2(y, x)`` in ``[-pi, pi]``.

    The inverse of :func:`spherical_to_cartesian`; ``xp`` as there. Normalise first: ``theta``
    reads ``z`` as the cosine.
    """
    return xp.arccos(xp.clip(direction[2], -1.0, 1.0)), xp.arctan2(direction[1], direction[0])


def rotation_onto(a, b, xp=jnp):
    """The ``(3, 3)`` rotation taking direction ``a`` onto direction ``b`` by the shortest arc,
    about ``a x b``; ``a`` and ``b`` need not be unit, and ``rotation_onto(a, a)`` is the identity.

    Built as two mirrors, through the plane normal to ``a`` and then the plane normal to the
    bisector ``a + b``, so it is orthonormal to rounding and smooth in ``a`` and ``b`` everywhere
    except at ``b = -a``, where the arc's axis is undefined. There the second mirror is normal to
    ``+x`` projected off ``a`` (``+y`` when ``|a_x| >= 0.9``), which makes it the half turn about
    ``a x`` that normal: ``+z`` onto ``-z`` is the half turn about ``+y``. ``xp`` as in
    :func:`spherical_to_cartesian`; with ``jax.numpy`` the gradient is finite everywhere and
    exact away from ``b = -a``.
    """
    a = a / xp.linalg.norm(a)
    b = b / xp.linalg.norm(b)
    bisector = a + b
    bisector_norm2 = xp.dot(bisector, bisector)
    antipodal = bisector_norm2 == 0
    helper = xp.where(xp.abs(a[0]) < 0.9, xp.array([1.0, 0.0, 0.0]), xp.array([0.0, 1.0, 0.0]))
    off_a = helper - xp.dot(helper, a) * a
    normal = xp.where(antipodal, off_a / xp.linalg.norm(off_a),
                      bisector / xp.sqrt(xp.where(antipodal, 1.0, bisector_norm2)))
    eye = xp.eye(3)
    return (eye - 2 * xp.outer(normal, normal)) @ (eye - 2 * xp.outer(a, a))

def sparse_to_full(sparse_indices, sparse_values, full_size):
    """Convert sparse representation back to full array with zeros.

    Parameters
    ----------
    sparse_indices : jnp.ndarray
        Indices of non-zero elements
    sparse_values : jnp.ndarray
        Values at the non-zero indices
    full_size : int
        Size of the output array

    Returns
    -------
    jnp.ndarray
        Full array with sparse values inserted at specified indices
    """
    full_data = jnp.zeros(full_size)
    return full_data.at[sparse_indices].set(sparse_values)


def smear_times(times, time_resolution=0.4, key=None):
    """
    Gaussianly smear input times.
    The default time resoluition is that of SK.
    Reference: https://arxiv.org/pdf/1307.0162

    Args:
        times: array of input times (e.g., per detector).
        time_resolution: standard deviation (in ns) of Gaussian smearing.
        key: JAX random key for reproducibility.
    Returns:
        smeared_times: Gaussian-smeared times.
    """
    noise = jax.random.normal(key, shape=times.shape) * time_resolution
    smeared_times = times + noise
    smeared_times = jnp.where((jnp.isfinite(smeared_times)), smeared_times, 1e6)

    return smeared_times


def smear_charges_SK_like(counts, key=None):
    """
    Gaussianly smear input charge counts according to Super-Kamiokande-like resolution.

    Reference: https://arxiv.org/pdf/1307.0162 (Table 2)

    Args:
        counts: array of input charge counts (e.g., number of hits per PMT).
        key: JAX random key for reproducibility.

    Returns:
        smeared_counts: Gaussian-smeared charge counts.
    """
    if key is None:
        raise ValueError("key must be provided for reproducibility.")

    #Define sigma according to the count range
    sigma = jnp.where(
        counts < 20,
        counts * 0.012,
        jnp.where(counts < 130, counts * 0.0075, counts * 0.005)
    )

    # Apply Gaussian smearing
    noise = jax.random.normal(key, shape=counts.shape) * sigma
    smeared_counts = counts + noise

    # Handle non-finite results (e.g., NaN or inf)
    smeared_counts = jnp.where(jnp.isfinite(smeared_counts), smeared_counts, 0.0)

    # Avoid negative or unphysical charge values
    smeared_counts = jnp.clip(smeared_counts, 0.0, None)

    return smeared_counts


def smear_charges_spe(counts, key=None, spe=None):
    """Per-photoelectron single-photoelectron (SPE) charge smearing (JAX).

    Physical alternative to :func:`smear_charges_SK_like`: instead of a single
    charge-dependent fractional Gaussian per sensor, each detected photoelectron
    draws a charge from a Gaussian+exponential SPE mixture (Bellamy 1994,
    NIM A 339 468) fit to WCSim's tabulated SK PMT SPE CDF. The reco charge is
    the exact N-fold sum (N = round(counts)):

        k ~ Binomial(N, 1-w)  Gaussian-core pes -> Normal(k*mu, sqrt(k)*sig)
        N-k                    exponential-tail pes -> Gamma(N-k, tau)

    normalized by the SPE ``fit_mean`` so <Q> = photoelectron count (unbiased);
    only the resolution/shape is imposed. JAX port of
    ``lucid.simulation.digitizer._sample_spe_charge`` (same ``_SPE_SK`` params).

    Args:
        counts: per-sensor detected photoelectron counts.
        key: JAX random key for reproducibility.
        spe: SPE mixture dict (w, mu, sig, tau, fit_mean); defaults to the SK PMT
             ``_SPE_SK`` from the digitizer.

    Returns:
        smeared_counts: SPE-sampled charge (p.e.), same shape as ``counts``.
    """
    if key is None:
        raise ValueError("key must be provided for reproducibility.")
    if spe is None:
        from lucid.simulation.digitizer import _SPE_SK as spe
    w, mu, sig, tau, fit_mean = (spe["w"], spe["mu"], spe["sig"],
                                 spe["tau"], spe["fit_mean"])

    # Photoelectron count per sensor (>=1 so the gamma/normal draws are well
    # defined; empty sensors are masked out by the caller's nonzero gate).
    n = jnp.maximum(1.0, jnp.round(counts)).astype(jnp.float32)
    k_key, g_key, t_key = jax.random.split(key, 3)

    k = jax.random.binomial(k_key, n, 1.0 - w)            # Gaussian-core pe count
    nmk = n - k                                           # exponential-tail pe count

    # Sum of k Gaussians = Normal(k*mu, sqrt(k)*sig); Sum of nmk Exp(tau) = Gamma.
    gauss = k * mu + jnp.sqrt(k) * sig * jax.random.normal(g_key, shape=counts.shape)
    gamma = jax.random.gamma(t_key, jnp.maximum(nmk, 1.0), shape=counts.shape) * tau
    tail = jnp.where(nmk > 0.0, gamma, 0.0)

    smeared_counts = (gauss + tail) / fit_mean
    smeared_counts = jnp.where(jnp.isfinite(smeared_counts), smeared_counts, 0.0)
    return jnp.clip(smeared_counts, 0.0, None)


# ---------------------------------------------------------------------------
# Shared math helpers (moved from generate.py during Phase 2.2 refactor)
# ---------------------------------------------------------------------------

def normalize(v, epsilon=1e-8):
    """Normalize a vector (or batch of vectors) with numerical stability.

    Parameters
    ----------
    v : jnp.ndarray
        Input vector or batch of vectors to normalize.
    epsilon : float, optional
        Small constant for numerical stability, by default 1e-8.

    Returns
    -------
    jnp.ndarray
        Normalized vector(s), same shape as input.
    """
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + epsilon)


def create_local_frame(z):
    """Right-handed orthonormal frame with *z* as the z-axis.

    Returns ``stack([x, y, z])`` with the basis vectors as **rows**, so the local→world map for a
    locally-sampled direction is ``frame.T @ local_dir`` (``local_dirs @ frame`` for a batch of
    row vectors); ``frame @ local_dir`` is the inverse rotation.
    """
    z = normalize(z)
    t = jnp.where(
        jnp.abs(z[0]) < 0.9,
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
    )
    x = normalize(jnp.cross(t, z))
    y = jnp.cross(z, x)
    return jnp.stack([x, y, z])
