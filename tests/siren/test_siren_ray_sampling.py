"""SIREN ray sampling: the Cherenkov cone, and the trained net's angle."""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.sources.siren_rays import generate_random_cone_vectors


# ---- the Cherenkov cone ---------------------------------------------------------------------------

# Oblique axes: an axis-aligned one cannot tell the basis from its transpose.
CONE_AXES = [np.array([1.0, 2.0, -2.0]) / 3.0,
             np.array([0.95, 0.1, 0.3]) / np.linalg.norm([0.95, 0.1, 0.3])]


@pytest.mark.parametrize('axis', CONE_AXES, ids=['oblique-a', 'oblique-b'])
@pytest.mark.parametrize('theta_deg', [41.2, 10.0, 120.0])
def test_cone_vectors_sit_at_the_opening_angle_with_uniform_azimuth(axis, theta_deg):
    """Every ray makes angle theta with the track axis, and the azimuth is uniform, so
    E[v] = cos(theta) a and E[v v^T] = cos^2 a a^T + sin^2/2 (I - a a^T).

    The azimuthal moments have standard error sin(theta)/sqrt(2N) <= 1.6e-3 at N = 2e5.
    A cone rotated by the inverse basis keeps |v| = 1 but misses v.a = cos(theta) by
    up to ~1 for these axes."""
    th = np.deg2rad(theta_deg)
    v = np.asarray(generate_random_cone_vectors(jnp.asarray(2.0 * axis), th, 200_000,
                                                jax.random.PRNGKey(3)), np.float64)
    npt.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    npt.assert_allclose(v @ axis, np.cos(th), atol=1e-5)
    npt.assert_allclose(v.mean(axis=0), np.cos(th) * axis, atol=8e-3)
    P = np.outer(axis, axis)
    npt.assert_allclose(v.T @ v / len(v), np.cos(th)**2 * P + np.sin(th)**2 / 2 * (np.eye(3) - P),
                        atol=6e-3)


def test_cone_takes_a_per_ray_opening_angle():
    """theta may be one value per ray (the SIREN-sampled angles): ray k sits at theta_k."""
    axis = CONE_AXES[0]
    th = np.linspace(0.1, 2.5, 1000)
    v = np.asarray(generate_random_cone_vectors(jnp.asarray(axis), jnp.asarray(th), 1000,
                                                jax.random.PRNGKey(0)))
    npt.assert_allclose(v @ axis, np.cos(th), atol=1e-5)


# ---- the grid-bin draw ---------------------------------------------------------------------------

# Empty bins at both ends and in the middle, a bin of float32 rounding noise next to a bright one
# (it owns no interval of the cumulative sum), and a dim bin whose interval rounding widens.
GRID_W = np.array([0.0, 0.0, 3.0, 0.0, 1e4, 1e-9, 3e-3, 5.0, 0.0], np.float32)


def test_a_drawn_bin_has_positive_mass_and_empty_bins_are_never_drawn():
    """Draws at u = 0, on every interval edge and just below the total land in a bin whose drawn
    mass is positive. A bin of zero weight is never drawn: before, u = 0 drew the empty bin 0 and
    the 1e-30 guard divided the jittered point's density by 1e-30 (the GPU's parallel cumsum also
    hands such bins intervals; measured: rays at 1/q up to 3.7e8, one ray taking a key's light)."""
    from lucid.sources.siren_rays import draw_bins
    cw = np.asarray(jax.lax.cummax(jnp.cumsum(jnp.asarray(GRID_W))))
    total = cw[-1]
    u = np.concatenate([[0.0], cw, [np.nextafter(total, 0, dtype=np.float32)]]).astype(np.float32)
    u = u[u < total]                                 # the draws' range is [0, total)
    idx, mass = map(np.asarray, draw_bins(jnp.asarray(GRID_W), jnp.asarray(u)))
    assert (mass > 0).all(), (idx, mass)
    assert (GRID_W[idx] > 0).all(), idx


def test_the_drawn_mass_is_the_probability_of_the_draw():
    """Over a lattice of u filling [0, total), each bin is drawn in the fraction mass / total of the
    draws: the importance weight target / mass is exact for the draw that ran, rounding included
    (the noise bin is never drawn; the dim bin's mass is its rounded interval, not its weight)."""
    from lucid.sources.siren_rays import draw_bins
    total = float(np.asarray(jax.lax.cummax(jnp.cumsum(jnp.asarray(GRID_W))))[-1])
    n = 2_000_000
    u = jnp.asarray((np.arange(n) + 0.5) / n * total, jnp.float32)
    idx, mass = map(np.asarray, draw_bins(jnp.asarray(GRID_W), u))
    freq = np.bincount(idx, minlength=GRID_W.size) / n
    drawn_mass = np.zeros(GRID_W.size)
    drawn_mass[idx] = mass
    npt.assert_allclose(freq, drawn_mass / total, atol=2.0 / n)
    assert freq[5] == 0.0


# ---- the energy gradient of the sampled shape -----------------------------------------------------

# An analytic emitter: a ring at the opening angle mu(E) = 0.5 + 2e-4 E rad, width 0.08 rad.
_MU0, _DMU, _SIG = 0.5, 2e-4, 0.08


def _ring_density(e, params, energy, angle, s_over_smax):
    return jnp.exp(-0.5 * ((angle - (_MU0 + _DMU * energy)) / _SIG) ** 2) * (1.0 + s_over_smax)


def _ring_mean_cos(energy):
    """E_f[cos angle] by a float64 quadrature of the ring over the emitter's window."""
    a = np.linspace(0.2, 1.4, 200_001)
    f = np.exp(-0.5 * ((a - (_MU0 + _DMU * energy)) / _SIG) ** 2)
    return np.sum(f * np.cos(a)) / np.sum(f)


def test_the_shapes_energy_gradient_is_the_derivative_of_its_expectation(monkeypatch):
    """d/dE of the sampled E[cos angle] (key-averaged autodiff through the importance sampler)
    equals the derivative of the true expectation. The draw is held fixed under differentiation,
    so this needs the proposal mass detached: differentiating it as well subtracts the proposal's
    score, which cancels the f'/f the grid proposal copies. Measured: 1.002 x the quadrature's
    derivative (key-mean se 0.5%); with the mass differentiated, 1e-4 x (zero within noise)."""
    import types
    import lucid.sources.siren_rays as sr
    monkeypatch.setattr(sr, 'density', _ring_density)
    e = types.SimpleNamespace(grid_bins=64, lo=(0.0, 0.2, 0.0), hi=(5000.0, 1.4, 1.0),
                              along_track_m=lambda s, E: s * E * 1e-3, n_photons=lambda E: 10.0 * E)
    rays = sr.make_cherenkov_surrogate_fn(e)

    def mean_cos(E, key):
        v, _, inten = rays(jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]), E, 20_000, None, key)
        return jnp.sum(inten * v[:, 2]) / jnp.sum(inten)

    E0 = 1000.0
    want = (_ring_mean_cos(E0 + 1.0) - _ring_mean_cos(E0 - 1.0)) / 2.0      # -1.28e-4 /MeV
    grad = jax.jit(jax.grad(mean_cos))
    got = np.array([float(grad(jnp.float32(E0), jax.random.PRNGKey(k))) for k in range(8)])
    npt.assert_allclose(got.mean(), want, rtol=0.03)


# ---- the trained Cherenkov SIREN, when present -------------------------------------------------

_MUON_MASS_MEV = 105.6583755     # PDG 2022
_ELECTRON_MASS_MEV = 0.51099895


def _cherenkov_angle_deg(n, kinetic_mev, mass_mev):
    gamma = 1.0 + kinetic_mev / mass_mev
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    return np.degrees(np.arccos(1.0 / (n * beta)))


@pytest.mark.parametrize('particle,mass,energy', [
    pytest.param(p, m, e, marks=pytest.mark.requires_emitter(p, 'water'))
    for p, m, e in [('muon', _MUON_MASS_MEV, 300.0), ('muon', _MUON_MASS_MEV, 1000.0),
                    ('muon', _MUON_MASS_MEV, 5000.0), ('electron', _ELECTRON_MASS_MEV, 1000.0)]])
def test_siren_angular_pdf_peaks_at_the_cherenkov_angle(particle, mass, energy):
    """The SIREN's photon-angle marginal peaks at the Cherenkov angle acos(1/(n beta)) of the
    particle's starting velocity. Water's index over the Cherenkov-weighted band is 1.33
    (589 nm) to 1.345 (380 nm) (IAPWS 1997), so the peak must fall between
    acos(1/(1.33 beta)) and acos(1/(1.345 beta)), to within 1 degree (the grid step is
    0.45 degree). Measured: muon 39.3/41.2/41.6 deg at 0.3/1/5 GeV, electron 41.6 deg;
    the 300 MeV muon (beta = 0.966) sits 2.3 deg below the beta = 1 angle."""
    from lucid.siren.emitter import density, load_emitter
    e = load_emitter(particle, 'water', 'cherenkov')
    a = jnp.linspace(e.lo[1], e.hi[1], 400)
    s = jnp.linspace(e.lo[2], e.hi[2], 200)
    A, S = jnp.meshgrid(a, s, indexing='ij')
    w = np.asarray(density(e, e.params, energy, A.ravel(), S.ravel())).reshape(400, 200)
    peak = np.degrees(float(a[np.argmax(w.sum(axis=1))]))
    lo = _cherenkov_angle_deg(1.33, energy, mass) - 1.0
    hi = _cherenkov_angle_deg(1.345, energy, mass) + 1.0
    assert lo <= peak <= hi, (peak, lo, hi)
