"""Physics tests for one photon transport step (``lucid.simulation.photon_step``).

Two steps are tested: the expected-value step ``photon_iteration_update_factors``
(the differentiable forward) and the sampled step ``photon_iteration_sample``
(the truth/data generator). The oracles are closed forms of a photon crossing a
distance D to a surface in a medium with scatter rate mu = 1/L_R + 1/L_M,
absorption length L_A and surface reflectance R:

- reach the surface unscattered: exp(-mu D); detected there: exp(-mu D)(1-R_sensor)exp(-D/L_A), the sensor's R even for a wall hit
- time of flight D/c, position x + D d nudged 1e-4 m inward
- continuing weight E[cont] = mu/(mu+a) (1 - e^{-(mu+a) D}) + R e^{-(mu+a) D}, a = 1/L_A
- scatter direction about an oblique photon direction: E[d] = p_M g d0 and
  E[d d^T] = (1-m2)/2 I + (3 m2 - 1)/2 d0 d0^T, with p_M = (1/L_M)/mu,
  m2 = (1-p_M) 2/5 + p_M (1+2g^2)/3 (Rayleigh and Henyey-Greenstein moments)

The sampled step must reproduce the expected-value step's means; both are
checked against the closed forms, at 2e5 vmapped keys. Nothing here builds a
detector, so it runs in the default lane.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.simulation.photon_step import (
    photon_iteration_sample,
    photon_iteration_update_factors,
)
from lucid.simulation.reflection import ScalarReflection

N_MC = 200_000
OBLIQUE = np.array([1.0, 2.0, -2.0]) / 3.0


def _base_args(key, **overrides):
    # Convenience: tests pass scalar wall/sensor rates; pack them into the refl_params pytree.
    wall = overrides.pop('wall_reflection_rate', 0.5)
    sensor = overrides.pop('sensor_reflection_rate', 0.3)
    args = dict(
        position=jnp.array([0.0, 0.0, 0.0]),
        direction=jnp.array([0.0, 0.0, 1.0]),
        time=0.0,
        surface_distance=2.0,
        normal=jnp.array([0.0, 0.0, -1.0]),
        scatter_length=50.0,
        mie_scatter_length=1.0e9,  # ≈ no Mie channel
        g=0.0,
        refl_params=ScalarReflection(wall_rate=jnp.asarray(wall),
                                     sensor_rate=jnp.asarray(sensor)),
        absorption_length=100.0,
        hit_sensor=True,
        lam=jnp.asarray(0.0),
        rng_key=key,
        speed_of_light=0.3,  # m/ns
    )
    args.update(overrides)
    return args


def _ev(**overrides):
    return photon_iteration_update_factors(**_base_args(jax.random.PRNGKey(42), **overrides))


def _vmapped(step, n=N_MC, seed=0, **overrides):
    """The step evaluated on n independent keys (jitted), as NumPy float64 arrays."""
    args = _base_args(jax.random.PRNGKey(0), **overrides)
    f = jax.jit(jax.vmap(lambda k: step(**{**args, 'rng_key': k})))
    out = f(jax.random.split(jax.random.PRNGKey(seed), n))
    return [np.asarray(o, dtype=np.float64) for o in out]


def detect_closed_form(D, S, M, L, R):
    return np.exp(-D * (1 / S + 1 / M)) * (1 - R) * np.exp(-D / L)


def continuing_closed_form(D, S, M, L, R):
    """E[continuing weight]: scattered before the surface (absorption survival to the
    scatter point, integrated over the free path) plus reflected at the surface."""
    mu, a = 1 / S + 1 / M, 1 / L
    return mu / (mu + a) * (1 - np.exp(-(mu + a) * D)) + R * np.exp(-(mu + a) * D)


# Scattering off: with both mean free paths at 1e12 m the sampled free path is
# ~1e11 m, so the step takes the surface branch for any key and its distance is
# exactly `surface_distance`.
NO_SCATTER = dict(scatter_length=1.0e12, mie_scatter_length=1.0e12)

# Scattering always: a surface 1e4 m away with ~1 m free paths.
ALWAYS_SCATTER = dict(surface_distance=1.0e4, absorption_length=1.0e9)


class TestAbsorptionAttenuation:
    """Beer-Lambert: the absorption factor in detect_prob is exp(-D/L_A).

    It is isolated as detect_prob(L_A) / detect_prob(inf), which divides out the
    L_A-independent reach x (1 - R).
    """

    @pytest.mark.parametrize("D,L_abs", [(2.0, 100.0), (5.0, 1.0), (2.0, 1e6)])
    def test_attenuation_value(self, D, L_abs):
        dp = float(_ev(surface_distance=D, absorption_length=L_abs)[3])
        dp_inf = float(_ev(surface_distance=D, absorption_length=1e12)[3])
        npt.assert_allclose(dp / dp_inf, np.exp(-D / L_abs), rtol=1e-5)

    def test_reflection_attenuation_is_folded_into_detect_prob(self):
        npt.assert_allclose(_ev()[4], 1.0)


class TestDetectionProbability:
    """detect_prob = exp(-D (1/L_R + 1/L_M)) (1 - R_sensor) exp(-D/L_A).

    The deposit is charge on a sensor, so R is the SENSOR's reflectance whether or not the ray
    hit the sensor's sphere: under a soft overlap a ray landing on the wall just outside a sensor
    still deposits on it, and the wall's reflectance must not enter.
    """

    @pytest.mark.parametrize("L_mie", [1e9, 20.0], ids=["no-mie", "mie-20m"])
    @pytest.mark.parametrize("hit_sensor", [True, False], ids=["sensor", "wall"])
    @pytest.mark.parametrize("R", [0.3, 1.0], ids=["R0.3", "mirror-sensor"])
    def test_detection_prob_formula(self, L_mie, hit_sensor, R):
        D, S, L_abs = 2.0, 50.0, 100.0
        dp = _ev(surface_distance=D, scatter_length=S, mie_scatter_length=L_mie,
                 absorption_length=L_abs, hit_sensor=hit_sensor,
                 sensor_reflection_rate=R, wall_reflection_rate=0.9)[3]
        npt.assert_allclose(float(dp), detect_closed_form(D, S, L_mie, L_abs, R),
                            rtol=1e-5, atol=1e-7)


class TestTimeAndPosition:
    """On the surface branch both steps advance t by D/c and x by D d (+1e-4 m inward)."""

    @pytest.mark.parametrize("step", [photon_iteration_update_factors, photon_iteration_sample],
                             ids=["expected", "sampled"])
    @pytest.mark.parametrize("D", [2.0, 4.0])
    def test_time_and_position(self, step, D):
        c, t0 = 0.3, 5.0
        pos = jnp.array([0.5, -0.2, 0.1])
        direction = jnp.array([0.0, 0.6, 0.8])
        normal = jnp.array([0.0, 0.0, 1.0])            # outward; the nudge is along -normal
        for i in range(3):                              # any key: the branch is fixed
            new_pos, _, new_time, *_ = step(**_base_args(
                jax.random.PRNGKey(i), surface_distance=D, speed_of_light=c, time=t0,
                position=pos, direction=direction, normal=normal, **NO_SCATTER))
            npt.assert_allclose(float(new_time), t0 + D / c, rtol=1e-6)
            npt.assert_allclose(new_pos, pos + D * direction - 1e-4 * normal, atol=1e-6)


class TestContinuingFactor:
    """continuing_factor is the weight a photon carries into the next step."""

    def test_continuing_plus_detect_is_the_absorption_survival(self):
        """With scattering off, detect + continue = exp(-D/L_A): the photon reaching the
        surface is either detected (1-R) or reflected (R), with the same survival."""
        D, L_abs, R = 2.0, 100.0, 0.3
        _, _, _, dp, _, cont, _, _ = _ev(surface_distance=D, absorption_length=L_abs,
                                         sensor_reflection_rate=R, **NO_SCATTER)
        survival = np.exp(-D / L_abs)
        npt.assert_allclose(float(dp), (1.0 - R) * survival, rtol=1e-5)
        npt.assert_allclose(float(cont), R * survival, rtol=1e-5)


class TestSampleVsExpectedValue:
    """The sampled step is a Monte-Carlo draw of the expected-value step.

    Its detected weight is detect_prob * reflection_attenuation (the sampled
    absorption Bernoulli rides on reflection_attenuation). With p ~ 0.6 and 2e5
    keys the standard error of a mean is 1.1e-3, so atol 5e-3 is ~4.5 sigma.
    Dropping the absorption draw moves the sampled mean by 0.015; swapping the
    scatter and absorption lengths moves it by >= 0.027.
    """

    CASES = [dict(scatter_length=50.0, mie_scatter_length=1e9),
             dict(scatter_length=50.0, mie_scatter_length=20.0),
             dict(scatter_length=5.0, mie_scatter_length=8.0, absorption_length=3.0,
                  surface_distance=1.5)]

    def test_sampled_detection_is_binary(self):
        _, _, _, d, a, _, _, _ = _vmapped(photon_iteration_sample, n=2000)
        assert set(np.unique(d)) <= {0.0, 1.0}
        assert set(np.unique(a)) <= {0.0, 1.0}

    @pytest.mark.parametrize("case", CASES, ids=["no-mie", "mie-20m", "short-lengths"])
    def test_sampled_detected_weight_matches_expected_value(self, case):
        args = _base_args(None, **case)
        D, S, M, L = (args['surface_distance'], args['scatter_length'],
                      args['mie_scatter_length'], args['absorption_length'])
        expected = detect_closed_form(D, S, M, L, 0.3)
        npt.assert_allclose(float(_ev(**case)[3]), expected, rtol=1e-5)
        _, _, _, d, a, _, _, _ = _vmapped(photon_iteration_sample, **case)
        npt.assert_allclose((d * a).mean(), expected, atol=5e-3)

    @pytest.mark.parametrize("case", CASES, ids=["no-mie", "mie-20m", "short-lengths"])
    @pytest.mark.parametrize("step", [photon_iteration_update_factors, photon_iteration_sample],
                             ids=["expected", "sampled"])
    def test_mean_continuing_weight(self, step, case):
        """Both steps' mean continuing weight equals the closed form (module docstring).
        The weight is in [0, 1], so its standard error at 2e5 keys is < 1.1e-3."""
        args = _base_args(None, **case)
        expected = continuing_closed_form(args['surface_distance'], args['scatter_length'],
                                          args['mie_scatter_length'],
                                          args['absorption_length'], 0.3)
        cont = _vmapped(step, **case)[5]
        npt.assert_allclose(cont.mean(), expected, atol=5e-3)


class TestScatterDirection:
    """The direction after a scatter, about an oblique photon direction, in both steps.

    The Mie/Rayleigh choice is a Bernoulli with p_M = (1/L_M)/(1/L_R + 1/L_M);
    with L_R = 1, L_M = 3 it is 1/4 (a swapped ratio gives 3/4). Standard errors at
    2e5 keys are <= 1.4e-3 on the mean and <= 7e-4 on the second moment.
    """

    @pytest.mark.parametrize("L_mie,g", [(1e9, 0.0), (3.0, 0.9)], ids=["rayleigh", "mie-mix"])
    @pytest.mark.parametrize("step", [photon_iteration_update_factors, photon_iteration_sample],
                             ids=["expected", "sampled"])
    def test_direction_moments(self, step, L_mie, g):
        S = 1.0
        p_m = (1 / L_mie) / (1 / S + 1 / L_mie)
        m2 = (1 - p_m) * 0.4 + p_m * (1 + 2 * g * g) / 3
        out = _vmapped(step, direction=jnp.asarray(OBLIQUE), scatter_length=S,
                       mie_scatter_length=L_mie, g=g, **ALWAYS_SCATTER)
        new_pos, new_dir = out[0], out[1]
        # every photon scattered: it stopped short of the surface 1e4 m away
        assert np.all(np.linalg.norm(new_pos, axis=1) < 1e3)
        npt.assert_allclose(np.linalg.norm(new_dir, axis=1), 1.0, atol=1e-5)
        npt.assert_allclose(new_dir.mean(axis=0), p_m * g * OBLIQUE, atol=7e-3)
        expected_ddT = (1 - m2) / 2 * np.eye(3) + (3 * m2 - 1) / 2 * np.outer(OBLIQUE, OBLIQUE)
        npt.assert_allclose(new_dir.T @ new_dir / len(new_dir), expected_ddT, atol=4e-3)

    @pytest.mark.parametrize("step", [photon_iteration_update_factors, photon_iteration_sample],
                             ids=["expected", "sampled"])
    def test_scatter_point_is_exponential(self, step):
        """The free path to a scatter is Exp(mu): mean 1/mu = 0.75 m at L_R=1, L_M=3
        (std 0.75, so the 2e5-key mean is good to 1.7e-3)."""
        new_pos = _vmapped(step, scatter_length=1.0, mie_scatter_length=3.0,
                           **ALWAYS_SCATTER)[0]
        npt.assert_allclose(new_pos[:, 2].mean(), 0.75, atol=8e-3)
        npt.assert_allclose(new_pos[:, :2], 0.0, atol=1e-6)


class TestGradients:
    """d detect_prob / d length = detect_prob * D / length^2 for each of the three lengths
    (the derivative of exp(-D/length)); the expected-value step is pathwise in all three."""

    @pytest.mark.parametrize("name", ["scatter_length", "mie_scatter_length", "absorption_length"])
    def test_length_gradient(self, name):
        D = 2.0
        base = dict(surface_distance=D, scatter_length=50.0, mie_scatter_length=20.0,
                    absorption_length=100.0)

        def dp(x):
            return _ev(**{**base, name: x})[3]

        L = base[name]
        g = float(jax.grad(dp)(jnp.asarray(L)))
        npt.assert_allclose(g, float(dp(L)) * D / L**2, rtol=1e-4)

    def test_surface_distance_gradient(self):
        """d detect_prob / dD = -(1/L_R + 1/L_M + 1/L_A) detect_prob: the track gradient
        flows pathwise through D."""
        S, M, L = 50.0, 20.0, 100.0

        def dp(D):
            return _ev(surface_distance=D, scatter_length=S, mie_scatter_length=M,
                       absorption_length=L)[3]

        g = float(jax.grad(dp)(jnp.asarray(2.0)))
        npt.assert_allclose(g, -(1 / S + 1 / M + 1 / L) * float(dp(2.0)), rtol=1e-4)
