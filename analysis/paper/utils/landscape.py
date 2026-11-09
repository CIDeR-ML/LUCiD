"""The 2-D Neyman loss surface of the loss-geometry figure, defined once.

Single collimated laser, scalar_mix detector, all parameters at truth except the 405 nm
scattering length ``v[0]`` and absorption length ``v[1]``. Truth data and model use INDEPENDENT
key sets (no common random numbers): the truth is a fixed noisy dataset, so the realized minimum
sits displaced from truth by the residual Monte-Carlo noise. The surface stage
(``compute_2d_neyman.py``) and the trajectory stage (``traj_2d_neyman.py``) both build from here,
so every iterate lies on the plotted surface.
"""
import jax
import jax.numpy as jnp

from analysis.paper.utils import calibration as C
from lucid.detector_params import DetectorParams
from lucid.fitting.transforms import damped_matrix


@jax.jit
def _gn_solve(F, g, lam, mu):
    return jnp.linalg.solve(damped_matrix(F, lam=lam, mu=mu), g)


def gauss_newton_step(F, g, lam, mu):
    """The joint fit's damped step ``A^-1 g`` (``lucid.fitting.transforms.damped_matrix``), without
    the sign: the streamline field wants the descent direction, the trajectory the step."""
    return _gn_solve(jnp.asarray(F, jnp.float32), jnp.asarray(g, jnp.float32), lam, mu)


class Surface:
    """``L(v)`` (Neyman chi^2 on charge), its gradient, and the per-point Fisher ``2 J^T J / Q``."""

    def __init__(self, recipe):
        truth = C.effective_truth(C.REP_WL)
        sim = C._cal_sim(recipe['n_ph'], recipe['k'], recipe['grid'])
        src = C._laser_sources(recipe['intensity'])[0]
        self.tx, self.ty = float(truth['scatter_length']), float(truth['absorption_length'])

        def build_dp(v):
            return DetectorParams.from_flat(
                scatter_length=v[0], mie_scatter_length=1e6, g=0.9,
                wall_reflection_rate=C.WALL_R_MIX, sensor_reflection_rate=C.SENSOR_R_MIX,
                wall_fspec=C.WALL_FSPEC, sensor_fspec=C.SENSOR_FSPEC,
                absorption_length=v[1], qe=truth['qe'], qe_corrections=jnp.ones(C.NS))

        tkeys = jax.random.split(jax.random.PRNGKey(3), recipe['nk_truth'])
        skeys = jax.random.split(jax.random.PRNGKey(101), recipe['nk_sim'])

        @jax.jit
        def mean_over(v, keys):        # accumulate, do not stack: peak memory does not scale with NK
            acc = sim(src, build_dp(v), keys[0])[0]
            for k in keys[1:]:
                acc = acc + sim(src, build_dp(v), k)[0]
            return acc / len(keys)

        @jax.jit
        def M(v):
            return mean_over(v, skeys)

        Q = jax.lax.stop_gradient(mean_over(jnp.array([self.tx, self.ty]), tkeys))
        Winv = 1.0 / jnp.maximum(Q, 1e-9)                            # Neyman weight 1/Q

        @jax.jit
        def L(v):
            return jnp.sum((M(v) - Q) ** 2 * Winv)

        jac = jax.jacfwd(M)
        self.loss_and_grad = jax.jit(jax.value_and_grad(L))
        # The Gauss-Newton / Fisher matrix at v.
        self.fisher = jax.jit(lambda v: 2.0 * jnp.einsum('np,nq,n->pq', jac(v), jac(v), Winv))
