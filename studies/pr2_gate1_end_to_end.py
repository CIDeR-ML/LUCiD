"""GATE 1 (end-to-end) — wire the factory step into the FULL simulator and confirm the
complete forward + gradient is byte-identical.

Single-medium: factory(has_interface=False) must give bit-identical charges/times/grad to the
legacy step through a whole event. Two-medium: factory(has_interface=True) must match the
legacy nested path. We swap the module-level step functions the simulator imports, rebuild,
and compare.
"""
import jax, jax.numpy as jnp
import numpy as np
from lucid.detector_params import DetectorParams
from lucid.sources import isotropic_source
import lucid.simulation.simulator as S
from lucid.simulation.reflection import scalar_reflection
from lucid.simulation.photon_step_factory import make_photon_step

DP = DetectorParams.from_flat(scatter_length=50.0, mie_scatter_length=1e6, absorption_length=50.0,
                              wall_reflection_rate=0.2, sensor_reflection_rate=0.2, qe=0.065,
                              qe_corrections=jnp.ones(10000))
SRC = isotropic_source(position=[0., 0., 13.0], intensity=5e7, wavelength=430.0)
KEY = jax.random.PRNGKey(0)


def build(geom, phys, dtype, expected):
    return S.setup_event_simulator(geom, 150000, temperature=None, K=10, is_calibration=True,
                                   detector_type=dtype, wavelength_mode=True, physics_config=phys,
                                   use_expected_value=expected)


def run(sim):
    c, t = sim(SRC, DP, KEY)
    return np.asarray(c), np.asarray(t)


def grad_charge(sim):
    g = jax.grad(lambda dp: jnp.sum(sim(SRC, dp, KEY)[0]))(DP)
    return np.asarray(g.absorption.absorption_length)


def patch_single():
    S.photon_iteration_sample = make_photon_step('sample', False)
    S.make_photon_iteration_update_factors_safe = lambda rfn=scalar_reflection: \
        make_photon_step('update_factors', False, rfn)


def patch_nested():
    S.photon_iteration_sample_nested = make_photon_step('sample', True)
    S.make_photon_iteration_update_factors_nested_safe = lambda rfn=scalar_reflection: \
        make_photon_step('update_factors', True, rfn)


def main():
    print("backend:", jax.default_backend())
    GS, PS = 'config/JUNO_geom_config.json', 'config/JUNO_physics_config.json'
    GN, PN = 'config/JUNO_nested_labls_geom_config.json', 'config/JUNO_nested_labls_physics_config.json'

    # ---- ORIGINAL (before patching) ----
    c_s_ev, t_s_ev = run(build(GS, PS, 'sphere', True))
    c_s_mc, t_s_mc = run(build(GS, PS, 'sphere', False))
    g_s = grad_charge(build(GS, PS, 'sphere', True))
    c_n_ev, t_n_ev = run(build(GN, PN, 'nested_sphere', True))
    g_n = grad_charge(build(GN, PN, 'nested_sphere', True))

    # ---- FACTORY (after patching) ----
    patch_single(); patch_nested()
    fc_s_ev, ft_s_ev = run(build(GS, PS, 'sphere', True))
    fc_s_mc, ft_s_mc = run(build(GS, PS, 'sphere', False))
    fg_s = grad_charge(build(GS, PS, 'sphere', True))
    fc_n_ev, ft_n_ev = run(build(GN, PN, 'nested_sphere', True))
    fg_n = grad_charge(build(GN, PN, 'nested_sphere', True))

    def be(a, b):
        return bool(np.array_equal(a, b))
    r = {
        'single expected charge': be(c_s_ev, fc_s_ev), 'single expected time': be(t_s_ev, ft_s_ev),
        'single sampling charge': be(c_s_mc, fc_s_mc), 'single sampling time': be(t_s_mc, ft_s_mc),
        'single expected grad':   be(g_s, fg_s),
        'nested expected charge': be(c_n_ev, fc_n_ev), 'nested expected time': be(t_n_ev, ft_n_ev),
        'nested expected grad':   be(g_n, fg_n),
    }
    for k, v in r.items():
        print(f"  {k:26s}: {v}")
    print("\nGATE 1 end-to-end:", "PASS" if all(r.values()) else "FAIL")


if __name__ == "__main__":
    main()
