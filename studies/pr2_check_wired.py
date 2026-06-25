"""Verify the factory-WIRED simulator is byte-identical to the legacy reference (pr2_capture_reference.py).
Run AFTER editing simulator.py."""
import jax, jax.numpy as jnp
import numpy as np
from lucid.detector_params import DetectorParams
from lucid.sources import isotropic_source
from lucid.simulation import setup_event_simulator

REF = "/lscratch/omara/tmp/claude-46143/-sdf-group-neutrino-omara-LUCiD/be77fc5c-ff33-43b9-8edc-363c1c12a9d6/scratchpad/pr2_reference.npz"
DP = DetectorParams.from_flat(scatter_length=50.0, mie_scatter_length=1e6, absorption_length=50.0,
                              wall_reflection_rate=0.2, sensor_reflection_rate=0.2, qe=0.065,
                              qe_corrections=jnp.ones(10000))
SRC = isotropic_source(position=[0., 0., 13.0], intensity=5e7, wavelength=430.0)
KEY = jax.random.PRNGKey(0)
GS, PS = 'config/JUNO_geom_config.json', 'config/JUNO_physics_config.json'
GN, PN = 'config/JUNO_nested_labls_geom_config.json', 'config/JUNO_nested_labls_physics_config.json'


def build(geom, phys, dtype, expected):
    return setup_event_simulator(geom, 150000, temperature=None, K=10, is_calibration=True,
                                 detector_type=dtype, wavelength_mode=True, physics_config=phys,
                                 use_expected_value=expected)


def run(sim):
    c, t = sim(SRC, DP, KEY)
    return np.asarray(c), np.asarray(t)


def grad_charge(sim):
    g = jax.grad(lambda dp: jnp.sum(sim(SRC, dp, KEY)[0]))(DP)
    return np.asarray(g.absorption.absorption_length)


def main():
    print("backend:", jax.default_backend())
    ref = np.load(REF)
    cur = {}
    cur['c_s_ev'], cur['t_s_ev'] = run(build(GS, PS, 'sphere', True))
    cur['c_s_mc'], cur['t_s_mc'] = run(build(GS, PS, 'sphere', False))
    cur['g_s'] = grad_charge(build(GS, PS, 'sphere', True))
    cur['c_n_ev'], cur['t_n_ev'] = run(build(GN, PN, 'nested_sphere', True))
    cur['g_n'] = grad_charge(build(GN, PN, 'nested_sphere', True))
    ok = True
    for k in cur:
        eq = bool(np.array_equal(ref[k], cur[k]))
        ok = ok and eq
        print(f"  {k:8s} byte-identical: {eq}")
    print("\nWIRED == LEGACY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
