"""Phase-2c reconciliation test — scintillation emission (refactor-v2 wbls merge).

Validates the emission-process machinery added on top of the unification forward:
  - medium.emission_processes gating (water Cherenkov-only, wbls Cherenkov+Scint, ice Cherenkov-only)
  - scintillation scalars inherited from the material JSON into DetectorParams (water stays neutral)
  - [net-guarded] the wbls track forward fires, scales with the light yield S, and carries
    correctly-signed Chou gradients (dq/dS>0, dq/dkB<0, dq/dC<0) while the timing constants
    tau_rise/tau_fall do NOT move total charge (they are a timing-only observable).

The forward/gradient test needs the downloaded dE/dx SIREN net (data/wbls) and is skipped
when it is absent, matching the rest of the suite's treatment of the trained nets.
"""
import os
import pytest
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WBLS_GEOM = os.path.join(ROOT, 'config/SK_like_wbls_geom_config.json')
WBLS_PHYS = os.path.join(ROOT, 'config/SK_like_wbls_physics_config.json')
_DEDX_NET = os.path.join(ROOT, 'data/wbls/muon/dedx_siren_training/trained_model/dedx_siren_weights.npz')
_needs_net = pytest.mark.skipif(
    not os.path.exists(_DEDX_NET),
    reason="wbls dE/dx SIREN net not downloaded (scripts/download_data.sh)")


def test_emission_process_gating():
    """The medium tuple decides which emitters run; water/ice stay Cherenkov-only."""
    from lucid.wavelength.medium import make_medium
    assert make_medium('water').emission_processes == ('cherenkov',)
    assert make_medium('ice').emission_processes == ('cherenkov',)
    wbls = make_medium('wbls')
    assert wbls.emission_processes == ('cherenkov', 'scintillation')
    assert 0.0 < wbls.cherenkov_fraction < 1.0
    assert wbls.scintillation_lambda_min < wbls.scintillation_lambda_max


def test_scint_scalars_inherited_from_material():
    """wbls physics config inherits S/kB/C/tau/moyal from the material JSON; water is neutral."""
    from lucid.detector_params import load_physics_config
    dp, _, _ = load_physics_config(WBLS_PHYS)
    s = dp.scintillation
    assert float(s.S) == pytest.approx(1387.3, rel=1e-4)
    assert float(s.tau_rise) == pytest.approx(0.13, rel=1e-4)
    assert float(s.tau_fall) == pytest.approx(2.7, rel=1e-4)
    assert float(s.moyal_loc) == pytest.approx(373.26120442, rel=1e-6)
    assert float(s.kB) > 0.0 and float(s.C) > 0.0
    # Water (no scintillation block) → neutral S=0, no light.
    water_phys = os.path.join(ROOT, 'config/SK_like_physics_config.json')
    if os.path.exists(water_phys):
        wdp, _, _ = load_physics_config(water_phys)
        assert float(wdp.scintillation.S) == 0.0


@_needs_net
@pytest.mark.slow
def test_wbls_forward_scales_and_gradients():
    """The wbls track forward fires; charge grows with S; Chou gradients are correctly signed."""
    from lucid.simulation.simulator import setup_event_simulator
    from lucid.detector_params import ParticleParams

    sim = setup_event_simulator(WBLS_GEOM, 20000, K=4, physics_config=WBLS_PHYS,
                                wavelength_mode=True, default_detector_params=True)
    dp = sim.default_detector_params
    pp = ParticleParams(energy=jnp.array(500.0), position=jnp.array([0., 0., 0.]),
                        theta=jnp.array(0.1), phi=jnp.array(0.2), t0=jnp.array(0.0))
    key = jax.random.PRNGKey(0)
    sim_u = setup_event_simulator(WBLS_GEOM, 20000, K=4, physics_config=WBLS_PHYS,
                                  wavelength_mode=True)  # unbound: (pp, dp, key)

    def q_tot_for_S(S):
        d2 = dp._replace(scintillation=dp.scintillation._replace(S=jnp.array(S)))
        return float(jnp.sum(sim_u(pp, d2, key)[3]))   # hits[3] = per-PMT charge

    q0, q1, q2 = q_tot_for_S(0.0), q_tot_for_S(1387.3), q_tot_for_S(2774.6)
    assert q0 > 0.0                      # Cherenkov baseline (the wbls Cherenkov fraction)
    assert q1 > q0 and q2 > q1           # scintillation adds light, monotone in S
    assert jnp.isfinite(jnp.array([q0, q1, q2])).all()

    def mk(field):
        def f(x):
            d2 = dp._replace(scintillation=dp.scintillation._replace(**{field: x}))
            return jnp.sum(sim_u(pp, d2, key)[3])
        return f

    assert float(jax.grad(mk('S'))(jnp.array(1387.3))) > 0.0     # more yield → more light
    assert float(jax.grad(mk('kB'))(jnp.array(1.65e-5))) < 0.0   # Birks quenching → less light
    assert float(jax.grad(mk('C'))(jnp.array(1.33e-9))) < 0.0    # bimolecular quenching → less light
    # tau_rise/tau_fall are emission-TIMING only — total charge must be insensitive to them.
    assert float(jax.grad(mk('tau_rise'))(jnp.array(0.13))) == 0.0
    assert float(jax.grad(mk('tau_fall'))(jnp.array(2.7))) == 0.0
