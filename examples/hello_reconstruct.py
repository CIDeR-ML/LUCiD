"""hello_reconstruct — recover a muon's energy from its observed charge.

Reconstruction scans the differentiable forward over energy and reports the best fit —
this is Stage 0 (energy-from-light) of the production pipeline. The forward is smooth and
monotonic in energy, so the charge-loss landscape has a clean minimum at the truth.

NOTE: full multi-parameter track reconstruction (vertex/direction/t0/energy by gradient
descent) lives in lucid.optimization and is the subject of docs/RECON_CONSOLIDATION.md —
the energy gradient through the SIREN emitter carries DiCE-score noise, so robust gradient
recon needs the conditioning + loss treatment documented there (not this 20-line demo).
Run:  python examples/hello_reconstruct.py   (writes hello_reconstruct.png)
"""
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
sim = setup_event_simulator(GEOM, 200_000, K=4, physics_config=PHYS,
                            default_detector_params=True, particle='muon',
                            n_cap=40, n_angular=80, n_height=40)
pos, dirn, key = jnp.array([0., 0., 0.]), jnp.array([0., 0., 1.]), jax.random.PRNGKey(1)
fwd = lambda E: sim(ParticleParams.from_cartesian(E, pos, dirn, 0.), key)[3]   # common keys

obs = fwd(1000.)                                                # truth = 1 GeV
grid = np.linspace(600., 1500., 13)
sse = np.array([float(jnp.sum((jnp.sqrt(fwd(E)+1e-6) - jnp.sqrt(obs+1e-6))**2)) for E in grid])
best = grid[sse.argmin()]
print(f'reconstructed energy = {best:.0f} MeV (truth 1000)')

plt.plot(grid, sse, 'o-'); plt.axvline(1000, ls='--', c='k', label='truth')
plt.axvline(best, ls=':', c='r', label=f'best {best:.0f}')
plt.xlabel('energy [MeV]'); plt.ylabel('charge √-SSE loss'); plt.legend()
plt.title('hello_reconstruct: energy from light'); plt.tight_layout()
plt.savefig('hello_reconstruct.png', dpi=110); print('wrote hello_reconstruct.png')
