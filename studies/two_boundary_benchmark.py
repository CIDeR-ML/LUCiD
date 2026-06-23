"""Runtime + gradient-time benchmark for the two-medium (LAB-LS/water) nested engine on GPU.

Reports, per (K, N): one-time JIT compile time, steady-state forward time, and steady-state
gradient time (jax.grad of a scalar loss wrt the full DetectorParams pytree — the calibration
gradient). K and N convergence (from the convergence studies): K≈32 is fully converged
including TIR whispering-gallery photons; per-sensor MC error ~ N^-1/2 (≈2.5% at 16M).
"""
import os, time
import numpy as np
import jax, jax.numpy as jnp
from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source

GEOM = "config/JUNO_nested_labls_geom_config.json"
PHYS = "config/JUNO_nested_labls_physics_config.json"


def bench(K, N, reps=5):
    dp = DetectorParams.from_flat(scatter_length=50.0, mie_scatter_length=1e6, absorption_length=50.0,
                                  wall_reflection_rate=0.2, sensor_reflection_rate=0.2, qe=0.065,
                                  qe_corrections=jnp.ones(10000))
    sim = setup_event_simulator(GEOM, int(N), temperature=None, K=K, is_calibration=True,
                                detector_type='nested_sphere', wavelength_mode=True, physics_config=PHYS)
    src = isotropic_source(position=[0.0, 0.0, 16.5], intensity=5e7, wavelength=430.0)
    key = jax.random.PRNGKey(0)

    def fwd(p):
        c, t = sim(src, p, key); return c
    loss = lambda p: jnp.sum(fwd(p))
    grad = jax.jit(jax.grad(loss))

    # compile (timed)
    t = time.perf_counter(); fwd(dp).block_until_ready(); t_cf = time.perf_counter() - t
    t = time.perf_counter(); jax.block_until_ready(grad(dp)); t_cg = time.perf_counter() - t
    # steady state
    t = time.perf_counter()
    for _ in range(reps):
        fwd(dp).block_until_ready()
    t_f = (time.perf_counter() - t) / reps
    t = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(grad(dp))
    t_g = (time.perf_counter() - t) / reps
    return t_cf, t_cg, t_f, t_g


def main():
    print("backend:", jax.default_backend())
    print(f"{'K':>3} {'N':>9} | {'compile fwd':>11} {'compile grad':>12} | "
          f"{'forward':>9} {'gradient':>9} {'grad/fwd':>8}")
    for K, N in [(24, 1_000_000), (32, 1_000_000), (32, 4_000_000)]:
        tcf, tcg, tf, tg = bench(K, N)
        print(f"{K:>3} {N:>9} | {tcf:>10.1f}s {tcg:>11.1f}s | "
              f"{tf*1e3:>7.1f}ms {tg*1e3:>7.1f}ms {tg/tf:>7.1f}x")


if __name__ == "__main__":
    main()
