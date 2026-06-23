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


def _dp():
    return DetectorParams.from_flat(scatter_length=50.0, mie_scatter_length=1e6, absorption_length=50.0,
                                    wall_reflection_rate=0.2, sensor_reflection_rate=0.2, qe=0.065,
                                    qe_corrections=jnp.ones(10000))


def _sim(K, N):
    return setup_event_simulator(GEOM, int(N), temperature=None, K=K, is_calibration=True,
                                 detector_type='nested_sphere', wavelength_mode=True, physics_config=PHYS)


def time_call(fn, dp, reps=5):
    """Return (compile_s, steady_ms) or (None, None) on OOM."""
    try:
        t = time.perf_counter(); jax.block_until_ready(fn(dp)); t_c = time.perf_counter() - t
        t = time.perf_counter()
        for _ in range(reps):
            jax.block_until_ready(fn(dp))
        return t_c, (time.perf_counter() - t) / reps * 1e3
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
            return None, None
        raise


def main():
    print("backend:", jax.default_backend(), "| nested LAB-LS / water, wavelength mode\n")
    src = isotropic_source(position=[0.0, 0.0, 16.5], intensity=5e7, wavelength=430.0)
    key = jax.random.PRNGKey(0)
    dp = _dp()

    print("FORWARD:")
    print(f"  {'K':>3} {'N':>9} {'compile':>9} {'forward':>9}")
    for K, N in [(24, 1_000_000), (32, 1_000_000), (32, 4_000_000)]:
        sim = _sim(K, N)
        tc, tf = time_call(lambda p: sim(src, p, key)[0], dp)
        msg = "OOM" if tf is None else f"{tc:>7.1f}s {tf:>7.1f}ms"
        print(f"  {K:>3} {N:>9} {msg}")

    print("\nGRADIENT  (jax.grad of Σcharge wrt full DetectorParams):")
    print(f"  {'K':>3} {'N':>9} {'compile':>9} {'gradient':>9} {'peak mem':>9}")
    for K, N in [(32, 250_000), (32, 500_000), (32, 1_000_000)]:
        sim = _sim(K, N)
        g = jax.jit(jax.grad(lambda p: jnp.sum(sim(src, p, key)[0])))
        tc, tg = time_call(g, dp)
        msg = ">11 GB (OOM)" if tg is None else f"{tc:>7.1f}s {tg:>7.1f}ms"
        print(f"  {K:>3} {N:>9} {msg}")


if __name__ == "__main__":
    main()
