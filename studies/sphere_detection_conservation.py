"""Conservation check for the sphere sensor model (PRE-EXISTING, single-medium).

With no absorption, no scattering, and no wall reflection, every photon travels straight
from the source and hits the outer sphere EXACTLY ONCE; it is detected only if it lands on
a sensor. Sensors tile a fixed fraction of the sphere (~coverage), and for a quasi-uniform
(Fibonacci) layout the detected fraction must be ~coverage INDEPENDENT of source position.

This script shows it is NOT: the detected fraction grows from ~41% (center, correct) to
~72% (source near the wall) — physically impossible, since sensors only cover ~41%. The
sphere sensor-capture model (compute_sensor_intersections_base + overlap) over-counts at
GRAZING incidence (a ray skimming the surface is captured by one or more sensors it should
miss). Off-center sources — and refracted rays in the two-medium engine — produce more
grazing hits, so this biases any position-dependent charge on a sphere.

Run:  JAX_PLATFORM_NAME=cpu python studies/sphere_detection_conservation.py
"""
import json
import numpy as np
import jax, jax.numpy as jnp
from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source

R, NS, SR = 19.5, 10000, 0.25
INT, QE = 5e7, 0.065
COVERAGE = NS * np.pi * SR**2 / (4 * np.pi * R**2)   # geometric sensor coverage fraction


def main():
    cfg = {"material": "water", "detector_type": "sphere",
           "geometry_definitions": {"radius": R, "n_sensors": NS, "sensor_radius": SR}}
    path = "/lscratch/omara/tmp/sph_cons.json"
    open(path, "w").write(json.dumps(cfg))
    d = DetectorParams.from_flat(
        scatter_length=1e8, mie_scatter_length=1e8, absorption_length=1e8,
        wall_reflection_rate=0.0, sensor_reflection_rate=0.0, qe=QE, qe_corrections=jnp.ones(NS))
    sim = setup_event_simulator(path, 200000, temperature=None, K=8, is_calibration=True,
                                detector_type='sphere', wavelength_mode=False)
    print(f"geometric sensor coverage = {COVERAGE*100:.1f}%  (detected fraction must not exceed this)")
    print(f"{'r_s':>5} {'detected frac':>14} {'vs coverage':>12}")
    for r in [0.0, 5.0, 10.0, 13.0, 16.0, 18.0]:
        src = isotropic_source(position=[0, 0, float(r)], intensity=INT, wavelength=420.0)
        q = np.mean([np.asarray(sim(src, d, jax.random.PRNGKey(11 + b))[0]).sum() for b in range(4)])
        frac = q / (INT * QE)
        flag = "  <-- UNPHYSICAL" if frac > COVERAGE * 1.05 else ""
        print(f"{r:>5.1f} {frac*100:>13.1f}% {frac/COVERAGE:>11.2f}x{flag}")


if __name__ == "__main__":
    main()
