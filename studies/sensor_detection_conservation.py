"""Sensor-detection conservation — the PMT cosθ angular-acceptance check.

A photosensor's capture cross-section is the PROJECTED photocathode area π·r²·cosθ, not a
θ-independent π·r² disk. Without the cosθ factor, grazing rays (which dominate near the far
wall when the source sits off-centre) are over-counted, so the detected fraction climbs with
source position instead of staying equal to the geometric sensor coverage.

This script isolates the propagator GEOMETRY (no photon step / optics): it fires isotropic
rays from a source and sums the propagator's per-sensor weights. With the cosθ acceptance
(applied via each geometry's `sensor_normals`), the detected fraction must be FLAT and equal
to the coverage (Σ sensor area / total surface area) for every source position — on the
sphere AND the cylinder.

Run:  JAX_PLATFORM_NAME=cpu python studies/sensor_detection_conservation.py
"""
import numpy as np
import jax
import jax.numpy as jnp

from lucid.geometry.detector_geometry import DetectorGeometry

N_RAYS = 200_000


def _cfg(d):
    """Write a tmp geometry config (with the cosθ acceptance OPTED IN) and return its path.

    The PMT cosθ acceptance is opt-in per detector (default off); this study demonstrates the
    cosθ-ON conservation, so it sets ``apply_angular_acceptance`` itself rather than relying on
    which standing configs have opted in.
    """
    import os, json, tempfile
    d = dict(d); d["apply_angular_acceptance"] = True
    p = os.path.join(tempfile.mkdtemp(), "geom.json")
    json.dump(d, open(p, "w"))
    return p


SPHERE_CFG = _cfg({"material": "water", "detector_type": "sphere",
                   "geometry_definitions": {"radius": 17.5, "n_sensors": 10000, "sensor_radius": 0.25}})
CYL_CFG = _cfg({"material": "water", "detector_type": "cylinder",
                "geometry_definitions": {"radius": 8.0, "height": 16.0, "n_sensors": 5000, "sensor_radius": 0.25}})


def _isotropic(origin, n, seed):
    r = np.random.RandomState(seed)
    d = r.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.broadcast_to(np.asarray(origin, np.float32), (n, 3))
    return jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32)


def detected_fraction(propagator, origin, seed=0):
    o, d = _isotropic(origin, N_RAYS, seed)
    res = propagator(o, d)
    # sensor_weights: (max_cand, n_rays). With temperature=None the per-sensor weight is the
    # (cosθ-projected) hard overlap; summing over candidates+rays / n_rays = detected fraction.
    return float(jnp.sum(res['sensor_weights'])) / N_RAYS


def sphere_coverage(dg):
    R = float(np.linalg.norm(dg.sensor_points, axis=1).mean())
    n = dg.sensor_points.shape[0]
    rs = dg.sensor_radius
    return n * np.pi * rs**2 / (4 * np.pi * R**2)


def cylinder_coverage(dg):
    p = np.asarray(dg.sensor_points)
    n = p.shape[0]
    rs = dg.sensor_radius
    R = float(np.linalg.norm(p[:, :2], axis=1).max())
    H = float(p[:, 2].max() - p[:, 2].min())
    area = 2 * np.pi * R * H + 2 * np.pi * R**2          # barrel + two caps
    return n * np.pi * rs**2 / area


def run(name, config, dtype, radii, coverage_fn):
    dg = DetectorGeometry.from_config(config, detector_type=dtype, temperature=None)
    cov = coverage_fn(dg)
    print(f"\n{name}  (coverage = {cov:.4f})")
    print(f"  {'source':>10}   {'detected':>9}   {'/coverage':>9}")
    fracs, dist = [], []
    for rr in radii:
        f = detected_fraction(dg.propagator, rr)
        fracs.append(f); dist.append(float(np.linalg.norm(rr)))
        print(f"  {str(np.round(rr,2)):>14}   {f:>9.4f}   {f/cov:>9.3f}")
    flat = (max(fracs) - min(fracs)) / np.mean(fracs)
    ok = flat < 0.05 and abs(np.mean(fracs) / cov - 1.0) < 0.10
    print(f"  spread={flat:.3f} (flat if <0.05),  mean/coverage={np.mean(fracs)/cov:.3f}  -> {'PASS' if ok else 'FAIL'}")
    return ok, np.array(dist), np.array(fracs), cov


def _plot(panels, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.2))
    for ax, (name, dist, fracs, cov) in zip(np.atleast_1d(axes), panels):
        ax.plot(dist, fracs, 'o-', color='C0', label='detected fraction')
        ax.axhline(cov, ls='--', color='C3', label=f'geometric coverage = {cov:.3f}')
        ax.set_xlabel('source distance from centre  [m]')
        ax.set_ylabel('detected fraction')
        ax.set_title(name)
        ax.set_ylim(0, max(fracs.max(), cov) * 1.25)
        ax.legend(loc='lower left', fontsize=8)
        ax.text(0.5, 0.92, 'flat ⇒ conservation holds (cosθ acceptance)',
                transform=ax.transAxes, ha='center', fontsize=8, color='0.3')
    fig.suptitle('PMT cosθ angular acceptance — detection conserved as the source moves through the detector')
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    print("\nsaved figure:", outpath)


def main():
    print("backend:", jax.default_backend(), " — detected fraction should be FLAT = coverage")
    ok_s, ds, fs, cs = run("SPHERE (R=17.5)", SPHERE_CFG, "sphere",
                           [[0, 0, float(z)] for z in np.linspace(0, 16.5, 12)], sphere_coverage)
    ok_c, dc, fc, cc = run("CYLINDER (R=8, H=16)", CYL_CFG, "cylinder",
                           [[float(x), 0, 0] for x in np.linspace(0, 7, 10)], cylinder_coverage)
    print("\nCONSERVATION:", "PASS" if (ok_s and ok_c) else "FAIL")
    _plot([("Sphere (JUNO)", ds, fs, cs), ("Cylinder (SK-like)", dc, fc, cc)],
          "studies/out/sensor_detection_conservation.png")


if __name__ == "__main__":
    main()
