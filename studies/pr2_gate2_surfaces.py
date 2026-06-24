"""GATE 2 — the surface list reproduces intersect_two_spheres_forward exactly (0 mismatches)."""
import jax, jax.numpy as jnp
import numpy as np
from lucid.propagation.nested_sphere import intersect_two_spheres_forward
from lucid.propagation.surfaces import nearest_interface, sphere_forward_t, region_of_spheres
from lucid.geometry import generate_detector

R_IN, R_OUT = 17.5, 19.5
CENTERS = jnp.array([[0., 0., 0.], [0., 0., 0.]])   # inner (interface), outer
RADII = jnp.array([R_IN, R_OUT])


def main():
    print("backend:", jax.default_backend())
    r = np.random.RandomState(0)
    # rays from inside the LS and from the water shell, random directions
    n = 4000
    rad = r.uniform(0, R_OUT, n)
    u = r.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    origins = jnp.asarray(rad[:, None] * u, jnp.float32)
    dirs = jnp.asarray(r.normal(size=(n, 3)), jnp.float32)
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)

    # legacy: t_hit, hit_inner
    t_leg, hi_leg, _, _ = jax.vmap(lambda o, d: intersect_two_spheres_forward(o, d, R_IN, R_OUT))(origins, dirs)
    # surface list: nearest among [inner, outer]; inner index = 0
    t_new, which, hit, pt, nrm = jax.vmap(lambda o, d: nearest_interface(o, d, CENTERS, RADII))(origins, dirs)
    # but nearest_interface treats BOTH as "interfaces"; for the comparison we want the
    # min over both surfaces with hit_inner = (which==0). t should match the legacy nearest.
    hi_new = (which == 0)

    dt = np.max(np.abs(np.asarray(t_leg) - np.asarray(t_new)))
    dhi = int(np.sum(np.asarray(hi_leg) != np.asarray(hi_new)))
    print(f"max |t_legacy - t_surface| = {dt:.2e}   hit_inner mismatches = {dhi} / {n}")

    # region_of: legacy r>=r_inner vs surface region (inner sphere only)
    pts = jnp.asarray(r.uniform(-R_OUT, R_OUT, (5000, 3)), jnp.float32)
    det = generate_detector('config/JUNO_nested_geom_config.json')
    reg_leg = np.asarray(det.region_of(pts))                                  # 0 inner / 1 outer
    reg_new = np.asarray(region_of_spheres(pts, CENTERS[:1], RADII[:1]))      # one inner sphere → 0/1
    dreg = int(np.sum(reg_leg != reg_new))
    print(f"region_of mismatches = {dreg} / 5000")

    # offset-sphere sanity: the legacy _sphere_forward_t can't do off-centre; surfaces can
    o = jnp.array([-5., 0., 0.]); d = jnp.array([1., 0., 0.]); ctr = jnp.array([4., 0., 0.])
    t_off, _ = sphere_forward_t(o, d, ctr, 1.0)
    print(f"offset sphere t = {float(t_off):.2f} (expect 8.0; origin-only model would give 4.0)")

    # jit/vmap compiles
    _ = jax.jit(lambda o, d: jax.vmap(lambda a, b: nearest_interface(a, b, CENTERS, RADII))(o, d))(origins[:64], dirs[:64])
    ok = (dt < 1e-3) and (dhi == 0) and (dreg == 0) and (abs(float(t_off) - 8.0) < 1e-4)
    print("\nGATE 2:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
