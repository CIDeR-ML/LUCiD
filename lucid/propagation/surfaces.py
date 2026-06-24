"""Region/interface surfaces — a JAX-array surface list for the inner geometry (PR2).

The inner region(s) are described by a stack of surfaces (currently spheres, generalizable),
each tagged as an interface. Per step we take the NEAREST forward surface among the interface
surfaces (the outer instrumented surface is handled by the existing sensor propagator). This
replaces the hardcoded `intersect_two_spheres_forward` and reproduces it exactly for two
concentric spheres, while supporting non-concentric / different inner shapes.

Single-medium = empty surface list ⇒ no interface ever hit.
"""
import jax
import jax.numpy as jnp

_LARGE = 1e10


def sphere_forward_t(origin, direction, center, radius):
    """Smallest strictly-positive forward crossing of a (possibly off-centre) sphere.

    Generalises `_sphere_forward_t` with an explicit `center` (the origin-centred version is
    the `center=0` case). Returns (t, valid)."""
    eps = 1e-6
    oc = origin - center
    a = jnp.sum(direction * direction)
    b = 2.0 * jnp.sum(oc * direction)
    c = jnp.sum(oc * oc) - radius * radius
    disc = b * b - 4.0 * a * c
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    t_small = (-b - sq) / (2.0 * a)
    t_large = (-b + sq) / (2.0 * a)
    t = jnp.where(t_small > eps, t_small, jnp.where(t_large > eps, t_large, _LARGE))
    valid = (disc >= 0.0) & (t < _LARGE)
    return jnp.where(valid, t, _LARGE), valid


def nearest_interface(origin, direction, centers, radii):
    """Nearest forward interface surface among a stack of spheres.

    centers (S,3), radii (S,). Returns (t, which, hit, point, normal). `hit` is whether any
    interface is forward (False ⇒ no interface in this step). For S=0 (single-medium) callers
    skip this entirely.
    """
    ts, valids = jax.vmap(lambda c, r: sphere_forward_t(origin, direction, c, r))(centers, radii)
    which = jnp.argmin(ts)
    t = ts[which]
    hit = valids[which] & (t < _LARGE)
    point = origin + t * direction
    nrm = point - centers[which]
    normal = nrm / (jnp.linalg.norm(nrm) + 1e-10)   # outward (sign-agnostic for the interface)
    return t, which, hit, point, normal


def region_of_spheres(positions, centers, radii):
    """Region id for points given inner spheres: 0..S-1 = inside the k-th inner sphere
    (first enclosing), S = outside all (outermost region). Matches `r >= r_inner` for one
    inner sphere centred at origin."""
    # inside[k] = |p - c_k| < r_k
    d = positions[:, None, :] - centers[None, :, :]           # (N, S, 3)
    inside = jnp.linalg.norm(d, axis=-1) < radii[None, :]     # (N, S)
    S = centers.shape[0]
    first = jnp.argmax(inside, axis=1)                        # first True, or 0 if none
    any_in = jnp.any(inside, axis=1)
    return jnp.where(any_in, first, S).astype(jnp.int32)
