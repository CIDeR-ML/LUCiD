"""The propagation primitives must run under `jax_enable_x64`, not only at float32.

Why this exists
---------------
`jax_enable_x64` is never switched on in this repository, so every gradient the reconstruction
takes is float32. That is a performance choice, not a correctness one -- and it hid a defect that
made float64 IMPOSSIBLE rather than merely unused:

    lucid/propagation/cylinder.py   parallel_side_branch  -> jnp.array(LARGE, dtype=jnp.float32)
    lucid/propagation/sphere.py     no_intersection_branch -> jnp.array(LARGE, dtype=jnp.float32)

`lax.cond` requires both branches to return identical types. The sibling branch's `tval_` follows
the input precision. With x64 off everything is float32 and the two agree by accident; with x64 on
the sibling becomes float64 and JAX raises

    TypeError: true_fun output and false_fun output must have identical types,
               got ... float32[] vs. float64[]

It surfaced while trying to compare a gradient at float32 against the same gradient at float64 --
the direct test of whether float32 cancellation was corrupting it. The comparison could not run at
all, which is a poor reason not to be able to check your own arithmetic.

These tests set x64 in a SUBPROCESS. The flag is process-global and must be set before jax touches
an array, so it cannot be toggled inside a session that has already imported and used jax without
contaminating every other test in the run.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run_x64(body: str):
    """Execute `body` in a fresh interpreter with jax_enable_x64 on."""
    src = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO)!r})
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        import numpy as np
        assert jnp.zeros(1).dtype == jnp.float64, 'x64 did not take effect'
{textwrap.indent(textwrap.dedent(body), ' ' * 8)}
        print('OK')
    """)
    r = subprocess.run([sys.executable, '-c', src], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith('OK'), \
        f'x64 subprocess failed\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr[-3000:]}'
    return r.stdout


def test_cylinder_wall_intersection_runs_at_float64():
    """The branch that used to pin float32. A ray along +z takes the parallel branch."""
    _run_x64("""
        from lucid.propagation.cylinder import intersect_cylinder_wall
        # purely axial: |a| < 1e-12 selects parallel_side_branch, the one that was pinned
        o = jnp.array([0.0, 0.0, -5.0]); d = jnp.array([0.0, 0.0, 1.0])
        hit, t = intersect_cylinder_wall(o, d, 5.0, 20.0)
        assert t.dtype == jnp.float64, t.dtype
        # and the other branch, so both are exercised in the same precision
        o2 = jnp.array([0.0, 0.0, 0.0]); d2 = jnp.array([1.0, 0.0, 0.0])
        hit2, t2 = intersect_cylinder_wall(o2, d2, 5.0, 20.0)
        assert t2.dtype == jnp.float64, t2.dtype
        assert bool(hit2) and np.isfinite(float(t2))
    """)


def test_sphere_intersection_runs_at_float64():
    """The same defect in the sphere primitive; a miss takes the pinned branch."""
    _run_x64("""
        from lucid.propagation.sphere import intersect_sphere
        c = jnp.array([0.0, 0.0, 0.0])
        # a MISS takes `no_intersection_branch`, the one that was pinned
        o = jnp.array([0.0, 0.0, -50.0]); d = jnp.array([1.0, 0.0, 0.0])
        hit, t = intersect_sphere(o, d, c, 5.0)
        assert t.dtype == jnp.float64, t.dtype
        assert not bool(hit)
        # and a hit, so both branches run in the same precision
        o2 = jnp.array([0.0, 0.0, 0.0]); d2 = jnp.array([0.0, 0.0, 1.0])
        hit2, t2 = intersect_sphere(o2, d2, c, 5.0)
        assert t2.dtype == jnp.float64, t2.dtype
        assert np.isfinite(float(t2))
    """)


def test_box_grid_assignment_runs_at_float64():
    """The third pin of the same class, and the one that made the fix incomplete.

    `assign_off_surface` returns int32 while its sibling builds `indices` around
    `jnp.argmin(...)`, which is int64 once x64 is on -- so `lax.cond` rejected the pair and box
    geometries could not be built in double precision at all. Unlike the cylinder and sphere
    versions of the same pin, which survive because every index component is cast to int32.
    """
    _run_x64("""
        from lucid.geometry import generate_detector
        from lucid.propagation.box import assign_sensors_to_box_grid
        det = generate_detector('config/MidBox_geom_config.json')
        # signature: (sensors, sensor_radius, length, width, height, n_x, n_y, n_z)
        out = assign_sensors_to_box_grid(
            jnp.array(det.all_points), det.S_radius, det.L, det.W, det.H, 5, 5, 5)
        # int32 is the CORRECT answer here: the grid indices are deliberately int32 and the
        # off-surface branch matches them. The bug was the two disagreeing, not the width.
        assert out.dtype == jnp.int32, out.dtype
        assert out.shape[-1] == 3, out.shape
    """)


def test_float32_is_unchanged_by_the_dtype_fix():
    """The fix must be invisible at float32, which is the precision everything actually runs in."""
    import jax.numpy as jnp
    import numpy as np

    from lucid.propagation.cylinder import intersect_cylinder_wall
    o = jnp.array([0.0, 0.0, -5.0]); d = jnp.array([0.0, 0.0, 1.0])
    hit, t = intersect_cylinder_wall(o, d, 5.0, 20.0)
    assert t.dtype == jnp.float32
    assert not bool(hit)
    assert np.isfinite(float(t))
