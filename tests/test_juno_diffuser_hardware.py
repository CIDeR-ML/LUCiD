"""Deterministic contracts for the JUNO diffuser hardware geometry."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from lucid.geometry import generate_detector
from lucid.geometry.juno_diffuser import (
    DIFFUSER_FIBER,
    DIFFUSER_PEEK,
    DIFFUSER_PTFE,
    DiffuserHardware,
    intersect_diffuser_hardware,
)
from lucid.propagation.nested_sphere import create_nested_sphere_propagator
from lucid.simulation.engineering_surfaces import (
    SURFACE_NAMES,
    engineering_surface_parameters,
)


def test_junosw_ls_diffuser_dimensions():
    """The effective dimensions reproduce J26.5.1 laserdiffuser.gdml."""
    hardware = DiffuserHardware.junosw_j26_ls()

    assert hardware.ball_radius_m == 0.010
    assert hardware.peek_radius_m == 0.002
    assert hardware.peek_half_length_m == 0.004
    assert hardware.peek_center_axial_m == 0.010
    assert hardware.fiber_radius_m == 0.0005
    assert hardware.fiber_half_length_m == 0.100
    assert hardware.fiber_center_axial_m == 0.110


def test_returning_ray_hits_ptfe_ball_but_outgoing_ray_does_not():
    """The emitter surface must not immediately self-intersect outward."""
    hardware = DiffuserHardware.junosw_j26_ls()
    origins = jnp.asarray([[0.020, 0.0, 0.0], [0.010, 0.0, 0.0]])
    directions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    hit = intersect_diffuser_hardware(
        origins,
        directions,
        hardware,
        center=jnp.zeros(3),
        axis=jnp.asarray([0.0, 0.0, 1.0]),
    )

    npt.assert_array_equal(hit.valid, [True, False])
    npt.assert_array_equal(hit.surface_codes, [DIFFUSER_PTFE, -1])
    npt.assert_allclose(hit.distance_m[0], 0.010, atol=1e-7)
    npt.assert_allclose(hit.positions[0], [0.010, 0.0, 0.0], atol=1e-7)
    npt.assert_allclose(
        hit.transport_normals[0], [-1.0, 0.0, 0.0], atol=1e-7
    )


def test_peek_and_fiber_are_finite_capped_cylinders():
    """Side intersections select the PEEK tube and PE fibre independently."""
    hardware = DiffuserHardware.junosw_j26_ls()
    origins = jnp.asarray([
        [0.003, 0.0, 0.012],
        [0.002, 0.0, 0.100],
    ])
    directions = jnp.asarray([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    hit = intersect_diffuser_hardware(
        origins,
        directions,
        hardware,
        center=jnp.zeros(3),
        axis=jnp.asarray([0.0, 0.0, 1.0]),
    )

    npt.assert_array_equal(hit.valid, [True, True])
    npt.assert_array_equal(
        hit.surface_codes, [DIFFUSER_PEEK, DIFFUSER_FIBER]
    )
    npt.assert_allclose(hit.distance_m, [0.001, 0.0015], atol=1e-7)


def test_diffuser_pose_translates_and_rotates_with_source():
    """The hardware follows the source centre and its configured local axis."""
    hardware = DiffuserHardware.junosw_j26_ls()
    center = jnp.asarray([1.0, 2.0, 3.0])
    axis = jnp.asarray([1.0, 0.0, 0.0])
    origins = jnp.asarray([[1.012, 2.003, 3.0]])
    directions = jnp.asarray([[0.0, -1.0, 0.0]])

    hit = intersect_diffuser_hardware(
        origins, directions, hardware, center=center, axis=axis
    )

    assert bool(np.asarray(hit.valid[0]))
    assert int(np.asarray(hit.surface_codes[0])) == DIFFUSER_PEEK
    npt.assert_allclose(hit.distance_m[0], 0.001, atol=1e-7)


def test_detector_loader_has_independent_diffuser_hardware_switch():
    """The checked-in baseline can enable or disable hardware alone."""
    config = "config/JUNO_laser_geom_config.json"

    enabled = generate_detector(config, diffuser_hardware_enabled=True)
    disabled = generate_detector(config, diffuser_hardware_enabled=False)

    assert enabled.diffuser_hardware is not None
    assert enabled.diffuser_hardware_geometry == "juno_ls_diffuser_hardware"
    assert disabled.diffuser_hardware is None
    assert disabled.diffuser_hardware_geometry is None


def test_diffuser_competes_with_detector_surfaces_in_propagator():
    """A nearby hardware hit wins the common nearest-surface selection."""
    sensors = jnp.asarray([
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, -2.0],
    ])
    propagate = create_nested_sphere_propagator(
        sensors,
        0.1,
        1.0,
        2.0,
        n_divisions=4,
        max_candidates_per_ray=1,
        temperature=None,
        r_acrylic=1.1,
        diffuser_hardware=DiffuserHardware.junosw_j26_ls(),
    )

    result = propagate(
        jnp.asarray([[0.020, 0.0, 0.0]]),
        jnp.asarray([[-1.0, 0.0, 0.0]]),
        jnp.zeros(3),
        jnp.asarray([0.0, 0.0, 1.0]),
    )

    npt.assert_array_equal(result["engineering_surface_code"], [4])
    npt.assert_array_equal(result["hit_diffuser_hardware"], [True])
    npt.assert_allclose(result["positions"], [[0.010, 0.0, 0.0]], atol=1e-7)


def test_diffuser_surface_parameters_are_complete():
    """All enabled diffuser parts have validated A/R/T parameters."""
    physics = {
        "diffuser_ptfe_reflection_rate": 0.95,
        "diffuser_ptfe_transmission_rate": 0.0,
        "diffuser_ptfe_fspec": 1.0,
        "diffuser_peek_reflection_rate": 0.5,
        "diffuser_peek_transmission_rate": 0.0,
        "diffuser_peek_fspec": 0.0,
        "diffuser_fiber_reflection_rate": 0.1,
        "diffuser_fiber_transmission_rate": 0.0,
        "diffuser_fiber_fspec": 0.0,
    }

    reflectance, transmission, fspec = engineering_surface_parameters(
        physics, required_codes=(4, 5, 6)
    )

    assert SURFACE_NAMES[4:7] == (
        "diffuser_ptfe", "diffuser_peek", "diffuser_fiber"
    )
    npt.assert_allclose(reflectance[4:7], [0.95, 0.5, 0.1])
    npt.assert_allclose(transmission[4:7], [0.0, 0.0, 0.0])
    npt.assert_allclose(fspec[4:7], [1.0, 0.0, 0.0])
