"""Tests for the reproducible static JUNO geometry renderer."""

import numpy as np
import numpy.testing as npt

from lucid.geometry.juno_structures import acrylic_bowl_radius
from scripts.render_juno_known_structures import build_bowl_surface


def test_static_bowl_surface_uses_transport_arc_profile():
    """Static 3D bowls must use the production R=776 mm profile."""
    axial = np.linspace(0.0, 0.140, 5)
    x_coord, y_coord, z_coord = build_bowl_surface(
        attachment=np.array([0.0, 0.0, 17.824]),
        outward_axis=np.array([0.0, 0.0, 1.0]),
        axial_samples=axial,
        azimuth_samples=8,
    )

    radii = np.sqrt(x_coord**2 + y_coord**2)
    expected = np.asarray(acrylic_bowl_radius(axial))
    expected_radii = np.broadcast_to(expected[:, None], radii.shape)
    expected_z = np.broadcast_to(
        17.824 + axial[:, None], z_coord.shape
    )
    npt.assert_allclose(radii, expected_radii, atol=1e-12)
    npt.assert_allclose(z_coord, expected_z)
