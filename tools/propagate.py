"""
Main propagation module - provides unified interface for all detector geometries.

This module imports and exposes the geometry-specific propagation functions in a clean way.
It maintains backward compatibility while providing the improved, organized structure.
"""

# Import geometry-specific modules
from tools.propagate_cylinder import (
    create_photon_propagator as create_cylinder_photon_propagator,
    intersect_cylinder,
    intersect_cylinder_with_grid,
    batch_intersect_cylinder_with_grid,
    find_intersected_sensors_differentiable as find_intersected_cylinder_sensors_differentiable,
    assign_sensors_to_grid as assign_sensors_to_cylinder_grid,
    create_sensor_grid_map as create_cylinder_sensor_grid_map,
    calculate_grid_centers as calculate_cylinder_grid_centers,
    create_inverted_sensor_map as create_cylinder_inverted_sensor_map
)

from tools.propagate_sphere import (
    create_sphere_photon_propagator,
    intersect_sphere,
    intersect_sphere_with_grid,
    batch_intersect_sphere_with_grid,
    find_intersected_sphere_sensors_differentiable,
    assign_sensors_to_sphere_grid,
    create_sensor_sphere_grid_map,
    calculate_sphere_grid_centers,
    create_inverted_sphere_sensor_map
)

from tools.propagate_box import (
    create_box_photon_propagator,
    intersect_box,
    intersect_box_with_grid,
    batch_intersect_box_with_grid,
    find_intersected_box_sensors_differentiable,
    assign_sensors_to_box_grid,
    create_sensor_box_grid_map,
    calculate_box_grid_centers
)

from tools.propagate_base import (
    process_intersection_normals,
    calculate_weighted_sensor_properties,
    calculate_hit_properties,
    compute_sensor_intersections_base,
    find_closest_sensors
)

# For backward compatibility, maintain the original interface
# These aliases preserve the existing API
create_photon_propagator = create_cylinder_photon_propagator
find_intersected_sensors_differentiable = find_intersected_cylinder_sensors_differentiable
assign_sensors_to_grid = assign_sensors_to_cylinder_grid
create_sensor_grid_map = create_cylinder_sensor_grid_map
calculate_grid_centers = calculate_cylinder_grid_centers
create_inverted_sensor_map = create_cylinder_inverted_sensor_map


def create_propagator_by_geometry(geometry_type, sensor_positions, sensor_radius, **kwargs):
    """
    Factory function to create the appropriate propagator based on geometry type.
    
    Parameters
    ----------
    geometry_type : str
        Type of geometry ('cylinder', 'sphere', or 'box')
    sensor_positions : ndarray
        Array of sensor positions
    sensor_radius : float
        Radius of each sensor
    **kwargs : dict
        Geometry-specific parameters
        
    Returns
    -------
    callable
        JIT-compiled photon propagation function
        
    Raises
    ------
    ValueError
        If geometry_type is not recognized
    """
    if geometry_type.lower() == 'cylinder':
        return create_cylinder_photon_propagator(sensor_positions, sensor_radius, **kwargs)
    elif geometry_type.lower() == 'sphere':
        return create_sphere_photon_propagator(sensor_positions, sensor_radius, **kwargs)
    elif geometry_type.lower() == 'box':
        return create_box_photon_propagator(sensor_positions, sensor_radius, **kwargs)
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}. "
                        "Supported types are: 'cylinder', 'sphere', 'box'")


# Expose all the individual intersection functions for direct use if needed
__all__ = [
    # Factory function
    'create_propagator_by_geometry',
    
    # Backward compatibility - Cylinder (default)
    'create_photon_propagator',
    'find_intersected_sensors_differentiable', 
    'assign_sensors_to_grid',
    'create_sensor_grid_map',
    'calculate_grid_centers',
    'create_inverted_sensor_map',
    'intersect_cylinder',
    'intersect_cylinder_with_grid',
    'batch_intersect_cylinder_with_grid',
    
    # Cylinder-specific
    'create_cylinder_photon_propagator',
    'find_intersected_cylinder_sensors_differentiable',
    'assign_sensors_to_cylinder_grid',
    'create_cylinder_sensor_grid_map',
    'calculate_cylinder_grid_centers',
    'create_cylinder_inverted_sensor_map',
    
    # Sphere-specific
    'create_sphere_photon_propagator',
    'intersect_sphere',
    'intersect_sphere_with_grid',
    'batch_intersect_sphere_with_grid',
    'find_intersected_sphere_sensors_differentiable',
    'assign_sensors_to_sphere_grid',
    'create_sensor_sphere_grid_map',
    'calculate_sphere_grid_centers',
    'create_inverted_sphere_sensor_map',
    
    # Box-specific
    'create_box_photon_propagator',
    'intersect_box',
    'intersect_box_with_grid',
    'batch_intersect_box_with_grid',
    'find_intersected_box_sensors_differentiable',
    'assign_sensors_to_box_grid',
    'create_sensor_box_grid_map',
    'calculate_box_grid_centers',
    
    # Base functions
    'process_intersection_normals',
    'calculate_weighted_sensor_properties',
    'calculate_hit_properties',
    'compute_sensor_intersections_base',
    'find_closest_sensors'
]


# Legacy support warning function
def _show_migration_info():
    """
    Print information about the refactored propagate module structure.
    """
    print("=" * 70)
    print("LUCID Propagate Module - Refactored Structure")
    print("=" * 70)
    print()
    print("The propagate module has been refactored for better organization:")
    print()
    print("📁 File Structure:")
    print("  • propagate.py          - Main interface (this file)")
    print("  • propagate_base.py     - Shared functionality")
    print("  • propagate_cylinder.py - Cylinder-specific functions")
    print("  • propagate_sphere.py   - Sphere-specific functions (with fixes)")
    print("  • propagate_box.py      - Box detector functions (NEW)")
    print()
    print("🔧 Key Improvements:")
    print("  • Fixed sphere geometry indexing wraparound issues")
    print("  • Added Box detector geometry support")
    print("  • Consistent function naming across geometries")
    print("  • Better code organization and maintainability")
    print()
    print("🚀 Usage:")
    print("  # Backward compatible (cylinder)")
    print("  propagator = create_photon_propagator(detector_positions, radius)")
    print()
    print("  # Geometry-specific")
    print("  propagator = create_propagator_by_geometry('sphere', positions, radius)")
    print("  propagator = create_propagator_by_geometry('box', positions, radius,")
    print("                                             length=4, width=4, height=6)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    _show_migration_info()