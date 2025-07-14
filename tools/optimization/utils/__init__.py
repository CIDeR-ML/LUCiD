"""
Common utilities for LUCiD optimization.
"""

from .geometry import (
    get_cherenkov_angle, create_cylinder_surface, create_sphere_surface,
    create_box_surface, compute_cone_cylinder_intersection,
    compute_cone_sphere_intersection, compute_cone_box_intersection
)
from .visualization import (
    create_event_visualization, print_summary_statistics,
    create_convergence_plots, create_summary_plots
)

__all__ = [
    'get_cherenkov_angle',
    'create_cylinder_surface',
    'create_sphere_surface', 
    'create_box_surface',
    'compute_cone_cylinder_intersection',
    'compute_cone_sphere_intersection',
    'compute_cone_box_intersection',
    'create_event_visualization',
    'print_summary_statistics',
    'create_convergence_plots',
    'create_summary_plots'
]