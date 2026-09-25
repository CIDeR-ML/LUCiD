"""
Base functionality shared across all detector geometries for photon propagation.
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
from ..overlap import create_overlap_prob


def process_intersection_normals(ray_origins, ray_directions, intersection_point,
                                 t_geometry, sensor_normals, sensor_hit_positions,
                                 inside_sensor, geometry_normals):
    """
    Main function to process all normal and position calculations for intersections.
    
    This is a generic version that works for any geometry type.
    
    Parameters
    ----------
    ray_origins : ndarray
        Starting points of rays
    ray_directions : ndarray
        Direction vectors of rays
    intersection_point : ndarray
        Points of intersection with geometry
    t_geometry : ndarray
        Intersection times with geometry
    sensor_normals : ndarray
        Normal vectors for sensor intersections
    sensor_hit_positions : ndarray
        Hit positions for sensor intersections
    inside_sensor : ndarray
        Boolean array indicating which rays hit inside sensors
    geometry_normals : ndarray
        Normal vectors for the geometry surface
        
    Returns
    -------
    dict
        Contains hit positions and normals
    """
    # Calculate weighted sensor properties
    # Keep only the first sensor entered. `inside_sensor` is evaluated per candidate
    # independently, so a ray threading several spheres sets several flags. The mask-mean in
    # `calculate_weighted_sensor_properties` is an exact gather only when one flag is set; with
    # several it averages the entry points, putting the stop on no sensor's surface (possibly
    # inside a sphere) and corrupting the leg length, attenuation, arrival time, reflection normal
    # and next origin. Taking the earliest entry makes the geometry match the first-hit semantics
    # of `first_hit_survival` in shared.py (ordered by arrival time, ties broken by slot).
    #
    # A hard selection (`argmin` over entry times): it feeds hit positions and normals only, not
    # the charge weights, and `hit_sensor = any(...)` is unchanged since exactly one flag survives
    # whenever any did.
    _t_entry = jnp.sum((sensor_hit_positions - ray_origins[None, :, :])
                       * ray_directions[None, :, :], axis=-1)          # (C, N)
    _ordered = jnp.where(inside_sensor, _t_entry, jnp.inf)
    _first = jnp.argmin(_ordered, axis=0)                              # (N,)
    _slot = jnp.arange(inside_sensor.shape[0])[:, None]
    inside_first = inside_sensor & (_slot == _first[None, :])

    weighted_sensor_normals, weighted_sensor_positions = calculate_weighted_sensor_properties(
        sensor_normals, sensor_hit_positions, inside_first)

    # Calculate final hit properties
    hit_positions, final_normals = calculate_hit_properties(
        ray_origins, ray_directions, t_geometry, inside_first,
        weighted_sensor_normals, weighted_sensor_positions,
        geometry_normals)
    
    return {
        'positions': hit_positions,
        'normals': final_normals
    }


def calculate_weighted_sensor_properties(sensor_normals, sensor_hit_positions,
                                          inside_sensor):
    """
    Calculate weighted normals and positions for sensor hits.
    
    Parameters
    ----------
    sensor_normals : ndarray
        Normal vectors for all potential sensor intersections
    sensor_hit_positions : ndarray
        Hit positions for all potential sensor intersections
    inside_sensor : ndarray
        Boolean array indicating which rays hit inside sensors
        
    Returns
    -------
    tuple
        Weighted sensor normals and positions
    """
    sensor_weights = inside_sensor[..., None]
    
    # Calculate weighted normals
    weighted_normals = jnp.sum(sensor_normals * sensor_weights, axis=0)
    sensor_weights_sum = jnp.sum(inside_sensor, axis=0)[..., None]
    weighted_normals = weighted_normals / (sensor_weights_sum + 1e-10)
    
    # Calculate weighted positions
    weighted_positions = jnp.sum(sensor_hit_positions * sensor_weights, axis=0)
    weighted_positions = weighted_positions / (sensor_weights_sum + 1e-10)
    
    return weighted_normals, weighted_positions


def calculate_hit_properties(ray_origins, ray_directions, t_geometry, inside_sensor,
                             weighted_sensor_normals, weighted_sensor_positions,
                             geometry_normals):
    """
    Calculate final hit positions and normals based on intersection type.
    
    Parameters
    ----------
    ray_origins : ndarray
        Starting points of rays
    ray_directions : ndarray
        Direction vectors of rays
    t_geometry : ndarray
        Intersection times with geometry
    inside_sensor : ndarray
        Boolean array indicating which rays hit inside sensors
    weighted_sensor_normals : ndarray
        Weighted normal vectors for sensor hits
    weighted_sensor_positions : ndarray
        Weighted positions for sensor hits
    geometry_normals : ndarray
        Normal vectors for geometry hits
        
    Returns
    -------
    tuple
        Final hit positions and normals
    """
    # Calculate geometry hit positions
    geometry_hit_positions = ray_origins + t_geometry[:, None] * ray_directions
    
    # Determine if any sensor was hit
    hit_sensor = jnp.any(inside_sensor, axis=0)
    
    # Select appropriate hit position
    hit_positions = jnp.where(hit_sensor[:, None],
                              weighted_sensor_positions,
                              geometry_hit_positions)
    
    # Select appropriate normal
    final_normals = jnp.where(hit_sensor[:, None],
                              weighted_sensor_normals,
                              geometry_normals)
    
    return hit_positions, final_normals


def compute_sensor_intersections_base(sensor_idx, sensor_positions, sensor_radius,
                                        ray_origins, ray_directions, geometry_bounds_check,
                                        overlap_prob, t_geometry=None):
    """
    Base function to compute sensor intersections for any geometry.
    
    Parameters
    ----------
    sensor_idx : ndarray
        Indices of potential sensors
    sensor_positions : ndarray
        Positions of all sensors
    sensor_radius : float
        Radius of each sensor
    ray_origins : ndarray
        Starting points of rays
    ray_directions : ndarray
        Direction vectors of rays
    geometry_bounds_check : callable
        Function that checks if a point is within the geometry bounds
    overlap_prob : callable
        Function that calculates overlap probability
        
    t_geometry : array (N,) or None
        Distance along each (unit) ray to where it leaves the detector. None, the default, leaves
        the deposit on the unbounded ray line; an array bounds it to the travelled leg (see
        `deposit_leg_bound` in shared.create_propagator).

    Returns
    -------
    tuple
        weights, times, sensor_idx, normals, inside_sensor, points
    """
    valid = sensor_idx != -1
    sphere_centers = jnp.where(valid[:, None], sensor_positions[sensor_idx], jnp.zeros(3))
    
    # Find closest approach of ray to sensor center
    oc = ray_origins - sphere_centers
    ray_d = ray_directions / (jnp.linalg.norm(ray_directions, axis=1, keepdims=True) + 1e-10)
    
    # Calculate closest approach for all rays (stable for gradients)
    t_closest = -jnp.sum(oc * ray_d, axis=1, keepdims=True)
    closest = ray_origins + t_closest * ray_d
    to_sensor = closest - sphere_centers
    # Safe norm: eps inside the sqrt. `jnp.linalg.norm(v)` differentiates to v/|v|, which is 0/0 at
    # v == 0, and an epsilon added afterwards protects the division that follows but not the norm's
    # own derivative (the same failure is documented for the surface distance in simulator.py).
    # `to_sensor` is zero when a ray's closest approach lands exactly on a sensor centre, an
    # ordinary coincidence in float32. The later `jnp.where(valid, ...)` masks the value but lets
    # the NaN cotangent through.
    #
    # Sentinel slots (`sensor_idx == -1`, centre at the origin) do not occur on this engine: map
    # construction fills every slot via `add_closest`. The string propagator, where they do occur,
    # carries the same fix.
    #
    # Not a double-`where`: it gives identical values through a different HLO graph, which XLA can
    # reduce differently, and the reconstruction fit is chaotic under rounding (a double-`where`
    # moved a fitted result by 9.6%, this form by 5.6e-09).
    distance = jnp.sqrt(jnp.sum(to_sensor ** 2, axis=1) + 1e-12)
    
    # Calculate normal vectors for closest approach
    # Negate so normals point outward from detector wall (matching geometry normal convention).
    # Raw to_sensor points from sensor center toward interior (inward); negating gives outward.
    normals_closest = -to_sensor / (distance[:, None] + 1e-10)   # reuse the SAFE norm above
    
    # Ray-sphere intersection coefficients
    a = jnp.sum(ray_d * ray_d, axis=1)  # Should be 1 for normalized directions
    b = 2.0 * jnp.sum(oc * ray_d, axis=1)
    c = jnp.sum(oc * oc, axis=1) - sensor_radius ** 2
    
    # Discriminant determines if intersection exists
    discriminant = b ** 2 - 4 * a * c
    
    # Calculate actual intersection for rays that hit the sensor
    sqrt_term = jnp.sqrt(jnp.maximum(1e-10, discriminant))
    
    # Use numerically stable quadratic formula to prevent NaN gradients
    q = jnp.where(
        b > 0,
        -0.5 * (b + sqrt_term),
        -0.5 * (b - sqrt_term)
    )
    t1 = q / (a + 1e-10)
    # The epsilon's sign must not vanish: `jnp.sign(0.0)` is 0.0, so `q + jnp.sign(q) * 1e-10` would
    # divide by zero at q == 0. q == 0 is currently unreachable (`sqrt_term` is clamped to >= 1e-5
    # by the `jnp.maximum(1e-10, ...)` above, so |q| >= 5e-6), but that clamp is otherwise the only
    # guard. `where(q < 0, -1, +1)` equals `sign` everywhere except at +-0.0.
    t2 = c / (q + jnp.where(q < 0, -1.0, 1.0) * 1e-10)
    
    t_intersect = jnp.where((t1 > 0) & (t2 > 0), 
                    jnp.minimum(t1, t2),  # Both positive - take smaller
                    jnp.where(t1 > 0, t1,  # Only t1 positive
                           jnp.where(t2 > 0, t2, -1)))  # Only t2 positive or neither
    
    # Calculate intersection points
    intersection_points = ray_origins + t_intersect[:, None] * ray_d
    
    # Calculate normals at intersection points
    # Negate so normals point outward from detector wall (matching geometry normal convention).
    to_intersection = intersection_points - sphere_centers
    normals_intersect = -to_intersection / (jnp.linalg.norm(to_intersection, axis=1, keepdims=True) + 1e-10)
    
    # Determine if ray intersects with sensor (add small epsilon for stability)
    intersects = (discriminant > 1e-6) & (t_intersect > 0)
   
    # Apply overlap function to get weights
    # Only sensors AHEAD of the photon take CHARGE. `t_closest = -(oc . d_hat)` is the along-ray
    # parameter of closest approach, so `t_closest <= 0` means the sensor is BEHIND -- its closest
    # approach lies on the backward-extended ray line. Depositing there is unphysical: a photon
    # emitted just past a sensor and moving away would otherwise collect a hit with a NEGATIVE
    # transport time. The dominant victim is a photon that has just reflected off a PMT, which sits
    # at that PMT moving away from it with impact parameter ~0 and would take a full spurious
    # re-deposit on the tube it just bounced off.
    _ahead = t_closest[:, 0] > 0.0

    # OPTIONAL LEG BOUND (`t_geometry` not None), off by default.
    #
    # `distance` is the perpendicular distance from the photon's ray LINE, and that line does not
    # stop at the wall. For a ray at incidence theta it continues outside the detector and passes
    # within a sensor radius of sensors displaced ALONG the wall from where the photon landed.
    # Writing the landing point E and a candidate centre C = E + a*t + b*s, the code sees
    # perp^2 = a^2 cos^2(theta) + b^2 while the photon's closest approach on the path it travelled
    # is sqrt(a^2 + b^2): the cos(theta) foreshortening makes a downstream sensor look nearer to the
    # line than it ever was to the photon, over-counting hits by (1 - cos theta)/2 per ray.
    #
    # Bounding the deposit to the travelled leg [0, t_geometry] replaces `distance` with the closest
    # approach over that segment. Only the FRONT end is clamped: `_ahead` already zeroes
    # candidates behind the photon, so the back-end term of the segment distance is redundant.
    #
    # The branch is on a Python `None`, not a traced predicate, so with the bound off the graph is
    # exactly `overlap_prob(distance)`. A `jnp.where` would change the graph even when unused, and
    # a different graph can reduce differently in XLA (see the safe norm above). Off must be
    # bit-exact, not close.
    if t_geometry is None:
        d_eff = distance
    else:
        _past = jnp.maximum(t_closest[:, 0] - t_geometry, 0.0)
        d_eff = jnp.sqrt(distance ** 2 + _past ** 2 + 1e-12)

    weights = jnp.where(valid & _ahead, overlap_prob(d_eff), 0.0)
    # Check if point is inside sensor (keep as boolean)
    # Not gated by `_ahead`: the geometry flag uses the same distance `d_eff` as the weight above (so
    # with the leg bound on, charge and geometry agree), but not the same gate. A behind candidate
    # can still be flagged, carrying no charge, when the photon's origin lies inside its sphere (the
    # forward ray exits with t > 0). This does not arise after a sensor reflection (photon_step
    # nudges the next origin 1e-4 outside the sphere) or for bulk emission, only for origins inside
    # a PMT, which the sphere model does not describe. Gating it would change the reflection path's
    # result there, so it is left ungated.
    inside_spherical_sensor = d_eff < sensor_radius
    
    # Check if intersection point is within geometry bounds
    inside_detector_volume = geometry_bounds_check(intersection_points)
    
    # # Use correct normals based on whether ray intersects or not
    # normals = jnp.where(intersects[:, None], normals_intersect, normals_closest)

    # Final sensor condition - point must be inside both sensor and geometry
    inside_sensor = inside_spherical_sensor & inside_detector_volume
    
    # Combine boolean conditions first, then add dimension
    intersects_and_inside = (intersects & inside_sensor)
    
    # Now use this combined condition
    times   = jnp.where(intersects_and_inside[:, None], t_intersect[:, None], t_closest)
    points  = jnp.where(intersects_and_inside[:, None], intersection_points, closest)
    normals = jnp.where(intersects_and_inside[:, None], normals_intersect, normals_closest)

    #return weights, times, sensor_idx, normals, inside_sensor, points

    # In principle we should not be using anything if intersects_and_inside is false, reflecting instead out of the detector surface.
    return weights, times, sensor_idx, normals, intersects_and_inside, points


@partial(jax.jit, static_argnums=(1,))
def calculate_linear_index_base(indices, grid_dims, index_map):
    """
    Calculate linear index for grid cells.
    
    Parameters
    ----------
    indices : tuple
        Grid indices for each dimension
    grid_dims : tuple
        Dimensions of the grid
    index_map : callable
        Function that maps grid indices to linear index
        
    Returns
    -------
    int
        Linear index
    """
    return index_map(indices, grid_dims)


@partial(jax.jit, static_argnums=(2,))
def find_closest_sensors(grid_centers, sensor_positions, max_candidates_per_ray):
    """Find closest sensors to each grid cell center"""
    squared_distances = jnp.sum(
        (grid_centers[:, None, :] - sensor_positions[None, :, :]) ** 2,
        axis=2
    )
    return jax.lax.top_k(-squared_distances, max_candidates_per_ray)[1]