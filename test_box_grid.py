import jax.numpy as jnp
from tools.propagate.box import assign_sensors_to_box_grid

# Test with a few sensors on different faces  
sensors = jnp.array([
    [2.0, 0.0, 0.0],   # Right face (+x)
    [-2.0, 0.0, 0.0],  # Left face (-x)
    [0.0, 2.0, 0.0],   # Front face (+y)
    [0.0, -2.0, 0.0],  # Back face (-y)
    [0.0, 0.0, 3.0],   # Top face (+z)
    [0.0, 0.0, -3.0],  # Bottom face (-z)
])

sensor_radius = 0.1
length, width, height = 4.0, 4.0, 6.0
n_x, n_y, n_z = 10, 10, 10

assignments = assign_sensors_to_box_grid(sensors, sensor_radius, length, width, height, n_x, n_y, n_z)
print('Grid assignments shape:', assignments.shape)
print('Sample assignments:')
for i, sensor in enumerate(sensors):
    valid_mask = assignments[i][:, 0] != -1
    valid_assignments = assignments[i][valid_mask]
    num_valid = jnp.sum(valid_mask)
    print(f'Sensor {i} at {sensor}: {num_valid} assignments')
    if num_valid > 0:
        first = valid_assignments[0]
        print(f'  First assignment: i={first[0]}, j={first[1]}, face={first[2]}')