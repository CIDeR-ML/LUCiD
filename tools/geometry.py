import numpy as np
import json
from scipy.spatial.transform import Rotation

def generate_dataset_origins(center, heights, radii, divisions):
    """Creates a collection of origins for the dataset generation."""
    xs, ys, zs = [], [], []

    for Z in heights:
        for R, N in zip(radii, divisions):
            theta = np.linspace(0, 2*np.pi, N, endpoint=False)
            x = R * np.cos(theta) + center[0]
            y = R * np.sin(theta) + center[1]

            xs.extend(x)
            ys.extend(y)
            zs.extend([Z] * len(x))

    return xs, ys, zs

def rotate_vector(vector, axis, angle):
    """ Rotate a vector around an axis by a given angle in radians. """
    axis = normalize(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product = np.cross(axis, vector)
    dot_product = np.dot(axis, vector) * (1 - cos_angle)
    return cos_angle * vector + sin_angle * cross_product + dot_product * axis
        
def normalize(v):
    """ Normalize a vector. """
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def generate_isotropic_random_vectors(N=1):
    """ A function to generate N isotropic random vectors. """
    # Generate random azimuthal angles (phi) in the range [0, 2*pi)
    phi = 2 * np.pi * np.random.rand(N)

    # Generate random polar angles (theta) in the range [0, pi)
    theta = np.arccos(2 * np.random.rand(N) - 1)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Stack the Cartesian coordinates into a 2D array
    vectors = np.column_stack((x, y, z))

    # Normalize the vectors
    vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors_normalized

def generate_vectors_on_cone_surface(R, theta, num_vectors=10):
    """ Generate vectors on the surface of a cone around R. """
    R = normalize(R)

    # Generate random azimuthal angles from 0 to 2pi
    phi_values = np.random.uniform(0, 2 * np.pi, num_vectors)

    # Spherical to Cartesian coordinates in the local system
    x_values = np.sin(theta) * np.cos(phi_values)
    y_values = np.sin(theta) * np.sin(phi_values)
    z_value = np.cos(theta)

    local_vectors = np.column_stack((x_values, y_values, z_value * np.ones_like(x_values)))
    #local_vectors = np.column_stack((x_values, y_values, z_values))

    # Find rotation axis and angle to align local z-axis with R
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, R)
    non_zero_indices = np.linalg.norm(axis, axis=-1) != 0  # Check for non-zero norms

    # If R is not already along z-axis
    #angles = np.arccos(np.einsum('ij,ij->i', z_axis, R[non_zero_indices]))
    angles = np.arccos(np.sum(z_axis * R[non_zero_indices], axis=-1))


    # Apply rotation to vectors
    rotated_vectors = rotate_vector_batch(local_vectors[non_zero_indices], axis[non_zero_indices], angles)

    # Update the original local vectors with rotated vectors
    local_vectors[non_zero_indices] = rotated_vectors

    # Convert local vectors to global coordinates
    vectors = np.dot(local_vectors, np.linalg.norm(R))

    return vectors

def rotate_vector_batch(vectors, axes, angles):
    """ Rotate multiple vectors by specified angles around the given axes. """
    norms = np.linalg.norm(axes, axis=-1)
    axes_normalized = axes / norms[:, np.newaxis]
    quaternion = Rotation.from_rotvec(axes_normalized * angles[:, np.newaxis]).as_quat()

    # Reshape vectors to (50000, 3) if needed
    if vectors.shape[0] == 1:
        vectors = vectors.reshape(-1, 3)

    rotated_vectors = Rotation.from_quat(quaternion).apply(vectors)

    return rotated_vectors


class Cylinder:
    """Manage the detector geometry"""
    def __init__(self, center, axis, radius, height, barrel_grid, cap_rings, cyl_sensor_radius):

        self.C = center
        self.A = axis
        self.r = radius
        self.H = height 
        self.S_radius = cyl_sensor_radius

        self.place_photosensors(barrel_grid,cap_rings)

    def place_photosensors(self, barrel_grid, cap_rings):
        """Position the photo sensor centers in the cylinder surface."""
        # barrel ----
        b_rows = barrel_grid[0]
        b_cols = barrel_grid[1]

        theta = np.linspace(0, 2*np.pi, b_cols, endpoint=False)  # Generate N angles from 0 to 2pi
        x = self.r * np.cos(theta) + self.C[0]
        y = self.r * np.sin(theta) + self.C[1]
        z = [(i+1)*self.H/(b_rows+1)-self.H/2 + self.C[2] for i in range(b_rows)]

        barr_points = np.array([[x[j],y[j],z[i]] for i in range(b_rows) for j in range(b_cols)])
        self.barr_points = barr_points

        del x,y,z,theta # ensure no values are passed to the caps.
        # -----------

        # caps ----
        Nrings = len(cap_rings)

        tcap_points = []
        bcap_points = []
        for i_ring, N_sensors_in_ring in enumerate(cap_rings):
            theta = np.linspace(0, 2*np.pi, N_sensors_in_ring, endpoint=False) + 2*np.pi / N_sensors_in_ring  # Generate N angles from 0 to 2pi
            x = self.r*((Nrings-(i_ring+1))/Nrings)* np.cos(theta) + self.C[0]
            y = self.r*((Nrings-(i_ring+1))/Nrings)* np.sin(theta) + self.C[1]
            top_z = [ self.H/2 + self.C[2] for i in range(N_sensors_in_ring)]
            bot_z = [-self.H/2 + self.C[2] for i in range(N_sensors_in_ring)]

            for i_sensor in range(N_sensors_in_ring):
                tcap_points.append([x[i_sensor],y[i_sensor],top_z[i_sensor]])
                bcap_points.append([x[i_sensor],y[i_sensor],bot_z[i_sensor]])

        self.tcap_points = np.array(tcap_points)
        self.bcap_points = np.array(bcap_points)

        self.all_points = np.concatenate([self.barr_points, self.tcap_points, self.bcap_points],axis=0)

        # let's make this generic format... ID to 3D pos dictionary
        self.ID_to_position = {i:self.all_points[i] for i in range(len(self.all_points))}

        self.ID_to_case = {}
        Nbarr = len(self.barr_points)
        Ntcap = len(self.tcap_points)
        Nbcap = len(self.bcap_points)
        for i in range(len(self.all_points)):
            if i<Nbarr:
                self.ID_to_case[i] = 0
            elif Nbarr<=i<Ntcap+Nbarr:
                self.ID_to_case[i] = 1
            elif Ntcap+Nbarr<=i<Nbcap+Ntcap+Nbarr:
                self.ID_to_case[i] = 2
            else:
                print("check: place_photosensors! this should not be happening: ", Nbarr, Ntcap, Nbcap, i)

        # -----------

# 
def load_config(file_path):
    """Function to load configuration from JSON file"""
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def load_cyl_geom(file_path):
    
    # Load configuration from JSON file
    config = load_config(file_path)

    # Extract values from the loaded configuration
    cyl_center            = np.array(config['geometry_definitions']['center'])
    cyl_axis              = np.array(config['geometry_definitions']['axis'])
    cyl_radius            = config['geometry_definitions']['radius']
    cyl_height            = config['geometry_definitions']['height']
    cyl_barrel_grid       = config['geometry_definitions']['barrel_grid']
    cyl_cap_rings         = config['geometry_definitions']['cap_rings']
    cyl_sensor_radius     = config['geometry_definitions']['sensor_radius']

    return cyl_center, cyl_axis, cyl_radius, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius


def generate_detector(file_path):
    """Function to generate cylinder from json config"""
    cyl_center, cyl_axis, cyl_radius, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius = load_cyl_geom(file_path)
    return Cylinder(cyl_center, cyl_axis, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius)


def generate_dataset_point_grid(file_path):
    """Function to generate dataset point grid from json config"""
    
    # Load configuration from JSON file
    config = load_config(file_path)

    # Extract values from the loaded configuration
    cylinder_center       = config['cylinder_parameters']['center']
    cylinder_heights      = config['cylinder_parameters']['heights']
    cylinder_radii        = config['cylinder_parameters']['radii']
    cylinder_divisions    = config['cylinder_parameters']['divisions']

    return generate_dataset_origins(cylinder_center, cylinder_heights, cylinder_radii, cylinder_divisions)
















