"""
Box (rectangular prism) detector geometry implementation.
"""

import numpy as np
import plotly.graph_objects as go
from .base import Detector


class Box(Detector):
    """Box (rectangular prism) detector geometry"""
    
    def __init__(self, length, width, height, n_sensors, sensor_radius):
        """
        Initialize box detector.
        
        Parameters:
        -----------
        length : float
            Length of the box (x-dimension)
        width : float
            Width of the box (y-dimension)
        height : float
            Height of the box (z-dimension)
        n_sensors : int
            Number of photosensors
        sensor_radius : float
            Radius of individual sensors
        """
        super().__init__(n_sensors, sensor_radius)
        self.L = length
        self.W = width
        self.H = height
        self.place_photosensors()

    def place_photosensors(self):
        """Position the photo sensor centers proportionally by surface area."""
        # Calculate surface areas for each face
        front_back_area = 2 * self.L * self.H
        left_right_area = 2 * self.W * self.H
        top_bottom_area = 2 * self.L * self.W
        total_area = front_back_area + left_right_area + top_bottom_area
        
        # Distribute sensors proportionally
        n_front_back = int(self.n_sensors * front_back_area / total_area)
        n_left_right = int(self.n_sensors * left_right_area / total_area)
        n_top_bottom = self.n_sensors - n_front_back - n_left_right
        
        n_per_front_back = n_front_back // 2
        n_per_left_right = n_left_right // 2
        n_per_top_bottom = n_top_bottom // 2
        
        # Place sensors on each face
        self.front_points = self._place_face_sensors(n_per_front_back, self.L, self.H, 'front')
        self.back_points = self._place_face_sensors(n_per_front_back, self.L, self.H, 'back')
        self.left_points = self._place_face_sensors(n_per_left_right, self.W, self.H, 'left')
        self.right_points = self._place_face_sensors(n_per_left_right, self.W, self.H, 'right')
        self.top_points = self._place_face_sensors(n_per_top_bottom, self.L, self.W, 'top')
        self.bottom_points = self._place_face_sensors(n_per_top_bottom, self.L, self.W, 'bottom')
        
        # Combine all points
        self.all_points = np.concatenate([
            self.front_points, self.back_points,
            self.left_points, self.right_points,
            self.top_points, self.bottom_points
        ], axis=0)
        
        # Create ID mappings
        self.ID_to_position = {i: self.all_points[i] for i in range(len(self.all_points))}
        
        # Create case mappings (0=front, 1=back, 2=left, 3=right, 4=top, 5=bottom)
        self.ID_to_case = {}
        n_front = len(self.front_points)
        n_back = len(self.back_points)
        n_left = len(self.left_points)
        n_right = len(self.right_points)
        n_top = len(self.top_points)
        n_bottom = len(self.bottom_points)
        
        cumulative = 0
        for i in range(len(self.all_points)):
            if i < n_front:
                self.ID_to_case[i] = 0
            elif i < n_front + n_back:
                self.ID_to_case[i] = 1
            elif i < n_front + n_back + n_left:
                self.ID_to_case[i] = 2
            elif i < n_front + n_back + n_left + n_right:
                self.ID_to_case[i] = 3
            elif i < n_front + n_back + n_left + n_right + n_top:
                self.ID_to_case[i] = 4
            else:
                self.ID_to_case[i] = 5

    def _place_face_sensors(self, n_sensors, dim1, dim2, face):
        """Place sensors on a rectangular face with regular grid."""
        if n_sensors == 0:
            return np.array([]).reshape(0, 3)
            
        # Calculate effective dimensions (with margins)
        dim1_eff = dim1 - 3 * self.S_radius
        dim2_eff = dim2 - 3 * self.S_radius
        
        if dim1_eff <= 0 or dim2_eff <= 0:
            return np.array([]).reshape(0, 3)
        
        # Find optimal rows and columns for approximately square spacing
        aspect_ratio = dim1_eff / dim2_eff
        n_rows = int(np.sqrt(n_sensors / aspect_ratio))
        n_cols = n_sensors // n_rows if n_rows > 0 else n_sensors
        
        # Adjust to get closer to target
        while n_rows * n_cols < n_sensors and n_rows > 1:
            if (n_rows + 1) * n_cols <= n_sensors:
                n_rows += 1
            elif n_rows * (n_cols + 1) <= n_sensors:
                n_cols += 1
            else:
                break
        
        # Generate grid
        pos1 = np.linspace(-dim1_eff/2, dim1_eff/2, n_cols)
        pos2 = np.linspace(-dim2_eff/2, dim2_eff/2, n_rows)
        
        points = []
        for p2 in pos2:
            for p1 in pos1:
                if face == 'front':  # +y face
                    points.append([p1 + self.C[0], self.W/2 + self.C[1], p2 + self.C[2]])
                elif face == 'back':  # -y face
                    points.append([p1 + self.C[0], -self.W/2 + self.C[1], p2 + self.C[2]])
                elif face == 'left':  # -x face
                    points.append([-self.L/2 + self.C[0], p1 + self.C[1], p2 + self.C[2]])
                elif face == 'right':  # +x face
                    points.append([self.L/2 + self.C[0], p1 + self.C[1], p2 + self.C[2]])
                elif face == 'top':  # +z face
                    points.append([p1 + self.C[0], p2 + self.C[1], self.H/2 + self.C[2]])
                elif face == 'bottom':  # -z face
                    points.append([p1 + self.C[0], p2 + self.C[1], -self.H/2 + self.C[2]])
        
        return np.array(points[:n_sensors])  # Trim to exact count

    def visualize_geometry_wireframe(self, show_sensors=True):
        """Visualize the box as a wireframe with detectors"""
        fig = go.Figure()

        # Define box vertices
        x_min, x_max = -self.L/2 + self.C[0], self.L/2 + self.C[0]
        y_min, y_max = -self.W/2 + self.C[1], self.W/2 + self.C[1]
        z_min, z_max = -self.H/2 + self.C[2], self.H/2 + self.C[2]
        
        # Create box edges
        edges = [
            # Bottom face edges
            [[x_min, x_max], [y_min, y_min], [z_min, z_min]],
            [[x_max, x_max], [y_min, y_max], [z_min, z_min]],
            [[x_max, x_min], [y_max, y_max], [z_min, z_min]],
            [[x_min, x_min], [y_max, y_min], [z_min, z_min]],
            # Top face edges
            [[x_min, x_max], [y_min, y_min], [z_max, z_max]],
            [[x_max, x_max], [y_min, y_max], [z_max, z_max]],
            [[x_max, x_min], [y_max, y_max], [z_max, z_max]],
            [[x_min, x_min], [y_max, y_min], [z_max, z_max]],
            # Vertical edges
            [[x_min, x_min], [y_min, y_min], [z_min, z_max]],
            [[x_max, x_max], [y_min, y_min], [z_min, z_max]],
            [[x_max, x_max], [y_max, y_max], [z_min, z_max]],
            [[x_min, x_min], [y_max, y_max], [z_min, z_max]],
        ]
        
        # Add edges as lines
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))

        # Add semi-transparent faces
        face_colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightcoral', 'lightcoral']
        face_names = ['Front', 'Back', 'Left', 'Right', 'Top', 'Bottom']
        
        # Front face (+y)
        x_face = [x_min, x_max, x_max, x_min, x_min]
        z_face = [z_min, z_min, z_max, z_max, z_min]
        fig.add_trace(go.Scatter3d(
            x=x_face, y=[y_max]*5, z=z_face,
            mode='lines', fill='toself',
            line=dict(color='lightblue', width=1),
            opacity=0.1, name='Front Face'
        ))
        
        # Back face (-y)
        fig.add_trace(go.Scatter3d(
            x=x_face, y=[y_min]*5, z=z_face,
            mode='lines', fill='toself',
            line=dict(color='lightblue', width=1),
            opacity=0.1, name='Back Face'
        ))

        if show_sensors:
            # Color code sensors by face
            colors = ['blue', 'darkblue', 'green', 'darkgreen', 'red', 'darkred']
            face_names_full = ['Front', 'Back', 'Left', 'Right', 'Top', 'Bottom']
            
            for case in range(6):
                indices = [i for i, c in self.ID_to_case.items() if c == case]
                if indices:
                    face_points = self.all_points[indices]
                    fig.add_trace(go.Scatter3d(
                        x=face_points[:, 0], 
                        y=face_points[:, 1], 
                        z=face_points[:, 2],
                        mode='markers',
                        marker=dict(size=4, color=colors[case], opacity=0.8),
                        name=f'{face_names_full[case]} Sensors ({len(indices)})'
                    ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            title=f'Box Detector Geometry (L={self.L}, W={self.W}, H={self.H})',
            height=800
        )

        fig.show()