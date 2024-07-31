import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

def create_color_gradient(max_cnts, colormap='viridis'):
    """Define the color scale in the 2D event display"""

    # Define the colormap and normalization
    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=max_cnts)

    # Create a scalar mappable
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    return scalar_mappable

def show_2D_display(ID_to_position, ID_to_PE, ID_to_case, cyl_sensor_radius, cyl_radius, cyl_height, file_name=None):
    """Do the 2D event display"""
    max_PE = np.max(list(ID_to_PE.values()))
    color_gradient = create_color_gradient(max_PE)

    fig, ax = plt.subplots(figsize=(8,8),facecolor='black')

    corr = cyl_radius/cyl_height
    caps_offset = -0.1

    all_pos = []


    for ID in list(ID_to_position.keys()):
        pos   = ID_to_position[ID]
        PE    = ID_to_PE[ID]
        case  = ID_to_case[ID]

        if PE:
            all_pos.append(pos)

        #barrel
        if case ==0:
            theta = np.arctan(pos[1]/pos[0]) if pos[0] != 0 else np.pi/2
            theta += np.pi/2
            if pos[0]>0:
                theta += np.pi
            theta /=2
            z = pos[2]/cyl_height

            ax.add_patch(plt.Circle((theta, z), cyl_sensor_radius/cyl_height, color=color_gradient.to_rgba(PE)))

        elif case ==1:
            ax.add_patch(plt.Circle((corr*pos[0]/cyl_height+np.pi/2, 1+corr*(caps_offset+pos[1]/cyl_height)), cyl_sensor_radius/cyl_height, color=color_gradient.to_rgba(PE)))

        elif case ==2:
            ax.add_patch(plt.Circle((corr*pos[0]/cyl_height+np.pi/2,-1+corr*(-caps_offset-pos[1]/cyl_height)), cyl_sensor_radius/cyl_height, color=color_gradient.to_rgba(PE)))

    margin = 0.05

    all_pos = np.array(all_pos)
    print(min(all_pos[:,0]),max(all_pos[:,0]))
    print(min(all_pos[:,1]),max(all_pos[:,1]))
    print(min(all_pos[:,2]),max(all_pos[:,2]))

    ax.set_facecolor("black")

    #hide x-axis
    ax.get_xaxis().set_visible(False)
    #hide y-axis 
    ax.get_yaxis().set_visible(False)
    plt.axis('equal')
    fig.tight_layout()
    if file_name:
        plt.savefig(file_name)
    plt.show()
