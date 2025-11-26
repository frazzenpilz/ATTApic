import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pic_modules.parameters import Parameters
from pic_modules.mesh import Mesh
import sys

def visualize_mesh_3d():
    print("Initializing Parameters...")
    parameters = Parameters("inputs.txt")
    parameters.assign_inputs()
    
    print("Generating Mesh...")
    mesh = Mesh(parameters, "PIC")
    
    print(f"Mesh Stats: {mesh.num_nodes} nodes, {mesh.num_cells} cells")
    
    # Extract Node Coordinates
    x_coords = []
    y_coords = []
    z_coords = [] # All 0 for 2D mesh
    
    for node in mesh.nodes:
        x_coords.append(node.x)
        y_coords.append(node.y)
        z_coords.append(0.0)
        
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if parameters.axisymmetric:
        print("Axisymmetric mode detected. Revolving mesh for 3D visualization...")
        import numpy as np
        
        # Revolve around X-axis for horizontal cylinder orientation
        # Simulation coordinates (after fix): node.x = radial (r), node.y = axial (z)
        # Plot mapping for intuitive visualization:
        #   - Sim node.y (axial, z, length) -> Plot X-axis (horizontal, long direction)
        #   - Sim node.x (radial, r, height) -> Plot Y-Z plane (radius from axis)
        
        angles = np.linspace(0, 2*np.pi, 16) # 16 slices
        
        for angle in angles:
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            # Plot Cells
            for cell in mesh.cells:
                pts = []
                for node_id in cell.node_ids:
                    node = mesh.nodes[node_id - 1]
                    # After fix: node.x is Radial (r), node.y is Axial (z)
                    
                    # Plot coordinates for horizontal cylinder:
                    # X-axis = axial (length), Y-Z plane = radial (radius)
                    px = node.y              # Axial (z) along X-axis
                    py = node.x * cos_a      # Radial (r) in Y-direction
                    pz = node.x * sin_a      # Radial (r) in Z-direction
                    
                    pts.append((px, py, pz))
                
                pts.append(pts[0])
                
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]
                
                ax.plot(xs, ys, zs, c='b', alpha=0.1, linewidth=0.5)
        
        # Plot Nodes (Revolved)
        # Just plot the generating plane nodes to avoid clutter, or a few slices?
        # Let's plot the nodes on the Z-Y plane (angle=pi/2) and Z-X plane (angle=0)
        # Or just all of them as dots? Might be too many.
        # Let's plot nodes for angle=0 (The slice)
        
        slice_x = []
        slice_y = []
        slice_z = []
        for node in mesh.nodes:
            # Plot nodes at angle=0 (slice in X-Y plane, Z=0)
            # X = axial (node.y), Y = radial (node.x), Z = 0
            slice_x.append(node.y)   # Axial along X
            slice_y.append(node.x)   # Radial in Y
            slice_z.append(0)        # At Z=0
            
        ax.scatter(slice_x, slice_y, slice_z, c='r', marker='o', s=10, label='Nodes (Slice)')

        ax.set_xlabel("X (Axial, z) [m]")
        ax.set_ylabel("Y (Radial, r) [m]")
        ax.set_zlabel("Z (Radial, r) [m]")
        limits = max(parameters.domain_height, parameters.domain_length)
        ax.set_xlim(0, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(-limits, limits)
        ax.set_title('PIC-IPD Mesh Visualization (Axisymmetric/Cylindrical)')
        ax.legend()
        
    else:
        # Cartesian 2D in 3D
        # Plot Nodes
        ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o', s=10, label='Nodes')
        
        # Plot Edges (Cells)
        print("Plotting Cells...")
        for cell in mesh.cells:
            # Get nodes for this cell
            # cell.node_ids are 1-based
            pts = []
            for node_id in cell.node_ids:
                node = mesh.nodes[node_id - 1]
                pts.append((node.x, node.y, 0.0))
            
            # Close the loop
            pts.append(pts[0])
            
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            
            ax.plot(xs, ys, zs, c='b', alpha=0.5, linewidth=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('PIC-IPD Mesh Visualization (3D View)')
        ax.legend()
    
    # Set aspect ratio to be equal
    # import numpy as np
    # ax.set_box_aspect((np.ptp(x_coords), np.ptp(y_coords), 1)) # Requires numpy
    
    output_file = "mesh_visualization_3d.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    visualize_mesh_3d()
