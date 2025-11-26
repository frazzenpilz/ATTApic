"""
Mesh Module - Grid generation and management for PIC/FDTD simulations

Based on C++ PIC-IPD implementation with proper Node/Cell/Face structures.
"""

import math
from .utils import log_messages, log_brief
import sys

class Node:
    """
    Node class representing a grid point.
    
    Attributes based on C++ Nodes class (Nodes.h):
    - EMfield[6]: [Ex, Ey, Ez, Bx, By, Bz]  
    - current[2]: [Jx, Jy]
    - phi: Electric potential
    - rho: Charge density
    """
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        
        # Fields
        self.phi = 0.0                                      # Electric potential
        self.rho = 0.0                                      # Charge density
        self.em_field = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # [Ex, Ey, Ez, Bx, By, Bz]
        self.current = [0.0, 0.0, 0.0]                      # [Jx, Jy, Jz]
        
        # Boundary classification
        self.boundary_type = "internal"
        
        # Neighbor node IDs (1-based, 0 means boundary)
        self.left_node_id = 0
        self.right_node_id = 0
        self.top_node_id = 0
        self.bottom_node_id = 0
        
        # Periodic connectivity
        self.periodic_x1_node_id = 0
        self.periodic_x2_node_id = 0

class Face:
    """Face class representing an edge between cells."""
    def __init__(self, id, node_ids, left_cell_id, right_cell_id):
        self.id = id
        self.node_ids = node_ids  # [node1, node2]
        self.left_cell_id = left_cell_id
        self.right_cell_id = right_cell_id
        self.type = 0

class Cell:
    """Cell class representing a computational cell."""
    def __init__(self, id, node_ids):
        self.id = id
        self.node_ids = node_ids  # [node1, node2, node3, node4] (quad cell)
        self.face_ids = [-1, -1, -1, -1]  # [left, bottom, right, top]
        self.boundary_type = "internal"
        
        # Particle tracking
        self.particles = []
        self.num_neutrals = 0
        self.num_ions = 0
        
        # Geometry (bounding box)
        self.left = 0.0
        self.right = 0.0
        self.top = 0.0
        self.bottom = 0.0
        
        # Neighbor cell IDs (1-based, -1 means boundary)
        self.left_cell_id = -1
        self.right_cell_id = -1
        self.top_cell_id = -1
        self.bottom_cell_id = -1
        
        # Periodic connectivity
        self.periodic_x1_cell_id = -1
        self.periodic_x2_cell_id = -1

class Mesh:
    """
    Mesh class for structured grid generation.
    
    Generates uniform rectangular grid for PIC or FDTD simulations.
    """
    def __init__(self, parameters, mesh_type="PIC"):
        self.parameters = parameters
        self.mesh_type = mesh_type
        self.nodes = []
        self.cells = []
        self.faces = []
        self.num_nodes = 0
        self.num_cells = 0
        self.num_faces = 0
        self.h = 0.0  # Cell spacing
        
        self.generate_mesh()

    def generate_mesh(self):
        """Generate structured rectangular mesh."""
        log_messages(f"Generating {self.mesh_type} mesh", __file__, sys._getframe().f_lineno, 1)
        
        # Select spacing
        if self.mesh_type == "PIC":
            self.h = self.parameters.pic_spacing
        elif self.mesh_type == "FDTD":
            self.h = self.parameters.fdtd_spacing
        else:
            self.h = self.parameters.pic_spacing
            
        domain_length = self.parameters.domain_length
        domain_height = self.parameters.domain_height
        
        # Calculate grid dimensions (number of cells)
        # X = radial (r), Y = axial (z) in cylindrical coordinates
        self.num_cells_x = int(round(domain_height / self.h))   # Radial direction (r)
        self.num_cells_y = int(round(domain_length / self.h))   # Axial direction (z)
        
        self.num_cells = self.num_cells_x * self.num_cells_y
        self.num_nodes = (self.num_cells_x + 1) * (self.num_cells_y + 1)
        
        # Create nodes
        self._create_nodes()
        
        # Set node neighbors
        self._set_node_neighbors()
        
        # Create cells
        self._create_cells()
        
        log_brief(f"Mesh generated: {self.num_nodes} nodes, {self.num_cells} cells", 1)

    def _create_nodes(self):
        """Create grid nodes."""
        domain_length = self.parameters.domain_length
        domain_height = self.parameters.domain_height
        
        node_id = 1
        for i in range(self.num_cells_x + 1):
            for j in range(self.num_cells_y + 1):
                x = self.h * i                      # Radial coordinate (r)
                # Top-down indexing: j=0 is top (y=L), j=num_cells_y is bottom (y=0)
                y = domain_length - self.h * j      # Axial coordinate (z)
                
                node = Node(node_id, x, y)
                
                # Classify boundary nodes
                is_left = abs(x) <= 1e-10                       # r = 0 (symmetry axis)
                is_right = abs(x - domain_height) <= 1e-10      # r = max (outer wall)
                is_top = abs(y - domain_length) <= 1e-10        # z = max (top boundary)
                is_bottom = abs(y) <= 1e-10                     # z = 0 (bottom boundary)
                
                if is_left and is_top:
                    node.boundary_type = "TL"
                elif is_left and is_bottom:
                    node.boundary_type = "BL"
                elif is_right and is_top:
                    node.boundary_type = "TR"
                elif is_right and is_bottom:
                    node.boundary_type = "BR"
                elif is_left:
                    node.boundary_type = "L"
                elif is_right:
                    node.boundary_type = "R"
                elif is_top:
                    node.boundary_type = "T"
                elif is_bottom:
                    node.boundary_type = "B"
                else:
                    node.boundary_type = "internal"
                
                self.nodes.append(node)
                node_id += 1

    def _set_node_neighbors(self):
        """Set neighbor node IDs."""
        for i in range(self.num_cells_x + 1):
            for j in range(self.num_cells_y + 1):
                # Node indexing: index = i * (num_cells_y + 1) + j
                current_idx = i * (self.num_cells_y + 1) + j
                node = self.nodes[current_idx]
                
                # Left neighbor (i-1, j)
                if i > 0:
                    node.left_node_id = (i - 1) * (self.num_cells_y + 1) + j + 1
                
                # Right neighbor (i+1, j)
                if i < self.num_cells_x:
                    node.right_node_id = (i + 1) * (self.num_cells_y + 1) + j + 1
                
                # Top neighbor (i, j-1)
                if j > 0:
                    node.top_node_id = i * (self.num_cells_y + 1) + (j - 1) + 1
                
                # Bottom neighbor (i, j+1)
                if j < self.num_cells_y:
                    node.bottom_node_id = i * (self.num_cells_y + 1) + (j + 1) + 1

    def _create_cells(self):
        """Create computational cells."""
        for cell_id in range(1, self.num_cells + 1):
            # Cell indexing: cell_id = i * num_cells_y + j + 1 (1-based)
            i = (cell_id - 1) // self.num_cells_y
            j = (cell_id - 1) % self.num_cells_y
            
            # Get corner node indices (0-based)
            # Cell connects: (i,j), (i,j+1), (i+1,j+1), (i+1,j)
            n1 = i * (self.num_cells_y + 1) + j
            n2 = i * (self.num_cells_y + 1) + j + 1
            n3 = (i + 1) * (self.num_cells_y + 1) + j + 1
            n4 = (i + 1) * (self.num_cells_y + 1) + j
            
            # Node IDs are 1-based
            node_ids = [n1 + 1, n2 + 1, n3 + 1, n4 + 1]
            
            cell = Cell(cell_id, node_ids)
            
            # Set geometry
            cell.left = self.nodes[n1].x
            cell.right = self.nodes[n3].x
            cell.top = self.nodes[n1].y
            cell.bottom = self.nodes[n2].y
            
            # Classify boundary cells
            is_left = (i == 0)
            is_right = (i == self.num_cells_x - 1)
            is_top = (j == 0)
            is_bottom = (j == self.num_cells_y - 1)
            
            if is_left and is_top:
                cell.boundary_type = "TL"
            elif is_left and is_bottom:
                cell.boundary_type = "BL"
            elif is_right and is_top:
                cell.boundary_type = "TR"
            elif is_right and is_bottom:
                cell.boundary_type = "BR"
            elif is_left:
                cell.boundary_type = "L"
            elif is_right:
                cell.boundary_type = "R"
            elif is_top:
                cell.boundary_type = "T"
            elif is_bottom:
                cell.boundary_type = "B"
            else:
                cell.boundary_type = "internal"
            
            self.cells.append(cell)

    def add_particle_to_cell(self, cell_id, particle_id, particle_type):
        """Add particle to cell tracking."""
        if 1 <= cell_id <= len(self.cells):
            self.cells[cell_id - 1].particles.append(particle_id)
            if particle_type == 0:
                self.cells[cell_id - 1].num_neutrals += 1
            elif particle_type == 1:
                self.cells[cell_id - 1].num_ions += 1

    def remove_particle_from_cell(self, cell_id, particle_id, particle_type):
        """Remove particle from cell tracking."""
        if 1 <= cell_id <= len(self.cells):
            try:
                self.cells[cell_id - 1].particles.remove(particle_id)
                if particle_type == 0:
                    self.cells[cell_id - 1].num_neutrals -= 1
                elif particle_type == 1:
                    self.cells[cell_id - 1].num_ions -= 1
            except ValueError:
                pass  # Particle not in list
