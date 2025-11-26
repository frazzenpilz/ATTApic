"""
Solver Module for PIC-IPD Simulation
Contains the PICSolver class and JIT-compiled kernels for particle pushing, field interpolation, and source projection.
"""

import numpy as np
import math
from numba import jit, prange
from .constants import EPSILON_0

# --- JIT Kernels ---

@jit(nopython=True, parallel=True)
def pusher_kernel(x, y, z, vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, mass, charge, active, dt, domain_length, domain_height, bc_type, bc_value):
    n = len(x)
    for i in prange(n):
        if not active[i]:
            continue
            
        # Boris Pusher
        q_m = charge[i] / mass[i]
        k = q_m * dt * 0.5
        
        # t vector
        tx = k * Bx[i]
        ty = k * By[i]
        tz = k * Bz[i]
        
        t_mag2 = tx*tx + ty*ty + tz*tz
        
        # s vector
        sx = 2.0 * tx / (1.0 + t_mag2)
        sy = 2.0 * ty / (1.0 + t_mag2)
        sz = 2.0 * tz / (1.0 + t_mag2)
        
        # v_minus
        vm_x = vx[i] + k * Ex[i]
        vm_y = vy[i] + k * Ey[i]
        vm_z = vz[i] + k * Ez[i]
        
        # v_prime
        vp_x = vm_x + (vm_y * tz - vm_z * ty)
        vp_y = vm_y + (vm_z * tx - vm_x * tz)
        vp_z = vm_z + (vm_x * ty - vm_y * tx)
        
        # v_plus
        v_new_x = vm_x + (vp_y * sz - vp_z * sy)
        v_new_y = vm_y + (vp_z * sx - vp_x * sz)
        v_new_z = vm_z + (vp_x * sy - vp_y * sx)
        
        # Final velocity update
        vx[i] = v_new_x + k * Ex[i]
        vy[i] = v_new_y + k * Ey[i]
        vz[i] = v_new_z + k * Ez[i]
        
        # Position update
        new_x = x[i] + vx[i] * dt
        new_y = y[i] + vy[i] * dt
        new_z = z[i] + vz[i] * dt
        
        # Boundary Conditions
        # x is Radial (0 to R=domain_height)
        # y is Axial (0 to L=domain_length)
        
        # Radial BCs (x)
        # Inner radius (x=0) -> Reflection (Axis of symmetry)
        if new_x < 0:
            new_x = -new_x
            vx[i] = -vx[i]
            
        # Outer radius (x=R) -> Wall
        if new_x >= domain_height: # domain_height is Radius
            if bc_type == 1: # Dirichlet (Absorption)
                active[i] = False
                continue
            elif bc_type == 2: # Periodic
                new_x -= domain_height
            elif bc_type == 3: # Reflect
                new_x = 2*domain_height - new_x
                vx[i] = -vx[i]
                
        # Axial BCs (y)
        # Inlet/Bottom (y=0)
        if new_y < 0:
            if bc_type == 1: # Dirichlet
                active[i] = False
                continue
            elif bc_type == 2: # Periodic
                new_y += domain_length
            elif bc_type == 3: # Reflect
                new_y = -new_y
                vy[i] = -vy[i]
        
        # Outlet/Top (y=L)
        if new_y >= domain_length: # domain_length is Length
            if bc_type == 1: # Dirichlet
                active[i] = False
                continue
            elif bc_type == 2: # Periodic
                new_y -= domain_length
            elif bc_type == 3: # Reflect
                new_y = 2*domain_length - new_y
                vy[i] = -vy[i]
                
        x[i] = new_x
        y[i] = new_y
        z[i] = new_z

@jit(nopython=True)
def projector_kernel(x, y, charge, weight, active, node_rho, h, num_cells_x, num_cells_y):
    n = len(x)
    num_nodes_x = num_cells_x + 1
    
    for i in range(n):
        if not active[i]:
            continue
            
        # Find cell index
        col = int(x[i] / h)
        row = int(y[i] / h)
        
        # Bounds check
        if col < 0: col = 0
        if col >= num_cells_x: col = num_cells_x - 1
        if row < 0: row = 0
        if row >= num_cells_y: row = num_cells_y - 1
        
        # Normalized coordinates in cell
        dx = (x[i] - col * h) / h
        dy = (y[i] - row * h) / h
        
        # Bilinear weighting
        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy
        
        q_eff = charge[i]
        
        # Node indices
        n00 = row * num_nodes_x + col
        n10 = row * num_nodes_x + (col + 1)
        n01 = (row + 1) * num_nodes_x + col
        n11 = (row + 1) * num_nodes_x + (col + 1)
        
        # Scatter charge
        node_rho[n00] += q_eff * w00
        node_rho[n10] += q_eff * w10
        node_rho[n01] += q_eff * w01
        node_rho[n11] += q_eff * w11

@jit(nopython=True)
def project_current_kernel(x, y, vx, vy, vz, charge, active, node_jx, node_jy, node_jz, h, num_cells_x, num_cells_y):
    n = len(x)
    num_nodes_x = num_cells_x + 1
    
    for i in range(n):
        if not active[i]:
            continue
            
        col = int(x[i] / h)
        row = int(y[i] / h)
        
        if col < 0: col = 0
        if col >= num_cells_x: col = num_cells_x - 1
        if row < 0: row = 0
        if row >= num_cells_y: row = num_cells_y - 1
        
        dx = (x[i] - col * h) / h
        dy = (y[i] - row * h) / h
        
        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy
        
        q_eff = charge[i]
        jx_val = q_eff * vx[i]
        jy_val = q_eff * vy[i]
        jz_val = q_eff * vz[i]
        
        n00 = row * num_nodes_x + col
        n10 = row * num_nodes_x + (col + 1)
        n01 = (row + 1) * num_nodes_x + col
        n11 = (row + 1) * num_nodes_x + (col + 1)
        
        node_jx[n00] += jx_val * w00
        node_jx[n10] += jx_val * w10
        node_jx[n01] += jx_val * w01
        node_jx[n11] += jx_val * w11
        
        node_jy[n00] += jy_val * w00
        node_jy[n10] += jy_val * w10
        node_jy[n01] += jy_val * w01
        node_jy[n11] += jy_val * w11
        
        node_jz[n00] += jz_val * w00
        node_jz[n10] += jz_val * w10
        node_jz[n01] += jz_val * w01
        node_jz[n11] += jz_val * w11

@jit(nopython=True)
def interpolator_kernel(x, y, Ex, Ey, Ez, Bx, By, Bz, active, 
                        node_Ex, node_Ey, node_Ez, node_Bx, node_By, node_Bz, 
                        h, num_cells_x, num_cells_y):
    n = len(x)
    num_nodes_x = num_cells_x + 1
    
    for i in range(n):
        if not active[i]:
            continue
            
        col = int(x[i] / h)
        row = int(y[i] / h)
        
        if col < 0: col = 0
        if col >= num_cells_x: col = num_cells_x - 1
        if row < 0: row = 0
        if row >= num_cells_y: row = num_cells_y - 1
        
        dx = (x[i] - col * h) / h
        dy = (y[i] - row * h) / h
        
        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy
        
        n00 = row * num_nodes_x + col
        n10 = row * num_nodes_x + (col + 1)
        n01 = (row + 1) * num_nodes_x + col
        n11 = (row + 1) * num_nodes_x + (col + 1)
        
        # Interpolate Ex
        Ex[i] = (node_Ex[n00] * w00 + node_Ex[n10] * w10 + 
                 node_Ex[n01] * w01 + node_Ex[n11] * w11)
        # Interpolate Ey
        Ey[i] = (node_Ey[n00] * w00 + node_Ey[n10] * w10 + 
                 node_Ey[n01] * w01 + node_Ey[n11] * w11)
        # Interpolate Ez
        Ez[i] = (node_Ez[n00] * w00 + node_Ez[n10] * w10 + 
                 node_Ez[n01] * w01 + node_Ez[n11] * w11)
                 
        # Interpolate Bx
        Bx[i] = (node_Bx[n00] * w00 + node_Bx[n10] * w10 + 
                 node_Bx[n01] * w01 + node_Bx[n11] * w11)
        # Interpolate By
        By[i] = (node_By[n00] * w00 + node_By[n10] * w10 + 
                 node_By[n01] * w01 + node_By[n11] * w11)
        # Interpolate Bz
        Bz[i] = (node_Bz[n00] * w00 + node_Bz[n10] * w10 + 
                 node_Bz[n01] * w01 + node_Bz[n11] * w11)

class PICSolver:
    def __init__(self, parameters):
        self.parameters = parameters
        self.bc_map = {
            "dirichlet": 1,
            "periodic": 2,
            "neumann": 3 # Using 3 for reflect/neumann for now
        }
        
    def solve_poisson(self, patch):
        mesh = patch.mesh
        h = self.parameters.pic_spacing
        epsilon = EPSILON_0
        
        # Simple Gauss-Seidel
        
        # Iterations
        for _ in range(self.parameters.max_solver_iterations):
            max_res = 0.0
            for node in mesh.nodes:
                # Get radial position (x = r in our coordinate system)
                r = node.x
                
                phi_old = node.phi
                rho = node.rho
                
                # Determine if we should update this node
                should_update = False
                
                if node.boundary_type == "internal":
                    should_update = True
                elif node.boundary_type == "L":
                    if self.parameters.left_bc_type != "dirichlet": should_update = True
                elif node.boundary_type == "R":
                    if self.parameters.right_bc_type != "dirichlet": should_update = True
                elif node.boundary_type == "T":
                    if self.parameters.top_bc_type != "dirichlet": should_update = True
                elif node.boundary_type == "B":
                    if self.parameters.bottom_bc_type != "dirichlet": should_update = True
                
                if not should_update:
                    continue

                # Neighbor indices (1-based in node object, convert to 0-based)
                n_left = node.left_node_id - 1
                n_right = node.right_node_id - 1
                n_top = node.top_node_id - 1
                n_bottom = node.bottom_node_id - 1
                
                phi_left = mesh.nodes[n_left].phi if n_left >= 0 else 0.0
                phi_right = mesh.nodes[n_right].phi if n_right >= 0 else 0.0
                phi_top = mesh.nodes[n_top].phi if n_top >= 0 else 0.0
                phi_bottom = mesh.nodes[n_bottom].phi if n_bottom >= 0 else 0.0
                
                # Cylindrical Laplacian discretization
                # ∇²φ = (1/r)(∂/∂r)(r ∂φ/∂r) + ∂²φ/∂z² ≈ -(ρ/ε₀)
                
                if r < h * 0.5:  # Near axis (r ≈ 0) - Applies to 'internal' or 'L' nodes at axis
                    # L'Hôpital's rule: lim(r→0) (1/r)(∂/∂r)(r ∂φ/∂r) = 2(∂²φ/∂r²)
                    # Stencil: φ_new = (1/6)[4*φ_right + φ_top + φ_bottom + (ρ/ε₀)h²]
                    # Note: For 'L' boundary at axis, 'left' neighbor doesn't exist, which is fine as formula uses 'right'
                    phi_new = (1.0/6.0) * (4.0*phi_right + phi_top + phi_bottom + (rho / epsilon) * h * h)
                else:
                    # General radial point
                    # Coefficients for (1/r)(∂/∂r)(r ∂φ/∂r) discretization
                    a_left = 1.0 - h / (2.0 * r)   # Coefficient for φ(r-Δr)
                    a_right = 1.0 + h / (2.0 * r)  # Coefficient for φ(r+Δr)
                    # Axial coefficients remain 1.0
                    a_top = 1.0
                    a_bottom = 1.0
                    
                    # Handle Neumann/Symmetry BCs by mirroring neighbors if needed
                    # If we are at a boundary and updating, it means it's Neumann/Symmetry
                    if node.boundary_type == "L": # Left boundary (not axis, or axis handled above?)
                         # If r > 0 and we are at L, it's a wall. Neumann means phi_left = phi_right?
                         # Or Forward difference = 0 => phi_node = phi_right?
                         # Standard Neumann stencil: modify coefficients.
                         # Simple approach: Mirror neighbor
                         phi_left = phi_right # Mirror
                    elif node.boundary_type == "R":
                         phi_right = phi_left # Mirror
                    elif node.boundary_type == "T":
                         phi_top = phi_bottom # Mirror
                    elif node.boundary_type == "B":
                         phi_bottom = phi_top # Mirror

                    # Normalization factor
                    norm = 1.0 / (a_left + a_right + a_top + a_bottom)
                    
                    # Update formula
                    phi_new = norm * (a_left*phi_left + a_right*phi_right + 
                                      a_top*phi_top + a_bottom*phi_bottom + 
                                      (rho / epsilon) * h * h)
                
                # SOR (Successive Over-Relaxation)
                phi_new = (1.0 - self.parameters.sor_parameter) * phi_old + self.parameters.sor_parameter * phi_new
                
                node.phi = phi_new
                res = abs(phi_new - phi_old)
                if res > max_res:
                    max_res = res
                    
            if max_res < self.parameters.residual_tolerance:
                break
        
        # Diagnostics: Track max phi, E, rho
        max_phi = max(abs(node.phi) for node in mesh.nodes)
        max_rho = max(abs(node.rho) for node in mesh.nodes)
        print(f"[POISSON] max|phi|={max_phi:.2e} V, max|rho|={max_rho:.2e} C/m^3")
                
        # Calculate E-field from Phi for ALL nodes (internal AND boundary)
        # C++ calculates E for all nodes using one-sided differences at boundaries
        for node in mesh.nodes:
            n_left = node.left_node_id - 1
            n_right = node.right_node_id - 1
            n_top = node.top_node_id - 1
            n_bottom = node.bottom_node_id - 1
            
            phi_left = mesh.nodes[n_left].phi
            phi_right = mesh.nodes[n_right].phi
            phi_top = mesh.nodes[n_top].phi
            phi_bottom = mesh.nodes[n_bottom].phi
            phi_node = node.phi
            
            # Calculate E-field based on boundary type
            if node.boundary_type == "internal":
                # Central difference (2nd order accurate)
                Ex = -(phi_right - phi_left) / (2 * h)
                Ey = -(phi_top - phi_bottom) / (2 * h)
            
            elif node.boundary_type == "L":  # Left boundary (inner radial wall, r=0)
                # One-sided difference in radial (x) direction
                Ex = -(phi_right - phi_node) / h  # Forward difference
                Ey = -(phi_top - phi_bottom) / (2 * h)  # Central
            
            elif node.boundary_type == "R":  # Right boundary (outer radial wall, r=R)
                # One-sided difference in radial (x) direction
                Ex = -(phi_node - phi_left) / h  # Backward difference
                Ey = -(phi_top - phi_bottom) / (2 * h)  # Central
            
            elif node.boundary_type == "T":  # Top boundary (axial outlet, z=L)
                Ex = -(phi_right - phi_left) / (2 * h)  # Central
                Ey = -(phi_node - phi_bottom) / h  # Backward difference
            
            elif node.boundary_type == "B":  # Bottom boundary (axis, z=0)
                Ex = -(phi_right - phi_left) / (2 * h)  # Central
                Ey = -(phi_top - phi_node) / h  # Forward difference
            
            elif node.boundary_type == "TL":  # Top-left corner
                Ex = -(phi_right - phi_node) / h  # Forward
                Ey = -(phi_node - phi_bottom) / h  # Backward
            
            elif node.boundary_type == "TR":  # Top-right corner
                Ex = -(phi_node - phi_left) / h  # Backward
                Ey = -(phi_node - phi_bottom) / h  # Backward
            
            elif node.boundary_type == "BL":  # Bottom-left corner
                Ex = -(phi_right - phi_node) / h  # Forward
                Ey = -(phi_top - phi_node) / h  # Forward
            
            elif node.boundary_type == "BR":  # Bottom-right corner
                Ex = -(phi_node - phi_left) / h  # Backward
                Ey = -(phi_top - phi_node) / h  # Forward
            
            else:
                # Unknown boundary type - use zero E-field
                Ex = 0.0
                Ey = 0.0
            
            # Set E-field (was already cleared to zero before solve)
            node.em_field[0] = Ex
            node.em_field[1] = Ey
        
        # Diagnostics: Track max E-field
        max_E = max(math.sqrt(node.em_field[0]**2 + node.em_field[1]**2) for node in mesh.nodes)
        print(f"[POISSON] max|E|={max_E:.2e} V/m\n")

    def pusher(self, patch):
        pl = patch.particle_list
        # bc_type = self.bc_map.get(self.parameters.left_bc_type, 1) # Simplified: assume same BC for all or pass specific
        # We pass integer BC code.
        bc_code = 1 # Default Dirichlet
        if self.parameters.left_bc_type == "periodic": bc_code = 2
        elif self.parameters.left_bc_type == "neumann": bc_code = 3
        
        pusher_kernel(
            pl.x, pl.y, pl.z, pl.vx, pl.vy, pl.vz, 
            pl.Ex, pl.Ey, pl.Ez, pl.Bx, pl.By, pl.Bz, 
            pl.mass, pl.charge, pl.active, 
            self.parameters.time_step, 
            self.parameters.domain_length, self.parameters.domain_height,
            bc_code, 0.0
        )

    def projector(self, patch):
        pl = patch.particle_list
        mesh = patch.mesh
        
        # Reset node rho
        num_nodes = len(mesh.nodes)
        node_rho = np.zeros(num_nodes)
        
        projector_kernel(
            pl.x, pl.y, pl.charge, pl.weight, pl.active,
            node_rho, self.parameters.pic_spacing,
            mesh.num_cells_x, mesh.num_cells_y
        )
        
        # Copy back to nodes and normalize by volume
        for i, rho in enumerate(node_rho):
            mesh.nodes[i].rho = rho
            vol = self.parameters.pic_spacing ** 2
            if self.parameters.axisymmetric:
                 # In cylindrical: x = r (radial), y = z (axial)
                 r = mesh.nodes[i].x  # FIX: Use x (radial), not y (axial)!
                 if r < 1e-10: r = self.parameters.pic_spacing / 2.0 # Avoid zero volume
                 vol = 2.0 * math.pi * r * self.parameters.pic_spacing**2
            
            mesh.nodes[i].rho /= vol

    def project_current(self, patch):
        pl = patch.particle_list
        mesh = patch.mesh
        
        num_nodes = len(mesh.nodes)
        node_jx = np.zeros(num_nodes)
        node_jy = np.zeros(num_nodes)
        node_jz = np.zeros(num_nodes)
        
        project_current_kernel(
            pl.x, pl.y, pl.vx, pl.vy, pl.vz, pl.charge, pl.active,
            node_jx, node_jy, node_jz, self.parameters.pic_spacing,
            mesh.num_cells_x, mesh.num_cells_y
        )
        
        for i in range(num_nodes):
            vol = self.parameters.pic_spacing ** 2
            if self.parameters.axisymmetric:
                 r = mesh.nodes[i].x # FIX: Use x (radial) here too!
                 if r < 1e-10: r = self.parameters.pic_spacing / 2.0
                 vol = 2.0 * math.pi * r * self.parameters.pic_spacing**2
            
            mesh.nodes[i].current[0] = node_jx[i] / vol
            mesh.nodes[i].current[1] = node_jy[i] / vol
            mesh.nodes[i].current[2] = node_jz[i] / vol

    def interpolator(self, patch):
        pl = patch.particle_list
        mesh = patch.mesh
        
        # Extract fields from nodes to arrays
        num_nodes = len(mesh.nodes)
        node_Ex = np.zeros(num_nodes)
        node_Ey = np.zeros(num_nodes)
        node_Ez = np.zeros(num_nodes)
        node_Bx = np.zeros(num_nodes)
        node_By = np.zeros(num_nodes)
        node_Bz = np.zeros(num_nodes)
        
        time = patch.time
        
        # Pre-calculate RF fields if enabled
        # Optimization: Calculate once per step for all nodes
        rf_Ex = np.zeros(num_nodes)
        rf_Ey = np.zeros(num_nodes)
        rf_Ez = np.zeros(num_nodes)
        rf_Bx = np.zeros(num_nodes)
        rf_By = np.zeros(num_nodes)
        rf_Bz = np.zeros(num_nodes)
        
        if patch.rf_coil:
             # Calculate current state
             I = patch.rf_coil.get_current(time)
             dIdt = patch.rf_coil.get_dI_dt(time)
             
             for i, node in enumerate(mesh.nodes):
                 E_rf = patch.rf_coil.get_E_field_at_position(node.x, node.y, time)
                 # E_rf is vector [Ex, Ey, Ez]
                 rf_Ex[i] = E_rf[0]
                 rf_Ey[i] = E_rf[1]
                 rf_Ez[i] = E_rf[2]
                 
                 B_rf = patch.rf_coil.get_field_at_position(node.x, node.y, time)
                 rf_Bx[i] = B_rf[0]
                 rf_By[i] = B_rf[1]
                 rf_Bz[i] = B_rf[2]
        
        for i, node in enumerate(mesh.nodes):
            # FDTD/Poisson fields + RF fields
            node_Ex[i] = node.em_field[0] + rf_Ex[i]
            node_Ey[i] = node.em_field[1] + rf_Ey[i]
            node_Ez[i] = node.em_field[2] + rf_Ez[i]
            node_Bx[i] = node.em_field[3] + rf_Bx[i]
            node_By[i] = node.em_field[4] + rf_By[i]
            node_Bz[i] = node.em_field[5] + rf_Bz[i]

        interpolator_kernel(
            pl.x, pl.y, pl.Ex, pl.Ey, pl.Ez, pl.Bx, pl.By, pl.Bz, pl.active,
            node_Ex, node_Ey, node_Ez, node_Bx, node_By, node_Bz,
            self.parameters.pic_spacing, mesh.num_cells_x, mesh.num_cells_y
        )
