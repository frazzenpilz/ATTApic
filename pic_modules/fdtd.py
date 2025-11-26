"""
FDTD Electromagnetic Solver Module (Optimized with Numba/JIT)

Solves Maxwell's equations on a finer mesh using Finite-Difference Time-Domain method.
Uses Yee grid (staggered E and B fields) for numerical stability.
Implements Axisymmetric TE Mode (Cylindrical Coordinates).

Maxwell's equations (Cylindrical TE Mode):
  ∂Bz/∂t = -(1/r) ∂(r E_theta)/∂r
  ∂Br/∂t = ∂E_theta/∂z
  ∂E_theta/∂t = c^2 (∂Br/∂z - ∂Bz/∂r) - J_theta/eps0

Mapping to Code Variables (Cartesian -> Cylindrical):
  x -> z (Axial)
  y -> r (Radial)
  z -> theta (Azimuthal)
  
  Ex -> Ez_phys (Axial E - not used in TE mode)
  Ey -> Er_phys (Radial E - not used in TE mode)
  Ez -> E_theta (Azimuthal E)
  
  Bx -> Bz_phys (Axial B)
  By -> Br_phys (Radial B)
  Bz -> B_theta (Azimuthal B - not used in TE mode)
  
  Jz -> J_theta (Azimuthal Current)
"""

import numpy as np
import math
from numba import jit, prange
from .constants import EPSILON_0, MU_0
from .mesh import Mesh, Node
from .utils import log_messages, log_brief
import sys

# --- JIT Kernels ---

@jit(nopython=True, parallel=True)
def interpolate_pic_to_fdtd_kernel(fdtd_x, fdtd_y, pic_em_field, pic_current, 
                                   pic_nx, pic_ny, pic_h, domain_height,
                                   fdtd_em_field, fdtd_current):
    """
    Interpolate E, B, and J from PIC mesh to FDTD mesh.
    """
    num_fdtd_nodes = fdtd_x.shape[0]
    pic_nx_nodes = pic_nx + 1
    
    for k in prange(num_fdtd_nodes):
        x = fdtd_x[k]
        y = fdtd_y[k]
        
        # Find containing PIC cell
        i = int(x / pic_h)
        j = int((domain_height - y) / pic_h)
        
        # Bounds check
        if i >= pic_nx: i = pic_nx - 1
        if j >= pic_ny: j = pic_ny - 1
        if i < 0: i = 0
        if j < 0: j = 0
        
        # Normalized position
        dx = (x - i * pic_h) / pic_h
        dy = ((domain_height - y) - j * pic_h) / pic_h
        
        # Clamp
        if dx < 0.0: dx = 0.0
        if dx > 1.0: dx = 1.0
        if dy < 0.0: dy = 0.0
        if dy > 1.0: dy = 1.0
        
        # Weights
        w_tl = (1.0 - dx) * (1.0 - dy)
        w_tr = dx * (1.0 - dy)
        w_bl = (1.0 - dx) * dy
        w_br = dx * dy
        
        # PIC node indices
        idx_tl = j * pic_nx_nodes + i
        idx_tr = j * pic_nx_nodes + min(i + 1, pic_nx)
        idx_bl = min(j + 1, pic_ny) * pic_nx_nodes + i
        idx_br = min(j + 1, pic_ny) * pic_nx_nodes + min(i + 1, pic_nx)
        
        # Interpolate E-field (Ex, Ey, Ez)
        for c in range(3):
            fdtd_em_field[k, c] = (
                w_tl * pic_em_field[idx_tl, c] +
                w_tr * pic_em_field[idx_tr, c] +
                w_bl * pic_em_field[idx_bl, c] +
                w_br * pic_em_field[idx_br, c]
            )
            
        # Interpolate Current (Jx, Jy, Jz)
        for c in range(3):
            fdtd_current[k, c] = (
                w_tl * pic_current[idx_tl, c] +
                w_tr * pic_current[idx_tr, c] +
                w_bl * pic_current[idx_bl, c] +
                w_br * pic_current[idx_br, c]
            )

@jit(nopython=True, parallel=True)
def interpolate_fdtd_to_pic_kernel(pic_x, pic_y, fdtd_em_field, 
                                   fdtd_nx, fdtd_ny, fdtd_h, domain_height,
                                   pic_em_field):
    """
    Interpolate E and B from FDTD mesh back to PIC mesh.
    """
    num_pic_nodes = pic_x.shape[0]
    fdtd_nx_nodes = fdtd_nx + 1
    
    for k in prange(num_pic_nodes):
        x = pic_x[k]
        y = pic_y[k]
        
        # Find containing FDTD cell
        i = int(x / fdtd_h)
        j = int((domain_height - y) / fdtd_h)
        
        # Bounds check
        if i >= fdtd_nx: i = fdtd_nx - 1
        if j >= fdtd_ny: j = fdtd_ny - 1
        if i < 0: i = 0
        if j < 0: j = 0
        
        # Normalized position
        dx = (x - i * fdtd_h) / fdtd_h
        dy = ((domain_height - y) - j * fdtd_h) / fdtd_h
        
        if dx < 0.0: dx = 0.0
        if dx > 1.0: dx = 1.0
        if dy < 0.0: dy = 0.0
        if dy > 1.0: dy = 1.0
        
        w_tl = (1.0 - dx) * (1.0 - dy)
        w_tr = dx * (1.0 - dy)
        w_bl = (1.0 - dx) * dy
        w_br = dx * dy
        
        idx_tl = j * fdtd_nx_nodes + i
        idx_tr = j * fdtd_nx_nodes + min(i + 1, fdtd_nx)
        idx_bl = min(j + 1, fdtd_ny) * fdtd_nx_nodes + i
        idx_br = min(j + 1, fdtd_ny) * fdtd_nx_nodes + min(i + 1, fdtd_nx)
        
        # Interpolate E and B (6 components)
        for c in range(6):
            pic_em_field[k, c] = (
                w_tl * fdtd_em_field[idx_tl, c] +
                w_tr * fdtd_em_field[idx_tr, c] +
                w_bl * fdtd_em_field[idx_bl, c] +
                w_br * fdtd_em_field[idx_br, c]
            )

@jit(nopython=True, parallel=True)
def update_b_field_kernel(em_field, nx, ny, h, dt, boundary_type, node_y):
    """
    Update B-field using Faraday's law (Cylindrical TE Mode).
    
    ∂Bx/∂t = -(1/y) ∂(y Ez)/∂y  (Mapped from ∂Bz/∂t = -(1/r) ∂(r E_theta)/∂r)
    ∂By/∂t = ∂Ez/∂x             (Mapped from ∂Br/∂t = ∂E_theta/∂z)
    ∂Bz/∂t = ...                (Not used in TE mode)
    """
    time_step_ratio = dt / (2.0 * h)
    nx_nodes = nx + 1
    
    # Iterate over all nodes
    # We can parallelize over linear index
    num_nodes = (nx + 1) * (ny + 1)
    
    for k in prange(num_nodes):
        # Decode i, j
        i = k // (ny + 1)
        j = k % (ny + 1)
        
        # Only update internal nodes (and check even/odd for Yee grid)
        # B-fields are at even nodes (k % 2 == 0) in the original code logic?
        # The original code iterated enumerate(nodes) and checked i % 2 == 0.
        # Wait, original code: `if i % 2 == 0` refers to the linear index in the list.
        # The list was constructed: for i in range(nx+1): for j in range(ny+1).
        # So linear index k corresponds to the loop order.
        
        if k % 2 == 0:
            # Check boundary (internal only)
            if i > 0 and i < nx and j > 0 and j < ny:
                # Neighbor indices
                # Top (j-1), Bottom (j+1), Left (i-1), Right (i+1)
                # Index = i * (ny+1) + j
                idx_top = i * (ny + 1) + (j - 1)
                idx_bottom = i * (ny + 1) + (j + 1)
                idx_left = (i - 1) * (ny + 1) + j
                idx_right = (i + 1) * (ny + 1) + j
                
                # (1) ∂Bx/∂t = -(1/y) ∂(y Ez)/∂y
                # Bx is index 3, Ez is index 2
                
                y = node_y[k]
                if abs(y) < 1e-10:
                    inv_y = 0.0
                else:
                    inv_y = 1.0 / y
                
                y_top = node_y[idx_top]
                y_bottom = node_y[idx_bottom]
                
                Ez_top = em_field[idx_top, 2]
                Ez_bottom = em_field[idx_bottom, 2]
                
                em_field[k, 3] -= time_step_ratio * inv_y * (
                    y_top * Ez_top - y_bottom * Ez_bottom
                )
                
                # (2) ∂By/∂t = ∂Ez/∂x
                # By is index 4
                Ez_right = em_field[idx_right, 2]
                Ez_left = em_field[idx_left, 2]
                
                em_field[k, 4] += time_step_ratio * (
                    Ez_right - Ez_left
                )
                
                # (3) ∂Bz/∂t (Not strictly needed for TE mode but kept for completeness/stability)
                # Bz is index 5, Ex(0), Ey(1)
                Ex_top = em_field[idx_top, 0]
                Ex_bottom = em_field[idx_bottom, 0]
                Ey_right = em_field[idx_right, 1]
                Ey_left = em_field[idx_left, 1]
                
                em_field[k, 5] += time_step_ratio * (
                    Ex_top - Ex_bottom - Ey_right + Ey_left
                )

@jit(nopython=True, parallel=True)
def update_e_field_kernel(em_field, current, nx, ny, h, dt, c_squared, node_y):
    """
    Update E-field using Ampère-Maxwell law.
    """
    time_step_ratio = dt / (2.0 * h)
    num_nodes = (nx + 1) * (ny + 1)
    
    for k in prange(num_nodes):
        i = k // (ny + 1)
        j = k % (ny + 1)
        
        if k % 2 == 1: # Odd nodes
             if i > 0 and i < nx and j > 0 and j < ny:
                idx_top = i * (ny + 1) + (j - 1)
                idx_bottom = i * (ny + 1) + (j + 1)
                idx_left = (i - 1) * (ny + 1) + j
                idx_right = (i + 1) * (ny + 1) + j
                
                # (4) ε₀∂Ex/∂t = ∂Bz/∂y - Jx
                Bz_top = em_field[idx_top, 5]
                Bz_bottom = em_field[idx_bottom, 5]
                
                em_field[k, 0] += time_step_ratio * c_squared * (
                    Bz_top - Bz_bottom
                ) - dt * current[k, 0] / EPSILON_0
                
                # (5) ε₀∂Ey/∂t = -∂Bz/∂x - Jy
                Bz_right = em_field[idx_right, 5]
                Bz_left = em_field[idx_left, 5]
                
                em_field[k, 1] -= time_step_ratio * c_squared * (
                    Bz_right - Bz_left
                ) + dt * current[k, 1] / EPSILON_0
                
                # (6) ε₀∂Ez/∂t = ∂By/∂x - ∂Bx/∂y - Jz
                # Ez is index 2. Bx(3), By(4). Jz(2).
                # Note: This equation is formally correct for E_theta in cylindrical too
                # if we interpret Bx, By as Bz, Br.
                # 1/c^2 dE_theta/dt = dBr/dz - dBz/dr - mu0 J_theta
                # dE_theta/dt = c^2 (dBr/dz - dBz/dr) - J_theta/eps0
                # Code: dEz/dt = c^2 (dBy/dx - dBx/dy) - Jz/eps0
                # Mapping: By->Br, x->z => dBy/dx -> dBr/dz. Correct.
                # Mapping: Bx->Bz, y->r => dBx/dy -> dBz/dr. Correct.
                
                By_right = em_field[idx_right, 4]
                By_left = em_field[idx_left, 4]
                Bx_top = em_field[idx_top, 3]
                Bx_bottom = em_field[idx_bottom, 3]
                
                em_field[k, 2] += time_step_ratio * c_squared * (
                    By_right - By_left - Bx_top + Bx_bottom
                ) - dt * current[k, 2] / EPSILON_0

@jit(nopython=True)
def apply_boundary_conditions_kernel(em_field, nx, ny, left_type, right_type, top_type, bottom_type):
    """
    Apply boundary conditions to E-fields.
    Simple implementation for Dirichlet (PEC) and Periodic.
    """
    # Left/Right (i=0, i=nx)
    # Indices: i * (ny+1) + j
    
    # Left
    if left_type == 1: # Dirichlet
        for j in range(ny + 1):
            idx = j
            em_field[idx, 1] = 0.0 # Ey
            em_field[idx, 2] = 0.0 # Ez
            
    # Right
    if right_type == 1: # Dirichlet
        base = nx * (ny + 1)
        for j in range(ny + 1):
            idx = base + j
            em_field[idx, 1] = 0.0
            em_field[idx, 2] = 0.0
            
    # Top (j=0)
    if top_type == 1: # Dirichlet
        for i in range(nx + 1):
            idx = i * (ny + 1)
            em_field[idx, 0] = 0.0 # Ex
            em_field[idx, 2] = 0.0 # Ez
            
    # Bottom (j=ny)
    if bottom_type == 1: # Dirichlet
        for i in range(nx + 1):
            idx = i * (ny + 1) + ny
            em_field[idx, 0] = 0.0
            em_field[idx, 2] = 0.0
            
    # Periodic X
    if left_type == 2: # Periodic
        for j in range(ny + 1):
            idx_left = j
            idx_right = nx * (ny + 1) + j
            
            for c in range(3): # E
                avg = 0.5 * (em_field[idx_left, c] + em_field[idx_right, c])
                em_field[idx_left, c] = avg
                em_field[idx_right, c] = avg
            for c in range(3, 6): # B
                avg = 0.5 * (em_field[idx_left, c] + em_field[idx_right, c])
                em_field[idx_left, c] = avg
                em_field[idx_right, c] = avg

@jit(nopython=True)
def apply_rf_coil_kernel(current, time, I0, f, h, nx, ny):
    """
    Apply RF coil source current to the mesh (Jz).
    """
    omega = 2.0 * math.pi * f
    current_val = I0 * math.sin(omega * time)
    area = h * h
    J_coil = current_val / area
    
    # Apply to top boundary nodes (j=0)
    # Index = i * (ny + 1)
    for i in range(nx + 1):
        idx = i * (ny + 1)
        current[idx, 2] += J_coil

# --- Main Class ---

class FDTDSolver:
    def __init__(self, parameters, pic_mesh):
        self.parameters = parameters
        self.pic_mesh = pic_mesh
        
        self.c_squared = 1.0 / (EPSILON_0 * MU_0)
        self.c = math.sqrt(self.c_squared)
        
        self.fdtd_dt = parameters.time_step / parameters.fdtd_iterations
        self.h = parameters.fdtd_spacing
        
        # Mesh dimensions
        self.nx = int(round(self.parameters.domain_length / self.h))
        self.ny = int(round(self.parameters.domain_height / self.h))
        self.num_nodes = (self.nx + 1) * (self.ny + 1)
        
        # Allocate Arrays (SoA)
        # em_field: [Ex, Ey, Ez, Bx, By, Bz]
        self.em_field = np.zeros((self.num_nodes, 6), dtype=np.float64)
        # current: [Jx, Jy, Jz]
        self.current = np.zeros((self.num_nodes, 3), dtype=np.float64)
        
        # Node coordinates cache
        self.node_x = np.zeros(self.num_nodes, dtype=np.float64)
        self.node_y = np.zeros(self.num_nodes, dtype=np.float64)
        
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                idx = i * (self.ny + 1) + j
                self.node_x[idx] = i * self.h
                self.node_y[idx] = self.parameters.domain_height - j * self.h
        
        # Boundary types map (1: Dirichlet, 2: Periodic, 0: None)
        self.bc_map = {
            "dirichlet": 1,
            "periodic": 2,
            "neumann": 0 # Not implemented
        }
        
        log_messages(f"FDTD Solver (JIT) initialized: {self.num_nodes} nodes", 
                    __file__, sys._getframe().f_lineno, 1)

    def interpolate_pic_to_fdtd(self):
        # Extract PIC data to arrays for JIT
        # This is the overhead, but it's once per particle step
        pic_nodes = self.pic_mesh.nodes
        num_pic = len(pic_nodes)
        
        # We need to ensure PIC mesh data is in arrays. 
        # Ideally PIC mesh should also be SoA, but for now we extract.
        # Optimization: Cache these arrays if PIC mesh doesn't change structure
        pic_em = np.zeros((num_pic, 3), dtype=np.float64)
        pic_J = np.zeros((num_pic, 3), dtype=np.float64)
        
        for k, node in enumerate(pic_nodes):
            pic_em[k, 0] = node.em_field[0]
            pic_em[k, 1] = node.em_field[1]
            pic_em[k, 2] = node.em_field[2]
            
            if hasattr(node, 'current'):
                pic_J[k, 0] = node.current[0]
                pic_J[k, 1] = node.current[1]
                if len(node.current) > 2:
                    pic_J[k, 2] = node.current[2]
        
        interpolate_pic_to_fdtd_kernel(
            self.node_x, self.node_y, 
            pic_em, pic_J,
            self.pic_mesh.num_cells_x, self.pic_mesh.num_cells_y, self.parameters.pic_spacing,
            self.parameters.domain_height,
            self.em_field, self.current
        )

    def interpolate_fdtd_to_pic(self):
        # Interpolate back
        pic_nodes = self.pic_mesh.nodes
        num_pic = len(pic_nodes)
        pic_em = np.zeros((num_pic, 6), dtype=np.float64)
        
        # Create pic_x, pic_y arrays
        pic_x = np.array([n.x for n in pic_nodes])
        pic_y = np.array([n.y for n in pic_nodes])
        
        interpolate_fdtd_to_pic_kernel(
            pic_x, pic_y, self.em_field,
            self.nx, self.ny, self.h, self.parameters.domain_height,
            pic_em
        )
        
        # Copy back to objects (Bottleneck, but unavoidable without full SoA refactor)
        for k, node in enumerate(pic_nodes):
            for c in range(6):
                node.em_field[c] = pic_em[k, c]

    def solve(self, time):
        # 1. Interpolate PIC -> FDTD
        self.interpolate_pic_to_fdtd()
        
        current_time = time
        dt = self.fdtd_dt
        
        # BC types
        l_bc = self.bc_map.get(self.parameters.left_bc_type, 0)
        r_bc = self.bc_map.get(self.parameters.right_bc_type, 0)
        t_bc = self.bc_map.get(self.parameters.top_bc_type, 0)
        b_bc = self.bc_map.get(self.parameters.bottom_bc_type, 0)
        
        # 2. Sub-cycle
        for _ in range(self.parameters.fdtd_iterations):
            current_time += dt
            
            # Apply RF Coil
            if self.parameters.use_rf_coil:
                apply_rf_coil_kernel(
                    self.current, current_time, 
                    self.parameters.coil_current, self.parameters.rf_frequency,
                    self.h, self.nx, self.ny
                )
            
            # Update B
            update_b_field_kernel(
                self.em_field, self.nx, self.ny, self.h, dt, 
                0, self.node_y # boundary_type logic moved inside kernel or simplified
            )
            
            # Update E
            update_e_field_kernel(
                self.em_field, self.current, self.nx, self.ny, self.h, dt, 
                self.c_squared, self.node_y
            )
            
            # BCs
            apply_boundary_conditions_kernel(
                self.em_field, self.nx, self.ny, l_bc, r_bc, t_bc, b_bc
            )
            
        # 3. Interpolate FDTD -> PIC
        self.interpolate_fdtd_to_pic()
