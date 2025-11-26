from .mesh import Mesh
from .particle import ParticleList
from .solver import PICSolver
from .mcc import MCC
from .rf_coil import RFCoil
from .fdtd import FDTDSolver
from .diagnostics import Diagnostics
from .utils import log_messages, log_brief
import sys
import os
import numpy as np
import h5py

class Patch:
    def __init__(self, parameters, patch_id):
        self.parameters = parameters
        self.patch_id = patch_id
        self.time = 0.0
        self.mesh = Mesh(parameters, "PIC")
        self.particle_list = ParticleList(parameters, self.mesh)
        self.solver = PICSolver(parameters)
        self.mcc = MCC(parameters)
        
        # Initialize RF coil if enabled
        self.rf_coil = RFCoil(parameters) if parameters.use_rf_coil else None
        
        # Initialize FDTD solver
        self.fdtd = FDTDSolver(parameters, self.mesh)
        
        # Initialize Diagnostics
        self.diagnostics = Diagnostics(parameters, self.mesh)
        
        self.num_errors = 0

    def start_pic(self):
        log_messages(f"Starting PIC loop for Patch {self.patch_id}", __file__, sys._getframe().f_lineno, 1)
        
        for step in range(1, self.parameters.maximum_number_of_iterations + 1):
            self.time += self.parameters.time_step
            
            # PIC Cycle
            
            # 1. Injection (Inlet)
            if self.parameters.inlet_source:
                self.particle_list.add_particles_to_sim(step)
            
            # 2. Projector (Particles -> Mesh: charge density)
            self.solver.projector(self)
            
            # 3. Current Projection (Particles -> Mesh: current density J)
            self.solver.project_current(self)
            
            # 4. FDTD Electromagnetic Solver (J -> E, B fields)
            if step % self.parameters.fdtd_frequency == 0:
                self.fdtd.solve(self.time)
            
            # 5. Poisson Solver (electrostatic φ for confinement)
            if self.parameters.use_poisson:
                # CRITICAL: Clear electrostatic E-field before solving
                # (C++ code does this via clearFields() - see Solver.cpp line 15)
                # Without this, E-field accumulates across timesteps → runaway instability!
                for node in self.mesh.nodes:
                    node.em_field[0] = 0.0  # Clear Ex  (radial)
                    node.em_field[1] = 0.0  # Clear Ey (axial)
                    # Do NOT clear em_field[2:6] - those are FDTD fields (Ez, Bx, By, Bz)
                
                self.solver.solve_poisson(self)
            
            # 6. Interpolator (Mesh -> Particles: E and B fields)
            self.solver.interpolator(self)
            
            # 7. Pusher (Move particles with Boris algorithm)
            self.solver.pusher(self)
            
            # 8. MCC (Collisions)
            if step % self.parameters.mcc_frequency == 0:
                self.mcc.collide(self.particle_list, self.parameters.time_step * self.parameters.mcc_frequency)
            
            # 9. Diagnostics Accumulation
            self.diagnostics.accumulate(self.particle_list, self.mesh.nodes, self.time)
            
            # 10. Output
            if step % self.parameters.plot_frequency == 0:
                log_brief(f"Time step {step}, Time: {self.time}", 1)
                self.generate_output(step)
                self.diagnostics.save_diagnostics()
        
        # End of loop - Save Diagnostics
        self.diagnostics.save_diagnostics()

    def generate_output(self, step):
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = f"{output_dir}/results_{step:05d}.h5"
        
        with h5py.File(filename, 'w') as f:
            # Global attributes
            f.attrs['time'] = self.time
            f.attrs['step'] = step
            f.attrs['patch_id'] = self.patch_id
            
            # Node Data
            num_nodes = len(self.mesh.nodes)
            node_ids = np.zeros(num_nodes, dtype=np.int32)
            node_x = np.zeros(num_nodes)
            node_y = np.zeros(num_nodes)
            node_phi = np.zeros(num_nodes)
            node_rho = np.zeros(num_nodes)
            node_ex = np.zeros(num_nodes)
            node_ey = np.zeros(num_nodes)
            node_ez = np.zeros(num_nodes)
            node_bx = np.zeros(num_nodes)
            node_by = np.zeros(num_nodes)
            node_bz = np.zeros(num_nodes)
            node_jx = np.zeros(num_nodes)
            node_jy = np.zeros(num_nodes)
            node_jz = np.zeros(num_nodes)
            
            for i, node in enumerate(self.mesh.nodes):
                node_ids[i] = node.id
                node_x[i] = node.x
                node_y[i] = node.y
                node_phi[i] = node.phi
                node_rho[i] = node.rho
                node_ex[i] = node.em_field[0]
                node_ey[i] = node.em_field[1]
                node_ez[i] = node.em_field[2]
                node_bx[i] = node.em_field[3] if len(node.em_field) > 3 else 0.0
                node_by[i] = node.em_field[4] if len(node.em_field) > 4 else 0.0
                node_bz[i] = node.em_field[5] if len(node.em_field) > 5 else 0.0
                node_jx[i] = node.current[0] if hasattr(node, 'current') else 0.0
                node_jy[i] = node.current[1] if hasattr(node, 'current') else 0.0
                node_jz[i] = node.current[2] if hasattr(node, 'current') and len(node.current) > 2 else 0.0
            
            g_nodes = f.create_group("nodes")
            g_nodes.create_dataset("id", data=node_ids)
            g_nodes.create_dataset("x", data=node_x)
            g_nodes.create_dataset("y", data=node_y)
            g_nodes.create_dataset("phi", data=node_phi)
            g_nodes.create_dataset("rho", data=node_rho)
            g_nodes.create_dataset("ex", data=node_ex)
            g_nodes.create_dataset("ey", data=node_ey)
            g_nodes.create_dataset("ez", data=node_ez)
            g_nodes.create_dataset("bx", data=node_bx)
            g_nodes.create_dataset("by", data=node_by)
            g_nodes.create_dataset("bz", data=node_bz)
            g_nodes.create_dataset("jx", data=node_jx)
            g_nodes.create_dataset("jy", data=node_jy)
            g_nodes.create_dataset("jz", data=node_jz)
            
            # Particle Data
            pl = self.particle_list
            n = pl.num_particles
            
            # Get active indices
            active_indices = np.where(pl.active[:n])[0]
            
            g_particles = f.create_group("particles")
            
            # Save only active particles
            g_particles.create_dataset("id", data=pl.particle_id[active_indices])
            g_particles.create_dataset("x", data=pl.x[active_indices])
            g_particles.create_dataset("y", data=pl.y[active_indices])
            g_particles.create_dataset("z", data=pl.z[active_indices])
            g_particles.create_dataset("vx", data=pl.vx[active_indices])
            g_particles.create_dataset("vy", data=pl.vy[active_indices])
            g_particles.create_dataset("vz", data=pl.vz[active_indices])
            g_particles.create_dataset("mass", data=pl.mass[active_indices])
            g_particles.create_dataset("charge", data=pl.charge[active_indices])
            g_particles.create_dataset("weight", data=pl.weight[active_indices])
            g_particles.create_dataset("species", data=pl.species_type[active_indices])
            
        log_brief(f"Output generated for step {step} (HDF5)", 1)

class VectorPatch:
    def __init__(self, parameters):
        self.parameters = parameters
        self.patches = []
        self.num_errors = 0
        
        # Create patches
        for i in range(parameters.number_of_patches):
            self.patches.append(Patch(parameters, i + 1))

    def start_pic(self):
        for patch in self.patches:
            patch.start_pic()
            self.num_errors += patch.num_errors
