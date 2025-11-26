import random
import math
from .constants import ELECTRON_CHARGE, ELECTRON_MASS_kg, XENON_MASS_kg
from .utils import log_messages, log_brief
import sys

class Particle:
    def __init__(self, parameters, mesh, cell_id, particle_id, index=0, p_type=None, position=None):
        self.particle_id = particle_id
        self.cell_id = cell_id
        self.parameters = parameters  # Store reference for thermal velocity calc
        self.particle_weight = parameters.specific_weight
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.old_velocity = [-1.0, -1.0, -1.0]
        self.em_field = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        self.species_type = 0 # 0: neutral, 1: ion, -1: electron
        self.mass = 0.0
        self.charge = 0.0

        # Determine properties based on type
        if p_type is None:
            # Default initialization logic
            if parameters.simulation_type == "electron":
                self.charge = ELECTRON_CHARGE * self.particle_weight
                self.mass = ELECTRON_MASS_kg * self.particle_weight
                self.species_type = -1
            else:
                if parameters.propellant == "xenon":
                    self.charge = 0.0
                    self.mass = XENON_MASS_kg * self.particle_weight
                    self.species_type = 0
        elif p_type == "electron":
            self.charge = ELECTRON_CHARGE * self.particle_weight
            self.mass = ELECTRON_MASS_kg * self.particle_weight
            self.species_type = -1
        elif p_type == "ion":
            if parameters.propellant == "xenon":
                self.charge = -ELECTRON_CHARGE * self.particle_weight
                self.mass = (XENON_MASS_kg - ELECTRON_MASS_kg) * self.particle_weight
            elif parameters.propellant == "argon":
                from .constants import ARGON_MASS_kg
                self.charge = -ELECTRON_CHARGE * self.particle_weight
                self.mass = (ARGON_MASS_kg - ELECTRON_MASS_kg) * self.particle_weight
            self.species_type = 1
        elif p_type == "neutral":
             if parameters.propellant == "xenon":
                self.charge = 0.0
                self.mass = XENON_MASS_kg * self.particle_weight
             elif parameters.propellant == "argon":
                from .constants import ARGON_MASS_kg
                self.charge = 0.0
                self.mass = ARGON_MASS_kg * self.particle_weight
             self.species_type = 0

        # Initialize Position
        cell = mesh.cells[cell_id - 1]
        if position is not None:
            # Inlet particle
             self.position[0] = 1e-5
             self.position[1] = position
             self.position[2] = 0.0
             
             self.velocity[0] = parameters.inlet_velocity
             self.velocity[1] = 0.0
             self.velocity[2] = 0.0
        else:
            # Random or Uniform distribution
            if parameters.particle_distribution == "precise":
                 self.position[0] = cell.left * (1 - parameters.initial_position[0]) + cell.right * parameters.initial_position[0]
                 self.position[1] = cell.top * parameters.initial_position[1] + cell.bottom * (1 - parameters.initial_position[1])
            elif parameters.particle_distribution == "random":
                r1 = random.random()
                r2 = random.random()
                self.position[0] = cell.left * (1 - r1) + cell.right * r1
                self.position[1] = cell.top * r2 + cell.bottom * (1 - r2)
            elif parameters.particle_distribution == "uniform":
                 # Simplified uniform distribution logic
                 xratio = (0.5 + (index % int(math.sqrt(parameters.initial_particles_per_cell)))) / math.sqrt(parameters.initial_particles_per_cell)
                 yratio = (0.5 + (index // int(math.sqrt(parameters.initial_particles_per_cell)))) / math.sqrt(parameters.initial_particles_per_cell)
                 self.position[0] = cell.left * (1 - xratio) + cell.right * xratio
                 self.position[1] = cell.top * yratio + cell.bottom * (1 - yratio)
            
            # Initialize Velocity with thermal distribution
            # Use Maxwell-Boltzmann: v_th = sqrt(k_B * T / m)
            from .constants import BOLTZMANN_CONSTANT
            
            # Thermal velocity magnitude
            v_th = math.sqrt(BOLTZMANN_CONSTANT * self.parameters.initial_temperature / (self.mass / self.particle_weight))
            
            # Random Maxwell-Boltzmann velocities (3D)
            # Each component is normally distributed with std = v_th/sqrt(3)
            import numpy as np
            std_v = v_th / math.sqrt(3.0)
            self.velocity[0] = np.random.normal(0, std_v) + parameters.initial_velocity[0]
            self.velocity[1] = np.random.normal(0, std_v) + parameters.initial_velocity[1]
            self.velocity[2] = np.random.normal(0, std_v)

        # Two-stream instability setup
        if parameters.two_stream:
            self.species_type = 1
            if random.random() >= 0.5:
                self.species_type = -1
                self.velocity[0] *= -1.0
                self.velocity[1] *= -1.0

    def velocity_magnitude(self):
        return math.sqrt(self.velocity[0]**2 + self.velocity[1]**2 + self.velocity[2]**2)

import numpy as np

class ParticleList:
    def __init__(self, parameters, mesh, capacity=200000):
        self.parameters = parameters
        self.mesh = mesh
        self.capacity = capacity
        self.num_particles = 0
        self.max_particle_id = 0
        
        # Structure of Arrays (SoA)
        self.x = np.zeros(capacity)
        self.y = np.zeros(capacity)
        self.z = np.zeros(capacity)
        
        self.vx = np.zeros(capacity)
        self.vy = np.zeros(capacity)
        self.vz = np.zeros(capacity)
        
        # Fields at particle position
        self.Ex = np.zeros(capacity)
        self.Ey = np.zeros(capacity)
        self.Ez = np.zeros(capacity)
        self.Bx = np.zeros(capacity)
        self.By = np.zeros(capacity)
        self.Bz = np.zeros(capacity)
        
        # Properties
        self.mass = np.zeros(capacity)
        self.charge = np.zeros(capacity)
        self.weight = np.zeros(capacity)
        self.species_type = np.zeros(capacity, dtype=np.int32)
        self.particle_id = np.zeros(capacity, dtype=np.int32)
        self.cell_id = np.zeros(capacity, dtype=np.int32)
        
        # Active flag to manage deletions efficiently
        self.active = np.zeros(capacity, dtype=bool)

        # Initialize particles if needed
        if parameters.initial_particles_per_cell > 0:
            self.initialize_particles()
            
        # Calculate inlet particles per step
        if self.parameters.inlet_source:
            mass = 0.0
            if self.parameters.simulation_type == "electron":
                mass = ELECTRON_MASS_kg
            elif self.parameters.propellant == "xenon":
                mass = XENON_MASS_kg
            
            # Total number of particles from inlet per time step
            if mass > 0:
                self.inlet_particles_per_step = int((self.parameters.inlet_flow_rate / mass) * 
                                                  (self.parameters.time_step / self.parameters.specific_weight))
            else:
                self.inlet_particles_per_step = 0
            
            log_messages(f"Inlet particles per step: {self.inlet_particles_per_step}", __file__, sys._getframe().f_lineno, 1)

    def _resize(self):
        new_capacity = self.capacity * 2
        log_messages(f"Resizing ParticleList from {self.capacity} to {new_capacity}", __file__, sys._getframe().f_lineno, 1)
        
        self.x = np.resize(self.x, new_capacity)
        self.y = np.resize(self.y, new_capacity)
        self.z = np.resize(self.z, new_capacity)
        self.vx = np.resize(self.vx, new_capacity)
        self.vy = np.resize(self.vy, new_capacity)
        self.vz = np.resize(self.vz, new_capacity)
        self.Ex = np.resize(self.Ex, new_capacity)
        self.Ey = np.resize(self.Ey, new_capacity)
        self.Ez = np.resize(self.Ez, new_capacity)
        self.Bx = np.resize(self.Bx, new_capacity)
        self.By = np.resize(self.By, new_capacity)
        self.Bz = np.resize(self.Bz, new_capacity)
        self.mass = np.resize(self.mass, new_capacity)
        self.charge = np.resize(self.charge, new_capacity)
        self.weight = np.resize(self.weight, new_capacity)
        self.species_type = np.resize(self.species_type, new_capacity)
        self.particle_id = np.resize(self.particle_id, new_capacity)
        self.cell_id = np.resize(self.cell_id, new_capacity)
        self.active = np.resize(self.active, new_capacity)
        
        self.capacity = new_capacity

    def add_particle_soa(self, x, y, z, vx, vy, vz, mass, charge, weight, species_type, cell_id):
        if self.num_particles >= self.capacity:
            self._resize()
            
        idx = self.num_particles
        self.x[idx] = x
        self.y[idx] = y
        self.z[idx] = z
        self.vx[idx] = vx
        self.vy[idx] = vy
        self.vz[idx] = vz
        self.mass[idx] = mass
        self.charge[idx] = charge
        self.weight[idx] = weight
        self.species_type[idx] = species_type
        self.cell_id[idx] = cell_id
        
        self.max_particle_id += 1
        self.particle_id[idx] = self.max_particle_id
        self.active[idx] = True
        
        self.num_particles += 1
        
        # Add to mesh (legacy support)
        self.mesh.add_particle_to_cell(cell_id, self.max_particle_id, species_type)
        
        return idx

    def initialize_particles(self):
        log_messages("Initializing particles (SoA)", __file__, sys._getframe().f_lineno, 1)
        
        count_e = 0
        count_i = 0
        
        # Pre-calculate thermal velocity constants
        from .constants import BOLTZMANN_CONSTANT, ARGON_MASS_kg
        
        # Electron properties
        m_e = ELECTRON_MASS_kg * self.parameters.specific_weight
        q_e = ELECTRON_CHARGE * self.parameters.specific_weight
        v_th_e = math.sqrt(BOLTZMANN_CONSTANT * self.parameters.initial_temperature / (ELECTRON_MASS_kg))
        std_v_e = v_th_e / math.sqrt(3.0)
        
        # Ion properties (Argon default)
        m_i = (ARGON_MASS_kg - ELECTRON_MASS_kg) * self.parameters.specific_weight
        q_i = -ELECTRON_CHARGE * self.parameters.specific_weight
        v_th_i = math.sqrt(BOLTZMANN_CONSTANT * self.parameters.initial_temperature / (ARGON_MASS_kg))
        std_v_i = v_th_i / math.sqrt(3.0)
        
        if self.parameters.propellant == "xenon":
             m_i = (XENON_MASS_kg - ELECTRON_MASS_kg) * self.parameters.specific_weight
             v_th_i = math.sqrt(BOLTZMANN_CONSTANT * self.parameters.initial_temperature / (XENON_MASS_kg))
             std_v_i = v_th_i / math.sqrt(3.0)

        for cell in self.mesh.cells:
            for i in range(self.parameters.initial_particles_per_cell):
                # Position (Random)
                r1 = random.random()
                r2 = random.random()
                x = cell.left * (1 - r1) + cell.right * r1
                y = cell.top * r2 + cell.bottom * (1 - r2)
                z = 0.0
                
                # Create Electron
                vx = np.random.normal(0, std_v_e) + self.parameters.initial_velocity[0]
                vy = np.random.normal(0, std_v_e) + self.parameters.initial_velocity[1]
                vz = np.random.normal(0, std_v_e)
                
                self.add_particle_soa(x, y, z, vx, vy, vz, m_e, q_e, self.parameters.specific_weight, -1, cell.id)
                count_e += 1
                
                # Create Ion (if not electron-only)
                if self.parameters.simulation_type == "electron":
                    vx_i = np.random.normal(0, std_v_i) + self.parameters.initial_velocity[0]
                    vy_i = np.random.normal(0, std_v_i) + self.parameters.initial_velocity[1]
                    vz_i = np.random.normal(0, std_v_i)
                    
                    self.add_particle_soa(x, y, z, vx_i, vy_i, vz_i, m_i, q_i, self.parameters.specific_weight, 1, cell.id)
                    count_i += 1
        
        log_brief(f"Initialized {count_e} electrons and {count_i} ions (SoA)", 1)

    def add_particle(self, particle):
        # Legacy adapter
        self.add_particle_soa(particle.position[0], particle.position[1], particle.position[2],
                            particle.velocity[0], particle.velocity[1], particle.velocity[2],
                            particle.mass, particle.charge, particle.particle_weight,
                            particle.species_type, particle.cell_id)

    def remove_particle(self, particle_id):
        # Find index (slow linear search, but okay for boundaries)
        # For high performance, we might need a map or just mark inactive
        # Since we use Numba, we can pass the 'active' array and just set to False
        
        # Vectorized search
        indices = np.where((self.particle_id == particle_id) & (self.active))[0]
        if len(indices) > 0:
            idx = indices[0]
            self.active[idx] = False
            
            # Remove from mesh (legacy)
            # We need to know cell_id and species_type
            # They are still in the arrays
            self.mesh.remove_particle_from_cell(self.cell_id[idx], particle_id, self.species_type[idx])
            
            # We don't decrement num_particles immediately to avoid hole management issues
            # But for simple append-only logic, we might want to swap-remove
            # Swap with last active?
            # For now, just mark inactive. Periodic cleanup or swap-remove is better.
            
            # Swap-remove implementation
            last_idx = self.num_particles - 1
            if idx != last_idx:
                # Move last particle to this slot
                self.x[idx] = self.x[last_idx]
                self.y[idx] = self.y[last_idx]
                self.z[idx] = self.z[last_idx]
                self.vx[idx] = self.vx[last_idx]
                self.vy[idx] = self.vy[last_idx]
                self.vz[idx] = self.vz[last_idx]
                self.Ex[idx] = self.Ex[last_idx]
                self.Ey[idx] = self.Ey[last_idx]
                self.Ez[idx] = self.Ez[last_idx]
                self.Bx[idx] = self.Bx[last_idx]
                self.By[idx] = self.By[last_idx]
                self.Bz[idx] = self.Bz[last_idx]
                self.mass[idx] = self.mass[last_idx]
                self.charge[idx] = self.charge[last_idx]
                self.weight[idx] = self.weight[last_idx]
                self.species_type[idx] = self.species_type[last_idx]
                self.particle_id[idx] = self.particle_id[last_idx]
                self.cell_id[idx] = self.cell_id[last_idx]
                self.active[idx] = self.active[last_idx]
            
            self.active[last_idx] = False
            self.num_particles -= 1

    def add_particles_to_sim(self, step=0):
        if not hasattr(self, 'inlet_particles_per_step') or self.inlet_particles_per_step <= 0:
            return

        # Pre-calculate properties
        from .constants import BOLTZMANN_CONSTANT, ARGON_MASS_kg
        m = ELECTRON_MASS_kg * self.parameters.specific_weight
        q = ELECTRON_CHARGE * self.parameters.specific_weight
        species = -1
        
        if self.parameters.propellant == "xenon" and self.parameters.simulation_type != "electron":
             m = XENON_MASS_kg * self.parameters.specific_weight
             q = 0.0
             species = 0
        
        v_th = math.sqrt(BOLTZMANN_CONSTANT * self.parameters.initial_temperature / (m / self.parameters.specific_weight))
        std_v = v_th / math.sqrt(3.0)

        for i in range(self.inlet_particles_per_step):
            r = random.random()
            position = r * self.parameters.inlet_size_percent * self.parameters.domain_height
            
            if not self.parameters.axisymmetric:
                position += self.parameters.domain_height * ((1.0 - self.parameters.inlet_size_percent) / 2.0)
            
            row = int(position / self.parameters.pic_spacing)
            if row >= self.mesh.num_cells_y: row = self.mesh.num_cells_y - 1
            if row < 0: row = 0
            
            cell_id = row * self.mesh.num_cells_x + 1
            
            x = 1e-5
            y = position
            z = 0.0
            
            vx = np.random.normal(0, std_v) + self.parameters.inlet_velocity
            vy = np.random.normal(0, std_v)
            vz = np.random.normal(0, std_v)
            
            self.add_particle_soa(x, y, z, vx, vy, vz, m, q, self.parameters.specific_weight, species, cell_id)

