"""
Monte Carlo Collision (MCC) Module
Optimized for speed with JIT compilation
"""
import numpy as np
import os
import math
from scipy import interpolate
from .utils import log_messages, log_brief
from .constants import ELECTRON_MASS_kg, ELECTRON_CHARGE, BOLTZMANN_CONSTANT
from numba import jit

@jit(nopython=True)
def collide_electrons_kernel(x, y, z, vx, vy, vz, mass, active,
                             sigma_elastic, sigma_excitation, sigma_ionization,
                             energy_grid_min, energy_grid_step, energy_grid_size,
                             neutral_density, time_step, max_id):
    """
    JIT-compiled electron collision kernel for maximum performance.
    """
    n = len(x)
    new_particles = []  # List of tuples: (x, y, z, vx, vy, vz)
    
    # Null collision frequency (conservative estimate)
    nu_max = 2.0e11
    P_null = 1.0 - math.exp(-nu_max * time_step)
    
    for i in range(n):
        if not active[i]:
            continue
        
        # Skip ions (mass check)
        if mass[i] > 2.0 * ELECTRON_MASS_kg:
            continue
        
        # Null collision check
        if np.random.random() > P_null:
            continue
        
        # Calculate electron energy
        v_sq = vx[i]**2 + vy[i]**2 + vz[i]**2
        v_mag = math.sqrt(v_sq)
        energy = 0.5 * mass[i] * v_sq / ELECTRON_CHARGE  # eV
        
        # Get cross sections from energy grid
        e_idx = int((energy - energy_grid_min) / energy_grid_step)
        if e_idx < 0: e_idx = 0
        if e_idx >= energy_grid_size: e_idx = energy_grid_size - 1
        
        # Total cross section * density
        s_ela = sigma_elastic[e_idx] * neutral_density
        s_exc = sigma_excitation[e_idx] * neutral_density
        s_ion = sigma_ionization[e_idx] * neutral_density
        s_tot = s_ela + s_exc + s_ion
        
        # Real collision frequency
        nu_real = v_mag * s_tot
        
        # Second null collision check
        if np.random.random() > (nu_real / nu_max):
            continue
        
        # Determine collision type
        r_type = np.random.random() * s_tot
        
        if r_type < s_ela:
            # Elastic scattering - isotropic
            theta = math.acos(1.0 - 2.0 * np.random.random())
            phi = 2.0 * math.pi * np.random.random()
            vx[i] = v_mag * math.sin(theta) * math.cos(phi)
            vy[i] = v_mag * math.sin(theta) * math.sin(phi)
            vz[i] = v_mag * math.cos(theta)
            
        elif r_type < s_ela + s_exc:
            # Excitation - lose 11.5 eV
            if energy > 11.5:
                new_energy = energy - 11.5
                v_new = math.sqrt(2 * new_energy * ELECTRON_CHARGE / mass[i])
                # Isotropic scatter
                theta = math.acos(1.0 - 2.0 * np.random.random())
                phi = 2.0 * math.pi * np.random.random()
                vx[i] = v_new * math.sin(theta) * math.cos(phi)
                vy[i] = v_new * math.sin(theta) * math.sin(phi)
                vz[i] = v_new * math.cos(theta)
                
        else:
            # Ionization - lose 15.8 eV, create secondary electron
            if energy > 15.8:
                remaining = energy - 15.8
                e_primary = remaining * np.random.random()
                e_secondary = remaining - e_primary
                
                # Update primary electron
                v_prim = math.sqrt(2 * e_primary * ELECTRON_CHARGE / mass[i])
                theta = math.acos(1.0 - 2.0 * np.random.random())
                phi = 2.0 * math.pi * np.random.random()
                vx[i] = v_prim * math.sin(theta) * math.cos(phi)
                vy[i] = v_prim * math.sin(theta) * math.sin(phi)
                vz[i] = v_prim * math.cos(theta)
                
                # Create secondary electron
                v_sec = math.sqrt(2 * e_secondary * ELECTRON_CHARGE / mass[i])
                theta_sec = math.acos(1.0 - 2.0 * np.random.random())
                phi_sec = 2.0 * math.pi * np.random.random()
                vx_new = v_sec * math.sin(theta_sec) * math.cos(phi_sec)
                vy_new = v_sec * math.sin(theta_sec) * math.sin(phi_sec)
                vz_new = v_sec * math.cos(theta_sec)
                
                new_particles.append((x[i], y[i], z[i], vx_new, vy_new, vz_new))
    
    return new_particles

@jit(nopython=True)
def collide_ions_kernel(vx, vy, vz, mass, active,
                        sigma_isotropic, sigma_backscatter,
                        energy_grid_min, energy_grid_step, energy_grid_size,
                        neutral_density, time_step):
    """
    JIT-compiled ion collision kernel.
    """
    n = len(vx)
    nu_max = 5.0e9
    P_null = 1.0 - math.exp(-nu_max * time_step)
    
    for i in range(n):
        if not active[i]:
            continue
        
        # Skip electrons
        if mass[i] < 2.0 * ELECTRON_MASS_kg:
            continue
        
        if np.random.random() > P_null:
            continue
        
        v_sq = vx[i]**2 + vy[i]**2 + vz[i]**2
        v_mag = math.sqrt(v_sq)
        energy = 0.5 * mass[i] * v_sq / ELECTRON_CHARGE
        
        e_idx = int((energy - energy_grid_min) / energy_grid_step)
        if e_idx < 0: e_idx = 0
        if e_idx >= energy_grid_size: e_idx = energy_grid_size - 1
        
        s_iso = sigma_isotropic[e_idx] * neutral_density
        s_back = sigma_backscatter[e_idx] * neutral_density
        s_tot = s_iso + s_back
        
        nu_real = v_mag * s_tot
        
        if np.random.random() > (nu_real / nu_max):
            continue
        
        if np.random.random() < (s_iso / s_tot):
            # Isotropic scatter
            theta = math.acos(1.0 - 2.0 * np.random.random())
            phi = 2.0 * math.pi * np.random.random()
            vx[i] = v_mag * math.sin(theta) * math.cos(phi)
            vy[i] = v_mag * math.sin(theta) * math.sin(phi)
            vz[i] = v_mag * math.cos(theta)
        else:
            # Backscatter
            vx[i] = -vx[i]
            vy[i] = -vy[i]
            vz[i] = -vz[i]

class MCC:
    """
    Monte Carlo Collision module with optimized JIT kernels.
    
    Neutral background: 3.2×10²¹ m⁻³ (10 mTorr argon at 300K)
    """
    def __init__(self, parameters):
        self.parameters = parameters
        
        # CORRECTED neutral density (was 27x too low!)
        # 10 mTorr = 1.33 Pa, n = P/(kT) = 1.33/(1.38e-23*300) = 3.2e21 m⁻³
        self.neutral_density = 3.2e21
        
        # Energy grid for cross sections
        self.energy_grid_min = 0.0
        self.energy_grid_max = 1000.0
        self.energy_grid_size = 10000
        self.energy_grid_step = (self.energy_grid_max - self.energy_grid_min) / (self.energy_grid_size - 1)
        self.energy_grid = np.linspace(self.energy_grid_min, self.energy_grid_max, self.energy_grid_size)
        
        # Initialize cross section arrays
        self.sigma_elastic = np.zeros_like(self.energy_grid)
        self.sigma_excitation = np.zeros_like(self.energy_grid)
        self.sigma_ionization = np.zeros_like(self.energy_grid)
        self.sigma_ion_isotropic = np.zeros_like(self.energy_grid)
        self.sigma_ion_backscatter = np.zeros_like(self.energy_grid)
        
        self.load_cross_sections()
    
    def load_cross_sections(self):
        """Load LXCAT cross section data."""
        log_messages("Loading MCC cross sections", __file__, 0, 1)
        base_dir = "lxcat_data"
        
        if os.path.exists(os.path.join(base_dir, "ar_electron_elastic_effective.txt")):
            self.sigma_elastic = self.load_lxcat_file(os.path.join(base_dir, "ar_electron_elastic_effective.txt"))
        if os.path.exists(os.path.join(base_dir, "ar_electron_excitation_11.5eV.txt")):
            self.sigma_excitation = self.load_lxcat_file(os.path.join(base_dir, "ar_electron_excitation_11.5eV.txt"))
        if os.path.exists(os.path.join(base_dir, "ar_electron_ionization.txt")):
            self.sigma_ionization = self.load_lxcat_file(os.path.join(base_dir, "ar_electron_ionization.txt"))
        if os.path.exists(os.path.join(base_dir, "ar_ion_isotropic.txt")):
            self.sigma_ion_isotropic = self.load_lxcat_file(os.path.join(base_dir, "ar_ion_isotropic.txt"))
        if os.path.exists(os.path.join(base_dir, "ar_ion_backscatter.txt")):
            self.sigma_ion_backscatter = self.load_lxcat_file(os.path.join(base_dir, "ar_ion_backscatter.txt"))
    
    def load_lxcat_file(self, filepath):
        """Parse LXCAT format cross section file."""
        data_energy = []
        data_sigma = []
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            parsing = False
            for line in lines:
                if "-----------------------------" in line:
                    parsing = not parsing
                    continue
                if parsing:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            e = float(parts[0])
                            s = float(parts[1])
                            data_energy.append(e)
                            data_sigma.append(s)
                        except ValueError:
                            continue
            if not data_energy:
                return np.zeros_like(self.energy_grid)
            f_interp = interpolate.interp1d(data_energy, data_sigma, kind='linear', 
                                           bounds_error=False, fill_value=0.0)
            return f_interp(self.energy_grid)
        except Exception as e:
            log_messages(f"Error loading {filepath}: {e}", __file__, 0, 3)
            return np.zeros_like(self.energy_grid)
    
    def collide(self, particle_list, time_step):
        """
        Main collision handler - calls JIT kernels for speed.
        """
        pl = particle_list
        
        # Electron collisions (returns new particles from ionization)
        new_particles_data = collide_electrons_kernel(
            pl.x, pl.y, pl.z, pl.vx, pl.vy, pl.vz, pl.mass, pl.active,
            self.sigma_elastic, self.sigma_excitation, self.sigma_ionization,
            self.energy_grid_min, self.energy_grid_step, self.energy_grid_size,
            self.neutral_density, time_step, pl.max_particle_id
        )
        
        # Add secondary electrons from ionization
        if new_particles_data:
            for p_data in new_particles_data:
                x, y, z, vx, vy, vz = p_data
                
                # Calculate cell_id
                col = int(x / self.parameters.pic_spacing)
                row = int(y / self.parameters.pic_spacing)
                
                if row < 0: row = 0
                if row >= pl.mesh.num_cells_y: row = pl.mesh.num_cells_y - 1
                if col < 0: col = 0
                if col >= pl.mesh.num_cells_x: col = pl.mesh.num_cells_x - 1
                
                cell_id = row * pl.mesh.num_cells_x + col + 1
                
                # Add secondary electron
                pl.add_particle_soa(x, y, z, vx, vy, vz,
                                  ELECTRON_MASS_kg * self.parameters.specific_weight,
                                  ELECTRON_CHARGE * self.parameters.specific_weight,
                                  self.parameters.specific_weight,
                                  -1, cell_id)
        
        # Ion collisions
        collide_ions_kernel(
            pl.vx, pl.vy, pl.vz, pl.mass, pl.active,
            self.sigma_ion_isotropic, self.sigma_ion_backscatter,
            self.energy_grid_min, self.energy_grid_step, self.energy_grid_size,
            self.neutral_density, time_step
        )
