import numpy as np
import os
import math
from .constants import ELECTRON_CHARGE, ELECTRON_MASS_kg, EPSILON_0, BOLTZMANN_CONSTANT
from .utils import log_messages, log_brief

class Diagnostics:
    def __init__(self, parameters, mesh):
        self.parameters = parameters
        self.mesh = mesh
        
        # Accumulators for time-averaged fields
        self.accumulated_steps = 0
        self.rho_sum = np.zeros(len(mesh.nodes))
        self.phi_sum = np.zeros(len(mesh.nodes))
        self.em_field_sum = np.zeros((len(mesh.nodes), 6)) # Ex, Ey, Ez, Bx, By, Bz
        
        # EEPF (Electron Energy Probability Function)
        self.energy_bins = np.linspace(0, 100, 200) # 0-100 eV
        self.eepf_counts = np.zeros(len(self.energy_bins) - 1)
        
        # History
        self.history_time = []
        self.history_num_electrons = []
        self.history_num_ions = []
        self.history_total_energy = []
        
    def accumulate(self, particle_list, nodes, time):
        self.accumulated_steps += 1
        
        # 1. Field Accumulation
        # This loop is over nodes, which is fine (40k iterations is fast enough in Python)
        # But we can vectorize it if node properties were arrays.
        # Since we are syncing arrays to nodes in solver, we can access nodes.
        # Or better: access the arrays if we had them in mesh.
        # For now, keep it as is, it's O(N_nodes).
        for i, node in enumerate(nodes):
            self.rho_sum[i] += node.rho
            self.phi_sum[i] += node.phi
            for k in range(6):
                self.em_field_sum[i][k] += node.em_field[k]
                
        # 2. Particle Diagnostics (EEPF & Global Counts)
        pl = particle_list
        active_mask = pl.active
        
        # Electrons
        e_mask = active_mask & (pl.species_type == -1)
        num_e_macro = np.count_nonzero(e_mask)
        num_e_real = np.sum(pl.weight[e_mask]) if num_e_macro > 0 else 0.0
        
        # Ions
        i_mask = active_mask & (pl.species_type == 1)
        num_i_macro = np.count_nonzero(i_mask)
        num_i_real = np.sum(pl.weight[i_mask]) if num_i_macro > 0 else 0.0
        
        total_energy = 0.0
        
        # EEPF
        if num_e_macro > 0:
            vx = pl.vx[e_mask]
            vy = pl.vy[e_mask]
            vz = pl.vz[e_mask]
            mass = pl.mass[e_mask]
            
            # Calculate energy in eV
            # E = 0.5 * m * v^2 / q_e
            v2 = vx**2 + vy**2 + vz**2
            energies = 0.5 * mass * v2 / abs(ELECTRON_CHARGE)
            
            total_energy = np.sum(energies)
            
            # For histogram, we want energy per REAL electron, not macroparticle
            # Energy of macroparticle = W * E_real
            # So E_real = E_macro / W
            # pl.mass = m_real * W
            # E_macro = 0.5 * (m_real * W) * v^2
            # E_real = 0.5 * m_real * v^2
            # So we should calculate E_real for the histogram
            
            mass_real = mass / pl.weight[e_mask]
            energies_real = 0.5 * mass_real * v2 / abs(ELECTRON_CHARGE)
            
            # Weighted histogram?
            # Ideally yes, but if weights are uniform, simple histogram of E_real is fine.
            # Let's use E_real for the histogram bins.
            
            hist, _ = np.histogram(energies_real, bins=self.energy_bins, weights=pl.weight[e_mask])
            self.eepf_counts += hist
            
        self.history_time.append(time)
        self.history_num_electrons.append(num_e_real)
        self.history_num_ions.append(num_i_real)
        self.history_total_energy.append(total_energy)
        
        # Simple diagnostic print
        if self.accumulated_steps % 100 == 0:
            avg_E = total_energy / num_e_real if num_e_real > 0 else 0.0
            print(f"Step {self.accumulated_steps}: Total E = {total_energy:.4e} eV, N_e_real = {num_e_real:.2e}, Avg E = {avg_E:.2f} eV")

    def save_diagnostics(self, output_dir="output"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 1. Save Averaged Fields
        if self.accumulated_steps > 0:
            factor = 1.0 / self.accumulated_steps
            
            # Save Node Data (Averaged)
            with open(os.path.join(output_dir, "averaged_fields.csv"), "w") as f:
                f.write("NodeID,X,Y,Rho_Avg,Phi_Avg,Ex_Avg,Ey_Avg,Ez_Avg,Bx_Avg,By_Avg,Bz_Avg\n")
                for i, node in enumerate(self.mesh.nodes):
                    f.write(f"{node.id},{node.x},{node.y},"
                            f"{self.rho_sum[i]*factor:.4e},{self.phi_sum[i]*factor:.4e},"
                            f"{self.em_field_sum[i][0]*factor:.4e},{self.em_field_sum[i][1]*factor:.4e},{self.em_field_sum[i][2]*factor:.4e},"
                            f"{self.em_field_sum[i][3]*factor:.4e},{self.em_field_sum[i][4]*factor:.4e},{self.em_field_sum[i][5]*factor:.4e}\n")
                            
        # 2. Save EEPF
        with open(os.path.join(output_dir, "eepf.dat"), "w") as f:
            f.write("# Energy[eV] Count Probability\n")
            total_counts = np.sum(self.eepf_counts)
            if total_counts == 0: total_counts = 1.0
            
            for i in range(len(self.eepf_counts)):
                energy = 0.5 * (self.energy_bins[i] + self.energy_bins[i+1])
                prob = self.eepf_counts[i] / total_counts
                f.write(f"{energy:.4f} {self.eepf_counts[i]} {prob:.4e}\n")
                
        # 3. Save History (Append mode)
        history_file = os.path.join(output_dir, "history.dat")
        file_exists = os.path.exists(history_file)
        
        with open(history_file, "a") as f:
            if not file_exists:
                f.write("# Time[s] Num_Electrons Num_Ions Total_Electron_Energy[eV]\n")
            
            for i in range(len(self.history_time)):
                f.write(f"{self.history_time[i]:.4e} {self.history_num_electrons[i]} {self.history_num_ions[i]} {self.history_total_energy[i]:.4e}\n")
        
        # Clear history buffers to free memory
        self.history_time = []
        self.history_num_electrons = []
        self.history_num_ions = []
        self.history_total_energy = []
                
        # 4. Save Info Report
        self.save_info_report(output_dir)
        
    def save_info_report(self, output_dir):
        with open(os.path.join(output_dir, "simulation_info.txt"), "w") as f:
            f.write("PIC-IPD Simulation Report\n")
            f.write("=========================\n")
            f.write(f"Time Steps Run: {self.accumulated_steps}\n")
            f.write(f"Final Time: {self.history_time[-1] if self.history_time else 0.0:.4e} s\n")
            f.write(f"Final Electron Count: {self.history_num_electrons[-1] if self.history_num_electrons else 0}\n")
            f.write(f"Final Ion Count: {self.history_num_ions[-1] if self.history_num_ions else 0}\n")
            
            # Calculate average density
            avg_rho = np.mean(self.rho_sum) / self.accumulated_steps if self.accumulated_steps > 0 else 0.0
            f.write(f"Average Charge Density: {avg_rho:.4e} C/m^3\n")
            
            # Plasma Frequency (approx)
            # n = rho / q
            n = abs(avg_rho / ELECTRON_CHARGE)
            if n > 0:
                wp = math.sqrt(n * ELECTRON_CHARGE**2 / (ELECTRON_MASS_kg * EPSILON_0))
                f.write(f"Approx. Plasma Frequency: {wp:.4e} rad/s\n")
            
            f.write("\nParameters:\n")
            f.write(f"Time Step: {self.parameters.time_step:.4e} s\n")
            f.write(f"Grid Spacing: {self.parameters.pic_spacing:.4e} m\n")

