"""
Example: Analyze Multiple Timesteps

This script demonstrates how to analyze multiple timesteps
to track the evolution of the plasma over time.
"""

from visualize_results import PICVisualizer
import matplotlib.pyplot as plt
import numpy as np

# Initialize visualizer
vis = PICVisualizer(output_dir="output")

# Analyze a series of timesteps
timesteps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7400]

print("Analyzing plasma evolution over time...\n")

# Track electron count and average energy over time
times = []
electron_counts = []
avg_energies = []

for step in timesteps:
    if vis.load_data(step):
        # Calculate electron count
        particles = vis.data['particles']
        electron_mask = particles['species'] == -1
        
        if np.any(electron_mask):
            # Count electrons (weighted)
            n_electrons = np.sum(particles['weight'][electron_mask])
            electron_counts.append(n_electrons)
            times.append(vis.time)
            
            # Calculate average electron energy
            vx = particles['vx'][electron_mask]
            vy = particles['vy'][electron_mask]
            vz = particles['vz'][electron_mask]
            mass = particles['mass'][electron_mask]
            weight = particles['weight'][electron_mask]
            
            v2 = vx**2 + vy**2 + vz**2
            mass_real = mass / weight
            energy_eV = 0.5 * mass_real * v2 / 1.602176634e-19
            
            avg_energy = np.average(energy_eV, weights=weight)
            avg_energies.append(avg_energy)
            
            print(f"Step {step:5d} | Time: {vis.time*1e9:6.2f} ns | "
                  f"Electrons: {n_electrons:.2e} | Avg Energy: {avg_energy:.2f} eV")
        
        # Generate visualization for this timestep
        vis.plot_densities(save_fig=True, show_fig=False)
        vis.plot_eedf(save_fig=True, show_fig=False)

# Plot evolution
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Electron count evolution
ax1.plot(np.array(times)*1e9, electron_counts, 'b-o', linewidth=2, markersize=6)
ax1.set_xlabel('Time (ns)', fontsize=12)
ax1.set_ylabel('Number of Electrons', fontsize=12)
ax1.set_title('Electron Population Evolution', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Average energy evolution
ax2.plot(np.array(times)*1e9, avg_energies, 'r-o', linewidth=2, markersize=6)
ax2.set_xlabel('Time (ns)', fontsize=12)
ax2.set_ylabel('Average Electron Energy (eV)', fontsize=12)
ax2.set_title('Average Electron Energy Evolution', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/plasma_evolution.png', dpi=300, bbox_inches='tight')
print(f"\nSaved evolution plot: output/plasma_evolution.png")
plt.show()

print("\nAnalysis complete!")
