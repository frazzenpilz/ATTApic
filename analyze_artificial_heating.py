"""
Analyze simulation results for artificial heating artifacts.

Artificial heating (numerical heating) is unphysical energy gain due to:
    1. Finite grid effects (interpolation errors)
    2. Poor time resolution (aliasing)
    3. Non-energy-conserving algorithms
    4. Statistical noise amplification
"""

import numpy as np
import matplotlib.pyplot as plt

# Load history data
data = np.loadtxt('output/history.dat')
time = data[:, 0]
n_electrons = data[:, 1]
n_ions = data[:, 2]
total_energy = data[:, 3]

# Calculate average energy per electron
avg_energy_eV = total_energy / n_electrons

# Calculate energy growth rate
# dE/dt should match RF power input - wall losses
time_ns = time * 1e9

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Particle count evolution
axes[0, 0].plot(time_ns, n_electrons, 'b-', label='Electrons', linewidth=2)
axes[0, 0].plot(time_ns, n_ions, 'r-', label='Ions', linewidth=2)
axes[0, 0].set_xlabel('Time (ns)', fontsize=12)
axes[0, 0].set_ylabel('Number of Particles', fontsize=12)
axes[0, 0].set_title('Particle Count Evolution', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Average energy evolution
axes[0, 1].plot(time_ns, avg_energy_eV, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Time (ns)', fontsize=12)
axes[0, 1].set_ylabel('Average Energy (eV)', fontsize=12)
axes[0, 1].set_title('Average Electron Energy', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# 3. Total energy evolution
axes[1, 0].plot(time_ns, total_energy, 'm-', linewidth=2)
axes[1, 0].set_xlabel('Time (ns)', fontsize=12)
axes[1, 0].set_ylabel('Total Energy (eV)', fontsize=12)
axes[1, 0].set_title('Total Electron Energy', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# 4. Energy per particle vs particle count (phase plot)
axes[1, 1].scatter(n_electrons, avg_energy_eV, c=time_ns, cmap='viridis', s=10)
axes[1, 1].set_xlabel('Number of Electrons', fontsize=12)
axes[1, 1].set_ylabel('Average Energy (eV)', fontsize=12)
axes[1, 1].set_title('Phase Space: Energy vs Population', fontsize=14)
cbar = plt.colorbar(axes[1, 1].scatter(n_electrons, avg_energy_eV, c=time_ns, cmap='viridis', s=10), ax=axes[1, 1])
cbar.set_label('Time (ns)', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/artificial_heating_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved analysis plot: output/artificial_heating_analysis.png")

# Statistical analysis
print("\n=== ARTIFICIAL HEATING DIAGNOSTIC ===\n")
print(f"Simulation Duration: {time_ns[-1]:.2f} ns ({len(data)} samples)")
print(f"Initial Avg Energy: {avg_energy_eV[0]:.3f} eV")
print(f"Final Avg Energy: {avg_energy_eV[-1]:.3f} eV")
print(f"Energy Gain: {avg_energy_eV[-1] - avg_energy_eV[0]:.3f} eV")
print(f"Heating Rate: {(avg_energy_eV[-1] - avg_energy_eV[0]) / time_ns[-1]:.4f} eV/ns")

# Check for runaway behavior (exponential growth)
# Fit exponential to second half of data
mid_idx = len(avg_energy_eV) // 2
t_fit = time_ns[mid_idx:]
E_fit = avg_energy_eV[mid_idx:]

# Linear fit in log space (if exponential: E = E0 * exp(gamma * t))
if np.all(E_fit > 0):
    log_E = np.log(E_fit)
    poly_coeffs = np.polyfit(t_fit, log_E, 1)
    gamma = poly_coeffs[0]  # Growth rate
    
    print(f"\nExponential Fit (second half):")
    print(f"  Growth rate γ: {gamma:.6f} ns⁻¹")
    
    if gamma > 0.001:
        print(f"  ⚠️  WARNING: Exponential growth detected!")
        print(f"  Doubling time: {np.log(2) / gamma:.2f} ns")
        print(f"  This suggests ARTIFICIAL HEATING (numerical instability)")
    else:
        print(f"  ✅ Growth is sub-exponential (likely physical)")

# Check for oscillations (should see RF frequency)
from scipy.fft import fft, fftfreq
if len(avg_energy_eV) > 100:
    # FFT of average energy
    dt_sample = (time[-1] - time[0]) / (len(time) - 1)
    yf = fft(avg_energy_eV - np.mean(avg_energy_eV))
    xf = fftfreq(len(avg_energy_eV), dt_sample)[:len(avg_energy_eV)//2]
    
    # Convert to MHz
    xf_MHz = xf / 1e6
    power = 2.0 / len(avg_energy_eV) * np.abs(yf[:len(avg_energy_eV)//2])
    
    # Find dominant frequency
    dominant_idx = np.argmax(power[1:]) + 1  # Skip DC component
    dominant_freq = xf_MHz[dominant_idx]
    
    print(f"\nFrequency Analysis:")
    print(f"  Dominant frequency: {dominant_freq:.2f} MHz")
    print(f"  RF frequency: 135.6 MHz")
    
    if abs(dominant_freq - 135.6) < 10:
        print(f"  ✅ Energy oscillates at RF frequency (physical heating)")
    elif dominant_freq < 1:
        print(f"  ⚠️  Low-frequency drift (could be numerical)")

# Particle conservation check
print(f"\nParticle Conservation:")
print(f"  Initial e⁻: {n_electrons[0]:.2e}")
print(f"  Final e⁻: {n_electrons[-1]:.2e}")
print(f"  Loss: {(1 - n_electrons[-1]/n_electrons[0])*100:.1f}%")
print(f"  Initial ions: {n_ions[0]:.2e}")
print(f"  Final ions: {n_ions[-1]:.2e}")
print(f"  Charge imbalance: {abs(n_electrons[-1] - n_ions[-1]) / n_electrons[0] * 100:.2f}%")

plt.show()
