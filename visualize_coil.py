"""
Visualize RF Coil and Magnetic Field
Shows 3D geometry of plasma chamber and coil, plus B-field slice
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pic_modules.rf_coil import RFCoil
from pic_modules.parameters import Parameters
from pic_modules.constants import MU_0

# Chamber and coil specifications
chamber_radius = 0.005  # 5 mm
chamber_length = 0.04   # 4 cm

coil_turns = 20
coil_current = 1.0      # 1 A
coil_radius = 0.006     # 6 mm (slightly larger than chamber)
coil_length = 0.05      # 5 cm (slightly longer than chamber)
wire_thickness = 0.001  # 1 mm
rf_frequency = 13.56e6  # Hz

print("Creating RF Coil Visualization")
print(f"Chamber: r={chamber_radius*1000:.1f}mm, l={chamber_length*100:.1f}cm")
print(f"Coil: N={coil_turns}, I={coil_current}A, R={coil_radius*1000:.1f}mm")

# Create parameters object with coil specs
params = Parameters.__new__(Parameters)  # Create without reading file
params.coil_turns = coil_turns
params.coil_radius = coil_radius
params.coil_length = coil_length
params.coil_current = coil_current
params.rf_frequency = rf_frequency
params.domain_length = coil_length
params.domain_height = coil_radius * 2
params.use_rf_coil = True

# Create RF coil
coil = RFCoil(params)

# Create figure with subplots
fig = plt.figure(figsize=(16, 6))

# ============================================================
# Plot 1: 3D Geometry (Chamber + Coil)
# ============================================================
ax1 = fig.add_subplot(131, projection='3d')

# Draw plasma chamber (cylinder)
theta = np.linspace(0, 2*np.pi, 50)
z_chamber = np.linspace(0, chamber_length, 50)
Theta, Z_chamber = np.meshgrid(theta, z_chamber)
X_chamber = chamber_radius * np.cos(Theta)
Y_chamber = chamber_radius * np.sin(Theta)

ax1.plot_surface(X_chamber*1000, Y_chamber*1000, Z_chamber*1000, 
                 alpha=0.3, color='lightblue', label='Plasma Chamber')

# Draw RF coil (helix)
z_coil_start = (chamber_length - coil_length) / 2
z_coil_end = z_coil_start + coil_length
z_helix = np.linspace(z_coil_start, z_coil_end, 500)
theta_helix = np.linspace(0, coil_turns * 2*np.pi, 500)
x_helix = coil_radius * np.cos(theta_helix)
y_helix = coil_radius * np.sin(theta_helix)

ax1.plot(x_helix*1000, y_helix*1000, z_helix*1000, 
         'r-', linewidth=2, label=f'RF Coil ({coil_turns} turns)')

ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.set_title('3D Geometry: Plasma Chamber + RF Coil')
ax1.legend()
ax1.set_box_aspect([1, 1, 2])

# ============================================================
# Plot 2: B-field Magnitude (r-z slice)
# ============================================================
ax2 = fig.add_subplot(132)

# Create grid in r-z plane
r_grid = np.linspace(0, coil_radius*1.5, 100)
z_grid = np.linspace(0, coil_length, 150)
R_grid, Z_grid = np.meshgrid(r_grid, z_grid)

# Calculate B-field at time when current is maximum (quarter period)
# I = I0 * sin(ωt), maximum at t = π/(2ω)
t_max = np.pi / (2 * coil.omega)

Br_grid = np.zeros_like(R_grid)
Bz_grid = np.zeros_like(R_grid)

for i in range(R_grid.shape[0]):
    for j in range(R_grid.shape[1]):
        r = R_grid[i, j]
        z = Z_grid[i, j]
        Br, Bz = coil.calculate_field_finite_solenoid(r, z, time=t_max)
        Br_grid[i, j] = Br
        Bz_grid[i, j] = Bz

# Calculate magnitude
B_mag = np.sqrt(Br_grid**2 + Bz_grid**2)
B_max = np.max(B_mag)

# Plot contour
if B_max > 0:
    levels = np.linspace(0, B_max*1.1, 20)
else:
    levels = 20
    
contour = ax2.contourf(Z_grid*1000, R_grid*1000, B_mag*1e6, 
                       levels=levels, cmap='plasma')
plt.colorbar(contour, ax=ax2, label='|B| (µT)')

# Add chamber outline
ax2.plot([0, chamber_length*1000], [chamber_radius*1000, chamber_radius*1000], 
         'w--', linewidth=2, label='Chamber boundary')
ax2.plot([0, 0], [0, chamber_radius*1000], 'w--', linewidth=2)
ax2.plot([chamber_length*1000, chamber_length*1000], [0, chamber_radius*1000], 'w--', linewidth=2)

# Add coil position indicators
z_start = coil.z_start * 1000
z_end = coil.z_end * 1000
r_coil = coil_radius * 1000
ax2.plot([z_start, z_start], [0, r_coil], 'r-', linewidth=3, label='Coil extent')
ax2.plot([z_end, z_end], [0, r_coil], 'r-', linewidth=3)
ax2.plot([z_start, z_end], [r_coil, r_coil], 'r-', linewidth=3)

ax2.set_xlabel('Z - Axial Position (mm)')
ax2.set_ylabel('R - Radial Position (mm)')
ax2.set_title(f'Magnetic Field Magnitude (Peak Current)\nCoil Current = {coil_current}A')
ax2.legend(loc='upper right')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# ============================================================
# Plot 3: B-field Components (on-axis)
# ============================================================
ax3 = fig.add_subplot(133)

# Calculate field on axis (r=0)
z_axis = np.linspace(0, coil_length, 200)
Bz_axis = np.zeros_like(z_axis)
Br_axis = np.zeros_like(z_axis)

for i, z in enumerate(z_axis):
    Br, Bz = coil.calculate_field_finite_solenoid(0.0, z, time=t_max)
    Br_axis[i] = Br
    Bz_axis[i] = Bz

ax3.plot(z_axis*1000, Bz_axis*1e6, 'b-', linewidth=2, label='Bz (axial)')
ax3.plot(z_axis*1000, Br_axis*1e6, 'r-', linewidth=2, label='Br (radial)')

# Show chamber extent
ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax3.axvline(chamber_length*1000, color='gray', linestyle='--', alpha=0.5)
ax3.axhline(0, color='k', linestyle='-', alpha=0.3)

# Show coil extent
ax3.axvline(coil.z_start*1000, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax3.axvline(coil.z_end*1000, color='red', linestyle='--', alpha=0.5, linewidth=2)

ax3.set_xlabel('Z - Axial Position (mm)')
ax3.set_ylabel('B-field (µT)')
ax3.set_title(f'On-Axis Magnetic Field Components\n(r=0)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add text with coil parameters
info_text = f"Coil Parameters:\n"
info_text += f"N = {coil_turns} turns\n"
info_text += f"I = {coil_current} A\n"
info_text += f"R = {coil_radius*1000:.1f} mm\n"
info_text += f"L = {coil_length*1000:.1f} mm\n"
info_text += f"f = {rf_frequency/1e6:.2f} MHz\n\n"
info_text += f"Turn density: {coil.n:.1f} turns/m\n"
info_text += f"Max |B|: {np.max(B_mag)*1e6:.2f} µT"

ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('rf_coil_visualization.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: rf_coil_visualization.png")
print(f"Maximum B-field: {np.max(B_mag)*1e6:.2f} µT")
print(f"B-field on axis at center: {Bz_axis[len(Bz_axis)//2]*1e6:.2f} µT")

plt.show()
