# PIC-IPD Visualization Guide

This guide explains how to use the `visualize_results.py` script to analyze and visualize the results from your PIC-IPD plasma simulation.

## Overview

The visualization script provides the following capabilities:

1. **Electron Density in Space** - 2D color map showing electron density distribution in cylindrical coordinates (r, z)
2. **Ion Density in Space** - 2D color map showing ion density distribution
3. **Neutral Density in Space** - 2D color map showing neutral particle density distribution
4. **EEDF (Electron Energy Distribution Function)** - Energy distribution of electrons for a given time and optional spatial region
5. **Spatial EEDF Map** - EEDF variation across different spatial regions

## Quick Start

### Basic Usage

To visualize results from the most recent timestep (7400):

```bash
python visualize_results.py --step 7400
```

This will generate and display:
- Density plots for electrons, ions, and neutrals
- EEDF plot

### Save Plots Without Displaying

```bash
python visualize_results.py --step 7400 --no-show
```

## Command Line Options

### Time Step Selection

```bash
--step STEP
```
Specify which timestep to visualize (default: 7400)

Example: `--step 5000` to visualize timestep 5000

### Output Directory

```bash
--output-dir DIR
```
Specify the directory containing HDF5 output files (default: `output`)

### Plot Types

```bash
--densities         # Plot density distributions (enabled by default)
--eedf             # Plot EEDF (enabled by default)
--eedf-spatial     # Plot spatial EEDF map (4x4 grid)
```

### Spatial Filtering for EEDF

You can calculate EEDF for a specific spatial region:

```bash
--r-min R_MIN      # Minimum radial position (meters)
--r-max R_MAX      # Maximum radial position (meters)
--z-min Z_MIN      # Minimum axial position (meters)
--z-max Z_MAX      # Maximum axial position (meters)
```

Example - EEDF only in central region:
```bash
python visualize_results.py --step 7400 --r-min 0 --r-max 0.01 --z-min 0.01 --z-max 0.03
```

### Display Control

```bash
--no-show          # Don't display plots, only save to files
```

## Examples

### 1. Analyze Final Timestep

```bash
python visualize_results.py --step 7400
```

### 2. Create All Plots and Save Without Display

```bash
python visualize_results.py --step 7400 --eedf-spatial --no-show
```

### 3. Analyze EEDF in Specific Spatial Region

Analyze electrons only in the radial range 0-10mm and axial range 10-30mm:

```bash
python visualize_results.py --step 7400 --r-min 0 --r-max 0.01 --z-min 0.01 --z-max 0.03
```

### 4. Generate Spatial EEDF Map

```bash
python visualize_results.py --step 7400 --eedf-spatial
```

This creates a 4×4 grid showing how the EEDF varies across different regions of the plasma.

### 5. Analyze Multiple Timesteps

Create a batch script to analyze multiple timesteps:

**Windows (PowerShell):**
```powershell
# analyze_all.ps1
for ($i=1000; $i -le 7400; $i+=1000) {
    python visualize_results.py --step $i --no-show
}
```

**Linux/Mac:**
```bash
# analyze_all.sh
for step in {1000..7400..1000}; do
    python visualize_results.py --step $step --no-show
done
```

## Output Files

The script generates the following files in the `output/` directory:

- `density_plots_step_XXXXX.png` - Electron, ion, and neutral density plots
- `eedf_step_XXXXX.png` - EEDF plot
- `eedf_spatial_map_step_XXXXX.png` - Spatial variation of EEDF (if --eedf-spatial is used)

Where `XXXXX` is the 5-digit timestep number.

## Understanding the Plots

### Density Plots

- **X-axis (r)**: Radial position in mm (cylindrical coordinates)
- **Y-axis (z)**: Axial position in mm
- **Color scale**: Particle density in m⁻³ (logarithmic scale)
- The plots show the spatial distribution of electrons, ions, and neutral particles

### EEDF Plot

- **X-axis**: Electron energy in eV
- **Y-axis**: Probability (normalized, logarithmic scale)
- **Blue solid line**: EEDF at the specified timestep
- **Red dashed line** (if shown): Time-averaged EEDF from all simulation steps
- Shows the energy distribution of electrons in the plasma

### Spatial EEDF Map

- A grid of EEDF plots showing how the electron energy distribution varies across space
- Each subplot shows the EEDF for a specific spatial region
- Useful for identifying spatial variations in electron heating

## Data Structure

The script reads HDF5 files with the following structure:

```
results_XXXXX.h5
├── /nodes/              # Mesh node data
│   ├── x, y            # Spatial coordinates (r, z)
│   ├── rho             # Charge density
│   ├── phi             # Electrostatic potential
│   └── ex, ey, ez      # Electric field components
│
└── /particles/          # Particle data
    ├── x, y, z         # Particle positions
    ├── vx, vy, vz      # Particle velocities
    ├── mass            # Particle mass
    ├── charge          # Particle charge
    ├── weight          # Macroparticle weight
    └── species         # Species type (-1: electron, 0: neutral, 1: ion)
```

## Python API Usage

You can also use the visualizer programmatically:

```python
from visualize_results import PICVisualizer

# Create visualizer
vis = PICVisualizer(output_dir="output")

# Load data for timestep 7400
vis.load_data(7400)

# Generate plots
vis.plot_densities(save_fig=True, show_fig=True)
vis.plot_eedf(save_fig=True, show_fig=True)

# Calculate EEDF with spatial filtering
r_range = (0, 0.01)  # 0-10 mm radially
z_range = (0.01, 0.03)  # 10-30 mm axially
vis.plot_eedf(r_range=r_range, z_range=z_range)

# Create spatial EEDF map
vis.plot_spatial_eedf_map(nr=4, nz=4)
```

## Troubleshooting

### "File not found" error

Make sure:
1. The HDF5 files exist in the `output/` directory
2. You're specifying the correct timestep number
3. You're running the script from the correct directory

### "No electrons found" warning

This can happen if:
1. The timestep is very early in the simulation
2. The spatial filter is too restrictive
3. All particles have been absorbed

### Memory issues with large datasets

If you have memory issues with large HDF5 files:
- Use the `--no-show` flag to avoid keeping plots in memory
- Reduce the grid resolution in the code (`nx`, `ny` parameters)
- Process timesteps one at a time rather than in batch

## Requirements

The script requires the following Python packages:
- `h5py` - HDF5 file reading
- `numpy` - Numerical computations
- `matplotlib` - Plotting
- `scipy` - Scientific computations (interpolation)

Install with:
```bash
pip install h5py numpy matplotlib scipy
```

## Tips and Best Practices

1. **Start with recent timesteps** - The plasma properties stabilize after several RF cycles
2. **Use --no-show for batch processing** - Faster when analyzing multiple timesteps
3. **Spatial filtering for EEDF** - Useful for comparing different plasma regions
4. **Check time-averaged EEDF** - The red dashed line shows accumulated statistics
5. **Export to other formats** - Modify the script to save as PDF or SVG for publications

## Contact

For questions or issues with the visualization script, please refer to the main PIC-IPD simulation documentation.
