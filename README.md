# PIC-IPD Simulation

A Particle-in-Cell (PIC) simulation for Inductively Coupled Plasma (ICP) using Python.

## Overview

This project implements a PIC simulation for modeling inductively coupled plasma dynamics. The simulation includes:

- **Particle dynamics**: Electron and ion particle tracking with Boris pusher
- **Field solver**: FDTD solver for electromagnetic fields in cylindrical coordinates (TE_θ mode)
- **Collision module**: Monte Carlo Collision (MCC) model with elastic, excitation, and ionization collisions
- **RF heating**: Time-harmonic RF coil source for inductive coupling
- **Diagnostics**: Real-time monitoring and comprehensive data output

## Features

- Axisymmetric 2D cylindrical geometry (r-z)
- Support for Argon plasma with cross-section data from LXCat
- Adaptive time-stepping with FDTD sub-cycling
- HDF5 output format for efficient data storage
- Visualization tools for results analysis

## Project Structure

- `main.py` - Main simulation entry point
- `pic_modules/` - Core simulation modules
  - `particles.py` - Particle management and Boris pusher
  - `fields.py` - Electric and magnetic field calculations
  - `fdtd.py` - FDTD electromagnetic solver
  - `mcc.py` - Monte Carlo Collision module
  - `rf_coil.py` - RF coil source implementation
  - `mesh.py` - Mesh generation and management
  - `solver.py` - Field solvers (Poisson, FDTD)
  - `diagnostics.py` - Diagnostic output and monitoring
- `lxcat_data/` - Cross-section data for collisions
- `visualize_*.py` - Visualization scripts
- `plot_evolution.py` - Evolution analysis tools

## Requirements

See `requirements.txt` for Python dependencies:
- numpy
- scipy
- matplotlib
- h5py (for HDF5 output)

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure simulation parameters in `inputs.txt`

3. Run the simulation:
   ```bash
   python main.py
   ```

4. Visualize results:
   ```bash
   python visualize_results.py
   python plot_evolution.py
   ```

## Configuration

Edit `inputs.txt` to adjust simulation parameters such as:
- Domain size and resolution
- Number of particles
- RF frequency and power
- Gas pressure and temperature
- Time step and simulation duration

## Output

Simulation results are saved to the `output/` directory:
- Particle data (positions, velocities, energies)
- Field data (E, B, J)
- Time history (density, energy, particle counts)
- Diagnostic plots

## References

- Robertz PhD Thesis (included in repository)
- LXCat database for collision cross-sections

## License

[Specify your license here]

## Author

Leona
