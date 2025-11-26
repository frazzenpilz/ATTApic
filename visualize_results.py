"""
Visualization Script for PIC-IPD Simulation Results

This script provides visualization capabilities for:
1. Electron density in space (r, z)
2. Ion density in space (r, z)
3. Neutral density in space (r, z)
4. EEDF (Electron Energy Distribution Function) for given time and space

Usage:
    python visualize_results.py --help
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse
import os
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# Physical constants
ELECTRON_CHARGE = -1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31   # kg


class PICVisualizer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.data = None
        self.current_step = None
        self.time = None
        
    def load_data(self, step):
        """Load HDF5 data for a specific timestep"""
        filename = f"{self.output_dir}/results_{step:05d}.h5"
        if not os.path.exists(filename):
            print(f"Error: File {filename} not found!")
            return False
            
        self.data = {}
        with h5py.File(filename, 'r') as f:
            # Load metadata
            self.time = f.attrs['time']
            self.current_step = f.attrs['step']
            
            # Load node data
            self.data['nodes'] = {
                'x': f['nodes/x'][:],
                'y': f['nodes/y'][:],
                'rho': f['nodes/rho'][:],
                'phi': f['nodes/phi'][:],
                'ex': f['nodes/ex'][:],
                'ey': f['nodes/ey'][:],
                'ez': f['nodes/ez'][:],
            }
            
            # Load particle data
            self.data['particles'] = {
                'x': f['particles/x'][:],
                'y': f['particles/y'][:],
                'z': f['particles/z'][:],
                'vx': f['particles/vx'][:],
                'vy': f['particles/vy'][:],
                'vz': f['particles/vz'][:],
                'mass': f['particles/mass'][:],
                'charge': f['particles/charge'][:],
                'weight': f['particles/weight'][:],
                'species': f['particles/species'][:],
            }
            
        print(f"Loaded data for step {step}, time = {self.time:.3e} s")
        return True
    
    def calculate_densities_on_grid(self, nx=100, ny=100):
        """Calculate electron, ion, and neutral densities on a structured grid"""
        particles = self.data['particles']
        
        # Get spatial bounds
        x_min, x_max = particles['x'].min(), particles['x'].max()
        y_min, y_max = particles['y'].min(), particles['y'].max()
        
        # Create grid (in cylindrical: x=r, y=z)
        r = np.linspace(x_min, x_max, nx)
        z = np.linspace(y_min, y_max, ny)
        R, Z = np.meshgrid(r, z)
        
        # Separate particles by species
        # species: -1 = electron, 0 = neutral, 1 = ion
        electron_mask = particles['species'] == -1
        ion_mask = particles['species'] == 1
        neutral_mask = particles['species'] == 0
        
        # Calculate densities by binning particles weighted by their weight
        def calc_density_grid(mask):
            if not np.any(mask):
                return np.zeros_like(R)
                
            x_p = particles['x'][mask]
            y_p = particles['y'][mask]
            w_p = particles['weight'][mask]
            
            # Create 2D histogram (density map)
            H, xedges, yedges = np.histogram2d(
                x_p, y_p, 
                bins=[nx, ny],
                range=[[x_min, x_max], [y_min, y_max]],
                weights=w_p
            )
            
            # Normalize by cell volume (in cylindrical coords: dV = 2π r dr dz)
            dr = (x_max - x_min) / nx
            dz = (y_max - y_min) / ny
            
            # Cell centers for r
            r_centers = (xedges[:-1] + xedges[1:]) / 2
            
            # Volume of each cell: dV = 2π * r * dr * dz
            volumes = 2 * np.pi * r_centers[:, np.newaxis] * dr * dz
            volumes = np.maximum(volumes, 1e-30)  # Avoid division by zero
            
            # Density = number / volume
            density = H / volumes
            
            return density.T  # Transpose to match meshgrid convention
        
        electron_density = calc_density_grid(electron_mask)
        ion_density = calc_density_grid(ion_mask)
        neutral_density = calc_density_grid(neutral_mask)
        
        return R, Z, electron_density, ion_density, neutral_density
    
    def plot_densities(self, save_fig=True, show_fig=True, nx=40, ny=10):
        """Plot electron, ion, and neutral densities in space
        
        Parameters:
        -----------
        nx, ny : int
            Grid resolution (default: 40x10 for smooth plots with ~600 particles)
            Increase for finer detail (grainier), decrease for smoother plots
        """
        if self.data is None:
            print("No data loaded! Call load_data() first.")
            return
        
        R, Z, n_e, n_i, n_n = self.calculate_densities_on_grid(nx=nx, ny=ny)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Convert to m^-3 for better readability
        vmin = 1e10
        
        # Electron density
        if np.max(n_e) > 0:
            im1 = axes[0].pcolormesh(R*1000, Z*1000, n_e, 
                                     cmap='plasma', 
                                     norm=colors.LogNorm(vmin=vmin, vmax=np.max(n_e)))
            axes[0].set_title(f'Electron Density (t={self.time*1e9:.2f} ns)')
            axes[0].set_xlabel('r (mm)')
            axes[0].set_ylabel('z (mm)')
            plt.colorbar(im1, ax=axes[0], label='Density (m$^{-3}$)')
        else:
            axes[0].text(0.5, 0.5, 'No electrons', ha='center', va='center')
            axes[0].set_title('Electron Density')
        
        # Ion density
        if np.max(n_i) > 0:
            im2 = axes[1].pcolormesh(R*1000, Z*1000, n_i, 
                                     cmap='viridis', 
                                     norm=colors.LogNorm(vmin=vmin, vmax=np.max(n_i)))
            axes[1].set_title(f'Ion Density (t={self.time*1e9:.2f} ns)')
            axes[1].set_xlabel('r (mm)')
            axes[1].set_ylabel('z (mm)')
            plt.colorbar(im2, ax=axes[1], label='Density (m$^{-3}$)')
        else:
            axes[1].text(0.5, 0.5, 'No ions', ha='center', va='center')
            axes[1].set_title('Ion Density')
        
        # Neutral density
        if np.max(n_n) > 0:
            im3 = axes[2].pcolormesh(R*1000, Z*1000, n_n, 
                                     cmap='cividis', 
                                     norm=colors.LogNorm(vmin=vmin, vmax=np.max(n_n)))
            axes[2].set_title(f'Neutral Density (t={self.time*1e9:.2f} ns)')
            axes[2].set_xlabel('r (mm)')
            axes[2].set_ylabel('z (mm)')
            plt.colorbar(im3, ax=axes[2], label='Density (m$^{-3}$)')
        else:
            axes[2].text(0.5, 0.5, 'No neutrals', ha='center', va='center')
            axes[2].set_title('Neutral Density')
        
        plt.tight_layout()
        
        if save_fig:
            fig_name = f'{self.output_dir}/density_plots_step_{self.current_step:05d}.png'
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {fig_name}")
        
        if show_fig:
            plt.show()
        else:
            plt.close()
    
    def calculate_eedf(self, r_range=None, z_range=None, energy_bins=None):
        """
        Calculate Electron Energy Distribution Function (EEDF)
        
        Parameters:
        -----------
        r_range : tuple, optional
            (r_min, r_max) in meters to filter particles spatially
        z_range : tuple, optional
            (z_min, z_max) in meters to filter particles spatially
        energy_bins : int or array, optional
            Number of bins or bin edges for energy histogram (default: 200 bins, 0-100 eV)
        
        Returns:
        --------
        energy_centers : array
            Energy bin centers in eV
        eedf : array
            Electron energy distribution function (normalized probability)
        """
        if self.data is None:
            print("No data loaded! Call load_data() first.")
            return None, None
        
        particles = self.data['particles']
        
        # Filter electrons
        electron_mask = particles['species'] == -1
        
        if not np.any(electron_mask):
            print("No electrons found!")
            return None, None
        
        # Apply spatial filters
        if r_range is not None:
            r_mask = (particles['x'] >= r_range[0]) & (particles['x'] <= r_range[1])
            electron_mask = electron_mask & r_mask
        
        if z_range is not None:
            z_mask = (particles['y'] >= z_range[0]) & (particles['y'] <= z_range[1])
            electron_mask = electron_mask & z_mask
        
        # Get electron data
        vx = particles['vx'][electron_mask]
        vy = particles['vy'][electron_mask]
        vz = particles['vz'][electron_mask]
        mass = particles['mass'][electron_mask]
        weight = particles['weight'][electron_mask]
        
        # Calculate kinetic energy per real particle (not macroparticle)
        v2 = vx**2 + vy**2 + vz**2
        mass_real = mass / weight  # Real particle mass
        energy_joules = 0.5 * mass_real * v2
        energy_eV = energy_joules / abs(ELECTRON_CHARGE)
        
        # Create energy bins
        if energy_bins is None:
            energy_bins = np.linspace(0, 100, 201)  # 0-100 eV, 200 bins
        elif isinstance(energy_bins, int):
            max_energy = np.max(energy_eV) if len(energy_eV) > 0 else 100
            energy_bins = np.linspace(0, max_energy, energy_bins + 1)
        
        # Calculate weighted histogram (EEDF)
        hist, bin_edges = np.histogram(energy_eV, bins=energy_bins, weights=weight)
        
        # Normalize to get probability distribution
        total = np.sum(hist)
        eedf = hist / total if total > 0 else hist
        
        # Energy bin centers
        energy_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        return energy_centers, eedf
    
    def plot_eedf(self, r_range=None, z_range=None, save_fig=True, show_fig=True):
        """Plot EEDF (Electron Energy Distribution Function)"""
        if self.data is None:
            print("No data loaded! Call load_data() first.")
            return
        
        energy, eedf = self.calculate_eedf(r_range=r_range, z_range=z_range)
        
        if energy is None:
            return
        
        # Also load and plot global EEDF from file if it exists
        eepf_file = f"{self.output_dir}/eepf.dat"
        global_energy = None
        global_eedf = None
        
        if os.path.exists(eepf_file):
            data = np.loadtxt(eepf_file)
            global_energy = data[:, 0]
            global_eedf = data[:, 2]  # Probability column
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot current timestep EEDF
        spatial_label = ""
        if r_range is not None or z_range is not None:
            spatial_label = " (spatial filter)"
            if r_range:
                spatial_label += f" r=[{r_range[0]*1000:.1f},{r_range[1]*1000:.1f}]mm"
            if z_range:
                spatial_label += f" z=[{z_range[0]*1000:.1f},{z_range[1]*1000:.1f}]mm"
        
        ax.semilogy(energy, eedf, 'b-', linewidth=2, 
                    label=f'Current (t={self.time*1e9:.2f} ns){spatial_label}')
        
        # Plot global accumulated EEDF if available
        if global_energy is not None:
            ax.semilogy(global_energy, global_eedf, 'r--', linewidth=1.5, 
                       alpha=0.7, label='Time-averaged (all steps)')
        
        ax.set_xlabel('Energy (eV)', fontsize=12)
        ax.set_ylabel('Probability (normalized)', fontsize=12)
        ax.set_title(f'Electron Energy Distribution Function\nStep {self.current_step}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        
        if save_fig:
            fig_name = f'{self.output_dir}/eedf_step_{self.current_step:05d}.png'
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {fig_name}")
        
        if show_fig:
            plt.show()
        else:
            plt.close()
    
    def plot_spatial_eedf_map(self, nr=5, nz=5, energy_max=50, save_fig=True, show_fig=True):
        """
        Plot spatial variation of EEDF across the simulation domain
        Divides domain into grid and shows EEDF for each region
        """
        if self.data is None:
            print("No data loaded! Call load_data() first.")
            return
        
        particles = self.data['particles']
        electron_mask = particles['species'] == -1
        
        if not np.any(electron_mask):
            print("No electrons found!")
            return
        
        # Get spatial bounds
        r_min, r_max = particles['x'][electron_mask].min(), particles['x'][electron_mask].max()
        z_min, z_max = particles['y'][electron_mask].min(), particles['y'][electron_mask].max()
        
        # Create spatial bins
        r_bins = np.linspace(r_min, r_max, nr + 1)
        z_bins = np.linspace(z_min, z_max, nz + 1)
        
        fig, axes = plt.subplots(nz, nr, figsize=(3*nr, 3*nz))
        if nr == 1 and nz == 1:
            axes = np.array([[axes]])
        elif nr == 1:
            axes = axes.reshape(-1, 1)
        elif nz == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(nz):
            for j in range(nr):
                z_range = (z_bins[nz-1-i], z_bins[nz-i])  # Flip z for plotting
                r_range = (r_bins[j], r_bins[j+1])
                
                energy, eedf = self.calculate_eedf(r_range=r_range, z_range=z_range, 
                                                   energy_bins=np.linspace(0, energy_max, 51))
                
                ax = axes[i, j]
                if energy is not None and np.sum(eedf) > 0:
                    ax.semilogy(energy, eedf, 'b-', linewidth=1)
                    ax.set_ylim(1e-5, 1)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_xlim(0, energy_max)
                ax.grid(True, alpha=0.3)
                
                # Labels only on edges
                if i == nz - 1:
                    ax.set_xlabel('E (eV)', fontsize=8)
                if j == 0:
                    ax.set_ylabel('EEDF', fontsize=8)
                
                # Title showing region
                ax.set_title(f'r=[{r_range[0]*1000:.1f},{r_range[1]*1000:.1f}]mm\n'
                           f'z=[{z_range[0]*1000:.1f},{z_range[1]*1000:.1f}]mm', 
                           fontsize=7)
        
        plt.suptitle(f'Spatial EEDF Map (t={self.time*1e9:.2f} ns)', fontsize=14, y=0.995)
        plt.tight_layout()
        
        if save_fig:
            fig_name = f'{self.output_dir}/eedf_spatial_map_step_{self.current_step:05d}.png'
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {fig_name}")
        
        if show_fig:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize PIC-IPD simulation results')
    parser.add_argument('--step', type=int, default=7400, 
                       help='Time step to visualize (default: 7400)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory containing HDF5 files (default: output)')
    parser.add_argument('--densities', action='store_true', default=True,
                       help='Plot density distributions (default: True)')
    parser.add_argument('--eedf', action='store_true', default=True,
                       help='Plot EEDF (default: True)')
    parser.add_argument('--eedf-spatial', action='store_true',
                       help='Plot spatial EEDF map (default: False)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save)')
    parser.add_argument('--r-min', type=float, default=None,
                       help='Minimum r for EEDF spatial filter (meters)')
    parser.add_argument('--r-max', type=float, default=None,
                       help='Maximum r for EEDF spatial filter (meters)')
    parser.add_argument('--z-min', type=float, default=None,
                       help='Minimum z for EEDF spatial filter (meters)')
    parser.add_argument('--z-max', type=float, default=None,
                       help='Maximum z for EEDF spatial filter (meters)')
    
    args = parser.parse_args()
    
    # Create visualizer
    vis = PICVisualizer(output_dir=args.output_dir)
    
    # Load data
    if not vis.load_data(args.step):
        return
    
    # Prepare spatial filters
    r_range = None
    z_range = None
    if args.r_min is not None or args.r_max is not None:
        r_range = (args.r_min or 0, args.r_max or 1e10)
    if args.z_min is not None or args.z_max is not None:
        z_range = (args.z_min or -1e10, args.z_max or 1e10)
    
    show_fig = not args.no_show
    
    # Plot densities
    if args.densities:
        print("\nGenerating density plots...")
        vis.plot_densities(save_fig=True, show_fig=show_fig)
    
    # Plot EEDF
    if args.eedf:
        print("\nGenerating EEDF plot...")
        vis.plot_eedf(r_range=r_range, z_range=z_range, save_fig=True, show_fig=show_fig)
    
    # Plot spatial EEDF map
    if args.eedf_spatial:
        print("\nGenerating spatial EEDF map...")
        vis.plot_spatial_eedf_map(nr=4, nz=4, save_fig=True, show_fig=show_fig)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
