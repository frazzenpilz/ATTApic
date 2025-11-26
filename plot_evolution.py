"""
Plasma Evolution Visualization Script

This script visualizes the temporal evolution of the plasma using the diagnostic
files generated during the simulation (history.dat and eepf.dat).

It shows:
1. Electron and ion population evolution over time
2. Average electron energy evolution
3. Total electron energy evolution
4. Time-averaged EEDF
5. RF cycle analysis (if applicable)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import os

# Physical constants
ELECTRON_CHARGE = 1.602176634e-19  # C (magnitude)
RF_FREQUENCY = 13.56e6  # Hz (typical ICP frequency)
RF_PERIOD = 1.0 / RF_FREQUENCY  # seconds


def load_history_data(filename="output/history.dat"):
    """Load history.dat file"""
    try:
        data = np.loadtxt(filename)
        history = {
            'time': data[:, 0],
            'num_electrons': data[:, 1],
            'num_ions': data[:, 2],
            'total_energy_eV': data[:, 3]
        }
        print(f"Loaded {len(history['time'])} timesteps from {filename}")
        return history
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def load_eepf_data(filename="output/eepf.dat"):
    """Load eepf.dat file"""
    try:
        data = np.loadtxt(filename)
        eepf = {
            'energy_eV': data[:, 0],
            'counts': data[:, 1],
            'probability': data[:, 2]
        }
        print(f"Loaded EEPF data with {len(eepf['energy_eV'])} energy bins from {filename}")
        return eepf
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def plot_plasma_evolution(history, save_fig=True, show_fig=True, output_dir="output"):
    """
    Create comprehensive plasma evolution plots
    """
    if history is None:
        print("No history data available!")
        return
    
    # Convert time to more convenient units
    time_ns = history['time'] * 1e9  # Convert to nanoseconds
    time_us = history['time'] * 1e6  # Convert to microseconds
    
    # Calculate derived quantities
    avg_energy_eV = history['total_energy_eV'] / np.maximum(history['num_electrons'], 1)
    
    # Determine time scale (use ns if < 1 us, otherwise use us)
    max_time = np.max(history['time'])
    if max_time < 1e-6:
        time_plot = time_ns
        time_label = 'Time (ns)'
    else:
        time_plot = time_us
        time_label = 'Time (μs)'
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Particle populations
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_plot, history['num_electrons'], 'b-', linewidth=2, label='Electrons')
    ax1.plot(time_plot, history['num_ions'], 'r-', linewidth=2, label='Ions')
    ax1.set_xlabel(time_label, fontsize=12)
    ax1.set_ylabel('Number of Particles', fontsize=12)
    ax1.set_title('Plasma Particle Population Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 2. Average electron energy
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_plot, avg_energy_eV, 'g-', linewidth=2)
    ax2.set_xlabel(time_label, fontsize=12)
    ax2.set_ylabel('Average Energy (eV)', fontsize=12)
    ax2.set_title('Average Electron Energy Evolution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=np.mean(avg_energy_eV), color='k', linestyle='--', alpha=0.5, 
                label=f'Mean: {np.mean(avg_energy_eV):.2f} eV')
    ax2.legend(fontsize=10)
    
    # 3. Total electron energy
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_plot, history['total_energy_eV'], 'purple', linewidth=2)
    ax3.set_xlabel(time_label, fontsize=12)
    ax3.set_ylabel('Total Energy (eV)', fontsize=12)
    ax3.set_title('Total Electron Energy Evolution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 4. Electron-to-Ion ratio
    ax4 = fig.add_subplot(gs[2, 0])
    ratio = history['num_electrons'] / np.maximum(history['num_ions'], 1)
    ax4.plot(time_plot, ratio, 'orange', linewidth=2)
    ax4.set_xlabel(time_label, fontsize=12)
    ax4.set_ylabel('Electron/Ion Ratio', fontsize=12)
    ax4.set_title('Charge Balance Evolution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect balance')
    ax4.legend(fontsize=10)
    
    # 5. Power deposition rate (derivative of total energy)
    ax5 = fig.add_subplot(gs[2, 1])
    if len(time_plot) > 1:
        dt = np.diff(history['time'])
        dE = np.diff(history['total_energy_eV'])
        power_eV_per_s = dE / dt
        power_plot = power_eV_per_s * ELECTRON_CHARGE  # Convert to Watts
        time_plot_mid = (time_plot[:-1] + time_plot[1:]) / 2
        
        ax5.plot(time_plot_mid, power_plot, 'm-', linewidth=1, alpha=0.7)
        ax5.set_xlabel(time_label, fontsize=12)
        ax5.set_ylabel('Power (W)', fontsize=12)
        ax5.set_title('Power Deposition Rate', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Show moving average for clarity
        if len(power_plot) > 100:
            window = min(100, len(power_plot) // 10)
            moving_avg = np.convolve(power_plot, np.ones(window)/window, mode='valid')
            time_avg = time_plot_mid[:len(moving_avg)]
            ax5.plot(time_avg, moving_avg, 'r-', linewidth=2, 
                    label=f'Moving avg (N={window})')
            ax5.legend(fontsize=10)
    
    plt.suptitle('Plasma Evolution Overview', fontsize=16, fontweight='bold', y=0.995)
    
    if save_fig:
        filename = os.path.join(output_dir, 'plasma_evolution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_rf_cycle_analysis(history, save_fig=True, show_fig=True, output_dir="output"):
    """
    Analyze plasma behavior over RF cycles
    """
    if history is None:
        print("No history data available!")
        return
    
    time = history['time']
    
    # Check if we have enough data for RF cycle analysis
    if np.max(time) < 2 * RF_PERIOD:
        print("Not enough data for RF cycle analysis (need at least 2 RF periods)")
        return
    
    # Find indices corresponding to last few RF cycles
    num_cycles_to_plot = min(5, int(np.max(time) / RF_PERIOD))
    time_start = np.max(time) - num_cycles_to_plot * RF_PERIOD
    idx_start = np.argmin(np.abs(time - time_start))
    
    time_cycle = time[idx_start:] - time[idx_start]
    time_cycle_ns = time_cycle * 1e9
    
    # Phase within RF cycle
    phase = (time_cycle / RF_PERIOD) % 1.0
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Particle numbers vs RF phase
    ax = axes[0, 0]
    ax.plot(time_cycle_ns, history['num_electrons'][idx_start:], 'b-', linewidth=2, label='Electrons')
    ax.plot(time_cycle_ns, history['num_ions'][idx_start:], 'r-', linewidth=2, label='Ions')
    ax.set_xlabel('Time (ns)', fontsize=11)
    ax.set_ylabel('Number of Particles', fontsize=11)
    ax.set_title(f'Particle Count Over {num_cycles_to_plot} RF Cycles', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Mark RF cycle boundaries
    for i in range(num_cycles_to_plot + 1):
        ax.axvline(x=i * RF_PERIOD * 1e9, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Average energy vs RF phase
    ax = axes[0, 1]
    avg_energy = history['total_energy_eV'][idx_start:] / np.maximum(history['num_electrons'][idx_start:], 1)
    ax.plot(time_cycle_ns, avg_energy, 'g-', linewidth=2)
    ax.set_xlabel('Time (ns)', fontsize=11)
    ax.set_ylabel('Average Electron Energy (eV)', fontsize=11)
    ax.set_title(f'Electron Energy Over {num_cycles_to_plot} RF Cycles', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for i in range(num_cycles_to_plot + 1):
        ax.axvline(x=i * RF_PERIOD * 1e9, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Phase-averaged particle count
    ax = axes[1, 0]
    phase_bins = np.linspace(0, 1, 51)
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    
    e_binned = []
    i_binned = []
    for i in range(len(phase_bins) - 1):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
        if np.any(mask):
            e_binned.append(np.mean(history['num_electrons'][idx_start:][mask]))
            i_binned.append(np.mean(history['num_ions'][idx_start:][mask]))
        else:
            e_binned.append(np.nan)
            i_binned.append(np.nan)
    
    ax.plot(phase_centers, e_binned, 'b-o', linewidth=2, markersize=4, label='Electrons')
    ax.plot(phase_centers, i_binned, 'r-o', linewidth=2, markersize=4, label='Ions')
    ax.set_xlabel('RF Phase (0-1)', fontsize=11)
    ax.set_ylabel('Average Particle Count', fontsize=11)
    ax.set_title('Phase-Averaged Particle Count', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Phase-averaged energy
    ax = axes[1, 1]
    energy_binned = []
    for i in range(len(phase_bins) - 1):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
        if np.any(mask):
            energy_binned.append(np.mean(avg_energy[mask]))
        else:
            energy_binned.append(np.nan)
    
    ax.plot(phase_centers, energy_binned, 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('RF Phase (0-1)', fontsize=11)
    ax.set_ylabel('Average Electron Energy (eV)', fontsize=11)
    ax.set_title('Phase-Averaged Electron Energy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'RF Cycle Analysis (f = {RF_FREQUENCY/1e6:.2f} MHz, T = {RF_PERIOD*1e9:.2f} ns)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_fig:
        filename = os.path.join(output_dir, 'rf_cycle_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_eepf(eepf, save_fig=True, show_fig=True, output_dir="output"):
    """
    Plot the time-averaged EEPF
    """
    if eepf is None:
        print("No EEPF data available!")
        return
    
    # Filter to non-zero probabilities
    mask = eepf['probability'] > 0
    if not np.any(mask):
        print("No non-zero EEPF data!")
        return
    
    energy = eepf['energy_eV'][mask]
    prob = eepf['probability'][mask]
    
    # Calculate mean energy
    mean_energy = np.sum(energy * prob * np.diff(eepf['energy_eV'][:len(energy)+1])[0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    ax1.plot(energy, prob, 'b-', linewidth=2)
    ax1.set_xlabel('Energy (eV)', fontsize=12)
    ax1.set_ylabel('Probability (normalized)', fontsize=12)
    ax1.set_title('Time-Averaged EEDF (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=mean_energy, color='r', linestyle='--', linewidth=2,
                label=f'Mean: {mean_energy:.2f} eV')
    ax1.legend(fontsize=11)
    ax1.set_xlim(left=0)
    
    # Log scale
    ax2.semilogy(energy, prob, 'b-', linewidth=2)
    ax2.set_xlabel('Energy (eV)', fontsize=12)
    ax2.set_ylabel('Probability (normalized)', fontsize=12)
    ax2.set_title('Time-Averaged EEDF (Log Scale)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.axvline(x=mean_energy, color='r', linestyle='--', linewidth=2,
                label=f'Mean: {mean_energy:.2f} eV')
    ax2.legend(fontsize=11)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=1e-6)
    
    plt.suptitle('Electron Energy Distribution Function',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_fig:
        filename = os.path.join(output_dir, 'eepf_time_averaged.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_all_in_one(history, eepf, save_fig=True, show_fig=True, output_dir="output"):
    """
    Create a comprehensive single-page summary
    """
    if history is None:
        print("No history data available!")
        return
    
    # Determine time scale
    max_time = np.max(history['time'])
    if max_time < 1e-6:
        time_plot = history['time'] * 1e9
        time_label = 'Time (ns)'
    else:
        time_plot = history['time'] * 1e6
        time_label = 'Time (μs)'
    
    avg_energy_eV = history['total_energy_eV'] / np.maximum(history['num_electrons'], 1)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # 1. Particle populations
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(time_plot, history['num_electrons'], 'b-', linewidth=2, label='Electrons')
    ax1.plot(time_plot, history['num_ions'], 'r-', linewidth=2, label='Ions')
    ax1.set_xlabel(time_label, fontsize=11)
    ax1.set_ylabel('Particle Count', fontsize=11)
    ax1.set_title('Particle Population Evolution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 2. Statistics box
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis('off')
    stats_text = f"""
    Simulation Statistics
    ───────────────────
    Total Time: {max_time*1e9:.2f} ns
    Final e⁻ count: {history['num_electrons'][-1]:.2e}
    Final ion count: {history['num_ions'][-1]:.2e}
    
    Average Energy:
      Mean: {np.mean(avg_energy_eV):.2f} eV
      Std: {np.std(avg_energy_eV):.2f} eV
      Min: {np.min(avg_energy_eV):.2f} eV
      Max: {np.max(avg_energy_eV):.2f} eV
    
    Net Ionization:
      Δe⁻: {history['num_electrons'][-1] - history['num_electrons'][0]:.0f}
      Δion: {history['num_ions'][-1] - history['num_ions'][0]:.0f}
    """
    ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 3. Average energy
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_plot, avg_energy_eV, 'g-', linewidth=2)
    ax3.set_xlabel(time_label, fontsize=11)
    ax3.set_ylabel('Avg Energy (eV)', fontsize=11)
    ax3.set_title('Electron Energy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Total energy
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_plot, history['total_energy_eV'], 'purple', linewidth=2)
    ax4.set_xlabel(time_label, fontsize=11)
    ax4.set_ylabel('Total Energy (eV)', fontsize=11)
    ax4.set_title('Total Electron Energy', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 5. Charge balance
    ax5 = fig.add_subplot(gs[1, 2])
    ratio = history['num_electrons'] / np.maximum(history['num_ions'], 1)
    ax5.plot(time_plot, ratio, 'orange', linewidth=2)
    ax5.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel(time_label, fontsize=11)
    ax5.set_ylabel('e⁻/ion Ratio', fontsize=11)
    ax5.set_title('Charge Balance', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. EEPF if available
    if eepf is not None:
        ax6 = fig.add_subplot(gs[2, :])
        mask = eepf['probability'] > 0
        if np.any(mask):
            energy = eepf['energy_eV'][mask]
            prob = eepf['probability'][mask]
            ax6.semilogy(energy, prob, 'b-', linewidth=2)
            ax6.set_xlabel('Energy (eV)', fontsize=11)
            ax6.set_ylabel('Probability', fontsize=11)
            ax6.set_title('Time-Averaged EEDF', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, which='both')
            ax6.set_xlim(left=0, right=min(20, np.max(energy)))
            ax6.set_ylim(bottom=1e-5)
    else:
        ax6 = fig.add_subplot(gs[2, :])
        ax6.text(0.5, 0.5, 'EEPF data not available', ha='center', va='center',
                transform=ax6.transAxes, fontsize=14)
        ax6.axis('off')
    
    plt.suptitle('Plasma Evolution Summary', fontsize=16, fontweight='bold', y=0.995)
    
    if save_fig:
        filename = os.path.join(output_dir, 'plasma_evolution_summary.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    if show_fig:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize plasma evolution from diagnostic files')
    parser.add_argument('--history', type=str, default='output/history.dat',
                       help='Path to history.dat file')
    parser.add_argument('--eepf', type=str, default='output/eepf.dat',
                       help='Path to eepf.dat file')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for plots')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save)')
    parser.add_argument('--rf-analysis', action='store_true',
                       help='Perform RF cycle analysis')
    parser.add_argument('--all-in-one', action='store_true', default=True,
                       help='Create comprehensive single-page summary')
    
    args = parser.parse_args()
    
    # Load data
    print("\nLoading diagnostic data...")
    history = load_history_data(args.history)
    eepf = load_eepf_data(args.eepf)
    
    if history is None:
        print("ERROR: Could not load history data!")
        return
    
    show_plots = not args.no_show
    
    # Generate plots
    print("\nGenerating evolution plots...")
    plot_plasma_evolution(history, save_fig=True, show_fig=show_plots, output_dir=args.output_dir)
    
    if eepf is not None:
        print("\nGenerating EEPF plot...")
        plot_eepf(eepf, save_fig=True, show_fig=show_plots, output_dir=args.output_dir)
    
    if args.rf_analysis:
        print("\nPerforming RF cycle analysis...")
        plot_rf_cycle_analysis(history, save_fig=True, show_fig=show_plots, output_dir=args.output_dir)
    
    if args.all_in_one:
        print("\nGenerating summary plot...")
        plot_all_in_one(history, eepf, save_fig=True, show_fig=show_plots, output_dir=args.output_dir)
    
    print("\nVisualization complete!")
    print(f"Plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
