"""
Live Monitor for PIC-IPD Simulation
Reads output/history.dat in real-time and displays updating plots
Run in a separate terminal while simulation is running
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import time

class LiveMonitor:
    def __init__(self, history_file='output/history.dat', update_interval=2000):
        """
        Initialize live monitor
        
        Args:
            history_file: Path to history.dat file
            update_interval: Update interval in milliseconds (default 2000ms = 2s)
        """
        self.history_file = history_file
        self.update_interval = update_interval
        
        # Create figure with subplots
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('PIC-IPD Simulation - Live Monitor', fontsize=14, fontweight='bold')
        
        # Initialize empty lines
        self.line_energy, = self.ax1.plot([], [], 'b-', linewidth=2, label='Avg Electron Energy')
        self.line_electrons, = self.ax2.plot([], [], 'r-', linewidth=2, label='Electrons')
        self.line_ions, = self.ax2.plot([], [], 'g-', linewidth=2, label='Ions')
        
        # Configure axes
        self.ax1.set_xlabel('Time (ns)')
        self.ax1.set_ylabel('Average Electron Energy (eV)')
        self.ax1.set_title('Electron Heating')
        self.ax1.grid(True)
        self.ax1.legend(loc='upper left')
        
        self.ax2.set_xlabel('Time (ns)')
        self.ax2.set_ylabel('Particle Count')
        self.ax2.set_title('Particle Counts')
        self.ax2.grid(True)
        self.ax2.legend(loc='upper left')
        
        plt.tight_layout()
        
        # Last file size to detect updates
        self.last_size = 0
        
    def read_data(self):
        """Read and parse history.dat file"""
        if not os.path.exists(self.history_file):
            return None
        
        # Check if file has been updated
        current_size = os.path.getsize(self.history_file)
        if current_size == self.last_size and self.last_size > 0:
            return None  # No new data
        self.last_size = current_size
        
        try:
            # Read data
            df = pd.read_csv(self.history_file, sep='\s+', comment='#',
                           names=['Time', 'N_e', 'N_i', 'Total_Energy_eV'])
            
            # Calculate average energy
            df['Avg_Energy_eV'] = df['Total_Energy_eV'] / df['N_e']
            df['Time_ns'] = df['Time'] * 1e9  # Convert to nanoseconds
            
            return df
        except Exception as e:
            print(f"Error reading data: {e}")
            return None
    
    def update(self, frame):
        """Update plot with new data"""
        df = self.read_data()
        
        if df is None or len(df) == 0:
            return self.line_energy, self.line_electrons, self.line_ions
        
        # Update energy plot
        self.line_energy.set_data(df['Time_ns'], df['Avg_Energy_eV'])
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update particle count plot
        self.line_electrons.set_data(df['Time_ns'], df['N_e'])
        self.line_ions.set_data(df['Time_ns'], df['N_i'])
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Update title with latest values
        if len(df) > 0:
            latest = df.iloc[-1]
            self.fig.suptitle(
                f'PIC-IPD Simulation - Live Monitor | '
                f'Step {len(df)} | Time: {latest["Time_ns"]:.2f} ns | '
                f'Avg E: {latest["Avg_Energy_eV"]:.1f} eV | '
                f'N_e: {int(latest["N_e"])}',
                fontsize=12, fontweight='bold'
            )
        
        return self.line_energy, self.line_electrons, self.line_ions
    
    def start(self):
        """Start the live monitor"""
        print(f"Starting live monitor...")
        print(f"Waiting for {self.history_file}...")
        
        # Wait for file to exist
        while not os.path.exists(self.history_file):
            time.sleep(1)
        
        print(f"File found! Monitoring with {self.update_interval/1000}s refresh rate...")
        print("Press Ctrl+C to stop")
        
        # Create animation
        anim = FuncAnimation(
            self.fig, self.update,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False
        )
        
        plt.show()

if __name__ == "__main__":
    import sys
    
    # Optional: specify custom history file path
    history_file = sys.argv[1] if len(sys.argv) > 1 else 'output/history.dat'
    
    # Create and start monitor
    monitor = LiveMonitor(history_file=history_file, update_interval=2000)
    
    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
        plt.close('all')
