import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def analyze_heating(run_id):
    base_dir = Path("results") / run_id
    picdata_path = base_dir / "picdata.npz"
    
    if not picdata_path.exists():
        print(f"Error: Could not find {picdata_path}")
        return

    print(f"Loading data from {picdata_path}...")
    data = np.load(picdata_path)
    
    # Extract electron velocities
    vx = data['vx_e']
    vy = data['vy_e']
    vz = data['vz_e']
    N_e = int(data['N_e'])
    
    # Constants
    E_MASS = 9.10938356e-31
    E_CHARGE = 1.60217662e-19
    EV_TO_J = E_CHARGE
    
    # Calculate energies in eV
    v_sqr = vx[:N_e]**2 + vy[:N_e]**2 + vz[:N_e]**2
    energies_eV = 0.5 * E_MASS * v_sqr / EV_TO_J
    
    mean_energy = np.mean(energies_eV)
    max_energy = np.max(energies_eV)
    
    print(f"\n--- Analysis Results ---")
    print(f"Number of Electrons: {N_e}")
    print(f"Mean Electron Energy: {mean_energy:.2f} eV")
    print(f"Max Electron Energy:  {max_energy:.2f} eV")
    
    # Check for heating in y-direction (where ICP field is applied)
    # Energy in y-component
    E_y_component = 0.5 * E_MASS * (vy[:N_e]**2) / EV_TO_J
    mean_Ey = np.mean(E_y_component)
    print(f"Mean Energy (Y-component only): {mean_Ey:.2f} eV")
    
    if mean_energy > 1.0: # Threshold depends on initial temp, but >1eV is a good sign of heating
        print("\n[SUCCESS] Significant electron heating detected!")
    else:
        print("\n[WARNING] Electron energy is low. Check heating amplitude.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_heating.py <run_id>")
        print("Example: python analyze_heating.py test_icp")
    else:
        analyze_heating(sys.argv[1])
