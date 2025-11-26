import numpy as np
import numba
import math
import sys
import os
import time as pytime # To avoid conflict with simulation Time
import scipy.interpolate # For interpolation

from pathlib import Path # For easier path manipulation
from tqdm import tqdm


# --- At the top of your script, after imports ---
RUN_ID = str(input("Name of this simulation? ")) # Example: you can make this dynamic, e.g., from command line or timestamp
BASE_OUTPUT_DIR = Path("results") / RUN_ID

# Create the directory if it doesn't exist
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f">> eduPIC: Output will be saved to: {BASE_OUTPUT_DIR}")

# Constants
PI = 3.141592653589793
TWO_PI = 2.0 * PI
E_CHARGE = 1.60217662e-19
EV_TO_J = E_CHARGE
E_MASS = 9.10938356e-31
AR_MASS = 6.63352090e-26
MU_ARAR = AR_MASS / 2.0
K_BOLTZMANN = 1.38064852e-23
EPSILON0 = 8.85418781e-12

# Simulation parameters
N_G = 400
N_T = 4000  # Time steps within an RF period
FREQUENCY = 13.56e6
VOLTAGE = 250.0
L = 0.025
PRESSURE = 10.0
TEMPERATURE = 350.0
WEIGHT = 7.0e4
ELECTRODE_AREA = 1.0e-4
N_INIT = 1000
MAX_PARTICLES_FACTOR = 600 # Factor to multiply N_INIT for initial array allocation
MAX_E_PARTICLES = N_INIT * MAX_PARTICLES_FACTOR
MAX_I_PARTICLES = N_INIT * MAX_PARTICLES_FACTOR


# Additional (derived) constants
PERIOD = 1.0 / FREQUENCY
DT_E = PERIOD / float(N_T)
N_SUB = 20
DT_I = N_SUB * DT_E
DX = L / float(N_G - 1)
INV_DX = 1.0 / DX
GAS_DENSITY = PRESSURE / (K_BOLTZMANN * TEMPERATURE)
OMEGA = TWO_PI * FREQUENCY

# Electron and ion cross sections
N_CS = 5
E_ELA = 0
E_EXC = 1
E_ION = 2
I_ISO = 3
I_BACK = 4
E_EXC_TH = 11.5
E_ION_TH = 15.8
CS_RANGES = 1000000
DE_CS = 0.001

# Cross section arrays (Numba requires NumPy arrays)
sigma = np.zeros((N_CS, CS_RANGES), dtype=np.float32)
sigma_tot_e = np.zeros(CS_RANGES, dtype=np.float32)
sigma_tot_i = np.zeros(CS_RANGES, dtype=np.float32)

# Particle coordinates (pre-allocate)
# Current number of particles (as 1-element arrays for pass-by-reference in Numba)
N_e_arr = np.array([0], dtype=np.int64)
N_i_arr = np.array([0], dtype=np.int64)

x_e = np.zeros(MAX_E_PARTICLES, dtype=np.float64)
vx_e = np.zeros(MAX_E_PARTICLES, dtype=np.float64)
vy_e = np.zeros(MAX_E_PARTICLES, dtype=np.float64)
vz_e = np.zeros(MAX_E_PARTICLES, dtype=np.float64)

x_i = np.zeros(MAX_I_PARTICLES, dtype=np.float64)
vx_i = np.zeros(MAX_I_PARTICLES, dtype=np.float64)
vy_i = np.zeros(MAX_I_PARTICLES, dtype=np.float64)
vz_i = np.zeros(MAX_I_PARTICLES, dtype=np.float64)

# Grid quantities
efield = np.zeros(N_G, dtype=np.float64)
pot = np.zeros(N_G, dtype=np.float64)
e_density = np.zeros(N_G, dtype=np.float64)
i_density = np.zeros(N_G, dtype=np.float64)
cumul_e_density = np.zeros(N_G, dtype=np.float64)
cumul_i_density = np.zeros(N_G, dtype=np.float64)

# Counters (as 1-element arrays for pass-by-reference in Numba)
N_e_abs_pow_arr = np.array([0], dtype=np.uint64)
N_e_abs_gnd_arr = np.array([0], dtype=np.uint64)
N_i_abs_pow_arr = np.array([0], dtype=np.uint64)
N_i_abs_gnd_arr = np.array([0], dtype=np.uint64)

# EEPF
N_EEPF = 2000
DE_EEPF = 0.05
eepf = np.zeros(N_EEPF, dtype=np.float64)

# IFED
N_IFED = 200
DE_IFED = 1.0
ifed_pow = np.zeros(N_IFED, dtype=np.int32) # Using int32 as in C++ array<int, N_IFED>
ifed_gnd = np.zeros(N_IFED, dtype=np.int32)
mean_i_energy_pow = 0.0 # These will be calculated later
mean_i_energy_gnd = 0.0

# XT distributions
N_BIN = 20
N_XT = N_T // N_BIN
pot_xt = np.zeros((N_G, N_XT), dtype=np.float64) # Reshaped for easier indexing
efield_xt = np.zeros((N_G, N_XT), dtype=np.float64)
ne_xt = np.zeros((N_G, N_XT), dtype=np.float64)
ni_xt = np.zeros((N_G, N_XT), dtype=np.float64)
ue_xt = np.zeros((N_G, N_XT), dtype=np.float64)
ui_xt = np.zeros((N_G, N_XT), dtype=np.float64)
je_xt = np.zeros((N_G, N_XT), dtype=np.float64)
ji_xt = np.zeros((N_G, N_XT), dtype=np.float64)
powere_xt = np.zeros((N_G, N_XT), dtype=np.float64)
poweri_xt = np.zeros((N_G, N_XT), dtype=np.float64)
meanee_xt = np.zeros((N_G, N_XT), dtype=np.float64)
meanei_xt = np.zeros((N_G, N_XT), dtype=np.float64)
counter_e_xt = np.zeros((N_G, N_XT), dtype=np.float64)
counter_i_xt = np.zeros((N_G, N_XT), dtype=np.float64)
ioniz_rate_xt = np.zeros((N_G, N_XT), dtype=np.float64)

# Time-resolved EEPF (for the central region MIN_X to MAX_X)
# Dimensions: (Time Bins in RF cycle, Energy Bins)
eepf_phase_resolved = np.zeros((N_XT, N_EEPF), dtype=np.float64)

mean_energy_accu_center_arr = np.array([0.0], dtype=np.float64)
mean_energy_counter_center_arr = np.array([0], dtype=np.uint64)
N_e_coll_arr = np.array([0], dtype=np.uint64)
N_i_coll_arr = np.array([0], dtype=np.uint64)

# Simulation state
Time_arr = np.array([0.0], dtype=np.float64) # Global time
cycle_arr = np.array([0], dtype=np.int32)
no_of_cycles_arr = np.array([0], dtype=np.int32)
cycles_done_arr = np.array([0], dtype=np.int32)
measurement_mode_arr = np.array([False], dtype=np.bool_) # Boolean flag

# Random number generator (pass this to JITted functions)
RNG = np.random.default_rng()

# --- Cross Section Functions (Python, JIT optional as called once) ---
def qmel(en):
    return 1e-20 * (abs(6.0 / pow(1.0 + (en / 0.1) + pow(en / 0.6, 2.0), 3.3)
                      - 1.1 * pow(en, 1.4) / (1.0 + pow(en / 15.0, 1.2)) / math.sqrt(1.0 + pow(en / 5.5, 2.5) + pow(en / 60.0, 4.1)))
                    + 0.05 / pow(1.0 + en / 10.0, 2.0) + 0.01 * pow(en, 3.0) / (1.0 + pow(en / 12.0, 6.0)))

def qexc(en):
    if en > E_EXC_TH:
        return 1e-20 * (0.034 * pow(en - 11.5, 1.1) * (1.0 + pow(en / 15.0, 2.8)) / (1.0 + pow(en / 23.0, 5.5))
                        + 0.023 * (en - 11.5) / pow(1.0 + en / 80.0, 1.9))
    else:
        return 0.0

def qion(en):
    if en > E_ION_TH:
        return 1e-20 * (970.0 * (en - 15.8) / pow(70.0 + en, 2.0) +
                        0.06 * pow(en - 15.8, 2.0) * math.exp(-en / 9.0))
    else:
        return 0.0

def load_individual_lxcat_file(filepath, target_energy_eV_grid, 
                               units_in_m2=True, 
                               header_skip_keyword="-----------------------------"):
    """
    Parses an individual LXCat cross-section file (containing one process)
    and interpolates it onto a target energy grid.

    Args:
        filepath (str or Path): Path to the LXCat data file.
        target_energy_eV_grid (np.ndarray): The simulation's energy grid (in eV).
        units_in_m2 (bool): True if LXCat data is in m^2, False if in cm^2 (for conversion).
        header_skip_keyword (str): A string that indicates the start of the data table.

    Returns:
        np.ndarray: Interpolated cross-sections on the target_energy_eV_grid.
    """
    energies_lxcat = []
    sigmas_lxcat = []
    data_section_found = False
    data_parsing_active = False

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue # Skip empty lines

                if header_skip_keyword in line:
                    if not data_section_found: # First occurrence is start of data
                        data_section_found = True
                        data_parsing_active = True
                        continue 
                    else: # Second occurrence is end of data
                        data_parsing_active = False
                        break # Stop reading after data block

                if data_parsing_active:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            energy = float(parts[0])
                            sigma_val = float(parts[1])
                            energies_lxcat.append(energy)
                            sigmas_lxcat.append(sigma_val)
                        except ValueError:
                            print(f"Warning: Could not parse data line in {filepath}: {line}")
                            continue
    except FileNotFoundError:
        print(f"ERROR: LXCat file not found: {filepath}")
        return np.zeros_like(target_energy_eV_grid, dtype=np.float32)
    except Exception as e:
        print(f"ERROR: Could not parse LXCat file {filepath}: {e}")
        return np.zeros_like(target_energy_eV_grid, dtype=np.float32)

    if not energies_lxcat:
        print(f"WARNING: No data parsed from {filepath}. Returning zeros.")
        return np.zeros_like(target_energy_eV_grid, dtype=np.float32)

    energies_lxcat = np.array(energies_lxcat, dtype=np.float32)
    sigmas_lxcat = np.array(sigmas_lxcat, dtype=np.float32)

    if not units_in_m2: # If units were cm^2
        sigmas_lxcat *= 1e-4 
        print(f"Loaded and converted {filepath} from cm^2 to m^2.")
    else:
        print(f"Loaded {filepath} (units assumed m^2).")

    # Interpolate
    if len(energies_lxcat) > 1:
        unique_indices = np.unique(energies_lxcat, return_index=True)[1]
        energies_lxcat = energies_lxcat[unique_indices]
        sigmas_lxcat = sigmas_lxcat[unique_indices]
        sort_indices = np.argsort(energies_lxcat)
        energies_lxcat = energies_lxcat[sort_indices]
        sigmas_lxcat = sigmas_lxcat[sort_indices]

        interp_func = scipy.interpolate.interp1d(
            energies_lxcat, sigmas_lxcat, kind='linear',
            bounds_error=False, fill_value=0.0 
        )
        interpolated_sigma = interp_func(target_energy_eV_grid).astype(np.float32)
        interpolated_sigma[interpolated_sigma < 0] = 0.0
    elif len(energies_lxcat) == 1: # Only one data point
        interpolated_sigma = np.zeros_like(target_energy_eV_grid, dtype=np.float32)
        # A simple way to handle a single point: apply it if energy matches, else zero
        # This might not be physically ideal for all cases of single-point CS.
        if energies_lxcat[0] in target_energy_eV_grid :
             idx_match = np.isclose(target_energy_eV_grid, energies_lxcat[0])
             interpolated_sigma[idx_match] = sigmas_lxcat[0]
        # Or, if the single point is a threshold, apply it for all energies above.
        # For now, this simple matching is used.
        print(f"Warning: Only one data point in {filepath}. Applied if energy matches grid point.")
    else: # No data points
        interpolated_sigma = np.zeros_like(target_energy_eV_grid, dtype=np.float32)
        
    return interpolated_sigma

      
def set_electron_cross_sections_ar():
    print(">> eduPIC: Setting e- / Ar cross sections from individual LXCat files")
    # Define your simulation's energy grid (in eV)
    # The energy grid for sigma array index `i` corresponds to energy `(i+0.5)*DE_CS` if DE_CS is bin width
    # Or, if DE_CS is the step, then energy `i*DE_CS` or `(i+1)*DE_CS`.
    # Let's assume DE_CS is the energy step, and CS_RANGES is the number of points.
    # Original code had: e = np.arange(CS_RANGES, dtype=np.float32) * DE_CS + DE_CS
    # This implies energy points are DE_CS, 2*DE_CS, ..., CS_RANGES*DE_CS
    # For interpolation, it's good to define the exact energy points for your sigma array.
    # If sigma[i] corresponds to energy E_i, then E_i = (i * DE_CS) for i=0 to CS_RANGES-1
    # or E_i = ((i+1) * DE_CS)
    # Let's use the mid-points of bins for a robust definition if DE_CS is bin width.
    # If DE_CS is the step size and sigma[0] is at energy DE_CS (like original analytical), then:
    sim_energy_grid_eV = (np.arange(CS_RANGES, dtype=np.float32) + 1) * DE_CS # Matches original analytical grid

    lxcat_data_dir = Path("./lxcat_data") # Store your individual files here

    # Ensure this directory exists
    if not lxcat_data_dir.is_dir():
        print(f"ERROR: LXCat data directory not found: {lxcat_data_dir}")
        print("Please create it and add your LXCat .txt files.")
        # Fallback to analytical if directory doesn't exist
        print(">> eduPIC: FALLBACK to analytical e- / Ar cross sections")
        e_analytical = np.arange(CS_RANGES, dtype=np.float32) * DE_CS + DE_CS
        sigma[E_ELA, :] = np.array([qmel(val) for val in e_analytical], dtype=np.float32)
        sigma[E_EXC, :] = np.array([qexc(val) for val in e_analytical], dtype=np.float32)
        sigma[E_ION, :] = np.array([qion(val) for val in e_analytical], dtype=np.float32)
        return

    # Define paths to your new individual files
    # The units in your Phelps file are already m^2.
    elastic_cs_file = lxcat_data_dir / "ar_electron_elastic_effective.txt" 
    excitation_cs_file = lxcat_data_dir / "ar_electron_excitation_11.5eV.txt"
    ionization_cs_file = lxcat_data_dir / "ar_electron_ionization.txt"

    sigma[E_ELA, :] = load_individual_lxcat_file(elastic_cs_file, sim_energy_grid_eV, units_in_m2=True)
    sigma[E_EXC, :] = load_individual_lxcat_file(excitation_cs_file, sim_energy_grid_eV, units_in_m2=True)
    sigma[E_ION, :] = load_individual_lxcat_file(ionization_cs_file, sim_energy_grid_eV, units_in_m2=True)

    # Thresholding (still good practice)
    # This ensures that below the known physical threshold, the cross-section is strictly zero,
    # even if interpolation or LXCat data has small non-zero values there.
    # Find the bin index corresponding to the threshold energy.
    # Bins are 0 to CS_RANGES-1. Energy for bin i is (i+1)*DE_CS.
    # So, index for E_thresh is roughly E_thresh/DE_CS - 1.
    
    exc_threshold_idx = math.floor(E_EXC_TH / DE_CS) # Bins below this index should be zero
    # Correct: sigma[E_EXC, :idx] zeros up to idx-1. We want to zero bins whose energy center is < E_EXC_TH
    # If energy_of_bin_i = (i+1)*DE_CS, then (i+1)*DE_CS < E_EXC_TH  => i < E_EXC_TH/DE_CS - 1
    # So, up to index floor(E_EXC_TH/DE_CS - 1) should be zero.
    # A simpler way: iterate and zero if energy < threshold.
    for i in range(CS_RANGES):
        energy_of_bin = (i + 1) * DE_CS
        if energy_of_bin < E_EXC_TH:
            sigma[E_EXC, i] = 0.0
        if energy_of_bin < E_ION_TH:
            sigma[E_ION, i] = 0.0
    
    # Optional plotting check 
    # import matplotlib.pyplot as plt # Make sure to import pyplot
    # plt.figure(figsize=(10, 6))
    # plt.plot(sim_energy_grid_eV, sigma[E_ELA,:], label=f"Elastic (LXCat - {elastic_cs_file.name})")
    # plt.plot(sim_energy_grid_eV, sigma[E_EXC,:], label=f"Excitation (LXCat - {excitation_cs_file.name})")
    # plt.plot(sim_energy_grid_eV, sigma[E_ION,:], label=f"Ionization (LXCat - {ionization_cs_file.name})")
    # plt.xlabel("Energy (eV)"); plt.ylabel("Cross Section (m^2)"); plt.loglog()
    # plt.legend(); plt.grid(True,which="both",ls="-"); plt.title("e-Ar CS from LXCat (Individual Files)")
    # plt.ylim(bottom=1e-24) # Adjust if needed
    # plot_path = BASE_OUTPUT_DIR / "lxcat_electron_cs_check.png"
    # plt.savefig(plot_path)
    # print(f"Saved electron CS plot to {plot_path}")
    # plt.close()

def qiso_ion(e_lab):
    return 2e-19 * pow(e_lab, -0.5) / (1.0 + e_lab) + \
           3e-19 * e_lab / pow(1.0 + e_lab / 3.0, 2.0)

def qmom_ion(e_lab):
    return 1.15e-18 * pow(e_lab, -0.1) * pow(1.0 + 0.015 / e_lab, 0.6)

      
def set_ion_cross_sections_ar():
    print(">> eduPIC: Setting Ar+ / Ar cross sections from individual LXCat files")
    # Ion energy grid for CS in eduPIC is defined differently from electrons in original analytical part
    # Original: e = np.arange(CS_RANGES, dtype=np.float32) * (2.0 * DE_CS) + (2.0*DE_CS)
    # This means points are 2*DE_CS, 4*DE_CS, ..., CS_RANGES*2*DE_CS
    # Let's stick to this definition for consistency with how ion energies might be indexed later.
    ion_sim_energy_grid_eV = (np.arange(CS_RANGES, dtype=np.float32) + 1) * (2.0 * DE_CS)

    lxcat_data_dir = Path("./lxcat_data")
    if not lxcat_data_dir.is_dir():
        print(f"ERROR: LXCat data directory not found: {lxcat_data_dir}")
        # Fallback to analytical
        print(">> eduPIC: FALLBACK to analytical Ar+ / Ar cross sections")
        e_analytical = (np.arange(CS_RANGES, dtype=np.float32) + 1) * (2.0 * DE_CS)
        _qiso_analytical = np.array([qiso_ion(val) for val in e_analytical], dtype=np.float32)
        _qmom_analytical = np.array([qmom_ion(val) for val in e_analytical], dtype=np.float32)
        sigma[I_ISO, :] = _qiso_analytical
        sigma[I_BACK, :] = (_qmom_analytical - _qiso_analytical) / 2.0
        sigma[I_BACK, sigma[I_BACK,:] < 0] = 0.0
        return

    ion_iso_cs_file = lxcat_data_dir / "ar_ion_isotropic.txt"
    ion_back_cs_file = lxcat_data_dir / "ar_ion_backscatter.txt"

    sigma_iso_loaded = load_individual_lxcat_file(ion_iso_cs_file, ion_sim_energy_grid_eV, units_in_m2=True)
    sigma_back_loaded = load_individual_lxcat_file(ion_back_cs_file, ion_sim_energy_grid_eV, units_in_m2=True)

    fallback_to_analytical = False
    if not ion_iso_cs_file.exists() or np.all(sigma_iso_loaded == 0):
        print(f"Warning: Problem loading or zero data for {ion_iso_cs_file}.")
        fallback_to_analytical = True
    if not ion_back_cs_file.exists() or np.all(sigma_back_loaded == 0):
        print(f"Warning: Problem loading or zero data for {ion_back_cs_file}.")
        fallback_to_analytical = True
        
    if fallback_to_analytical:
        print("Using analytical ion CS as fallback.")
        e_analytical = (np.arange(CS_RANGES, dtype=np.float32) + 1) * (2.0 * DE_CS)
        _qiso_analytical = np.array([qiso_ion(val) for val in e_analytical], dtype=np.float32)
        _qmom_analytical = np.array([qmom_ion(val) for val in e_analytical], dtype=np.float32)
        sigma[I_ISO, :] = _qiso_analytical
        sigma[I_BACK, :] = (_qmom_analytical - _qiso_analytical) / 2.0
        sigma[I_BACK, sigma[I_BACK,:] < 0] = 0.0
    else:
        sigma[I_ISO, :] = sigma_iso_loaded
        sigma[I_BACK, :] = sigma_back_loaded
        print("Loaded ion CS from individual LXCat files.")

    # Optional plotting for ion cross sections
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,6))
    # plt.plot(ion_sim_energy_grid_eV, sigma[I_ISO,:], label=f"Ion Isotropic (LXCat - {ion_iso_cs_file.name})")
    # plt.plot(ion_sim_energy_grid_eV, sigma[I_BACK,:], label=f"Ion Backscatter (LXCat - {ion_back_cs_file.name})")
    # plt.xlabel("Ion Energy (eV)"); plt.ylabel("Cross Section (m^2)"); plt.loglog()
    # plt.legend(); plt.grid(True,which="both",ls="-"); plt.title("Ar+-Ar CS from LXCat (Individual Files)")
    # plt.ylim(bottom=1e-22) # Adjust if needed
    # plot_path = BASE_OUTPUT_DIR / "lxcat_ion_cs_check.png"
    # plt.savefig(plot_path)
    # print(f"Saved ion CS plot to {plot_path}")
    # plt.close()
    
def calc_total_cross_sections():
    for i in range(CS_RANGES):
        sigma_tot_e[i] = (sigma[E_ELA, i] + sigma[E_EXC, i] + sigma[E_ION, i]) * GAS_DENSITY
        sigma_tot_i[i] = (sigma[I_ISO, i] + sigma[I_BACK, i]) * GAS_DENSITY

def test_cross_sections():
    with open("cross_sections.dat", "w") as f:
        energies = np.arange(CS_RANGES) * DE_CS
        for e_val in energies:
            f.write(f"{e_val}\n")
        for i in range(N_CS):
            for val in sigma[i,:]:
                f.write(f"{val}\n") # Note: This format might differ slightly from C++ iterator output

@numba.njit(fastmath=True)
def max_electron_coll_freq_jit(sigma_tot_e_arr): # Pass array explicitly
    nu_max = 0.0
    for i in range(CS_RANGES):
        e = i * DE_CS
        v = math.sqrt(2.0 * e * EV_TO_J / E_MASS)
        nu = v * sigma_tot_e_arr[i]
        if nu > nu_max:
            nu_max = nu
    return nu_max

@numba.njit(fastmath=True)
def max_ion_coll_freq_jit(sigma_tot_i_arr): # Pass array explicitly
    nu_max = 0.0
    for i in range(CS_RANGES):
        e = i * DE_CS # C++ uses 2*DE_CS here in generation, but index is still i
        g = math.sqrt(2.0 * e * EV_TO_J / MU_ARAR) # Energy in lab frame for cross-section
        nu = g * sigma_tot_i_arr[i]
        if nu > nu_max:
            nu_max = nu
    return nu_max

def init_particles(nseed):
    N_e_arr[0] = nseed
    N_i_arr[0] = nseed
    
    # Initialize only up to nseed
    x_e[:nseed] = RNG.random(nseed) * L
    vx_e[:nseed] = 0.0
    vy_e[:nseed] = 0.0
    vz_e[:nseed] = 0.0

    x_i[:nseed] = RNG.random(nseed) * L
    vx_i[:nseed] = 0.0
    vy_i[:nseed] = 0.0
    vz_i[:nseed] = 0.0

# --- JITted Core Physics Functions ---
@numba.njit(fastmath=True)
def RMB_numba(rng_state): # Numba compatible normal distribution
    # Box-Muller transform for normal distribution from uniform
    # This generates two normal variates, we use one.
    # Simpler: Numba supports np.random.normal directly if rng is passed.
    # However, to match C++ exactly, we'd need to know its internal state or use a known good one.
    # For simplicity, use Numba's built-in, it's good enough.
    # If using the passed `RNG` object directly in Numba (experimental)
    # return RNG.normal(0.0, math.sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS))
    # For a more standard Numba way if passing RNG object is problematic:
    u1 = rng_state.random()
    u2 = rng_state.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(TWO_PI * u2)
    return z0 * math.sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS)

# --- JITted Core Physics Functions ---
@numba.njit(fastmath=True)
def RMB_numba(rng_state):
    # ... (this function should be correct, no changes needed) ...
    u1 = rng_state.random()
    u2 = rng_state.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(TWO_PI * u2)
    return z0 * math.sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS)


# REPLACEMENT for collision_electron_jit
@numba.njit(fastmath=True)
def collision_electron_jit(k_colliding, # Index of the colliding electron in global arrays
                           eindex,
                           # Global particle arrays (for primary electron and for adding new ones)
                           N_e_arr_ref, N_i_arr_ref, 
                           x_e_global, vx_e_global, vy_e_global, vz_e_global,
                           x_i_global, vx_i_global, vy_i_global, vz_i_global,
                           MAX_E_PARTICLES_local, MAX_I_PARTICLES_local,
                           rng_state):
    F1 = E_MASS / (E_MASS + AR_MASS)
    F2 = AR_MASS / (E_MASS + AR_MASS)
    
    # Get current velocities of the colliding electron
    vxe = vx_e_global[k_colliding]
    vye = vy_e_global[k_colliding]
    vze = vz_e_global[k_colliding]
    xe_particle = x_e_global[k_colliding]

    gx = vxe
    gy = vye
    gz = vze
    g = math.sqrt(gx * gx + gy * gy + gz * gz)
    if g == 0:
      return 

    wx = F1 * vxe
    wy = F1 * vye
    wz = F1 * vze

    if gx == 0: theta = 0.5 * PI
    else: theta = math.atan2(math.sqrt(gy * gy + gz * gz), gx)
    
    if gy == 0:
        if gz > 0: phi = 0.5 * PI
        else: phi = -0.5 * PI
    else: phi = math.atan2(gz, gy)
        
    st_initial = math.sin(theta)
    ct_initial = math.cos(theta)
    sp_initial = math.sin(phi)
    cp_initial = math.cos(phi)

    t0 = sigma[E_ELA, eindex]
    t1 = t0 + sigma[E_EXC, eindex]
    t2 = t1 + sigma[E_ION, eindex]
    
    if t2 == 0: 
        return

    rnd = rng_state.random()
    chi = 0.0
    eta = 0.0
    
    g_primary = g

    if rnd < (t0 / t2):  # Elastic
        chi = math.acos(1.0 - 2.0 * rng_state.random())
        eta = TWO_PI * rng_state.random()
    elif rnd < (t1 / t2):  # Excitation
        energy_rel = 0.5 * E_MASS * g * g
        energy_after_loss = abs(energy_rel - E_EXC_TH * EV_TO_J)
        g_primary = math.sqrt(2.0 * energy_after_loss / E_MASS) if energy_after_loss > 0 else 0.0
        chi = math.acos(1.0 - 2.0 * rng_state.random())
        eta = TWO_PI * rng_state.random()
    else:  # Ionization
        energy_rel = 0.5 * E_MASS * g * g
        energy_after_threshold = abs(energy_rel - E_ION_TH * EV_TO_J)
        
        tan_arg = energy_after_threshold / EV_TO_J / 20.0
        if tan_arg < 0: tan_arg = 0.0
        
        e_ej_rel = 10.0 * math.tan(rng_state.random() * math.atan(tan_arg)) * EV_TO_J
        if e_ej_rel < 0: e_ej_rel = 0.0
        
        e_sc_rel = abs(energy_after_threshold - e_ej_rel)
        if e_sc_rel < 0: e_sc_rel = 0.0

        g_primary = math.sqrt(2.0 * e_sc_rel / E_MASS) if e_sc_rel > 0 else 0.0
        g2_new_e = math.sqrt(2.0 * e_ej_rel / E_MASS) if e_ej_rel > 0 else 0.0
        
        energy_val_sc = e_sc_rel / energy_after_threshold if energy_after_threshold > 0 else 0.0
        energy_val_ej = e_ej_rel / energy_after_threshold if energy_after_threshold > 0 else 0.0
        
        chi = math.acos(math.sqrt(min(max(energy_val_sc,0.0),1.0))) 
        chi2 = math.acos(math.sqrt(min(max(energy_val_ej,0.0),1.0)))

        eta = TWO_PI * rng_state.random()
        eta2 = eta + PI

        sc2 = math.sin(chi2)
        cc2 = math.cos(chi2)
        se2 = math.sin(eta2)
        ce2 = math.cos(eta2)

        gx_new_e_rel = g2_new_e * (ct_initial * cc2 - st_initial * sc2 * ce2)
        gy_new_e_rel = g2_new_e * (st_initial * cp_initial * cc2 + ct_initial * cp_initial * sc2 * ce2 - sp_initial * sc2 * se2)
        gz_new_e_rel = g2_new_e * (st_initial * sp_initial * cc2 + ct_initial * sp_initial * sc2 * ce2 + cp_initial * sc2 * se2)
        
        if N_e_arr_ref[0] < MAX_E_PARTICLES_local:
            idx_new_e = N_e_arr_ref[0]
            x_e_global[idx_new_e] = xe_particle
            vx_e_global[idx_new_e] = wx + F2 * gx_new_e_rel
            vy_e_global[idx_new_e] = wy + F2 * gy_new_e_rel
            vz_e_global[idx_new_e] = wz + F2 * gz_new_e_rel
            N_e_arr_ref[0] += 1
        
        if N_i_arr_ref[0] < MAX_I_PARTICLES_local:
            idx_new_i = N_i_arr_ref[0]
            x_i_global[idx_new_i] = xe_particle
            vx_i_global[idx_new_i] = RMB_numba(rng_state)
            vy_i_global[idx_new_i] = RMB_numba(rng_state)
            vz_i_global[idx_new_i] = RMB_numba(rng_state)
            N_i_arr_ref[0] += 1
            
    sc_chi = math.sin(chi)
    cc_chi = math.cos(chi)
    se_eta = math.sin(eta)
    ce_eta = math.cos(eta)
    
    gx_final_rel = g_primary * (ct_initial * cc_chi - st_initial * sc_chi * ce_eta)
    gy_final_rel = g_primary * (st_initial * cp_initial * cc_chi + ct_initial * cp_initial * sc_chi * ce_eta - sp_initial * sc_chi * se_eta)
    gz_final_rel = g_primary * (st_initial * sp_initial * cc_chi + ct_initial * sp_initial * sc_chi * ce_eta + cp_initial * sc_chi * se_eta)

    vx_e_global[k_colliding] = wx + F2 * gx_final_rel
    vy_e_global[k_colliding] = wy + F2 * gy_final_rel
    vz_e_global[k_colliding] = wz + F2 * gz_final_rel


# REPLACEMENT for collision_ion_jit
@numba.njit(fastmath=True)
def collision_ion_jit(k_colliding,
                      vx_target, vy_target, vz_target,
                      e_index, 
                      vx_i_global, vy_i_global, vz_i_global,
                      rng_state):
    vx1 = vx_i_global[k_colliding]
    vy1 = vy_i_global[k_colliding]
    vz1 = vz_i_global[k_colliding]

    gx = vx1 - vx_target
    gy = vy1 - vy_target
    gz = vz1 - vz_target
    g = math.sqrt(gx * gx + gy * gy + gz * gz)
    if g == 0: 
        return 

    wx = 0.5 * (vx1 + vx_target)
    wy = 0.5 * (vy1 + vy_target)
    wz = 0.5 * (vz1 + vz_target)

    if gx == 0: theta = 0.5 * PI
    else: theta = math.atan2(math.sqrt(gy * gy + gz * gz), gx)
    
    if gy == 0:
        if gz > 0: phi = 0.5 * PI
        else: phi = -0.5 * PI
    else: phi = math.atan2(gz, gy)

    t1 = sigma[I_ISO, e_index]
    t2 = t1 + sigma[I_BACK, e_index]
    if t2 == 0:
        return

    rnd = rng_state.random()
    chi = 0.0
    if rnd < (t1 / t2):
        chi = math.acos(1.0 - 2.0 * rng_state.random())
    else:
        chi = PI
        
    eta = TWO_PI * rng_state.random()
    st = math.sin(theta)
    ct = math.cos(theta)
    sp = math.sin(phi)
    cp = math.cos(phi)
    sc_chi = math.sin(chi)
    cc_chi = math.cos(chi)
    se_eta = math.sin(eta)
    ce_eta = math.cos(eta)

    gx_new = g * (ct * cc_chi - st * sc_chi * ce_eta)
    gy_new = g * (st * cp * cc_chi + ct * cp * sc_chi * ce_eta - sp * sc_chi * se_eta)
    gz_new = g * (st * sp * cc_chi + ct * sp * sc_chi * ce_eta + cp * sc_chi * se_eta)

    vx_i_global[k_colliding] = wx + 0.5 * gx_new
    vy_i_global[k_colliding] = wy + 0.5 * gy_new
    vz_i_global[k_colliding] = wz + 0.5 * gz_new

@numba.njit(fastmath=True)
def solve_Poisson_jit(pot_arr, efield_arr, rho1_arr, tt, N_G_local, DX_local, INV_DX_local, OMEGA_local, VOLTAGE_local, EPSILON0_local):
    A = 1.0
    B = -2.0
    C = 1.0
    S = 1.0 / (2.0 * DX_local)
    ALPHA = -DX_local * DX_local / EPSILON0_local
    
    # Numba doesn't allow creating arrays inside JIT function easily like this if size isn't const
    # Declare g, w, f outside or pass them if their size is N_G
    # For this specific Thomas algorithm, intermediate f,g,w are 1D and can be fixed size
    g_thomas = np.empty(N_G_local, dtype=np.float64) 
    w_thomas = np.empty(N_G_local, dtype=np.float64)
    f_thomas = np.empty(N_G_local, dtype=np.float64)


    pot_arr[0] = VOLTAGE_local * math.cos(OMEGA_local * tt)
    pot_arr[N_G_local - 1] = 0.0

    for i in range(1, N_G_local - 1):
        f_thomas[i] = ALPHA * rho1_arr[i]
    
    f_thomas[1] -= pot_arr[0]
    f_thomas[N_G_local - 2] -= pot_arr[N_G_local-1] # C++ N_G-2 is second to last, Python N_G-1 is last
                                               # So N_G-2 in C++ is N_G-2 in Python if 0-indexed

    if B == 0: return # Avoid division by zero; should not happen with B=-2.0

    w_thomas[1] = C / B
    g_thomas[1] = f_thomas[1] / B
    
    for i in range(2, N_G_local - 1): # Up to N_G-2 in Python (0 to N_G-1 is full range)
        denom = (B - A * w_thomas[i-1])
        if denom == 0: return # Avoid division by zero
        w_thomas[i] = C / denom
        g_thomas[i] = (f_thomas[i] - A * g_thomas[i-1]) / denom
        
    pot_arr[N_G_local - 2] = g_thomas[N_G_local - 2] # N_G-2 is correct for 0-indexed python
    # Corrected loop range: from N_G-3 down to 1 (inclusive)
    for i in range(N_G_local - 3, 0, -1): # Python range is exclusive at end, so 0 means loop until 1
        pot_arr[i] = g_thomas[i] - w_thomas[i] * pot_arr[i+1]

    for i in range(1, N_G_local - 1):
        efield_arr[i] = (pot_arr[i-1] - pot_arr[i+1]) * S
    
    efield_arr[0] = (pot_arr[0] - pot_arr[1]) * INV_DX_local - rho1_arr[0] * DX_local / (2.0 * EPSILON0_local)
    efield_arr[N_G_local-1] = (pot_arr[N_G_local-2] - pot_arr[N_G_local-1]) * INV_DX_local + \
                              rho1_arr[N_G_local-1] * DX_local / (2.0 * EPSILON0_local)


# REPLACEMENT for do_one_cycle_jit_kernels
@numba.njit(fastmath=True)
def do_one_cycle_jit_kernels(
    # ... (function signature/arguments remain the same as your file)
    Time_arr_ref, N_e_arr_ref, N_i_arr_ref,
    x_e, vx_e, vy_e, vz_e,
    x_i, vx_i, vy_i, vz_i,
    efield, pot, e_density, i_density, cumul_e_density, cumul_i_density,
    N_e_abs_pow_arr_ref, N_e_abs_gnd_arr_ref, N_i_abs_pow_arr_ref, N_i_abs_gnd_arr_ref,
    eepf, ifed_pow, ifed_gnd,
    pot_xt, efield_xt, ne_xt, ni_xt, ue_xt, ui_xt, meanee_xt, meanei_xt,
    counter_e_xt, counter_i_xt, ioniz_rate_xt,
    mean_energy_accu_center_arr_ref, mean_energy_counter_center_arr_ref,
    N_e_coll_arr_ref, N_i_coll_arr_ref,
    sigma_tot_e, sigma_tot_i,
    sigma_e_ela_cs, sigma_e_exc_cs, sigma_e_ion_cs,
    sigma_i_iso_cs, sigma_i_back_cs,
    measurement_mode, cycle_val,
    rng_state,
    # New argument for phase-resolved EEPF
    eepf_phase_resolved 
    ):

    DV = ELECTRODE_AREA * DX
    FACTOR_W = WEIGHT / DV
    FACTOR_E = DT_E / E_MASS * E_CHARGE
    FACTOR_I = DT_I / AR_MASS * E_CHARGE
    MIN_X = 0.45 * L
    MAX_X = 0.55 * L
    
    rho = np.empty(N_G, dtype=np.float64)

    for t_loop_idx in range(N_T):
        Time_arr_ref[0] += DT_E
        t_index_xt = t_loop_idx // N_BIN

        # Step 1: Compute densities (this part was mostly correct)
        e_density[:] = 0.0
        current_N_e = N_e_arr_ref[0]
        for k in range(current_N_e):
            pos_norm = x_e[k] * INV_DX
            p_float = math.floor(pos_norm)
            rmod = pos_norm - p_float
            p = int(p_float)
            
            if 0 <= p < N_G -1:
                 e_density[p]   += (1.0-rmod) * FACTOR_W
                 e_density[p+1] += rmod * FACTOR_W
            elif p == N_G -1:
                 e_density[p] += (1.0-rmod) * FACTOR_W

        e_density[0] *= 2.0
        e_density[N_G-1] *= 2.0
        cumul_e_density += e_density

        if (t_loop_idx % N_SUB) == 0:
            i_density[:] = 0.0
            current_N_i = N_i_arr_ref[0]
            for k in range(current_N_i):
                pos_norm = x_i[k] * INV_DX
                p_float = math.floor(pos_norm)
                rmod = pos_norm - p_float
                p = int(p_float)
                if 0 <= p < N_G -1:
                    i_density[p]   += (1.0-rmod) * FACTOR_W
                    i_density[p+1] += rmod * FACTOR_W
                elif p == N_G -1:
                    i_density[p] += (1.0-rmod) * FACTOR_W
            i_density[0] *= 2.0
            i_density[N_G-1] *= 2.0
        cumul_i_density += i_density

        # Step 2: Solve Poisson
        for i_rho in range(N_G):
            rho[i_rho] = E_CHARGE * (i_density[i_rho] - e_density[i_rho])
        solve_Poisson_jit(pot, efield, rho, Time_arr_ref[0], N_G, DX, INV_DX, OMEGA, VOLTAGE, EPSILON0)

        # --- CORRECTED PARTICLE LOOPS START HERE ---
        # Electrons
        k_e = 0
        num_electrons_to_process_this_timestep = N_e_arr_ref[0] 
        while k_e < num_electrons_to_process_this_timestep:
            pos_norm = x_e[k_e] * INV_DX
            p_float = math.floor(pos_norm)
            rmod = pos_norm - p_float
            p = int(p_float)
            
            e_x_field = 0.0
            if 0 <= p < N_G - 1:
                e_x_field = (1.0-rmod)*efield[p] + rmod*efield[p+1]
            elif p == N_G -1:
                e_x_field = efield[N_G-1]
            
            if measurement_mode:
                mean_v = vx_e[k_e] - 0.5 * e_x_field * FACTOR_E
                if 0 <= p < N_G-1:
                    counter_e_xt[p, t_index_xt]   += (1.0-rmod)
                    counter_e_xt[p+1, t_index_xt] += rmod
                    ue_xt[p, t_index_xt]   += (1.0-rmod) * mean_v
                    ue_xt[p+1, t_index_xt] += rmod * mean_v
                    v_sqr  = mean_v * mean_v + vy_e[k_e] * vy_e[k_e] + vz_e[k_e] * vz_e[k_e]
                    energy = 0.5 * E_MASS * v_sqr / EV_TO_J
                    meanee_xt[p, t_index_xt]   += (1.0-rmod) * energy
                    meanee_xt[p+1, t_index_xt] += rmod * energy
                    
                    energy_idx_cs = min(int(energy / DE_CS + 0.5), CS_RANGES - 1)
                    if energy_idx_cs < 0: energy_idx_cs = 0
                    
                    velocity = math.sqrt(v_sqr)
                    rate = sigma_e_ion_cs[energy_idx_cs] * velocity * DT_E * GAS_DENSITY
                    ioniz_rate_xt[p, t_index_xt]   += (1.0-rmod) * rate
                    ioniz_rate_xt[p+1, t_index_xt] += rmod * rate
                
                v_sqr_eepf = vx_e[k_e]**2 + vy_e[k_e]**2 + vz_e[k_e]**2
                energy_eepf = 0.5 * E_MASS * v_sqr_eepf / EV_TO_J
                if (MIN_X < x_e[k_e]) and (x_e[k_e] < MAX_X):
                    energy_idx_eepf = int(energy_eepf / DE_EEPF)
                    if 0 <= energy_idx_eepf < N_EEPF:
                        eepf[energy_idx_eepf] += 1.0
                        # Accumulate for phase-resolved EEPF
                        eepf_phase_resolved[t_index_xt, energy_idx_eepf] += 1.0
                    mean_energy_accu_center_arr_ref[0] += energy_eepf
                    mean_energy_counter_center_arr_ref[0] += 1
            
            vx_e[k_e] -= e_x_field * FACTOR_E
            x_e[k_e]  += vx_e[k_e] * DT_E

            removed = False
            if x_e[k_e] < 0:
                N_e_abs_pow_arr_ref[0] += 1
                removed = True
            elif x_e[k_e] > L:
                N_e_abs_gnd_arr_ref[0] += 1
                removed = True
            
            if removed:
                N_e_arr_ref[0] -= 1
                idx_of_last_active_particle = N_e_arr_ref[0]
                if k_e < idx_of_last_active_particle:
                    x_e[k_e] = x_e[idx_of_last_active_particle]
                    vx_e[k_e] = vx_e[idx_of_last_active_particle]
                    vy_e[k_e] = vy_e[idx_of_last_active_particle]
                    vz_e[k_e] = vz_e[idx_of_last_active_particle]
                num_electrons_to_process_this_timestep -= 1
            else: 
                v_sqr_coll = vx_e[k_e]**2 + vy_e[k_e]**2 + vz_e[k_e]**2
                velocity_coll = math.sqrt(v_sqr_coll)
                energy_coll = 0.5 * E_MASS * v_sqr_coll / EV_TO_J
                
                energy_idx_coll = min(int(energy_coll / DE_CS + 0.5), CS_RANGES - 1)
                if energy_idx_coll < 0: energy_idx_coll = 0 
                
                nu_coll = sigma_tot_e[energy_idx_coll] * velocity_coll
                p_coll = 1.0 - math.exp(-nu_coll * DT_E)

                if rng_state.random() < p_coll:
                    collision_electron_jit(k_e,
                                           energy_idx_coll,
                                           N_e_arr_ref, N_i_arr_ref, 
                                           x_e, vx_e, vy_e, vz_e,
                                           x_i, vx_i, vy_i, vz_i,
                                           MAX_E_PARTICLES, MAX_I_PARTICLES,
                                           rng_state)
                    N_e_coll_arr_ref[0] += 1
                
                k_e += 1
        
        # Ions (subcycled)
        if (t_loop_idx % N_SUB) == 0:
            k_i = 0
            num_ions_to_process_this_timestep = N_i_arr_ref[0]
            while k_i < num_ions_to_process_this_timestep:
                pos_norm_i = x_i[k_i] * INV_DX
                p_float_i = math.floor(pos_norm_i)
                rmod_i = pos_norm_i - p_float_i
                p_i = int(p_float_i)

                e_x_field_i = 0.0
                if 0 <= p_i < N_G - 1:
                     e_x_field_i = (1.0-rmod_i)*efield[p_i] + rmod_i*efield[p_i+1]
                elif p_i == N_G -1:
                     e_x_field_i = efield[N_G-1]

                if measurement_mode:
                    mean_v_i = vx_i[k_i] + 0.5 * e_x_field_i * FACTOR_I
                    if 0 <= p_i < N_G-1:
                        counter_i_xt[p_i, t_index_xt]   += (1.0-rmod_i)
                        counter_i_xt[p_i+1, t_index_xt] += rmod_i
                        ui_xt[p_i, t_index_xt]   += (1.0-rmod_i) * mean_v_i
                        ui_xt[p_i+1, t_index_xt] += rmod_i * mean_v_i
                        v_sqr_i  = mean_v_i**2 + vy_i[k_i]**2 + vz_i[k_i]**2
                        energy_i = 0.5 * AR_MASS * v_sqr_i / EV_TO_J
                        meanei_xt[p_i, t_index_xt]   += (1.0-rmod_i) * energy_i
                        meanei_xt[p_i+1, t_index_xt] += rmod_i * energy_i
                
                vx_i[k_i] += e_x_field_i * FACTOR_I
                x_i[k_i]  += vx_i[k_i] * DT_I

                removed_i = False
                v_sqr_bound_i = 0.0
                if x_i[k_i] < 0:
                    N_i_abs_pow_arr_ref[0] += 1
                    removed_i = True
                    v_sqr_bound_i = vx_i[k_i]**2 + vy_i[k_i]**2 + vz_i[k_i]**2
                    energy_bound_i = 0.5 * AR_MASS * v_sqr_bound_i / EV_TO_J
                    energy_idx_ifed = int(energy_bound_i / DE_IFED)
                    if 0 <= energy_idx_ifed < N_IFED:
                        ifed_pow[energy_idx_ifed] += 1
                elif x_i[k_i] > L:
                    N_i_abs_gnd_arr_ref[0] += 1
                    removed_i = True
                    v_sqr_bound_i = vx_i[k_i]**2 + vy_i[k_i]**2 + vz_i[k_i]**2
                    energy_bound_i = 0.5 * AR_MASS * v_sqr_bound_i / EV_TO_J
                    energy_idx_ifed = int(energy_bound_i / DE_IFED)
                    if 0 <= energy_idx_ifed < N_IFED:
                        ifed_gnd[energy_idx_ifed] += 1
                
                if removed_i:
                    N_i_arr_ref[0] -= 1
                    idx_of_last_active_ion = N_i_arr_ref[0]
                    if k_i < idx_of_last_active_ion:
                        x_i[k_i] = x_i[idx_of_last_active_ion]
                        vx_i[k_i] = vx_i[idx_of_last_active_ion]
                        vy_i[k_i] = vy_i[idx_of_last_active_ion]
                        vz_i[k_i] = vz_i[idx_of_last_active_ion]
                    num_ions_to_process_this_timestep -= 1
                else:
                    vx_a = RMB_numba(rng_state)
                    vy_a = RMB_numba(rng_state)
                    vz_a = RMB_numba(rng_state)
                    
                    gx_coll_i = vx_i[k_i] - vx_a
                    gy_coll_i = vy_i[k_i] - vy_a
                    gz_coll_i = vz_i[k_i] - vz_a
                    g_sqr_coll_i = gx_coll_i**2 + gy_coll_i**2 + gz_coll_i**2
                    g_coll_i = math.sqrt(g_sqr_coll_i)
                    energy_coll_i = 0.5 * MU_ARAR * g_sqr_coll_i / EV_TO_J
                    
                    energy_idx_coll_i = min(int(energy_coll_i / DE_CS + 0.5), CS_RANGES - 1)
                    if energy_idx_coll_i < 0: energy_idx_coll_i = 0

                    nu_coll_i = sigma_tot_i[energy_idx_coll_i] * g_coll_i
                    p_coll_i = 1.0 - math.exp(-nu_coll_i * DT_I)

                    if rng_state.random() < p_coll_i:
                        collision_ion_jit(k_i,
                                          vx_a, vy_a, vz_a, 
                                          energy_idx_coll_i, 
                                          vx_i, vy_i, vz_i,
                                          rng_state)
                        N_i_coll_arr_ref[0] += 1
                    k_i += 1

        if measurement_mode:
            for p_xt in range(N_G):
                pot_xt[p_xt, t_index_xt]    += pot[p_xt]
                efield_xt[p_xt, t_index_xt] += efield[p_xt]
                ne_xt[p_xt, t_index_xt]     += e_density[p_xt]
                ni_xt[p_xt, t_index_xt]     += i_density[p_xt]
        
        # replaced by tqdm progress bar
        #if (t_loop_idx % 1000) == 0:
        #     print(" c = ", cycle_val, "  t = ", t_loop_idx, "  #e = ", N_e_arr_ref[0], "  #i = ", N_i_arr_ref[0])

# MODIFICATION for do_one_cycle
def do_one_cycle():
    global Time_arr, N_e_arr, N_i_arr # To ensure module globals are used if re-assigned
    global x_e, vx_e, vy_e, vz_e, x_i, vx_i, vy_i, vz_i
    global efield, pot, e_density, i_density, cumul_e_density, cumul_i_density
    global N_e_abs_pow_arr, N_e_abs_gnd_arr, N_i_abs_pow_arr, N_i_abs_gnd_arr
    global eepf, ifed_pow, ifed_gnd
    global pot_xt, efield_xt, ne_xt, ni_xt, ue_xt, ui_xt, meanee_xt, meanei_xt
    global counter_e_xt, counter_i_xt, ioniz_rate_xt
    global mean_energy_accu_center_arr, mean_energy_counter_center_arr
    global N_e_coll_arr, N_i_coll_arr
    global sigma_tot_e, sigma_tot_i, sigma # Make sure these are accessible
    global measurement_mode_arr, cycle_arr # Pass current cycle for printing
    global eepf_phase_resolved 
    
    do_one_cycle_jit_kernels(
        Time_arr, N_e_arr, N_i_arr,
        x_e, vx_e, vy_e, vz_e,
        x_i, vx_i, vy_i, vz_i,
        efield, pot, e_density, i_density, cumul_e_density, cumul_i_density,
        N_e_abs_pow_arr, N_e_abs_gnd_arr, N_i_abs_pow_arr, N_i_abs_gnd_arr,
        eepf, ifed_pow, ifed_gnd,
        pot_xt, efield_xt, ne_xt, ni_xt, ue_xt, ui_xt, meanee_xt, meanei_xt,
        counter_e_xt, counter_i_xt, ioniz_rate_xt,
        mean_energy_accu_center_arr, mean_energy_counter_center_arr,
        N_e_coll_arr, N_i_coll_arr,
        sigma_tot_e, sigma_tot_i,
        sigma[E_ELA,:], sigma[E_EXC,:], sigma[E_ION,:],
        sigma[I_ISO,:], sigma[I_BACK,:],
        measurement_mode_arr[0], cycle_arr[0],
        RNG,
        eepf_phase_resolved
    )
    # The conv.dat writing part is also correct.
    conv_file_path = BASE_OUTPUT_DIR / "conv.dat"
    with open(conv_file_path, "a") as datafile:
        datafile.write(f"{cycle_arr[0]} {N_e_arr[0]} {N_i_arr[0]}\n")
    
    return N_e_arr[0], N_i_arr[0]
        
# --- File I/O and Post-processing (Python) ---
def save_particle_data():
    picdata_file_path = BASE_OUTPUT_DIR / "picdata.npz"
    np.savez(picdata_file_path,
             Time=Time_arr[0], cycles_done=cycles_done_arr[0],
             N_e=N_e_arr[0], N_i=N_i_arr[0],
             x_e=x_e[:N_e_arr[0]], vx_e=vx_e[:N_e_arr[0]], vy_e=vy_e[:N_e_arr[0]], vz_e=vz_e[:N_e_arr[0]],
             x_i=x_i[:N_i_arr[0]], vx_i=vx_i[:N_i_arr[0]], vy_i=vy_i[:N_i_arr[0]], vz_i=vz_i[:N_i_arr[0]]
            )
    print(f">> eduPIC: data saved : {N_e_arr[0]} electrons {N_i_arr[0]} ions, "
          f"{cycles_done_arr[0]} cycles completed, time is {Time_arr[0]:.4e} [s] to {picdata_file_path}")

def load_particle_data():
    global x_e, vx_e, vy_e, vz_e, x_i, vx_i, vy_i, vz_i
    picdata_file_path = BASE_OUTPUT_DIR / "picdata.npz"
    try:
        data = np.load(picdata_file_path)
        Time_arr[0] = data['Time']
        cycles_done_arr[0] = data['cycles_done']
        
        N_e_loaded = int(data['N_e']) # Ensure it's int
        N_i_loaded = int(data['N_i'])
        N_e_arr[0] = N_e_loaded
        N_i_arr[0] = N_i_loaded

        # Ensure arrays are large enough, then copy data
        # This simple load assumes MAX_PARTICLES is sufficient.
        # A more robust load would re-allocate if N_e_loaded > MAX_E_PARTICLES
        if N_e_loaded > MAX_E_PARTICLES or N_i_loaded > MAX_I_PARTICLES:
            print("Error: Loaded particle count exceeds pre-allocated array size.")
            print(f"Loaded N_e: {N_e_loaded}, Max allowed: {MAX_E_PARTICLES}")
            print(f"Loaded N_i: {N_i_loaded}, Max allowed: {MAX_I_PARTICLES}")
            print("Increase MAX_PARTICLES_FACTOR and re-run (or delete picdata.npz for fresh start).")
            sys.exit(1)

        x_e[:N_e_loaded] = data['x_e']
        vx_e[:N_e_loaded] = data['vx_e']
        vy_e[:N_e_loaded] = data['vy_e']
        vz_e[:N_e_loaded] = data['vz_e']

        x_i[:N_i_loaded] = data['x_i']
        vx_i[:N_i_loaded] = data['vx_i']
        vy_i[:N_i_loaded] = data['vy_i']
        vz_i[:N_i_loaded] = data['vz_i']

        print(f">> eduPIC: data loaded from {picdata_file_path}: {N_e_arr[0]} electrons {N_i_arr[0]} ions, "
              f"{cycles_done_arr[0]} cycles completed before, time is {Time_arr[0]:.4e} [s]")
    except FileNotFoundError:
        print(f">> eduPIC: ERROR: No particle data file ({picdata_file_path}) found.")
        print("           Try running initial cycle using argument '0'.")
        sys.exit(1)

def save_density():
    density_file_path = BASE_OUTPUT_DIR / "density.dat"
    with open(density_file_path, "w") as f:
        f.write(f"# x[m] e_density[m^-3] i_density[m^-3]\n") # Header
        # Ensure no_of_cycles_arr[0] is not zero
        c = 1.0
        if no_of_cycles_arr[0] > 0 and N_T > 0 :
            c = 1.0 / (float(no_of_cycles_arr[0]) * float(N_T))
        
        for i in range(N_G):
            f.write(f"{i*DX:.12e} {cumul_e_density[i]*c:.12e} {cumul_i_density[i]*c:.12e}\n")

def save_eepf():
    eepf_file_path = BASE_OUTPUT_DIR / "eepf.dat"
    global eepf # ensure we are using the global
    with open(eepf_file_path, "w") as f:
        f.write(f"# energy[eV] f(E)[eV^-3/2]\n") # Header
        h_sum = np.sum(eepf) * DE_EEPF
        if h_sum == 0: h_sum = 1.0 # Avoid division by zero if eepf is all zeros
        
        for i in range(N_EEPF):
            energy = (i + 0.5) * DE_EEPF
            val_eepf = 0.0
            if energy > 0: # Avoid division by zero for sqrt(energy)
                 val_eepf = eepf[i] / h_sum / math.sqrt(energy) if h_sum !=0 else 0.0
            f.write(f"{energy:.4e} {val_eepf:.4e}\n")

def save_ifed():
    global mean_i_energy_pow, mean_i_energy_gnd
    ifed_file_path = BASE_OUTPUT_DIR / "ifed.dat"
    with open(ifed_file_path, "w") as f:
        f.write(f"# energy[eV] ifed_pow[eV^-1] ifed_gnd[eV^-1]\n") # Header
        h_pow_sum = np.sum(ifed_pow) * DE_IFED
        h_gnd_sum = np.sum(ifed_gnd) * DE_IFED
        if h_pow_sum == 0: h_pow_sum = 1.0
        if h_gnd_sum == 0: h_gnd_sum = 1.0

        _mean_i_energy_pow = 0.0
        _mean_i_energy_gnd = 0.0
        for i in range(N_IFED):
            energy = (i + 0.5) * DE_IFED
            p = float(ifed_pow[i]) / h_pow_sum if h_pow_sum !=0 else 0.0
            g = float(ifed_gnd[i]) / h_gnd_sum if h_gnd_sum !=0 else 0.0
            f.write(f"{energy:.4e} {p:.4e} {g:.4e}\n")
            _mean_i_energy_pow += energy * p * DE_IFED # C++ version sums energy*p, here multiply by DE_IFED for correct sum
            _mean_i_energy_gnd += energy * g * DE_IFED
        mean_i_energy_pow = _mean_i_energy_pow
        mean_i_energy_gnd = _mean_i_energy_gnd


def save_xt_1(distr_xt, fname_suffix): # fname_suffix e.g., "pot_xt.dat"
    xt_file_path = BASE_OUTPUT_DIR / fname_suffix
    with open(xt_file_path, "w") as f:
        for i in range(N_G):
            # Correct way to format a row of numbers
            f.write(" ".join([f"{val:.8e}" for val in distr_xt[i,:]]) + "\n")

def norm_all_xt():
    # Ensure no_of_cycles_arr[0] is not zero before division
    num_cycles_run = float(no_of_cycles_arr[0])
    if num_cycles_run == 0: num_cycles_run = 1.0 # Avoid division by zero if no cycles run for measurement

    f1 = float(N_XT) / (num_cycles_run * float(N_T))
    f2_denominator = (num_cycles_run * (PERIOD / float(N_XT)))
    f2 = WEIGHT / (ELECTRODE_AREA * DX) / f2_denominator if f2_denominator != 0 else 0.0


    global pot_xt, efield_xt, ne_xt, ni_xt, ue_xt, ui_xt, je_xt, ji_xt
    global powere_xt, poweri_xt, meanee_xt, meanei_xt, ioniz_rate_xt
    global counter_e_xt, counter_i_xt # These are used for normalization

    pot_xt *= f1
    efield_xt *= f1
    ne_xt *= f1
    ni_xt *= f1

    # Numba does not support np.divide(..., where=...) well. Do it carefully.
    # ue_xt = np.divide(ue_xt, counter_e_xt, out=np.zeros_like(ue_xt), where=counter_e_xt!=0)
    for r in range(N_G):
      for c_xt in range(N_XT):
        if counter_e_xt[r,c_xt] > 0:
          ue_xt[r,c_xt] /= counter_e_xt[r,c_xt]
          meanee_xt[r,c_xt] /= counter_e_xt[r,c_xt]
          ioniz_rate_xt[r,c_xt] = (ioniz_rate_xt[r,c_xt] / counter_e_xt[r,c_xt]) * f2 # C++ version multiplies by f2 after div
        else:
          ue_xt[r,c_xt] = 0.0
          meanee_xt[r,c_xt] = 0.0
          ioniz_rate_xt[r,c_xt] = 0.0

        if counter_i_xt[r,c_xt] > 0:
          ui_xt[r,c_xt] /= counter_i_xt[r,c_xt]
          meanei_xt[r,c_xt] /= counter_i_xt[r,c_xt]
        else:
          ui_xt[r,c_xt] = 0.0
          meanei_xt[r,c_xt] = 0.0
          
    je_xt = -ue_xt * ne_xt * E_CHARGE
    ji_xt = ui_xt * ni_xt * E_CHARGE
    
    powere_xt = je_xt * efield_xt
    poweri_xt = ji_xt * efield_xt


def save_all_xt():
    save_xt_1(pot_xt, "pot_xt.dat")
    save_xt_1(efield_xt, "efield_xt.dat")
    save_xt_1(ne_xt, "ne_xt.dat")
    save_xt_1(ni_xt, "ni_xt.dat")
    save_xt_1(je_xt, "je_xt.dat")
    save_xt_1(ji_xt, "ji_xt.dat")
    save_xt_1(powere_xt, "powere_xt.dat")
    save_xt_1(poweri_xt, "poweri_xt.dat")
    save_xt_1(meanee_xt, "meanee_xt.dat")
    save_xt_1(meanei_xt, "meanei_xt.dat")
    save_xt_1(ioniz_rate_xt, "ioniz_xt.dat")

def check_and_save_info():
    info_file_path = BASE_OUTPUT_DIR / "info.txt"
    with open(info_file_path, "w") as f:
        line = "-" * 80
        f.write(f"########################## eduPIC simulation report (Python/Numba) ############################\n")
        
        num_cycles_run = float(no_of_cycles_arr[0])
        if num_cycles_run == 0: num_cycles_run = 1.0

        density_val = 0.0
        if num_cycles_run > 0 and N_T > 0:
            density_val = cumul_e_density[N_G // 2] / num_cycles_run / float(N_T)
        
        plas_freq_val = 0.0
        if density_val > 0:
             plas_freq_val = E_CHARGE * math.sqrt(density_val / EPSILON0 / E_MASS)
        
        meane_val = 0.0
        if mean_energy_counter_center_arr[0] > 0:
            meane_val = mean_energy_accu_center_arr[0] / float(mean_energy_counter_center_arr[0])
        
        kT_val = 2.0 * meane_val * EV_TO_J / 3.0
        
        debye_length_val = 0.0
        if density_val > 0 and kT_val > 0: # Ensure kT_val is positive
            debye_length_val = math.sqrt(EPSILON0 * kT_val / density_val) / E_CHARGE
        
        sim_time_val = num_cycles_run / FREQUENCY
        
        ecoll_freq_val = 0.0
        icoll_freq_val = 0.0
        if sim_time_val > 0:
            # N_e_arr[0] is current number of particles. For average, might need time-averaged N_e
            # Original C++ uses N_e (current N_e) at the end of simulation.
            # Let's use the current N_e and N_i counts, assuming they are representative.
            if N_e_arr[0] > 0 : ecoll_freq_val = float(N_e_coll_arr[0]) / sim_time_val / float(N_e_arr[0])
            if N_i_arr[0] > 0 : icoll_freq_val = float(N_i_coll_arr[0]) / sim_time_val / float(N_i_arr[0])


        f.write(f"Simulation parameters:\n")
        f.write(f"Gap distance                          = {L:.4e} [m]\n")
        f.write(f"# of grid divisions                   = {N_G}\n")
        f.write(f"Frequency                             = {FREQUENCY:.4e} [Hz]\n")
        f.write(f"# of time steps / period              = {N_T}\n")
        f.write(f"# of electron / ion time steps        = {N_SUB}\n")
        f.write(f"Voltage amplitude                     = {VOLTAGE:.4e} [V]\n")
        f.write(f"Pressure (Ar)                         = {PRESSURE:.4e} [Pa]\n")
        f.write(f"Temperature                           = {TEMPERATURE:.4e} [K]\n")
        f.write(f"Superparticle weight                  = {WEIGHT:.4e}\n")
        f.write(f"# of simulation cycles in this run    = {no_of_cycles_arr[0]}\n")
        f.write(line + "\n")
        f.write(f"Plasma characteristics:\n")
        f.write(f"Electron density @ center             = {density_val:.4e} [m^-3]\n")
        f.write(f"Plasma frequency @ center             = {plas_freq_val:.4e} [rad/s]\n")
        f.write(f"Debye length @ center                 = {debye_length_val:.4e} [m]\n")
        f.write(f"Electron collision frequency          = {ecoll_freq_val:.4e} [1/s]\n")
        f.write(f"Ion collision frequency               = {icoll_freq_val:.4e} [1/s]\n")
        f.write(line + "\n")
        f.write(f"Stability and accuracy conditions:\n")
        conditions_OK = True
        c_check = plas_freq_val * DT_E
        f.write(f"Plasma frequency @ center * DT_e      = {c_check:.4e} (OK if less than 0.20)\n")
        if c_check > 0.2: conditions_OK = False
        
        c_check = DX / debye_length_val if debye_length_val > 0 else float('inf')
        f.write(f"DX / Debye length @ center            = {c_check:.4e} (OK if less than 1.00)\n")
        if c_check > 1.0: conditions_OK = False
        
        c_check = max_electron_coll_freq_jit(sigma_tot_e) * DT_E
        f.write(f"Max. electron coll. frequency * DT_E  = {c_check:.4e} (OK if less than 0.05)\n")
        if c_check > 0.05: conditions_OK = False
        
        c_check = max_ion_coll_freq_jit(sigma_tot_i) * DT_I
        f.write(f"Max. ion coll. frequency * DT_I       = {c_check:.4e} (OK if less than 0.05)\n")
        if c_check > 0.05: conditions_OK = False
        
        if not conditions_OK:
            f.write(line + "\n")
            f.write("** STABILITY AND ACCURACY CONDITION(S) VIOLATED - REFINE SIMULATION SETTINGS! **\n")
            f.write(line + "\n")
            print(">> eduPIC: ERROR: STABILITY AND ACCURACY CONDITION(S) VIOLATED! ")
            print(">> eduPIC: for details see 'info.txt' and refine simulation settings!")
        else:
            v_max_cfl = DX / DT_E
            e_max_cfl = 0.5 * E_MASS * v_max_cfl**2 / EV_TO_J
            f.write(f"Max e- energy for CFL condition       = {e_max_cfl:.4e} [eV]\n")
            f.write("Check EEPF to ensure that CFL is fulfilled for the majority of the electrons!\n")
            f.write(line + "\n")

            print(">> eduPIC: saving diagnostics data")
            save_density()
            save_eepf()
            save_ifed() # Calculates mean_i_energy_pow/gnd
            norm_all_xt()
            save_all_xt()

            f.write(f"Particle characteristics at the electrodes:\n")
            denom_flux = (num_cycles_run * PERIOD)
            if denom_flux == 0: denom_flux = 1.0 # Avoid div by zero
            
            ion_flux_pow = float(N_i_abs_pow_arr[0]) * WEIGHT / ELECTRODE_AREA / denom_flux
            ion_flux_gnd = float(N_i_abs_gnd_arr[0]) * WEIGHT / ELECTRODE_AREA / denom_flux
            ele_flux_pow = float(N_e_abs_pow_arr[0]) * WEIGHT / ELECTRODE_AREA / denom_flux
            ele_flux_gnd = float(N_e_abs_gnd_arr[0]) * WEIGHT / ELECTRODE_AREA / denom_flux

            f.write(f"Ion flux at powered electrode         = {ion_flux_pow:.4e} [m^-2 s^-1]\n")
            f.write(f"Ion flux at grounded electrode        = {ion_flux_gnd:.4e} [m^-2 s^-1]\n")
            f.write(f"Mean ion energy at powered electrode  = {mean_i_energy_pow:.4e} [eV]\n") # from save_ifed
            f.write(f"Mean ion energy at grounded electrode = {mean_i_energy_gnd:.4e} [eV]\n") # from save_ifed
            f.write(f"Electron flux at powered electrode    = {ele_flux_pow:.4e} [m^-2 s^-1]\n")
            f.write(f"Electron flux at grounded electrode   = {ele_flux_gnd:.4e} [m^-2 s^-1]\n")
            
            power_e_avg = np.sum(powere_xt) / float(N_XT * N_G) if (N_XT * N_G > 0) else 0.0
            power_i_avg = np.sum(poweri_xt) / float(N_XT * N_G) if (N_XT * N_G > 0) else 0.0
            f.write(line + "\n")
            f.write(f"Absorbed power calculated as <j*E>:\n")
            f.write(f"Electron power density (average)      = {power_e_avg:.4e} [W m^-3]\n")
            f.write(f"Ion power density (average)           = {power_i_avg:.4e} [W m^-3]\n")
            f.write(f"Total power density (average)         = {power_e_avg + power_i_avg:.4e} [W m^-3]\n")
            f.write(line + "\n")
        print(f">> eduPIC: for details see '{info_file_path}' and refine simulation settings!")

# --- Main ---
if __name__ == "__main__":
    print(">> eduPIC (Python/Numba): starting...")
    # ... (copyright notice similar to C++)

    if len(sys.argv) < 2:
        print(">> eduPIC: error = need number_of_cycles argument (0 for init)")
        sys.exit(1)

    arg1_cycles = int(sys.argv[1])
    
    if len(sys.argv) > 2 and sys.argv[2] == "m":
        measurement_mode_arr[0] = True
    else:
        measurement_mode_arr[0] = False

    if measurement_mode_arr[0]:
        print(">> eduPIC: measurement mode: on")
    else:
        print(">> eduPIC: measurement mode: off")

    # Initialize cross sections
    set_electron_cross_sections_ar()
    set_ion_cross_sections_ar()
    calc_total_cross_sections()
    # test_cross_sections(); sys.exit() # For testing CS output

    # Clear conv.dat logic
    if arg1_cycles == 0:
        conv_file_path = BASE_OUTPUT_DIR / "conv.dat"
        picdata_file_path = BASE_OUTPUT_DIR / "picdata.npz"
        if os.path.exists(conv_file_path) and not os.path.exists(picdata_file_path):
            print(f">> eduPIC: Fresh initialization (arg 0), removing old {conv_file_path}")
            os.remove(conv_file_path)
        # If picdata.npz exists, the C++ version exits with a warning.
        # We should do something similar.

    start_time = pytime.time()

    if arg1_cycles == 0: # Init mode
        picdata_file_path = BASE_OUTPUT_DIR / "picdata.npz"
        if os.path.exists(picdata_file_path):
            print(">> eduPIC: Warning: Data from previous calculation (picdata.npz) detected.")
            print("           To start a new simulation from the beginning, please delete all output files (picdata.npz, conv.dat etc.)")
            print("           before running with argument '0'.")
            print("           To continue the existing calculation, please specify the number of cycles to run, e.g., ./eduPIC.py 100")
            sys.exit(0)
        
        no_of_cycles_arr[0] = 1 # Run 1 cycle for init
        cycle_arr[0] = 1
        init_particles(N_INIT)
        print(">> eduPIC: running initializing cycle")
        Time_arr[0] = 0.0
        # Clear cumulative diagnostics for a fresh init run if measurement is on
        if measurement_mode_arr[0]:
            cumul_e_density[:] = 0.0
            cumul_i_density[:] = 0.0
            eepf[:] = 0.0
            ifed_pow[:] = 0
            ifed_gnd[:] = 0
            # XT arrays
            pot_xt[:] = 0.0; efield_xt[:] = 0.0; ne_xt[:] = 0.0; ni_xt[:] = 0.0
            ue_xt[:] = 0.0; ui_xt[:] = 0.0; je_xt[:] = 0.0; ji_xt[:] = 0.0
            powere_xt[:] = 0.0; poweri_xt[:] = 0.0; meanee_xt[:] = 0.0; meanei_xt[:] = 0.0
            counter_e_xt[:] = 0.0; counter_i_xt[:] = 0.0; ioniz_rate_xt[:] = 0.0
            eepf_phase_resolved[:] = 0.0
            # Counters
            N_e_abs_pow_arr[0] = 0; N_e_abs_gnd_arr[0] = 0; N_i_abs_pow_arr[0] = 0; N_i_abs_gnd_arr[0] = 0
            mean_energy_accu_center_arr[0] = 0.0; mean_energy_counter_center_arr[0] = 0
            N_e_coll_arr[0] = 0; N_i_coll_arr[0] = 0


        do_one_cycle()
        cycles_done_arr[0] = 1
    else: # Continue run
        no_of_cycles_arr[0] = arg1_cycles
        load_particle_data() # Loads Time_arr, cycles_done_arr, N_e_arr, N_i_arr, particle data
                             # Diagnostics are NOT loaded/saved for continuation by C++ version, implies they are reset or accumulated fresh
        print(f">> eduPIC: running {no_of_cycles_arr[0]} cycle(s)")
        
        # If measurement_mode is on for a continued run, clear/reset accumulators for *this segment* of the run
        if measurement_mode_arr[0]:
            print(">> eduPIC: Measurement mode on for continued run. Resetting diagnostic accumulators for this run segment.")
            cumul_e_density[:] = 0.0 # These are for average density over *this run*
            cumul_i_density[:] = 0.0
            eepf[:] = 0.0
            ifed_pow[:] = 0
            ifed_gnd[:] = 0
            # XT arrays
            pot_xt[:] = 0.0; efield_xt[:] = 0.0; ne_xt[:] = 0.0; ni_xt[:] = 0.0
            ue_xt[:] = 0.0; ui_xt[:] = 0.0; je_xt[:] = 0.0; ji_xt[:] = 0.0
            powere_xt[:] = 0.0; poweri_xt[:] = 0.0; meanee_xt[:] = 0.0; meanei_xt[:] = 0.0
            counter_e_xt[:] = 0.0; counter_i_xt[:] = 0.0; ioniz_rate_xt[:] = 0.0
            eepf_phase_resolved[:] = 0.0
            # Counters related to fluxes and collisions should also be reset for the current measurement period
            N_e_abs_pow_arr[0] = 0; N_e_abs_gnd_arr[0] = 0; N_i_abs_pow_arr[0] = 0; N_i_abs_gnd_arr[0] = 0
            mean_energy_accu_center_arr[0] = 0.0; mean_energy_counter_center_arr[0] = 0
            N_e_coll_arr[0] = 0; N_i_coll_arr[0] = 0


        print(f">> eduPIC: running {no_of_cycles_arr[0]} cycle(s)")
    
        # Create the tqdm instance before the loop
        progress_bar = tqdm(range(no_of_cycles_arr[0]), desc="Simulating Cycles", unit="cycle")

        for i in progress_bar:
            cycle_arr[0] = cycles_done_arr[0] + 1 + i

            # Call the function and get the particle counts
            current_Ne, current_Ni = do_one_cycle()

            # Update the progress bar's description with the new counts
            # Using set_postfix is a clean way to add extra info
            progress_bar.set_postfix(Ne=current_Ne, Ni=current_Ni)

        cycles_done_arr[0] += no_of_cycles_arr[0]

    save_particle_data()
    if measurement_mode_arr[0]:
        check_and_save_info()
    
    end_time = pytime.time()
    print(f">> eduPIC: simulation of {no_of_cycles_arr[0]} cycle(s) is completed.")
    print(f">> eduPIC: Output data is in {BASE_OUTPUT_DIR}")
    print(f">> eduPIC: Total wall clock time: {end_time - start_time:.2f} seconds.")