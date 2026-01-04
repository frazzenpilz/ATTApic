import numpy as np
import numba
import math
import sys
import os
import time as pytime
import scipy.interpolate

from pathlib import Path
from tqdm import tqdm

# --- User Input ---
# If running non-interactively, you can hardcode this or use sys.argv
if len(sys.argv) > 3:
    RUN_ID = sys.argv[3]
else:
    RUN_ID = "ATTA"
    # Default to a timestamped run if not provided
    # import datetime
    # RUN_ID = f"ATTA_Source_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

BASE_OUTPUT_DIR = Path("results") / RUN_ID
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f">> eduPIC: Output will be saved to: {BASE_OUTPUT_DIR}")

# ===================================================================
# === PHYSICAL CONSTANTS (NO SCALING) ===
# ===================================================================
PI = 3.141592653589793
TWO_PI = 2.0 * PI
E_CHARGE = 1.60217662e-19
EV_TO_J = E_CHARGE
E_MASS = 9.10938356e-31
AR_MASS = 6.63352090e-26  # Argon-40
MU_ARAR = AR_MASS / 2.0
K_BOLTZMANN = 1.38064852e-23
EPSILON0 = 8.85418781e-12 # REAL PHYSICAL VACUUM PERMITTIVITY

# ===================================================================
# === ATTA SOURCE PARAMETERS (Ritterbusch Design) ===
# ===================================================================

# --- Simulation Domain & Resolution ---
# L = 6cm. 
# Stability Requirement: dx < Debye_Length (~60um at 2eV, 1e16 density)
N_G = 4000            # Grid points (dx = 60 microns)

# Time Step Requirement: w_p * dt < 0.2
# At 1e16 density, w_p ~ 1.8e11 rad/s. dt must be < 1.0 ps.
# T_RF (130MHz) = 7700 ps. 
# We need N_T >= 7700.
N_T = 4000            # Time steps (Reduced slightly for speed, w_p is lower at start)

# --- Gas Conditions (Estimated Source Pressure) ---
# Chamber is 2e-5 mbar, but tube interior must be higher for ignition.
# Standard RF discharge pressure: ~10 mTorr = 1.33 Pa.
PRESSURE = 0.5        # [Pa] CORRECTED PHYSICS
TEMPERATURE = 160.0   # [K]

# --- Geometry (Ceramic Tube) ---
TUBE_RADIUS = 0.005   # [m] 5mm inner radius
COIL_LENGTH = 0.045   # [m] 45mm active coil length
L = 0.060             # [m] 6 cm total simulation length
COIL_CENTER = L / 2.0 

# --- RF Driver ---
RF_FREQ_ICP = 150.0e6 # [Hz] 150 MHz
OMEGA_ICP = TWO_PI * RF_FREQ_ICP
VOLTAGE = 0.0         # Grounded shield

# --- Power Target ---
TARGET_POWER_DENSITY = 80 # [W/m^3] Increased target for higher density mode

# --- Initial Field ---
# Start weak. Let the feedback loop ramp it up.
E_INDUCED_AMPLITUDE = 100.0 # [V/m]

# --- Superparticles ---
# We need enough particles so we don't have empty cells with N_G=1000
# but not so many that it runs forever.
WEIGHT = 2.0e3        # Lower weight = More real particles per macroparticle (Higher precision)
N_INIT = 5_000       # Start with more particles to fill the fine grid
MAX_PARTICLES_FACTOR = 500 # Allow buffer for ionization growth
MAX_E_PARTICLES = int(N_INIT * MAX_PARTICLES_FACTOR)
MAX_I_PARTICLES = int(N_INIT * MAX_PARTICLES_FACTOR)
ELECTRODE_AREA = PI * TUBE_RADIUS**2 

# ===================================================================

# Derived Constants
PERIOD = 1.0 / RF_FREQ_ICP
DT_E = PERIOD / float(N_T)
N_SUB = 20 # Subcycling for ions
DT_I = N_SUB * DT_E
DX = L / float(N_G - 1)
INV_DX = 1.0 / DX
GAS_DENSITY = PRESSURE / (K_BOLTZMANN * TEMPERATURE)
OMEGA = 0.0 # DC/Capacitive frequency is 0

# --- Cross Sections ---
N_CS = 5
E_ELA = 0; E_EXC = 1; E_ION = 2; I_ISO = 3; I_BACK = 4
E_EXC_TH = 11.5; E_ION_TH = 15.8
CS_RANGES = 1000000
DE_CS = 0.001

# Memory Allocation
sigma = np.zeros((N_CS, CS_RANGES), dtype=np.float32)
sigma_tot_e = np.zeros(CS_RANGES, dtype=np.float32)
sigma_tot_i = np.zeros(CS_RANGES, dtype=np.float32)

# Particle Arrays
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

# Fields & Diagnostics
efield = np.zeros(N_G, dtype=np.float64)
pot = np.zeros(N_G, dtype=np.float64)
e_density = np.zeros(N_G, dtype=np.float64)
i_density = np.zeros(N_G, dtype=np.float64)
cumul_e_density = np.zeros(N_G, dtype=np.float64)
cumul_i_density = np.zeros(N_G, dtype=np.float64)

# Counters
N_e_abs_pow_arr = np.array([0], dtype=np.uint64)
N_e_abs_gnd_arr = np.array([0], dtype=np.uint64)
N_i_abs_pow_arr = np.array([0], dtype=np.uint64)
N_i_abs_gnd_arr = np.array([0], dtype=np.uint64)

# EEPF
N_EEPF = 2000; DE_EEPF = 0.05
eepf = np.zeros(N_EEPF, dtype=np.float64)

# IFED
N_IFED = 200; DE_IFED = 1.0
ifed_pow = np.zeros(N_IFED, dtype=np.int32)
ifed_gnd = np.zeros(N_IFED, dtype=np.int32)
mean_i_energy_pow = 0.0; mean_i_energy_gnd = 0.0

# Spatiotemporal (XT) Diagnostics
N_BIN = 20; N_XT = N_T // N_BIN
pot_xt = np.zeros((N_G, N_XT), dtype=np.float64)
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

eepf_phase_resolved = np.zeros((N_XT, N_EEPF), dtype=np.float64)

# Global accumulator for Te calculation (needed for Radial loss)
mean_energy_accu_center_arr = np.array([0.0], dtype=np.float64)
mean_energy_counter_center_arr = np.array([0], dtype=np.uint64)
N_e_coll_arr = np.array([0], dtype=np.uint64)
N_i_coll_arr = np.array([0], dtype=np.uint64)

# --- FEEDBACK STATE VARIABLES ---
E_amp_state = np.array([E_INDUCED_AMPLITUDE], dtype=np.float64)
Power_accum_state = np.array([0.0], dtype=np.float64)
Rad_loss_prob_state = np.array([0.0], dtype=np.float64)

# Simulation State
Time_arr = np.array([0.0], dtype=np.float64)
cycle_arr = np.array([0], dtype=np.int32)
no_of_cycles_arr = np.array([0], dtype=np.int32)

cycles_done_arr = np.array([0], dtype=np.int32)
cycles_session_arr = np.array([0], dtype=np.int32)
measurement_mode_arr = np.array([False], dtype=np.bool_)

RNG = np.random.default_rng()

# -------------------------------------------------------------------
# --- CROSS SECTION LOADERS (Standard LXCat) ---
# -------------------------------------------------------------------
def load_individual_lxcat_file(filepath, target_energy_eV_grid, units_in_m2=True, header_skip_keyword="-----------------------------"):
    energies_lxcat = []; sigmas_lxcat = []
    data_section_found = False; data_parsing_active = False
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if header_skip_keyword in line:
                    if not data_section_found:
                        data_section_found = True; data_parsing_active = True; continue
                    else:
                        data_parsing_active = False; break
                if data_parsing_active:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            e_val = float(parts[0]); s_val = float(parts[1])
                            energies_lxcat.append(e_val); sigmas_lxcat.append(s_val)
                        except ValueError: continue
    except: return np.zeros_like(target_energy_eV_grid, dtype=np.float32)

    if not energies_lxcat: return np.zeros_like(target_energy_eV_grid, dtype=np.float32)
    
    energies_lxcat = np.array(energies_lxcat, dtype=np.float32)
    sigmas_lxcat = np.array(sigmas_lxcat, dtype=np.float32)
    if not units_in_m2: sigmas_lxcat *= 1e-4

    if len(energies_lxcat) > 1:
        # Sort and Interpolate
        idx = np.argsort(energies_lxcat)
        interp_func = scipy.interpolate.interp1d(energies_lxcat[idx], sigmas_lxcat[idx], kind='linear', bounds_error=False, fill_value=0.0)
        return interp_func(target_energy_eV_grid).astype(np.float32)
    return np.zeros_like(target_energy_eV_grid, dtype=np.float32)

# Analytical fallbacks (just in case)
def qmel(en): return 1e-20 * (abs(6.0/pow(1.0+(en/0.1)+pow(en/0.6,2.0),3.3) - 1.1*pow(en,1.4)/(1.0+pow(en/15.0,1.2))/math.sqrt(1.0+pow(en/5.5,2.5)+pow(en/60.0,4.1))) + 0.05/pow(1.0+en/10.0,2.0) + 0.01*pow(en,3.0)/(1.0+pow(en/12.0,6.0)))
def qexc(en): return 1e-20 * (0.034*pow(en-11.5,1.1)*(1.0+pow(en/15.0,2.8))/(1.0+pow(en/23.0,5.5)) + 0.023*(en-11.5)/pow(1.0+en/80.0,1.9)) if en > E_EXC_TH else 0.0
def qion(en): return 1e-20 * (970.0*(en-15.8)/pow(70.0+en,2.0) + 0.06*pow(en-15.8,2.0)*math.exp(-en/9.0)) if en > E_ION_TH else 0.0
def qiso_ion(en): return 2e-19 * pow(en,-0.5)/(1.0+en) + 3e-19*en/pow(1.0+en/3.0,2.0)
def qmom_ion(en): return 1.15e-18 * pow(en,-0.1)*pow(1.0+0.015/en,0.6)

def set_cross_sections():
    print(">> eduPIC: Loading Cross Sections...")
    sim_e_grid = (np.arange(CS_RANGES, dtype=np.float32) + 1) * DE_CS
    # For Ions, we stick to the weird original scaling or just use same grid? Original used 2xDE_CS
    sim_i_grid = (np.arange(CS_RANGES, dtype=np.float32) + 1) * (2.0 * DE_CS)

    lxcat_dir = Path("./lxcat_data")
    
    # Electrons
    f_el = lxcat_dir / "ar_electron_elastic_effective.txt"
    f_ex = lxcat_dir / "ar_electron_excitation_11.5eV.txt"
    f_io = lxcat_dir / "ar_electron_ionization.txt"
    
    if f_el.exists() and f_ex.exists() and f_io.exists():
        sigma[E_ELA, :] = load_individual_lxcat_file(f_el, sim_e_grid)
        sigma[E_EXC, :] = load_individual_lxcat_file(f_ex, sim_e_grid)
        sigma[E_ION, :] = load_individual_lxcat_file(f_io, sim_e_grid)
    else:
        print(">> Warning: LXCat files missing, using analytical fallback for electrons.")
        sigma[E_ELA, :] = [qmel(x) for x in sim_e_grid]
        sigma[E_EXC, :] = [qexc(x) for x in sim_e_grid]
        sigma[E_ION, :] = [qion(x) for x in sim_e_grid]

    # Ions
    f_i_iso = lxcat_dir / "ar_ion_isotropic.txt"
    f_i_bak = lxcat_dir / "ar_ion_backscatter.txt"
    
    if f_i_iso.exists() and f_i_bak.exists():
        sigma[I_ISO, :] = load_individual_lxcat_file(f_i_iso, sim_i_grid)
        sigma[I_BACK, :] = load_individual_lxcat_file(f_i_bak, sim_i_grid)
    else:
        print(">> Warning: LXCat files missing, using analytical fallback for ions.")
        qiso = np.array([qiso_ion(x) for x in sim_i_grid])
        qmom = np.array([qmom_ion(x) for x in sim_i_grid])
        sigma[I_ISO, :] = qiso
        sigma[I_BACK, :] = np.maximum(0.0, (qmom - qiso)/2.0)

    # Calculate Totals
    for i in range(CS_RANGES):
        sigma_tot_e[i] = (sigma[E_ELA,i] + sigma[E_EXC,i] + sigma[E_ION,i]) * GAS_DENSITY
        sigma_tot_i[i] = (sigma[I_ISO,i] + sigma[I_BACK,i]) * GAS_DENSITY

# -------------------------------------------------------------------
# --- INITIALIZATION ---
# -------------------------------------------------------------------
def init_particles(nseed):
    print(">> eduPIC: Initializing particles...")
    N_e_arr[0] = nseed; N_i_arr[0] = nseed
    
    # Uniform spatial distribution
    x_e[:nseed] = RNG.random(nseed) * L
    x_i[:nseed] = RNG.random(nseed) * L
    
    # Maxwellian velocity at 2.0 eV for electrons (warm start)
    v_th_e = math.sqrt(E_CHARGE * 2.0 / E_MASS)
    vx_e[:nseed] = RNG.normal(0, v_th_e, nseed)
    vy_e[:nseed] = RNG.normal(0, v_th_e, nseed)
    vz_e[:nseed] = RNG.normal(0, v_th_e, nseed)
    
    # Maxwellian velocity at Gas Temp for ions
    v_th_i = math.sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS)
    vx_i[:nseed] = RNG.normal(0, v_th_i, nseed)
    vy_i[:nseed] = RNG.normal(0, v_th_i, nseed)
    vz_i[:nseed] = RNG.normal(0, v_th_i, nseed)

# -------------------------------------------------------------------
# --- NUMBA KERNELS ---
# -------------------------------------------------------------------
@numba.njit(fastmath=True, cache=True, nogil=True)
def RMB_numba(rng_state):
    u1 = rng_state.random(); u2 = rng_state.random()
    return math.sqrt(-2.0*math.log(u1)) * math.cos(TWO_PI*u2) * math.sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS)

@numba.njit(fastmath=True, nogil=True)
def solve_Poisson_jit(pot_arr, efield_arr, rho1_arr, N_G_local, DX_local, INV_DX_local, EPSILON0_local):
    # Standard Thomas Algorithm for Tridiagonal Matrix
    # ICP Condition: Grounded Walls => pot[0] = 0, pot[L] = 0
    # Equation: phi[i-1] - 2phi[i] + phi[i+1] = -rho[i]*dx^2/eps0
    
    # Arrays for Thomas algorithm
    c_prime = np.empty(N_G_local, dtype=np.float64)
    d_prime = np.empty(N_G_local, dtype=np.float64)
    
    # Boundary Conditions
    pot_arr[0] = 0.0
    pot_arr[N_G_local - 1] = 0.0
    
    # Coefficients
    # Main diag (b) = -2, Off diag (a, c) = 1
    # RHS (d) = -rho * dx^2 / eps0
    const = - (DX_local * DX_local) / EPSILON0_local
    
    # Forward Sweep (start from i=1)
    # c_prime[1] = c1 / b1 = 1 / -2 = -0.5
    c_prime[1] = -0.5
    # d_prime[1] = (d1 - a1*pot[0]) / b1 = d1 / -2
    d_prime[1] = (rho1_arr[1] * const) / -2.0
    
    for i in range(2, N_G_local - 1):
        # c_prime[i] = c[i] / (b[i] - a[i]*c_prime[i-1])
        temp = 1.0 / (-2.0 - 1.0 * c_prime[i-1])
        c_prime[i] = 1.0 * temp
        # d_prime[i] = (d[i] - a[i]*d_prime[i-1]) / ...
        d_prime[i] = ((rho1_arr[i] * const) - 1.0 * d_prime[i-1]) * temp
        
    # Back Substitution
    for i in range(N_G_local - 2, 0, -1):
        pot_arr[i] = d_prime[i] - c_prime[i] * pot_arr[i+1]
        
    # Electric Field E = -dphi/dx (Central Difference)
    for i in range(1, N_G_local - 1):
        efield_arr[i] = (pot_arr[i-1] - pot_arr[i+1]) * (0.5 * INV_DX_local)
    
    # Boundaries (Forward/Backward Diff)
    efield_arr[0] = (pot_arr[0] - pot_arr[1]) * INV_DX_local
    efield_arr[N_G_local-1] = (pot_arr[N_G_local-2] - pot_arr[N_G_local-1]) * INV_DX_local

@numba.njit(fastmath=True, nogil=True)
def collision_electron_jit(k, eidx, Ne_ref, Ni_ref, xe, vxe, vye, vze, xi, vxi, vyi, vzi, max_e, max_i, rng):
    # Simplified collision logic for brevity, assuming isotropic scattering mostly
    # Mass factors
    m_e = E_MASS; m_n = AR_MASS
    F1 = m_e / (m_e + m_n); F2 = m_n / (m_e + m_n)
    
    # Initial velocity
    vx = vxe[k]; vy = vye[k]; vz = vze[k]
    v_mag = math.sqrt(vx*vx + vy*vy + vz*vz)
    if v_mag == 0: return

    # Probabilities
    s_el = sigma[E_ELA, eidx]
    s_ex = sigma[E_EXC, eidx]
    s_io = sigma[E_ION, eidx]
    s_tot = s_el + s_ex + s_io
    if s_tot == 0: return

    r = rng.random()
    type = 0 # 0: elastic, 1: exc, 2: ion
    if r < s_el/s_tot: type = 0
    elif r < (s_el+s_ex)/s_tot: type = 1
    else: type = 2

    # Energy Loss
    if type == 1: # Excitation
        loss = E_EXC_TH * EV_TO_J
        kin = 0.5 * m_e * v_mag*v_mag
        if kin > loss:
            v_rem = math.sqrt(2.0*(kin-loss)/m_e)
            # Scale velocity vector
            scale = v_rem / v_mag
            vx *= scale; vy *= scale; vz *= scale
        else: return # Should not happen given cross section threshold
    
    elif type == 2: # Ionization
        loss = E_ION_TH * EV_TO_J
        kin = 0.5 * m_e * v_mag*v_mag
        if kin > loss:
            rem_E = kin - loss
            # Split energy randomly (simplification)
            r_split = rng.random()
            E_prim = rem_E * r_split
            E_sec = rem_E * (1.0 - r_split)
            
            # Update primary
            v_prim = math.sqrt(2.0*E_prim/m_e)
            
            # Create Secondary Electron
            if Ne_ref[0] < max_e:
                idx = Ne_ref[0]
                xe[idx] = xe[k]
                # Isotropic emission for secondary
                cos_th = 1.0 - 2.0*rng.random()
                sin_th = math.sqrt(1.0 - cos_th*cos_th)
                phi = TWO_PI * rng.random()
                v_sec = math.sqrt(2.0*E_sec/m_e)
                vxe[idx] = v_sec * sin_th * math.cos(phi)
                vye[idx] = v_sec * sin_th * math.sin(phi)
                vze[idx] = v_sec * cos_th
                Ne_ref[0] += 1
            
            # Create Ion
            if Ni_ref[0] < max_i:
                idxi = Ni_ref[0]
                xi[idxi] = xe[k]
                vxi[idxi] = RMB_numba(rng); vyi[idxi] = RMB_numba(rng); vzi[idxi] = RMB_numba(rng)
                Ni_ref[0] += 1
            
            # Scatter primary isotropically (simplification)
            cos_th = 1.0 - 2.0*rng.random()
            sin_th = math.sqrt(1.0 - cos_th*cos_th)
            phi = TWO_PI * rng.random()
            vx = v_prim * sin_th * math.cos(phi)
            vy = v_prim * sin_th * math.sin(phi)
            vz = v_prim * cos_th
            
        else: return

    # Elastic Scattering (Angle change)
    # Simple isotropic scattering for now
    cos_th = 1.0 - 2.0*rng.random()
    sin_th = math.sqrt(1.0 - cos_th*cos_th)
    phi = TWO_PI * rng.random()
    v_new_mag = math.sqrt(vx*vx + vy*vy + vz*vz)
    
    vxe[k] = v_new_mag * sin_th * math.cos(phi)
    vye[k] = v_new_mag * sin_th * math.sin(phi)
    vze[k] = v_new_mag * cos_th

@numba.njit(fastmath=True, nogil=True)
def collision_ion_jit(k, eidx, vx_i, vy_i, vz_i, rng):
    # Charge Exchange / Elastic
    # Simplified: Isotropic scattering + Thermalization
    # Get random neutral velocity
    vnx = RMB_numba(rng); vny = RMB_numba(rng); vnz = RMB_numba(rng)
    
    s_iso = sigma[I_ISO, eidx]
    s_back = sigma[I_BACK, eidx]
    
    if (s_iso + s_back) == 0: return
    
    # If backward (Charge Exchange), ion takes neutral velocity
    if rng.random() < s_back / (s_iso + s_back):
        vx_i[k] = vnx; vy_i[k] = vny; vz_i[k] = vnz
    else:
        # Isotropic: Elastic sphere collision approximation
        # Just thermalize for simplicity in 1D approximation
        vx_i[k] = (vx_i[k] + vnx) * 0.5
        vy_i[k] = (vy_i[k] + vny) * 0.5
        vz_i[k] = (vz_i[k] + vnz) * 0.5

@numba.njit(fastmath=True, nogil=True)
def do_one_cycle_jit_kernels(
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
    measurement_mode, cycle_val,
    rng_state,
    eepf_phase_resolved,
    E_amp_ref, Power_accum_ref, Rad_loss_prob_ref
    ):

    DV = ELECTRODE_AREA * DX
    FACTOR_W = WEIGHT / DV
    FACTOR_E = DT_E / E_MASS * E_CHARGE
    FACTOR_I = DT_I / AR_MASS * E_CHARGE
    MIN_X = 0.45 * L
    MAX_X = 0.55 * L
    
    # Coil geometry for s(x): Centered Solenoid
    # Flat top in the middle, decaying at edges
    COIL_HALF_LEN = COIL_LENGTH / 2.0
    
    Power_accum_ref[0] = 0.0
    current_E_amp = E_amp_ref[0]
    current_P_loss = Rad_loss_prob_ref[0]
    
    rho = np.empty(N_G, dtype=np.float64)

    for t_loop_idx in range(N_T):
        Time_arr_ref[0] += DT_E
        t_index_xt = t_loop_idx // N_BIN
        
        sin_omega_t = math.sin(OMEGA_ICP * Time_arr_ref[0])

        # --- DENSITY GATHER ---
        e_density[:] = 0.0
        for k in range(N_e_arr_ref[0]):
            pos_norm = x_e[k] * INV_DX
            p = int(pos_norm)
            rmod = pos_norm - p
            if 0 <= p < N_G - 1:
                e_density[p] += (1.0 - rmod) * FACTOR_W
                e_density[p+1] += rmod * FACTOR_W
        
        # Standard weighting boundary correction
        e_density[0] *= 2.0; e_density[N_G-1] *= 2.0
        cumul_e_density += e_density

        if (t_loop_idx % N_SUB) == 0:
            i_density[:] = 0.0
            for k in range(N_i_arr_ref[0]):
                pos_norm = x_i[k] * INV_DX
                p = int(pos_norm)
                rmod = pos_norm - p
                if 0 <= p < N_G - 1:
                    i_density[p] += (1.0 - rmod) * FACTOR_W
                    i_density[p+1] += rmod * FACTOR_W
            i_density[0] *= 2.0; i_density[N_G-1] *= 2.0
            
        cumul_i_density += i_density

        # --- FIELD SOLVE ---
        for i in range(N_G):
            rho[i] = E_CHARGE * (i_density[i] - e_density[i])
        solve_Poisson_jit(pot, efield, rho, N_G, DX, INV_DX, EPSILON0)

        # --- XT DIAGNOSTICS ACCUMULATION ---
        # Only accumulate if measurement mode is on
        if measurement_mode:
            for p_xt in range(N_G):
                pot_xt[p_xt, t_index_xt]    += pot[p_xt]
                efield_xt[p_xt, t_index_xt] += efield[p_xt]
                ne_xt[p_xt, t_index_xt]     += e_density[p_xt]
                ni_xt[p_xt, t_index_xt]     += i_density[p_xt]


        # --- ELECTRONS ---
        k = 0
        while k < N_e_arr_ref[0]:
            # Interpolate E_x
            pos_norm = x_e[k] * INV_DX
            p = int(pos_norm)
            rmod = pos_norm - p
            Ex = 0.0
            if 0 <= p < N_G - 1:
                Ex = (1.0 - rmod)*efield[p] + rmod*efield[p+1]
            
            # Inductive Field Profile s(x)
            # Solenoid: roughly uniform inside, decays outside.
            # Center of domain is L/2. Coil extends +/- COIL_HALF_LEN
            dist_from_center = abs(x_e[k] - COIL_CENTER)
            s_x = 0.0
            if dist_from_center < COIL_HALF_LEN:
                s_x = 1.0
            else:
                # Linear decay over 5mm edge
                decay_len = 0.005
                over = dist_from_center - COIL_HALF_LEN
                if over < decay_len:
                    s_x = 1.0 - (over / decay_len)
            
            Ey_ind = current_E_amp * s_x * sin_omega_t

            # --- PER-PARTICLE XT DIAGNOSTICS (Electrons) ---
            if measurement_mode:
                # Mean velocity at half-step for accurate flux calculation
                mean_v = vx_e[k] - 0.5 * Ex * FACTOR_E
                
                # Accumulate per-particle quantities (with linear weighting)
                if 0 <= p < N_G - 1:
                    counter_e_xt[p, t_index_xt]   += (1.0 - rmod)
                    counter_e_xt[p+1, t_index_xt] += rmod
                    ue_xt[p, t_index_xt]   += (1.0 - rmod) * mean_v
                    ue_xt[p+1, t_index_xt] += rmod * mean_v
                    
                    # Energy calculation
                    v_sqr = mean_v * mean_v + vy_e[k] * vy_e[k] + vz_e[k] * vz_e[k]
                    energy = 0.5 * E_MASS * v_sqr / EV_TO_J
                    meanee_xt[p, t_index_xt]   += (1.0 - rmod) * energy
                    meanee_xt[p+1, t_index_xt] += rmod * energy
                    
                    # Ionization rate
                    energy_idx_cs = min(int(energy / DE_CS + 0.5), CS_RANGES - 1)
                    if energy_idx_cs < 0: energy_idx_cs = 0
                    velocity = math.sqrt(v_sqr)
                    rate = sigma[E_ION, energy_idx_cs] * velocity * DT_E * GAS_DENSITY
                    ioniz_rate_xt[p, t_index_xt]   += (1.0 - rmod) * rate
                    ioniz_rate_xt[p+1, t_index_xt] += rmod * rate
                
                # EEPF accumulation (for central region)
                v_sqr_eepf = vx_e[k]**2 + vy_e[k]**2 + vz_e[k]**2
                en_ev = 0.5 * E_MASS * v_sqr_eepf / EV_TO_J
                if MIN_X < x_e[k] < MAX_X:
                    mean_energy_accu_center_arr_ref[0] += en_ev
                    mean_energy_counter_center_arr_ref[0] += 1
                    e_idx = int(en_ev / DE_EEPF)
                    if 0 <= e_idx < N_EEPF:
                        eepf[e_idx] += 1.0


            # Push
            vx_e[k] -= Ex * FACTOR_E
            
            vy_old = vy_e[k]
            vy_e[k] -= Ey_ind * FACTOR_E
            
            # Power Accumulation
            vy_avg = 0.5 * (vy_old + vy_e[k])
            # P = J.E = (-e * v_y) * E_y
            Power_accum_ref[0] += (-E_CHARGE * vy_avg * Ey_ind) * WEIGHT
            
            x_e[k] += vx_e[k] * DT_E
            
            # Boundaries & Losses
            removed = False
            # Axial (Walls)
            if x_e[k] <= 0 or x_e[k] >= L:
                removed = True
            
            # Radial (Virtual)
            if not removed:
                if rng_state.random() < current_P_loss:
                    removed = True
            
            if removed:
                last = N_e_arr_ref[0] - 1
                if k < last:
                    x_e[k] = x_e[last]; vx_e[k] = vx_e[last]; vy_e[k] = vy_e[last]; vz_e[k] = vz_e[last]
                N_e_arr_ref[0] -= 1
            else:
                # Collisions
                v_mag = math.sqrt(vx_e[k]**2 + vy_e[k]**2 + vz_e[k]**2)
                en_ev = 0.5 * E_MASS * v_mag*v_mag / EV_TO_J
                c_idx = int(en_ev / DE_CS)
                if c_idx >= CS_RANGES: c_idx = CS_RANGES - 1
                
                nu = sigma_tot_e[c_idx] * v_mag
                if rng_state.random() < (1.0 - math.exp(-nu * DT_E)):
                    collision_electron_jit(k, c_idx, N_e_arr_ref, N_i_arr_ref, x_e, vx_e, vy_e, vz_e, x_i, vx_i, vy_i, vz_i, MAX_E_PARTICLES, MAX_I_PARTICLES, rng_state)
                k += 1

        # --- IONS ---
        if (t_loop_idx % N_SUB) == 0:
            k = 0
            P_loss_i = 1.0 - math.exp(N_SUB * math.log(1.0 - current_P_loss))
            
            while k < N_i_arr_ref[0]:
                pos_norm = x_i[k] * INV_DX
                p = int(pos_norm)
                rmod = pos_norm - p
                Ex = 0.0
                if 0 <= p < N_G - 1:
                    Ex = (1.0 - rmod)*efield[p] + rmod*efield[p+1]
                
                # --- PER-PARTICLE XT DIAGNOSTICS (Ions) ---
                if measurement_mode:
                    # Mean velocity at half-step for accurate flux calculation
                    mean_v_i = vx_i[k] + 0.5 * Ex * FACTOR_I
                    
                    # Accumulate per-particle quantities (with linear weighting)
                    if 0 <= p < N_G - 1:
                        counter_i_xt[p, t_index_xt]   += (1.0 - rmod)
                        counter_i_xt[p+1, t_index_xt] += rmod
                        ui_xt[p, t_index_xt]   += (1.0 - rmod) * mean_v_i
                        ui_xt[p+1, t_index_xt] += rmod * mean_v_i
                        
                        # Energy calculation
                        v_sqr_i = mean_v_i**2 + vy_i[k]**2 + vz_i[k]**2
                        energy_i = 0.5 * AR_MASS * v_sqr_i / EV_TO_J
                        meanei_xt[p, t_index_xt]   += (1.0 - rmod) * energy_i
                        meanei_xt[p+1, t_index_xt] += rmod * energy_i
                
                # Push
                vx_i[k] += Ex * FACTOR_I
                x_i[k] += vx_i[k] * DT_I
                
                removed = False
                if x_i[k] <= 0 or x_i[k] >= L: removed = True
                if not removed and rng_state.random() < P_loss_i: removed = True
                
                if removed:
                    last = N_i_arr_ref[0] - 1
                    if k < last:
                        x_i[k] = x_i[last]; vx_i[k] = vx_i[last]; vy_i[k] = vy_i[last]; vz_i[k] = vz_i[last]
                    N_i_arr_ref[0] -= 1
                else:
                    # Collisions (Simplified)
                    # Use mean energy approximation for index for speed
                    v_mag = math.sqrt(vx_i[k]**2 + vy_i[k]**2 + vz_i[k]**2)
                    en_ev = 0.5 * MU_ARAR * v_mag*v_mag / EV_TO_J # Center of mass energy approx
                    c_idx = int(en_ev / DE_CS)
                    if c_idx >= CS_RANGES: c_idx = CS_RANGES - 1
                    nu = sigma_tot_i[c_idx] * v_mag
                    if rng_state.random() < (1.0 - math.exp(-nu * DT_I)):
                        collision_ion_jit(k, c_idx, vx_i, vy_i, vz_i, rng_state)
                    k += 1

def do_one_cycle():
    global E_amp_state, Power_accum_state, Rad_loss_prob_state
    
    # Increment session cycle counter for correct averaging
    cycles_session_arr[0] += 1
    
    # Update Radial Loss based on Te
    Te_curr = 2.0 # Default start
    if mean_energy_counter_center_arr[0] > 0:
        mean_E = mean_energy_accu_center_arr[0] / mean_energy_counter_center_arr[0]
        Te_curr = (2.0/3.0) * mean_E
    
    # Bohm Flux Model for Radial Loss
    u_Bohm = math.sqrt(E_CHARGE * Te_curr / AR_MASS)
    # Loss frequency = u_B / (R/2) * h_L (approx 0.5)
    nu_loss = (u_Bohm / (TUBE_RADIUS / 2.0)) * 0.4
    Rad_loss_prob_state[0] = 1.0 - math.exp(-nu_loss * DT_E)

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
        measurement_mode_arr[0], cycle_arr[0],
        RNG,
        eepf_phase_resolved,
        E_amp_state, Power_accum_state, Rad_loss_prob_state
    )

    # Power Controller
    sim_power = Power_accum_state[0] * (DT_E / PERIOD)
    vol = L * ELECTRODE_AREA # Use full domain volume for average density calc
    # Or better: Coil volume for deposition
    coil_vol = COIL_LENGTH * ELECTRODE_AREA
    sim_pd = abs(sim_power / coil_vol)
    
    # Feedback
    if sim_pd > 1e-12: # Lower threshold to catch low density
        scale = math.sqrt(TARGET_POWER_DENSITY / sim_pd)
        # New: Allow 50% jump per cycle during ignition
        scale = max(0.5, min(1.5, scale)) 
        E_amp_state[0] = E_amp_state[0] * scale
    
    # --- Peak Density Logging ---
    # cumulative density is sum of (n_e * FACTOR_W) every time step.
    # To get average physical density, divide by (Cycles_Session * N_T)
    # We use the session counter because cumul_e_density is reset at start of script.
    total_samples = float(cycles_session_arr[0]) * float(N_T) 
    c_norm = 1.0 / total_samples
    peak_ne = np.max(cumul_e_density) * c_norm

    with open(BASE_OUTPUT_DIR / "conv.dat", "a") as f:
        f.write(f"{cycle_arr[0]} {N_e_arr[0]} {N_i_arr[0]} {E_amp_state[0]:.2f} {sim_pd:.2f} {Te_curr:.2f} {peak_ne:.2e}\n")
    
    return N_e_arr[0], N_i_arr[0]

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
        e = i * DE_CS
        g = math.sqrt(2.0 * e * EV_TO_J / MU_ARAR) 
        nu = g * sigma_tot_i_arr[i]
        if nu > nu_max:
            nu_max = nu
    return nu_max
# -------------------------------------------------------------------
# --- FILE I/O AND POST-PROCESSING (UPDATED FOR RESTART) ---
# -------------------------------------------------------------------
def save_particle_data():
    picdata_file_path = BASE_OUTPUT_DIR / "picdata.npz"
    
    # We must save the current Field Amplitude so the PID controller
    # doesn't reset to the beginning.
    np.savez(picdata_file_path,
             Time=Time_arr[0], 
             cycles_done=cycles_done_arr[0],
             E_amp=E_amp_state[0],  # <--- CRITICAL SAVE
             N_e=N_e_arr[0], N_i=N_i_arr[0],
             x_e=x_e[:N_e_arr[0]], vx_e=vx_e[:N_e_arr[0]], vy_e=vy_e[:N_e_arr[0]], vz_e=vz_e[:N_e_arr[0]],
             x_i=x_i[:N_i_arr[0]], vx_i=vx_i[:N_i_arr[0]], vy_i=vy_i[:N_i_arr[0]], vz_i=vz_i[:N_i_arr[0]]
            )
    print(f">> eduPIC: State saved. Cycles: {cycles_done_arr[0]}, E_amp: {E_amp_state[0]:.2f}")

def load_particle_data():
    global x_e, vx_e, vy_e, vz_e, x_i, vx_i, vy_i, vz_i
    global E_amp_state # Need to update this global
    
    picdata_file_path = BASE_OUTPUT_DIR / "picdata.npz"
    try:
        data = np.load(picdata_file_path)
        Time_arr[0] = data['Time']
        cycles_done_arr[0] = int(data['cycles_done']) # Ensure int
        
        # --- RESTORE FEEDBACK STATE ---
        if 'E_amp' in data:
            E_amp_state[0] = float(data['E_amp'])
            print(f">> eduPIC: Restored Field Amplitude E = {E_amp_state[0]:.2f} V/m")
        else:
            print(f">> eduPIC: WARNING: Old save file format detected!")
            print(f">> eduPIC: E_amp NOT in save file. Using current default: {E_amp_state[0]:.2f} V/m")
            print(f">> eduPIC: Simulation will restart feedback from this value.")
            print(f">> eduPIC: To fix: Run a few cycles to re-stabilize, then save will include E_amp.")

        N_e_loaded = int(data['N_e'])
        N_i_loaded = int(data['N_i'])
        N_e_arr[0] = N_e_loaded
        N_i_arr[0] = N_i_loaded

        # Safety check for array size
        if N_e_loaded > MAX_E_PARTICLES or N_i_loaded > MAX_I_PARTICLES:
            print("Error: Loaded particle count exceeds allocation.")
            sys.exit(1)

        x_e[:N_e_loaded] = data['x_e']
        vx_e[:N_e_loaded] = data['vx_e']
        vy_e[:N_e_loaded] = data['vy_e']
        vz_e[:N_e_loaded] = data['vz_e']

        x_i[:N_i_loaded] = data['x_i']
        vx_i[:N_i_loaded] = data['vx_i']
        vy_i[:N_i_loaded] = data['vy_i']
        vz_i[:N_i_loaded] = data['vz_i']

        # --- CRITICAL: RECALCULATE TEMPERATURE IMMEDIATELY ---
        # If we don't do this, the first cycle assumes Te=2.0 eV (default)
        # instead of the hot plasma temp (e.g. 3.8 eV), causing wrong radial losses.
        
        # Calculate mean energy of loaded electrons matching the runtime window (0.45L - 0.55L)
        min_x_window = 0.45 * L
        max_x_window = 0.55 * L
        
        # Create mask for central particles
        # x_e is the global array, we care about the first N_e_loaded
        x_view = x_e[:N_e_loaded]
        mask = (x_view > min_x_window) & (x_view < max_x_window)
        
        count_center = np.count_nonzero(mask)
        
        if count_center > 0:
            vx_center = vx_e[:N_e_loaded][mask]
            vy_center = vy_e[:N_e_loaded][mask]
            vz_center = vz_e[:N_e_loaded][mask]
            
            v_sq = vx_center**2 + vy_center**2 + vz_center**2
            total_en_J = 0.5 * E_MASS * np.sum(v_sq)
            total_en_eV = total_en_J / EV_TO_J
            
            # Prime the accumulators
            mean_energy_accu_center_arr[0] = total_en_eV
            mean_energy_counter_center_arr[0] = count_center
            
            print(f">> eduPIC: Loaded {N_e_loaded} e-, {N_i_loaded} ions.")
            print(f">> eduPIC: Resuming from Cycle {cycles_done_arr[0]}. Calculated start Te (Center): {total_en_eV/count_center * (2/3):.2f} eV")
        else:
             # Fallback if no particles in center (unlikely for restart)
             mean_energy_accu_center_arr[0] = 0.0
             mean_energy_counter_center_arr[0] = 0
             print(f">> eduPIC: Loaded {N_e_loaded} e-, {N_i_loaded} ions.")
             print(f">> eduPIC: WARNING: No particles in central window on load.")

    except FileNotFoundError:
        print(f">> eduPIC: ERROR: No particle data file found at {picdata_file_path}")
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
    global eepf 
    with open(eepf_file_path, "w") as f:
        f.write(f"# energy[eV] f(E)[eV^-3/2]\n") # Header
        h_sum = np.sum(eepf) * DE_EEPF
        if h_sum == 0: h_sum = 1.0
        
        for i in range(N_EEPF):
            energy = (i + 0.5) * DE_EEPF
            val_eepf = 0.0
            if energy > 0: 
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
            _mean_i_energy_pow += energy * p * DE_IFED
            _mean_i_energy_gnd += energy * g * DE_IFED
        mean_i_energy_pow = _mean_i_energy_pow
        mean_i_energy_gnd = _mean_i_energy_gnd

def save_xt_1(distr_xt, fname_suffix): 
    xt_file_path = BASE_OUTPUT_DIR / fname_suffix
    with open(xt_file_path, "w") as f:
        for i in range(N_G):
            f.write(" ".join([f"{val:.8e}" for val in distr_xt[i,:]]) + "\n")

def norm_all_xt():
    num_cycles_run = float(no_of_cycles_arr[0])
    if num_cycles_run == 0: num_cycles_run = 1.0 

    f1 = float(N_XT) / (num_cycles_run * float(N_T))
    f2_denominator = (num_cycles_run * (PERIOD / float(N_XT)))
    f2 = WEIGHT / (ELECTRODE_AREA * DX) / f2_denominator if f2_denominator != 0 else 0.0

    global pot_xt, efield_xt, ne_xt, ni_xt, ue_xt, ui_xt, je_xt, ji_xt
    global powere_xt, poweri_xt, meanee_xt, meanei_xt, ioniz_rate_xt
    global counter_e_xt, counter_i_xt 

    pot_xt *= f1
    efield_xt *= f1
    ne_xt *= f1
    ni_xt *= f1

    for r in range(N_G):
      for c_xt in range(N_XT):
        if counter_e_xt[r,c_xt] > 0:
          ue_xt[r,c_xt] /= counter_e_xt[r,c_xt]
          meanee_xt[r,c_xt] /= counter_e_xt[r,c_xt]
          ioniz_rate_xt[r,c_xt] = (ioniz_rate_xt[r,c_xt] / counter_e_xt[r,c_xt]) * f2
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
        if density_val > 0 and kT_val > 0: 
            debye_length_val = math.sqrt(EPSILON0 * kT_val / density_val) / E_CHARGE
        
        sim_time_val = num_cycles_run / RF_FREQ_ICP
        
        ecoll_freq_val = 0.0
        icoll_freq_val = 0.0
        if sim_time_val > 0:
            if N_e_arr[0] > 0 : ecoll_freq_val = float(N_e_coll_arr[0]) / sim_time_val / float(N_e_arr[0])
            if N_i_arr[0] > 0 : icoll_freq_val = float(N_i_coll_arr[0]) / sim_time_val / float(N_i_arr[0])

        f.write(f"Simulation parameters:\n")
        f.write(f"Gap distance                          = {L:.4e} [m]\n")
        f.write(f"# of grid divisions                   = {N_G}\n")
        f.write(f"Frequency                             = {RF_FREQ_ICP:.4e} [Hz]\n")
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
            save_ifed() 
            norm_all_xt()
            save_all_xt()

            f.write(f"Particle characteristics at the electrodes:\n")
            denom_flux = (num_cycles_run * PERIOD)
            if denom_flux == 0: denom_flux = 1.0 
            
            ion_flux_pow = float(N_i_abs_pow_arr[0]) * WEIGHT / ELECTRODE_AREA / denom_flux
            ion_flux_gnd = float(N_i_abs_gnd_arr[0]) * WEIGHT / ELECTRODE_AREA / denom_flux
            ele_flux_pow = float(N_e_abs_pow_arr[0]) * WEIGHT / ELECTRODE_AREA / denom_flux
            ele_flux_gnd = float(N_e_abs_gnd_arr[0]) * WEIGHT / ELECTRODE_AREA / denom_flux

            f.write(f"Ion flux at powered electrode         = {ion_flux_pow:.4e} [m^-2 s^-1]\n")
            f.write(f"Ion flux at grounded electrode        = {ion_flux_gnd:.4e} [m^-2 s^-1]\n")
            f.write(f"Mean ion energy at powered electrode  = {mean_i_energy_pow:.4e} [eV]\n") 
            f.write(f"Mean ion energy at grounded electrode = {mean_i_energy_gnd:.4e} [eV]\n") 
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

# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eduPIC_ICP.py <cycles> [m] [RunID]")
        sys.exit(1)
        
    cycles_to_run = int(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == 'm':
        measurement_mode_arr[0] = True
        
    no_of_cycles_arr[0] = cycles_to_run  # Fix for save_density normalization
        
    set_cross_sections()
    
    # Init or Load
    if cycles_to_run == 0:
        if (BASE_OUTPUT_DIR / "picdata.npz").exists():
             print("Data exists. Delete to restart.")
             sys.exit(1)
        init_particles(N_INIT)
        cycles_done_arr[0] = 0
        cycle_arr[0] = 0
        # Warmup cycle to set Te and Fields
        do_one_cycle()
    else:
        load_particle_data()
        # CRITICAL FIX: Initialize cycle counter from loaded state
        # Without this, cycle_arr[0] starts at 0, causing the counter to reset
        cycle_arr[0] = cycles_done_arr[0]
    
    # Run Loop
    if cycles_to_run > 0:
        for i in tqdm(range(cycles_to_run)):
            cycle_arr[0] += 1
            n_e, n_i = do_one_cycle()
            cycles_done_arr[0] = cycle_arr[0]
            
            # Reseed if particles die out (Safety for low pressure)
            if n_e < 1000:
                print(">> Particles too low, reseeding...")
                # Add particles to center
                add_n = N_INIT - n_e
                if N_e_arr[0] + add_n < MAX_E_PARTICLES:
                    new_x = RNG.random(add_n) * 0.01 + (L/2.0 - 0.005) # Center
                    x_e[N_e_arr[0]:N_e_arr[0]+add_n] = new_x
                    # ... velocities ...
                    N_e_arr[0] += add_n
                    # Same for ions...
                    
    save_particle_data()
    if measurement_mode_arr[0]:
        check_and_save_info()