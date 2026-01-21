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
if len(sys.argv) > 3:
    RUN_ID = sys.argv[3]
else:
    RUN_ID = "ATTA_Restartable"

BASE_OUTPUT_DIR = Path("results") / RUN_ID
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f">> eduPIC: Output will be saved to: {BASE_OUTPUT_DIR}")

# ===================================================================
# === PHYSICAL CONSTANTS ===
# ===================================================================
PI = 3.141592653589793
TWO_PI = 2.0 * PI
E_CHARGE = 1.60217662e-19
EV_TO_J = E_CHARGE
E_MASS = 9.10938356e-31
AR_MASS = 6.63352090e-26  # Argon-40
MU_ARAR = AR_MASS / 2.0
K_BOLTZMANN = 1.38064852e-23
EPSILON0 = 8.85418781e-12

# ===================================================================
# === ATTA SOURCE PARAMETERS ===
# ===================================================================

# --- Simulation Domain ---
# L = 6cm. 
N_G = 1000            # Grid points

# --- Time Step ---
# At 1e17 density, wp ~ 1.8e10 rad/s. dt < 0.2/wp ~ 11ps.
# T_RF (150MHz) = 6666 ps.
N_T = 4000            # dt = 1.66 ps (Safe margin)

# --- Gas Conditions ---
PRESSURE = 0.1        # [Pa] (~3.75 mTorr) - Reverted to Research Note value
TEMPERATURE = 160.0   # [K] Wall temperature

# --- Geometry ---
TUBE_RADIUS = 0.005   # [m] 5mm inner radius
COIL_LENGTH = 0.045   # [m] 45mm active coil length
L = 0.060             # [m] 6 cm total simulation length
COIL_CENTER = L / 2.0 

# --- RF Driver ---
RF_FREQ_ICP = 150.0e6 # [Hz]
OMEGA_ICP = TWO_PI * RF_FREQ_ICP
VOLTAGE = 0.0         # Grounded shield

# --- Power Target ---
TARGET_POWER_DENSITY = 2.0e6  # [W/m^3] (2 MW/m^3)

# --- Initial Field ---
E_INDUCED_AMPLITUDE = 200.0 # [V/m] Start with a kick

# --- Superparticles ---
WEIGHT = 2.0e6        
N_INIT = 80_000      
MAX_PARTICLES_FACTOR = 200 
MAX_E_PARTICLES = int(N_INIT * MAX_PARTICLES_FACTOR)
MAX_I_PARTICLES = int(N_INIT * MAX_PARTICLES_FACTOR)
ELECTRODE_AREA = PI * TUBE_RADIUS**2 

# ===================================================================

# Derived Constants
PERIOD = 1.0 / RF_FREQ_ICP
DT_E = PERIOD / float(N_T)
N_SUB = 50 
DT_I = N_SUB * DT_E
DX = L / float(N_G - 1)
INV_DX = 1.0 / DX
# Initial Gas Density
GAS_DENSITY_BASE = PRESSURE / (K_BOLTZMANN * TEMPERATURE)
GAS_DENSITY_CURRENT = np.array([GAS_DENSITY_BASE], dtype=np.float64)

# --- Cross Sections ---
N_CS = 5
E_ELA = 0; E_EXC = 1; E_ION = 2; I_ISO = 3; I_BACK = 4
E_EXC_TH = 11.55; E_ION_TH = 15.76
CS_RANGES = 1000000
DE_CS = 0.001

# Memory Allocation
sigma = np.zeros((N_CS, CS_RANGES), dtype=np.float32)
# Totals are updated dynamicall based on gas density
sigma_tot_e_base = np.zeros(CS_RANGES, dtype=np.float32) 
sigma_tot_i_base = np.zeros(CS_RANGES, dtype=np.float32)
sigma_tot_e_current = np.zeros(CS_RANGES, dtype=np.float32)
sigma_tot_i_current = np.zeros(CS_RANGES, dtype=np.float32)

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

# --- NEW: Radial Loss Counters for Controller ---
N_e_loss_rad_arr = np.array([0], dtype=np.uint64)
N_i_loss_rad_arr = np.array([0], dtype=np.uint64)

# EEPF
N_EEPF = 2000; DE_EEPF = 0.05
eepf = np.zeros(N_EEPF, dtype=np.float64)

# IFED
N_IFED = 200; DE_IFED = 1.0
ifed_pow = np.zeros(N_IFED, dtype=np.int32)
ifed_gnd = np.zeros(N_IFED, dtype=np.int32)
mean_i_energy_pow = 0.0; mean_i_energy_gnd = 0.0

# Total Energy Diagnostics (for conservation check)
total_energy_history = np.zeros(N_T + 1, dtype=np.float64) 
kinetic_energy_e_history = np.zeros(N_T + 1, dtype=np.float64)
kinetic_energy_i_history = np.zeros(N_T + 1, dtype=np.float64)
electric_field_energy_history = np.zeros(N_T + 1, dtype=np.float64)
rf_power_input_history = np.zeros(N_T + 1, dtype=np.float64)
particle_loss_energy_history = np.zeros(N_T + 1, dtype=np.float64)
energy_conserved_history = np.zeros(N_T + 1, dtype=np.float64)
energy_counter = np.array([0], dtype=np.int32) 
cumulative_rf_energy = np.array([0.0], dtype=np.float64)
cumulative_particle_loss = np.array([0.0], dtype=np.float64)

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

# Global accumulator for Te calculation
mean_energy_accu_center_arr = np.array([0.0], dtype=np.float64)
mean_energy_counter_center_arr = np.array([0], dtype=np.uint64)
N_e_coll_arr = np.array([0], dtype=np.uint64)
N_i_coll_arr = np.array([0], dtype=np.uint64)

# --- FEEDBACK STATE VARIABLES ---
E_amp_state = np.array([E_INDUCED_AMPLITUDE], dtype=np.float64)
Power_accum_state = np.array([0.0], dtype=np.float64)
Rad_loss_prob_state = np.array([0.0], dtype=np.float64)
Phi_sheath_state = np.array([9.4], dtype=np.float64) 
# NEW: Gas Temperature State for Restart Consistency
T_gas_state = np.array([TEMPERATURE], dtype=np.float64)

# Simulation State
Time_arr = np.array([0.0], dtype=np.float64)
cycle_arr = np.array([0], dtype=np.int32)
no_of_cycles_arr = np.array([0], dtype=np.int32)
cycles_done_arr = np.array([0], dtype=np.int32)
cycles_session_arr = np.array([0], dtype=np.int32)
measurement_mode_arr = np.array([False], dtype=np.bool_)

RNG = np.random.default_rng()

# -------------------------------------------------------------------
# --- CROSS SECTION LOADERS ---
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
        idx = np.argsort(energies_lxcat)
        interp_func = scipy.interpolate.interp1d(energies_lxcat[idx], sigmas_lxcat[idx], kind='linear', bounds_error=False, fill_value=0.0)
        return interp_func(target_energy_eV_grid).astype(np.float32)
    return np.zeros_like(target_energy_eV_grid, dtype=np.float32)

# Fallbacks
def qmel(en): return 1e-20 * (abs(6.0/pow(1.0+(en/0.1)+pow(en/0.6,2.0),3.3) - 1.1*pow(en,1.4)/(1.0+pow(en/15.0,1.2))/math.sqrt(1.0+pow(en/5.5,2.5)+pow(en/60.0,4.1))) + 0.05/pow(1.0+en/10.0,2.0) + 0.01*pow(en,3.0)/(1.0+pow(en/12.0,6.0)))
def qexc(en): return 1e-20 * (0.034*pow(en-11.5,1.1)*(1.0+pow(en/15.0,2.8))/(1.0+pow(en/23.0,5.5)) + 0.023*(en-11.5)/pow(1.0+en/80.0,1.9)) if en > E_EXC_TH else 0.0
def qion(en): return 1e-20 * (970.0*(en-15.8)/pow(70.0+en,2.0) + 0.06*pow(en-15.8,2.0)*math.exp(-en/9.0)) if en > E_ION_TH else 0.0
def qiso_ion(en): return 2e-19 * pow(en,-0.5)/(1.0+en) + 3e-19*en/pow(1.0+en/3.0,2.0)
def qmom_ion(en): return 1.15e-18 * pow(en,-0.1)*pow(1.0+0.015/en,0.6)

def set_cross_sections():
    print(">> eduPIC: Loading Cross Sections...")
    sim_e_grid = (np.arange(CS_RANGES, dtype=np.float32) + 1) * DE_CS
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

    # Initialize Totals with Base Density
    for i in range(CS_RANGES):
        sigma_tot_e_base[i] = (sigma[E_ELA,i] + sigma[E_EXC,i] + sigma[E_ION,i]) * GAS_DENSITY_BASE
        sigma_tot_i_base[i] = (sigma[I_ISO,i] + sigma[I_BACK,i]) * GAS_DENSITY_BASE
    
    # Initialize current to base
    sigma_tot_e_current[:] = sigma_tot_e_base[:]
    sigma_tot_i_current[:] = sigma_tot_i_base[:]

# -------------------------------------------------------------------
# --- INITIALIZATION ---
# -------------------------------------------------------------------
def init_particles(nseed):
    print(">> eduPIC: Initializing particles...")
    N_e_arr[0] = nseed; N_i_arr[0] = nseed
    
    x_e[:nseed] = RNG.random(nseed) * L
    x_i[:nseed] = RNG.random(nseed) * L
    
    v_th_e = math.sqrt(E_CHARGE * 2.0 / E_MASS)
    vx_e[:nseed] = RNG.normal(0, v_th_e, nseed)
    vy_e[:nseed] = RNG.normal(0, v_th_e, nseed)
    vz_e[:nseed] = RNG.normal(0, v_th_e, nseed)
    
    v_th_i = math.sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS)
    vx_i[:nseed] = RNG.normal(0, v_th_i, nseed)
    vy_i[:nseed] = RNG.normal(0, v_th_i, nseed)
    vz_i[:nseed] = RNG.normal(0, v_th_i, nseed)

# -------------------------------------------------------------------
# --- NUMBA KERNELS ---
# -------------------------------------------------------------------
@numba.njit(fastmath=True, cache=True, nogil=True)
def RMB_numba(rng_state):
    # Thermal velocity for neutrals
    u1 = rng_state.random(); u2 = rng_state.random()
    return math.sqrt(-2.0*math.log(u1)) * math.cos(TWO_PI*u2) * math.sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS)

@numba.njit(fastmath=True, nogil=True)
def solve_Poisson_jit(pot_arr, efield_arr, rho1_arr, N_G_local, DX_local, INV_DX_local, EPSILON0_local):
    c_prime = np.empty(N_G_local, dtype=np.float64)
    d_prime = np.empty(N_G_local, dtype=np.float64)
    
    pot_arr[0] = 0.0
    pot_arr[N_G_local - 1] = 0.0
    const = - (DX_local * DX_local) / EPSILON0_local
    
    c_prime[1] = -0.5
    d_prime[1] = (rho1_arr[1] * const) / -2.0
    
    for i in range(2, N_G_local - 1):
        temp = 1.0 / (-2.0 - 1.0 * c_prime[i-1])
        c_prime[i] = 1.0 * temp
        d_prime[i] = ((rho1_arr[i] * const) - 1.0 * d_prime[i-1]) * temp
        
    for i in range(N_G_local - 2, 0, -1):
        pot_arr[i] = d_prime[i] - c_prime[i] * pot_arr[i+1]
        
    for i in range(1, N_G_local - 1):
        efield_arr[i] = (pot_arr[i-1] - pot_arr[i+1]) * (0.5 * INV_DX_local)
    
    efield_arr[0] = (pot_arr[0] - pot_arr[1]) * INV_DX_local
    efield_arr[N_G_local-1] = (pot_arr[N_G_local-2] - pot_arr[N_G_local-1]) * INV_DX_local

@numba.njit(fastmath=True, nogil=True)
def collision_electron_jit(k, eidx, Ne_ref, Ni_ref, xe, vxe, vye, vze, xi, vxi, vyi, vzi, max_e, max_i, rng):
    m_e = E_MASS
    vx = vxe[k]; vy = vye[k]; vz = vze[k]
    v_mag = math.sqrt(vx*vx + vy*vy + vz*vz)
    if v_mag == 0: return

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

    if type == 1: # Excitation
        loss = E_EXC_TH * EV_TO_J
        kin = 0.5 * m_e * v_mag*v_mag
        if kin > loss:
            v_rem = math.sqrt(2.0*(kin-loss)/m_e)
            scale = v_rem / v_mag
            vx *= scale; vy *= scale; vz *= scale
        else: return
    
    elif type == 2: # Ionization
        loss = E_ION_TH * EV_TO_J
        kin = 0.5 * m_e * v_mag*v_mag
        if kin > loss:
            rem_E = kin - loss
            r_split = rng.random()
            E_prim = rem_E * r_split
            E_sec = rem_E * (1.0 - r_split)
            v_prim = math.sqrt(2.0*E_prim/m_e)
            
            # Secondary Electron
            if Ne_ref[0] < max_e:
                idx = Ne_ref[0]
                xe[idx] = xe[k]
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
            
            # Scatter primary
            cos_th = 1.0 - 2.0*rng.random()
            sin_th = math.sqrt(1.0 - cos_th*cos_th)
            phi = TWO_PI * rng.random()
            vx = v_prim * sin_th * math.cos(phi)
            vy = v_prim * sin_th * math.sin(phi)
            vz = v_prim * cos_th
            
        else: return

    # Elastic / Post-Inelastic Scattering
    cos_th = 1.0 - 2.0*rng.random()
    sin_th = math.sqrt(1.0 - cos_th*cos_th)
    phi = TWO_PI * rng.random()
    v_new_mag = math.sqrt(vx*vx + vy*vy + vz*vz)
    
    vxe[k] = v_new_mag * sin_th * math.cos(phi)
    vye[k] = v_new_mag * sin_th * math.sin(phi)
    vze[k] = v_new_mag * cos_th

@numba.njit(fastmath=True, nogil=True)
def collision_ion_jit(k, eidx, vx_i, vy_i, vz_i, rng):
    # Charge Exchange (Backward) vs Elastic (Isotropic)
    vnx = RMB_numba(rng); vny = RMB_numba(rng); vnz = RMB_numba(rng)
    
    s_iso = sigma[I_ISO, eidx]
    s_back = sigma[I_BACK, eidx]
    
    if (s_iso + s_back) == 0: return
    
    # <--- FIXED: Charge Exchange means velocity replacement, not mixing -->
    if rng.random() < s_back / (s_iso + s_back):
        # Charge Exchange: Ion becomes fast neutral (lost), new neutral becomes slow ion
        vx_i[k] = vnx; vy_i[k] = vny; vz_i[k] = vnz
    else:
        # Elastic sphere
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
    E_amp_ref, Power_accum_ref, Rad_loss_prob_ref, Phi_sheath_ref,
    N_e_loss_rad_ref, N_i_loss_rad_ref  # <--- NEW: Loss counters
    ):

    DV = ELECTRODE_AREA * DX
    FACTOR_W = WEIGHT / DV
    FACTOR_E = DT_E / E_MASS * E_CHARGE
    FACTOR_I = DT_I / AR_MASS * E_CHARGE
    MIN_X = 0.45 * L
    MAX_X = 0.55 * L
    COIL_HALF_LEN = COIL_LENGTH / 2.0
    
    Power_accum_ref[0] = 0.0
    current_E_amp = E_amp_ref[0]
    current_P_loss = Rad_loss_prob_ref[0]
    
    rho = np.empty(N_G, dtype=np.float64)

    for t_loop_idx in range(N_T):
        Time_arr_ref[0] += DT_E
        t_index_xt = t_loop_idx // N_BIN
        sin_omega_t = math.sin(OMEGA_ICP * Time_arr_ref[0])

        # --- DENSITY ---
        e_density[:] = 0.0
        for k in range(N_e_arr_ref[0]):
            pos_norm = x_e[k] * INV_DX
            p = int(pos_norm)
            rmod = pos_norm - p
            if 0 <= p < N_G - 1:
                e_density[p] += (1.0 - rmod) * FACTOR_W
                e_density[p+1] += rmod * FACTOR_W
        
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

        # --- FIELD ---
        for i in range(N_G):
            rho[i] = E_CHARGE * (i_density[i] - e_density[i])
        solve_Poisson_jit(pot, efield, rho, N_G, DX, INV_DX, EPSILON0)

        # --- XT ---
        if measurement_mode:
            for p_xt in range(N_G):
                pot_xt[p_xt, t_index_xt]    += pot[p_xt]
                efield_xt[p_xt, t_index_xt] += efield[p_xt]
                ne_xt[p_xt, t_index_xt]     += e_density[p_xt]
                ni_xt[p_xt, t_index_xt]     += i_density[p_xt]

        # --- ELECTRONS ---
        k = 0
        while k < N_e_arr_ref[0]:
            pos_norm = x_e[k] * INV_DX
            p = int(pos_norm)
            rmod = pos_norm - p
            Ex = 0.0
            if 0 <= p < N_G - 1:
                Ex = (1.0 - rmod)*efield[p] + rmod*efield[p+1]
            
            dist_from_center = abs(x_e[k] - COIL_CENTER)
            s_x = 0.0
            if dist_from_center < COIL_HALF_LEN:
                s_x = 1.0
            else:
                decay_len = 0.005
                over = dist_from_center - COIL_HALF_LEN
                if over < decay_len: s_x = 1.0 - (over / decay_len)
            
            Ey_ind = current_E_amp * s_x * sin_omega_t

            if measurement_mode:
                mean_v = vx_e[k] - 0.5 * Ex * FACTOR_E
                if 0 <= p < N_G - 1:
                    counter_e_xt[p, t_index_xt]   += (1.0 - rmod)
                    counter_e_xt[p+1, t_index_xt] += rmod
                    ue_xt[p, t_index_xt]   += (1.0 - rmod) * mean_v
                    ue_xt[p+1, t_index_xt] += rmod * mean_v
                    v_sqr = mean_v * mean_v + vy_e[k] * vy_e[k] + vz_e[k] * vz_e[k]
                    energy = 0.5 * E_MASS * v_sqr / EV_TO_J
                    meanee_xt[p, t_index_xt]   += (1.0 - rmod) * energy
                    meanee_xt[p+1, t_index_xt] += rmod * energy
                    energy_idx_cs = min(int(energy / DE_CS + 0.5), CS_RANGES - 1)
                    if energy_idx_cs < 0: energy_idx_cs = 0
                    velocity = math.sqrt(v_sqr)
                    rate = sigma[E_ION, energy_idx_cs] * velocity * DT_E * GAS_DENSITY_CURRENT[0] # Use current density
                    ioniz_rate_xt[p, t_index_xt]   += (1.0 - rmod) * rate
                    ioniz_rate_xt[p+1, t_index_xt] += rmod * rate
                
                v_sqr_eepf = vx_e[k]**2 + vy_e[k]**2 + vz_e[k]**2
                en_ev = 0.5 * E_MASS * v_sqr_eepf / EV_TO_J
                if MIN_X < x_e[k] < MAX_X:
                    mean_energy_accu_center_arr_ref[0] += en_ev
                    mean_energy_counter_center_arr_ref[0] += 1
                    e_idx = int(en_ev / DE_EEPF)
                    if 0 <= e_idx < N_EEPF: eepf[e_idx] += 1.0

            vx_e[k] -= Ex * FACTOR_E
            vy_old = vy_e[k]
            vy_e[k] -= Ey_ind * FACTOR_E
            vy_avg = 0.5 * (vy_old + vy_e[k])
            Power_accum_ref[0] += (-E_CHARGE * vy_avg * Ey_ind) * WEIGHT
            x_e[k] += vx_e[k] * DT_E
            
            removed = False
            removed_radial = False
            if x_e[k] <= 0 or x_e[k] >= L:
                removed = True
            
            # <--- FIXED: Energy-dependent Radial Loss Logic --->
            if not removed:
                # Radial Energy = 0.5 * m * (vy^2 + vz^2)
                rad_en_ev = 0.5 * E_MASS * (vy_e[k]**2 + vz_e[k]**2) / EV_TO_J
                
                if rad_en_ev > Phi_sheath_ref[0]:
                     # Only remove if it hits the wall geometrically
                     if rng_state.random() < current_P_loss:
                         removed = True
                         removed_radial = True
            
            if removed:
                if removed_radial:
                    N_e_loss_rad_ref[0] += 1

                last = N_e_arr_ref[0] - 1
                if k < last:
                    x_e[k] = x_e[last]; vx_e[k] = vx_e[last]; vy_e[k] = vy_e[last]; vz_e[k] = vz_e[last]
                N_e_arr_ref[0] -= 1
            else:
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
                
                if measurement_mode:
                    mean_v_i = vx_i[k] + 0.5 * Ex * FACTOR_I
                    if 0 <= p < N_G - 1:
                        counter_i_xt[p, t_index_xt]   += (1.0 - rmod)
                        counter_i_xt[p+1, t_index_xt] += rmod
                        ui_xt[p, t_index_xt]   += (1.0 - rmod) * mean_v_i
                        ui_xt[p+1, t_index_xt] += rmod * mean_v_i
                        v_sqr_i = mean_v_i**2 + vy_i[k]**2 + vz_i[k]**2
                        energy_i = 0.5 * AR_MASS * v_sqr_i / EV_TO_J
                        meanei_xt[p, t_index_xt]   += (1.0 - rmod) * energy_i
                        meanei_xt[p+1, t_index_xt] += rmod * energy_i
                
                vx_i[k] += Ex * FACTOR_I
                x_i[k] += vx_i[k] * DT_I
                
                removed = False
                removed_radial = False
                if x_i[k] <= 0 or x_i[k] >= L: removed = True
                
                # Ion Radial Loss (Standard Bohm probability)
                if not removed and rng_state.random() < P_loss_i: 
                    removed = True
                    removed_radial = True
                
                if removed:
                    if removed_radial:
                        N_i_loss_rad_ref[0] += 1
                        
                    last = N_i_arr_ref[0] - 1
                    if k < last:
                        x_i[k] = x_i[last]; vx_i[k] = vx_i[last]; vy_i[k] = vy_i[last]; vz_i[k] = vz_i[last]
                    N_i_arr_ref[0] -= 1
                else:
                    v_mag = math.sqrt(vx_i[k]**2 + vy_i[k]**2 + vz_i[k]**2)
                    en_ev = 0.5 * MU_ARAR * v_mag*v_mag / EV_TO_J
                    c_idx = int(en_ev / DE_CS)
                    if c_idx >= CS_RANGES: c_idx = CS_RANGES - 1
                    nu = sigma_tot_i[c_idx] * v_mag
                    if rng_state.random() < (1.0 - math.exp(-nu * DT_I)):
                        collision_ion_jit(k, c_idx, vx_i, vy_i, vz_i, rng_state)
                    k += 1

def compute_total_energy():
    """
    Computes the total energy of the system including RF input and losses.
    
    Returns:
    --------
    total_energy : float [J]
        Total system energy = KE_electrons + KE_ions + E_field_energy
    kinetic_energy_e : float [J]
        Total kinetic energy of electrons
    kinetic_energy_i : float [J]
        Total kinetic energy of ions
    electric_field_energy : float [J]
        Electrostatic field energy integrated over domain
    """
    
    # Compute electron kinetic energy
    kinetic_energy_e = 0.0
    for k in range(N_e_arr[0]):
        v_sq = vx_e[k]**2 + vy_e[k]**2 + vz_e[k]**2
        kinetic_energy_e += 0.5 * E_MASS * v_sq * WEIGHT
    
    # Compute ion kinetic energy
    kinetic_energy_i = 0.0
    for k in range(N_i_arr[0]):
        v_sq = vx_i[k]**2 + vy_i[k]**2 + vz_i[k]**2
        kinetic_energy_i += 0.5 * AR_MASS * v_sq * WEIGHT
    
    # Compute electric field energy: E_field = 0.5 * epsilon_0 * integral(E^2 dV)
    # Using trapezoidal rule for integration
    electric_field_energy = 0.0
    for i in range(N_G):
        electric_field_energy += 0.5 * EPSILON0 * efield[i]**2 * ELECTRODE_AREA * DX
    
    total_energy = kinetic_energy_e + kinetic_energy_i + electric_field_energy
    
    return total_energy, kinetic_energy_e, kinetic_energy_i, electric_field_energy

def do_one_cycle():
    global E_amp_state, Power_accum_state, Rad_loss_prob_state, Phi_sheath_state
    global sigma_tot_e_current, sigma_tot_i_current, GAS_DENSITY_CURRENT, T_gas_state
    
    cycles_session_arr[0] += 1
    
    # --- 0. Reset Loss Counters ---
    N_e_loss_rad_arr[0] = 0
    N_i_loss_rad_arr[0] = 0

    # --- 1. Update Temperature & Bohm Velocity ---
    Te_curr = 2.0 
    if mean_energy_counter_center_arr[0] > 0:
        mean_E = mean_energy_accu_center_arr[0] / mean_energy_counter_center_arr[0]
        Te_curr = (2.0/3.0) * mean_E
    
    u_Bohm = math.sqrt(E_CHARGE * Te_curr / AR_MASS)
    nu_loss = (u_Bohm / (TUBE_RADIUS / 2.0)) * 0.4 # h_L factor approx 0.4-0.5
    Rad_loss_prob_state[0] = 1.0 - math.exp(-nu_loss * DT_E)

    # --- 2. Update Neutral Gas (Depletion) ---
    # Calc Power Density
    sim_power = Power_accum_state[0] * (DT_E / PERIOD)
    coil_vol = COIL_LENGTH * ELECTRODE_AREA
    sim_pd = abs(sim_power / coil_vol)
    
    # Estimate Gas Temp (Linear approx from Robertz/Lequette data)
    # T_gas rises with power density.
    T_gas_state[0] = TEMPERATURE + 0.05 * (sim_pd / 1e3) 
    if T_gas_state[0] > 1000.0: T_gas_state[0] = 1000.0
    
    # Update Density (P = nkT const)
    new_gas_density = PRESSURE / (K_BOLTZMANN * T_gas_state[0])
    
    # Rescale Cross Sections
    scale_factor = new_gas_density / GAS_DENSITY_BASE
    sigma_tot_e_current[:] = sigma_tot_e_base[:] * scale_factor
    sigma_tot_i_current[:] = sigma_tot_i_base[:] * scale_factor
    GAS_DENSITY_CURRENT[0] = new_gas_density

    # --- 3. Run Kernel ---
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
        sigma_tot_e_current, sigma_tot_i_current, # Pass scaled sigmas
        measurement_mode_arr[0], cycle_arr[0],
        RNG,
        eepf_phase_resolved,
        E_amp_state, Power_accum_state, Rad_loss_prob_state, Phi_sheath_state,
        N_e_loss_rad_arr, N_i_loss_rad_arr
    )

    # --- 4. Ambipolar Controller (Phi Feedback) ---
    n_loss_e = float(N_e_loss_rad_arr[0])
    n_loss_i = float(N_i_loss_rad_arr[0])
    
    if n_loss_i > 0:
        ratio = n_loss_e / n_loss_i
        # If losing too many electrons, raise barrier
        if ratio > 1.02:
            Phi_sheath_state[0] *= 1.002
        # If losing too few electrons, lower barrier
        elif ratio < 0.98:
            Phi_sheath_state[0] *= 0.998
    else:
        # Kickstart if no losses yet
        Phi_sheath_state[0] *= 0.999

    # Safety Clamps
    # Argon Phi_floating ~ 4.7 Te. Allow range 3.0 to 8.0.
    min_phi = 3.0 * Te_curr
    max_phi = 8.0 * Te_curr
    if Phi_sheath_state[0] < min_phi: Phi_sheath_state[0] = min_phi
    if Phi_sheath_state[0] > max_phi: Phi_sheath_state[0] = max_phi

    # --- 5. Power Feedback (PID) ---
    if sim_pd > 1e-12:
        scale = math.sqrt(TARGET_POWER_DENSITY / sim_pd)
        scale = max(0.9, min(1.1, scale)) # Gentle ramp
        E_amp_state[0] = E_amp_state[0] * scale
    
    total_samples = float(cycles_session_arr[0]) * float(N_T) 
    c_norm = 1.0 / total_samples
    peak_ne = np.max(cumul_e_density) * c_norm

    with open(BASE_OUTPUT_DIR / "conv.dat", "a") as f:
        f.write(f"{cycle_arr[0]} {N_e_arr[0]} {N_i_arr[0]} {E_amp_state[0]:.2f} {sim_pd:.2e} {Te_curr:.2f} {Phi_sheath_state[0]:.2f} {T_gas_state[0]:.1f} {peak_ne:.2e}\n")
    
    # --- ENERGY CONSERVATION DIAGNOSTIC ---
    total_energy, kinetic_energy_e, kinetic_energy_i, electric_field_energy = compute_total_energy()
    
    cumulative_rf_energy[0] += Power_accum_state[0] * DT_E 
    
    energy_conserved = total_energy + cumulative_particle_loss[0] - cumulative_rf_energy[0]
    
    idx = energy_counter[0]
    if idx < len(total_energy_history):
        total_energy_history[idx] = total_energy
        kinetic_energy_e_history[idx] = kinetic_energy_e
        kinetic_energy_i_history[idx] = kinetic_energy_i
        electric_field_energy_history[idx] = electric_field_energy
        rf_power_input_history[idx] = cumulative_rf_energy[0]
        particle_loss_energy_history[idx] = cumulative_particle_loss[0]
        energy_conserved_history[idx] = energy_conserved
        energy_counter[0] += 1
    
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
    
    np.savez(picdata_file_path,
             Time=Time_arr[0], 
             cycles_done=cycles_done_arr[0],
             E_amp=E_amp_state[0],
             Phi_sheath=Phi_sheath_state[0],
             T_gas=T_gas_state[0], # <--- PERSIST GAS TEMP
             N_e=N_e_arr[0], N_i=N_i_arr[0],
             x_e=x_e[:N_e_arr[0]], vx_e=vx_e[:N_e_arr[0]], vy_e=vy_e[:N_e_arr[0]], vz_e=vz_e[:N_e_arr[0]],
             x_i=x_i[:N_i_arr[0]], vx_i=vx_i[:N_i_arr[0]], vy_i=vy_i[:N_i_arr[0]], vz_i=vz_i[:N_i_arr[0]]
            )
    print(f">> eduPIC: State saved. Cycles: {cycles_done_arr[0]}")

def load_particle_data():
    global x_e, vx_e, vy_e, vz_e, x_i, vx_i, vy_i, vz_i
    global E_amp_state, Phi_sheath_state, T_gas_state
    global sigma_tot_e_current, sigma_tot_i_current, GAS_DENSITY_CURRENT
    
    picdata_file_path = BASE_OUTPUT_DIR / "picdata.npz"
    try:
        data = np.load(picdata_file_path)
        Time_arr[0] = data['Time']
        cycles_done_arr[0] = int(data['cycles_done'])
        
        if 'E_amp' in data: E_amp_state[0] = float(data['E_amp'])
        if 'Phi_sheath' in data: Phi_sheath_state[0] = float(data['Phi_sheath'])
        
        # --- Gas Temp Restore & Physics Rescale ---
        if 'T_gas' in data: 
            T_gas_state[0] = float(data['T_gas'])
        else:
            T_gas_state[0] = TEMPERATURE
            print(">> WARNING: T_gas not found in save, defaulting to BASE.")
            
        # IMMEDIATE UPDATE OF PHYSICS
        new_gas_density = PRESSURE / (K_BOLTZMANN * T_gas_state[0])
        scale_factor = new_gas_density / GAS_DENSITY_BASE
        sigma_tot_e_current[:] = sigma_tot_e_base[:] * scale_factor
        sigma_tot_i_current[:] = sigma_tot_i_base[:] * scale_factor
        GAS_DENSITY_CURRENT[0] = new_gas_density

        N_e_loaded = int(data['N_e'])
        N_i_loaded = int(data['N_i'])
        N_e_arr[0] = N_e_loaded
        N_i_arr[0] = N_i_loaded

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

        # Recalc temp for startup
        min_x_window = 0.45 * L
        max_x_window = 0.55 * L
        x_view = x_e[:N_e_loaded]
        mask = (x_view > min_x_window) & (x_view < max_x_window)
        count_center = np.count_nonzero(mask)
        
        if count_center > 0:
            vx_center = vx_e[:N_e_loaded][mask]
            vy_center = vy_e[:N_e_loaded][mask]
            vz_center = vz_e[:N_e_loaded][mask]
            v_sq = vx_center**2 + vy_center**2 + vz_center**2
            total_en_J = 0.5 * E_MASS * np.sum(v_sq)
            mean_energy_accu_center_arr[0] = total_en_J / EV_TO_J
            mean_energy_counter_center_arr[0] = count_center
            
    except FileNotFoundError:
        print(f">> eduPIC: ERROR: No particle data file found at {picdata_file_path}")
        sys.exit(1)

def save_density():
    density_file_path = BASE_OUTPUT_DIR / "density.dat"
    with open(density_file_path, "w") as f:
        f.write(f"# x[m] e_density[m^-3] i_density[m^-3]\n") # Header
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

def save_and_analyze_energy():
    """
    Save total energy history and analyze energy conservation.
    
    Saves complete energy accounting including RF input and losses.
    E_conserved = E_stored + E_losses - E_input_RF should remain constant.
    """
    energy_file_path = BASE_OUTPUT_DIR / "energy_conservation.dat"
    
    # Only save valid data points (up to energy_counter)
    n_samples = energy_counter[0]
    
    with open(energy_file_path, "w") as f:
        f.write("# Cycle  E_Stored[J]  KE_Electrons[J]  KE_Ions[J]  E_Field[J]  E_RF_Input[J]  E_Loss[J]  E_Conserved[J]\n")
        for i in range(n_samples):
            f.write(f"{i} {total_energy_history[i]:.6e} {kinetic_energy_e_history[i]:.6e} ")
            f.write(f"{kinetic_energy_i_history[i]:.6e} {electric_field_energy_history[i]:.6e} ")
            f.write(f"{rf_power_input_history[i]:.6e} {particle_loss_energy_history[i]:.6e} ")
            f.write(f"{energy_conserved_history[i]:.6e}\n")
    
    # Compute energy balance metrics
    if n_samples > 100:  # Use later part of simulation for steady-state analysis
        # Find steady-state region (last 25% of cycles)
        steady_state_start = int(0.75 * n_samples)
        # Slicing numpy arrays safely
        steady_state_energy = energy_conserved_history[steady_state_start:n_samples]
        
        # Calculate energy fluctuation in steady state (should be ~constant)
        mean_energy_ss = np.mean(steady_state_energy)
        std_energy_ss = np.std(steady_state_energy)
        
        if abs(mean_energy_ss) > 1e-20:
            fluctuation_percent = (std_energy_ss / abs(mean_energy_ss)) * 100.0
        else:
            fluctuation_percent = 0.0
        
        # Calculate energy conservation error
        conservation_error = std_energy_ss
        
        # Total RF energy input (last valid point)
        total_rf_input = rf_power_input_history[n_samples - 1]
        
        # Total energy stored at end
        final_stored = total_energy_history[n_samples - 1]
        
        # Particle loss energy
        final_loss = particle_loss_energy_history[n_samples - 1]
        
        # Current Conserved Value (E_stored + E_loss - E_input)
        conserved_value = energy_conserved_history[n_samples - 1]
        
        return fluctuation_percent, conservation_error, final_stored, total_rf_input, final_loss, conserved_value
    
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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
        
        c_check = max_electron_coll_freq_jit(sigma_tot_e_current) * DT_E
        f.write(f"Max. electron coll. frequency * DT_E  = {c_check:.4e} (OK if less than 0.05)\n")
        if c_check > 0.05: conditions_OK = False
        
        c_check = max_ion_coll_freq_jit(sigma_tot_i_current) * DT_I
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
            
            power_e_avg = np.sum(powere_xt) / float(N_XT * N_G) if (N_XT * N_G > 0) else 0.0
            power_i_avg = np.sum(poweri_xt) / float(N_XT * N_G) if (N_XT * N_G > 0) else 0.0
            f.write(line + "\n")
            f.write(f"Absorbed power calculated as <j*E>:\n")
            f.write(f"Electron power density (average)      = {power_e_avg:.4e} [W m^-3]\n")
            f.write(f"Ion power density (average)           = {power_i_avg:.4e} [W m^-3]\n")
            f.write(f"Total power density (average)         = {power_e_avg + power_i_avg:.4e} [W m^-3]\n")
            f.write(line + "\n")
            
            # Energy Conservation Diagnostics
            fluctuation_pct, conservation_error, final_stored, total_rf_input, final_loss, conserved_value = save_and_analyze_energy()
            f.write(f"Energy Balance Diagnostics (Energy Conservation):\n")
            f.write(f"Total RF power input over simulation    = {total_rf_input:.6e} [J]\n")
            f.write(f"Total energy stored in plasma at end    = {final_stored:.6e} [J]\n")
            f.write(f"Total energy lost by particles         = {final_loss:.6e} [J]\n")
            f.write(f"Energy balance: E_stored + E_loss - E_RF_input\n")
            f.write(f"Conserved quantity (should be const)    = {conserved_value:.6e} [J]\n")
            f.write(f"Steady-state fluctuation               = {fluctuation_pct:.4f} [%]\n")
            f.write(f"Energy conservation error              = {conservation_error:.6e} [J]\n")
            f.write(f"NOTE: E_conserved = E_stored + E_losses - E_input_RF should remain constant.\n")
            f.write(f"(See 'energy_conservation.dat' for cycle-by-cycle energy breakdown)\n")
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
        
    no_of_cycles_arr[0] = cycles_to_run 
        
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
        cycle_arr[0] = cycles_done_arr[0]
    
    # Run Loop
    if cycles_to_run > 0:
        for i in tqdm(range(cycles_to_run)):
            cycle_arr[0] += 1
            n_e, n_i = do_one_cycle()
            cycles_done_arr[0] = cycle_arr[0]
            
            # Reseed if particles die out
            if n_e < 1000:
                print(">> Particles too low, reseeding...")
                add_n = N_INIT - n_e
                if N_e_arr[0] + add_n < MAX_E_PARTICLES:
                    new_x = RNG.random(add_n) * 0.01 + (L/2.0 - 0.005) 
                    x_e[N_e_arr[0]:N_e_arr[0]+add_n] = new_x
                    v_th_e = math.sqrt(E_CHARGE * 2.0 / E_MASS)
                    vx_e[N_e_arr[0]:N_e_arr[0]+add_n] = RNG.normal(0, v_th_e, add_n)
                    vy_e[N_e_arr[0]:N_e_arr[0]+add_n] = RNG.normal(0, v_th_e, add_n)
                    vz_e[N_e_arr[0]:N_e_arr[0]+add_n] = RNG.normal(0, v_th_e, add_n)
                    N_e_arr[0] += add_n
                    
                    x_i[N_i_arr[0]:N_i_arr[0]+add_n] = new_x
                    v_th_i = math.sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS)
                    vx_i[N_i_arr[0]:N_i_arr[0]+add_n] = RNG.normal(0, v_th_i, add_n)
                    vy_i[N_i_arr[0]:N_i_arr[0]+add_n] = RNG.normal(0, v_th_i, add_n)
                    vz_i[N_i_arr[0]:N_i_arr[0]+add_n] = RNG.normal(0, v_th_i, add_n)
                    N_i_arr[0] += add_n
                    
    save_particle_data()
    if measurement_mode_arr[0]:
        check_and_save_info()