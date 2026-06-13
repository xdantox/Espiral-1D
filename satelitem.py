import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

# --- 1. PARÁMETROS FÍSICOS ---
norm = 0.05788 
Jnn  = 46.812805 / norm
dJnn = 44.873295 / norm
Jnnn = 2.60139 / norm
D_xx = 0.76 / norm        # D (Hard Axis - Bare)
K    = -45.4866 / norm
D_yy = 0.76  / norm  # D_plane (In-Plane - Modulado)
S_mag = 1.0

# --- PARÁMETROS DE GEOMETRÍA (Esquema Extendido / Físico) ---
q_real = 2.1103175969
gamma = -0.284375
alpha = 0.0036 

# --- TRUNCAMIENTO FLOQUET ---
# Incluye armónicos m = 0, ±1, ..., ±N_max
N_max = 8 
ANNOTATE_CELLS = False

# ====================================================================
# NUEVO: INTERRUPTOR DE DESVANECIMIENTO (THINNING)
# ====================================================================
APPLY_INTENSITY_THINNING = True  # Cambia a False para ver TODAS las bandas sin desvanecerse

# --- ARGUMENTOS DE BESSEL ---
eps_1 = 2 * abs(alpha) * np.sin(q_real)
eps_2_vec = eps_1 * 2 * np.cos(q_real) 
eps_doble = 2 * eps_1 

# NUEVO: Argumento puro on-site para D_yy
eps_aniso = 2 * abs(alpha) 

# --- CONFIGURACIÓN CELDA c=2 ---
c = 2
th_vca = np.zeros(c)
for i in range(c):
    m = i // 2
    if i % 2 == 0:
        th_vca[i] = 2 * m * q_real + gamma 
    else:
        th_vca[i] = (2 * m + 1) * q_real - gamma 

# =============================================================================
# FUNCIONES DE BLOQUE CORREGIDAS
# =============================================================================
def add_nn_block(M_k, i, j, phase, d_angle, J_link, K_val, bessel_funcs, mode, include_bare=False):
    
    if mode == 'cos':
        geo_simple = np.cos(d_angle) * bessel_funcs['J_eps']
    else:
        geo_simple = -np.sin(d_angle) * bessel_funcs['J_eps']

    if mode == 'cos':
        geo_double = np.cos(2 * d_angle) * bessel_funcs['J_2eps']
        if include_bare:
            term_bare = 1.0
        else:
            term_bare = 0.0
    else:
        geo_double = -np.sin(2 * d_angle) * bessel_funcs['J_2eps']
        term_bare = 0.0

    k_hop_u  = geo_simple
    k_mass_u = 0.5 * (term_bare + geo_double)

    k_hop_v  = geo_double
    k_mass_v = geo_double

    delta_diag = -J_link * geo_simple
    M_k[2*i, 2*i]     += delta_diag; M_k[2*i+1, 2*i+1] += delta_diag
    M_k[2*j, 2*j]     += delta_diag; M_k[2*j+1, 2*j+1] += delta_diag
    
    if mode == 'cos' and include_bare:
        M_k[2*i, 2*j] += J_link * phase
        M_k[2*j, 2*i] += J_link * np.conjugate(phase)
    
    M_k[2*i+1, 2*j+1] += J_link * geo_simple * phase
    M_k[2*j+1, 2*i+1] += J_link * geo_simple * np.conjugate(phase)

    val_mass_u = +2.0 * K_val * S_mag**2 * k_mass_u
    M_k[2*i, 2*i] += val_mass_u
    M_k[2*j, 2*j] += val_mass_u
    
    val_hop_u  = -2.0 * K_val * S_mag**2 * k_hop_u
    M_k[2*i, 2*j] += val_hop_u * phase
    M_k[2*j, 2*i] += val_hop_u * np.conjugate(phase)

    val_mass_v = +2.0 * K_val * S_mag**2 * k_mass_v
    M_k[2*i+1, 2*i+1] += val_mass_v
    M_k[2*j+1, 2*j+1] += val_mass_v
    
    val_hop_v  = -2.0 * K_val * S_mag**2 * k_hop_v
    M_k[2*i+1, 2*j+1] += val_hop_v * phase
    M_k[2*j+1, 2*i+1] += val_hop_v * np.conjugate(phase)

def add_nnn_block(M_k, i, j, phase, d_angle, J_link, bessel_val, mode,include_bare=False):
    if mode == 'cos':
        geo = np.cos(d_angle) * bessel_val
    else:
        geo = -np.sin(d_angle) * bessel_val
    if mode == 'cos' and include_bare:
        M_k[2*i, 2*j]     +=  J_link * phase
        M_k[2*j, 2*i]     +=  J_link * np.conjugate(phase)

    delta_diag = -J_link * geo
    M_k[2*i, 2*i] += delta_diag; M_k[2*i+1, 2*i+1] += delta_diag
    M_k[2*j, 2*j] += delta_diag; M_k[2*j+1, 2*j+1] += delta_diag
    
    M_k[2*i+1, 2*j+1] += J_link * geo * phase
    M_k[2*j+1, 2*i+1] += J_link * geo * np.conjugate(phase)

def get_block_aniso_correct(dm):
    r"""
    Construye el bloque de dispersión D_yy para un salto de Floquet \Delta m.
    dm: Salto de momento (m_destino - m_origen).
    """
    M_k = np.zeros((2*c, 2*c), dtype=complex)
    
    if dm % 2 != 0:
        return M_k
        
    order = abs(dm)
    p = order // 2
    parity_sign = (-1)**(p + 1)
    
    # La amplitud topológica pura (El Hessiano y el Coseno ya se cancelaron algebraicamente)
    bessel_term = jv(p - 1, eps_aniso) + parity_sign * jv(p + 1, eps_aniso)
    
    for i in range(c):
        phi_nu = th_vca[i] 
        phase = np.exp(1j * dm * phi_nu)
        
        val_v = -D_yy * bessel_term * phase
        val_u = 0.5 * val_v
        
        M_k[2*i+1, 2*i+1] += val_v
        M_k[2*i, 2*i]     += val_u
        
    return M_k

def get_block_4x4(k_val, order=0, is_diagonal=False):
    M_k = np.zeros((2*c, 2*c), dtype=complex)
    
    if order % 2 == 0:
        mode_exchange = 'cos'
    else:
        mode_exchange = 'sin'

    factor_exch = 1.0
    parity = (-1)**(abs(order) // 2) 

    bessel_exchange = {
        'J_eps':  parity * factor_exch * jv(order, eps_1),
        'J_2eps': parity * factor_exch * jv(order, eps_doble)
    }
    bes_J2 = parity * factor_exch * jv(order, eps_2_vec)

    if is_diagonal:
        include_bare_hopping = True 
        D_hard_term = 2 * D_xx
        D_plane_bare = -D_yy 
    else:
        include_bare_hopping = False
        D_hard_term = 0.0
        D_plane_bare = 0.0

    for i in range(c):
        th_i = th_vca[i]
        
        M_k[2*i, 2*i] += D_hard_term
        M_k[2*i, 2*i] += D_plane_bare

        J_right = Jnn + (-1)**i * dJnn
        j = (i + 1) % c
        d_ij = (th_vca[j] + (0 if i + 1 < c else c * q_real)) - th_i
        phase_nn = 1.0 + 0j if i + 1 < c else np.exp(1j * k_val)
        
        add_nn_block(M_k, i, j, phase_nn, d_ij, J_right, K, bessel_exchange, 
                     mode=mode_exchange, include_bare=include_bare_hopping)

        l = (i + 2) % c
        offset_angle = 0 if i + 2 < c else c * q_real
        d_il = (th_vca[l] + offset_angle) - th_i
        phase_nnn = 1.0 + 0j if i + 2 < c else np.exp(1j * k_val )
        
        add_nnn_block(M_k, i, l, phase_nnn, d_il, Jnnn, bes_J2, 
                      mode=mode_exchange, include_bare=include_bare_hopping)

    return M_k

def get_floquet_matrix(k, N_max=1):
    block_size = 2 * c
    ms = np.arange(-N_max, N_max + 1)
    n_blocks = len(ms)
    H = np.zeros((block_size * n_blocks, block_size * n_blocks), dtype=complex)
    
    q_cell = 2.0 * q_real 

    for idx, m in enumerate(ms):
        r0, r1 = idx * block_size, (idx + 1) * block_size
        M_m = get_block_4x4(k + m * q_cell, order=0, is_diagonal=True) 
        M_m += get_block_aniso_correct(dm=0)
        H[r0:r1, r0:r1] = M_m

    max_possible_dist = n_blocks - 1
    for dist in range(1, max_possible_dist + 1):
        for idx in range(n_blocks - dist):
            r0, r1 = idx * block_size, (idx + 1) * block_size
            c0, c1 = (idx + dist) * block_size, (idx + dist + 1) * block_size

            dm = ms[idx + dist] - ms[idx]

            V_exch = get_block_4x4(k + ms[idx] * q_cell, order=dm, is_diagonal=False) 
            V_aniso = get_block_aniso_correct(dm)
            V_total = V_exch + V_aniso

            H[c0:c1, r0:r1] = V_total
            H[r0:r1, c0:c1] = V_total.conj().T

    return H
            
# --- SOLVER ---
Sigma_small = np.zeros((2*c, 2*c), dtype=complex)
for i in range(c):
    Sigma_small[2*i, 2*i+1] = 1.0; Sigma_small[2*i+1, 2*i] = -1.0
Sigma_small /= S_mag
Sigma_big = np.kron(np.eye(2 * N_max + 1), Sigma_small)

q_vals = np.linspace(- 2*np.pi,   2*np.pi, 7001)

def fold_k(k):
    """Fuerza a la zona del dímero [-pi/2, pi/2]"""
    return ((k + np.pi/2) % np.pi) - np.pi/2

k_plot = []
w_plot = []
weights_plot = []

block_size = 2 * c
center_offset = N_max * block_size

for k in q_vals:
    H_F = get_floquet_matrix(k, N_max=N_max)
    Dyn = 1j * Sigma_big @ H_F
    evals, evecs = np.linalg.eig(Dyn)

    k_phys = k / 2.0

    for idx in range(evals.size):
        val = evals[idx]
        if np.real(val) > 1e-6:
            # === CAMBIO A THz ===
            w_rad_s = np.real(val) * 1.7e11
            w_thz = w_rad_s / (2 * np.pi * 1e12)
            
            X = evecs[:, idx]

            symp_norm = np.imag(np.vdot(X, Sigma_big @ X))
            if np.abs(symp_norm) > 1e-15:
                X = X / np.sqrt(np.abs(symp_norm))
            else:
                continue

            X0 = X[center_offset:center_offset + block_size]
            u_A, v_A, u_B, v_B = X0[0], X0[1], X0[2], X0[3]

            W_u = np.abs(u_A)**2 + np.abs(u_B)**2
            W_v = 0.5 * (np.abs(v_A)**2 + np.abs(v_B)**2)

            k_plot.append(fold_k(k_phys))
            w_plot.append(w_thz) 
            weights_plot.append(W_u)

            k_plot.append(fold_k(k_phys + q_real))
            w_plot.append(w_thz) 
            weights_plot.append(W_v)

            k_plot.append(fold_k(k_phys - q_real))
            w_plot.append(w_thz) 
            weights_plot.append(W_v)

k_plot = np.array(k_plot)
w_plot = np.array(w_plot)
weights_plot = np.array(weights_plot)
weights_norm = weights_plot / np.max(weights_plot)

# ====================================================================
# LÓGICA DE FILTRADO (Controlada por el interruptor APPLY_INTENSITY_THINNING)
# ====================================================================
W_log10 = np.log10(weights_norm + 1e-15)

LOG_FLOOR = -8
log_floor_mask = W_log10 >= LOG_FLOOR
if not np.any(log_floor_mask):
    log_floor_mask = np.ones_like(W_log10, dtype=bool)

vmax_val = float(np.percentile(W_log10[log_floor_mask], 99.4))
W_raw_log = W_log10 - vmax_val
vmin_val = float(np.percentile(W_raw_log[log_floor_mask], 4))
vmax_val = 0.0

if APPLY_INTENSITY_THINNING:
    mask = log_floor_mask & (W_raw_log > (vmin_val - 1.0))
else:
    mask = np.ones_like(W_raw_log, dtype=bool)

k_plot_mask = k_plot[mask]
w_plot_mask = w_plot[mask]
W_plot_mask = W_raw_log[mask]

sort_idx = np.argsort(W_plot_mask)
k_plot_sorted = k_plot_mask[sort_idx]
w_plot_sorted = w_plot_mask[sort_idx]
W_plot_sorted = W_plot_mask[sort_idx]

# Configuración del tamaño de los puntos
if APPLY_INTENSITY_THINNING:
    W_norm = np.clip((W_plot_sorted - vmin_val) / (vmax_val - vmin_val), 0.0, 1.0)
    s_min = 0.1  
    s_max = 3.5  
    point_sizes = s_min + (s_max - s_min) * (W_norm ** 2)
else:
    point_sizes = 2.0  # Tamaño fijo si el filtro está desactivado

# ====================================================================
# AJUSTES DE PRESENTACIÓN (LETRAS GRANDES)
# ====================================================================
TITLE_SIZE = 22
LABEL_SIZE = 18
TICK_SIZE = 16
TEXT_SIZE = 16
CBAR_LABEL_SIZE = 18

plt.figure(figsize=(12, 7.5))

scatter = plt.scatter(
    k_plot_sorted,
    w_plot_sorted,
    c=W_plot_sorted,
    cmap='plasma',
    s=point_sizes,  
    alpha=1.0,
    edgecolors='none',
    vmin=vmin_val,
    vmax=vmax_val,
)

plt.xlabel(r'Wave Vector $k$ (Dimer BZ)', fontsize=LABEL_SIZE)
plt.ylabel(r'Frequency $\nu$ [THz]', fontsize=LABEL_SIZE)
plt.title('MoI3 LSWT M/L Dispersion and band structure for c=2 (Floquet)', fontsize=TITLE_SIZE, pad=15)

plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

cbar = plt.colorbar(scatter)
cbar.set_label(r'$\log_{10} \mathcal{S}(k, \nu)$ Relative Intensity', fontsize=CBAR_LABEL_SIZE)
cbar.ax.tick_params(labelsize=TICK_SIZE)

plt.grid(True, alpha=0.3)
plt.xlim(-np.pi/2, np.pi/2)
plt.ylim(0, np.max(w_plot) * 1.05)
ax = plt.gca()

ax.text(
    0.02, 0.98, f"N_max = {N_max}", transform=ax.transAxes,
    ha="left", va="top", fontsize=TEXT_SIZE, color="white",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="white"),
)

if ANNOTATE_CELLS:
    omega_max = float(np.max(w_plot) * 1.05)
    # y_base adaptado a la escala física (aprox 32 THz)
    y_base = min(32.0, 0.84 * omega_max)
    y_crys = min(y_base + 0.10 * omega_max, 0.95 * omega_max)
    y_mag = y_base

    q_plot = np.abs(((q_real + np.pi/2) % np.pi) - np.pi/2)

    ax.annotate("", xy=(-np.pi/2, y_crys), xytext=(np.pi/2, y_crys), arrowprops=dict(arrowstyle="<->", color="white", lw=1.4))
    ax.text(0.0, y_crys - 0.035 * omega_max, r"Crystallographic Cell: (dimer, $d=2a$): $k\in[-\pi/2,\pi/2]$",
            color="white", ha="center", va="top", fontsize=TEXT_SIZE - 2, bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="white"))

    ax.axvline(+q_plot, color="cyan", lw=1.5, ls="--", alpha=0.95)
    ax.axvline(-q_plot, color="cyan", lw=1.5, ls="--", alpha=0.95)

    ax.annotate("", xy=(-q_plot, y_mag), xytext=(q_plot, y_mag), arrowprops=dict(arrowstyle="<->", color="cyan", lw=1.6))
    ax.text(0.0, y_mag - 0.035 * omega_max, rf"Magnetic Cell: $k_{{mag}}\in[-q_{{inc}},q_{{inc}}],\ q_{{inc}}={q_plot:.4f}$",
            color="cyan", ha="center", va="top", fontsize=TEXT_SIZE - 2, bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="cyan"))

plt.gca().set_facecolor('#110022')
plt.show()