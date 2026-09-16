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
D_yy = 0.76  / norm       # D_plane (In-Plane - Modulado)
S_mag = 1.0

# --- PARÁMETROS DE GEOMETRÍA (Rigurosos del LLG) ---
q_real = 5.2514862797
gamma  = 1.286581         # CORREGIDO: Signo positivo del mínimo global
alpha  = 0.002230 
beta   = -0.004351 

# --- TRUNCAMIENTO FLOQUET ---
N_max = 6
APPLY_INTENSITY_THINNING = False

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
# FUNCIONES DE BLOQUE ESTABLES
# =============================================================================
def add_nn_block(M_k, i, j, phase_fwd, phase_rev, d_angle, J_link, K_val, bessel_funcs, mode, include_bare=False):
    if mode == 'cos':
        geo_simple = np.cos(d_angle) * bessel_funcs['J_eps']
        geo_double = np.cos(2 * d_angle) * bessel_funcs['J_2eps']
        term_bare = 1.0 if include_bare else 0.0
    else:
        geo_simple = -np.sin(d_angle) * bessel_funcs['J_eps']
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
        M_k[2*i, 2*j] += J_link * phase_fwd
        M_k[2*j, 2*i] += J_link * phase_rev
    
    M_k[2*i+1, 2*j+1] += J_link * geo_simple * phase_fwd
    M_k[2*j+1, 2*i+1] += J_link * geo_simple * phase_rev

    val_mass_u = +2.0 * K_val * S_mag**2 * k_mass_u
    M_k[2*i, 2*i] += val_mass_u
    M_k[2*j, 2*j] += val_mass_u
    
    val_hop_u  = -2.0 * K_val * S_mag**2 * k_hop_u
    M_k[2*i, 2*j] += val_hop_u * phase_fwd
    M_k[2*j, 2*i] += val_hop_u * phase_rev

    val_mass_v = +2.0 * K_val * S_mag**2 * k_mass_v
    M_k[2*i+1, 2*i+1] += val_mass_v
    M_k[2*j+1, 2*j+1] += val_mass_v
    
    val_hop_v  = -2.0 * K_val * S_mag**2 * k_hop_v
    M_k[2*i+1, 2*j+1] += val_hop_v * phase_fwd
    M_k[2*j+1, 2*i+1] += val_hop_v * phase_rev

def add_nnn_block(M_k, i, j, phase_fwd, phase_rev, d_angle, J_link, bessel_val, mode, include_bare=False):
    if mode == 'cos':
        geo = np.cos(d_angle) * bessel_val
    else:
        geo = -np.sin(d_angle) * bessel_val

    if mode == 'cos' and include_bare:
        M_k[2*i, 2*j] += J_link * phase_fwd
        M_k[2*j, 2*i] += J_link * phase_rev

    delta_diag = -J_link * geo
    M_k[2*i, 2*i]     += delta_diag; M_k[2*i+1, 2*i+1] += delta_diag
    M_k[2*j, 2*j]     += delta_diag; M_k[2*j+1, 2*j+1] += delta_diag
    
    M_k[2*i+1, 2*j+1] += J_link * geo * phase_fwd
    M_k[2*j+1, 2*i+1] += J_link * geo * phase_rev

def get_block_aniso_correct(dm):
    M_k = np.zeros((2*c, 2*c), dtype=complex)
    p = dm
    parity_sign = (-1)**(p + 1)
    
    A_def = np.sqrt(alpha**2 + beta**2)
    eps_aniso = 2.0 * A_def
    
    for i in range(c):
        # Fase matemática real de la deformación in-situ
        Phi_i = np.arctan2(-beta * ((-1.0)**i), -alpha)
        Delta_i = Phi_i - 2.0 * gamma * ((-1.0)**i)
        
        # Funciones de Bessel con el desfase complejo riguroso
        term_minus = jv(p - 1, eps_aniso) * np.exp(1j * (p - 1) * Delta_i)
        term_plus  = parity_sign * jv(p + 1, eps_aniso) * np.exp(1j * (p + 1) * Delta_i)
        bessel_complex = term_minus + term_plus
        
        phi_nu = th_vca[i]
        phase = np.exp(1j * 2.0 * dm * phi_nu)
        
        # Rigurosamente acorde al LSWT original
        val_v = -1.0 * D_yy * bessel_complex * phase
        val_u = 0.5 * val_v
        
        M_k[2*i+1, 2*i+1] += val_v
        M_k[2*i, 2*i]     += val_u
        
    return M_k

def get_block_4x4(k_val, order=0, is_diagonal=False):
    M_k = np.zeros((2*c, 2*c), dtype=complex)
    mode_exchange = 'cos' if order % 2 == 0 else 'sin'
    parity = (-1)**(order // 2)

    if is_diagonal:
        include_bare_hopping = True
        D_hard_term = 2 * D_xx
        D_plane_bare = -D_yy   
    else:
        include_bare_hopping = False
        D_hard_term = 0.0
        D_plane_bare = 0.0

    q_cell = 2.0 * c * q_real
    k_out = k_val + order * q_cell

    for i in range(c):
        th_i = th_vca[i]
        M_k[2*i, 2*i] += D_hard_term
        M_k[2*i, 2*i] += D_plane_bare

        # =========================================================
        # Primeros vecinos (NN)
        # =========================================================
        eps_local_nn = -2.0 * alpha * np.sin(q_real) + 2.0 * beta * np.cos(q_real) * ((-1.0)**i)
        eps_local_2eps = 2.0 * eps_local_nn
        
        J_right = Jnn + ((-1)**i) * dJnn
        j = (i + 1) % c
        offset_nn = 0 if i + 1 < c else c * q_real
        th_j_eff = th_vca[j] + offset_nn
        
        d_ij = th_j_eff - th_i
        sum_ij = th_j_eff + th_i  
        phase_floquet_nn = np.exp(1j * order * sum_ij)
        
        bessel_exchange_local = {
            'J_eps':  parity * jv(order, eps_local_nn) * phase_floquet_nn,
            'J_2eps': parity * jv(order, eps_local_2eps) * phase_floquet_nn
        }

        if i + 1 < c:
            phase_fwd_nn = 1.0 + 0j
            phase_rev_nn = 1.0 + 0j
        else:
            phase_fwd_nn = np.exp(1j * k_val)  
            phase_rev_nn = np.exp(-1j * k_out) 

        add_nn_block(M_k, i, j, phase_fwd_nn, phase_rev_nn, d_ij, J_right, K, bessel_exchange_local,
                     mode=mode_exchange, include_bare=include_bare_hopping)

        # =========================================================
        # Segundos vecinos (NNN)
        # =========================================================
        l = (i + 2) % c
        offset_nnn = 0 if i + 2 < c else c * q_real
        th_l_eff = th_vca[l] + offset_nnn
        
        d_il = th_l_eff - th_i
        sum_il = th_l_eff + th_i  
        phase_floquet_nnn = np.exp(1j * order * sum_il)
        
        # Rigurosa inclusión de Delta_n para NNN
        A_def = np.sqrt(alpha**2 + beta**2)
        Phi_i_nnn = np.arctan2(-beta * ((-1.0)**i), -alpha)
        Delta_i_nnn = Phi_i_nnn - 2.0 * gamma * ((-1.0)**i)
        
        eps_local_nnn = 2.0 * A_def * np.sin(2.0 * q_real)
        bes_J2_local = parity * jv(order, eps_local_nnn) * np.exp(1j * order * Delta_i_nnn) * phase_floquet_nnn
        
        if i + 2 < c:
            phase_fwd_nnn = 1.0 + 0j
            phase_rev_nnn = 1.0 + 0j
        else:
            phase_fwd_nnn = np.exp(1j * k_val)
            phase_rev_nnn = np.exp(-1j * k_out)

        add_nnn_block(M_k, i, l, phase_fwd_nnn, phase_rev_nnn, d_il, Jnnn, bes_J2_local,
                      mode=mode_exchange, include_bare=include_bare_hopping)

    return M_k

def get_floquet_matrix(k, N_max=1):
    block_size = 2 * c
    ms = np.arange(-N_max, N_max + 1)
    n_blocks = len(ms)
    H = np.zeros((block_size * n_blocks, block_size * n_blocks), dtype=complex)
    
    q_cell = 2.0 * c * q_real

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
            
            V_total =  V_exch + V_aniso

            H[c0:c1, r0:r1] = V_total
            H[r0:r1, c0:c1] = V_total.conj().T

    return H
            
# --- SOLVER Y PROYECCIÓN ESPECTRAL ---
Sigma_small = np.zeros((2*c, 2*c), dtype=complex)
for i in range(c):
    Sigma_small[2*i, 2*i+1] = 1.0
    Sigma_small[2*i+1, 2*i] = -1.0
Sigma_small /= S_mag
Sigma_big = np.kron(np.eye(2 * N_max + 1), Sigma_small)

q_vals = np.linspace(-2*np.pi, 2*np.pi, 7001)

def fold_k(k):
    return ((k + np.pi/2) % np.pi) - np.pi/2

k_plot = []
w_plot = []
weights_plot = []

block_size = 2 * c
ms_range = np.arange(-N_max, N_max + 1)

for k in q_vals:
    H_F = get_floquet_matrix(k, N_max=N_max)
    Dyn = 1j * Sigma_big @ H_F
    evals, evecs = np.linalg.eig(Dyn)

    k_phys = k / 2.0

    for idx in range(evals.size):
        val = evals[idx]
        if np.real(val) > 1e-8:
            w_rad_s = np.real(val) * 1.7e11
            w_thz = w_rad_s / (2 * np.pi * 1e12)
            
            X = evecs[:, idx]
            symp_norm = np.imag(np.vdot(X, Sigma_big @ X))
            if np.abs(symp_norm) > 1e-15:
                X = X / np.sqrt(np.abs(symp_norm))
            else:
                continue

            for m_idx, m_val in enumerate(ms_range):
                if abs(m_val) > 1:
                    continue
                
                offset = m_idx * block_size
                Xm = X[offset:offset + block_size]
                u_A, v_A, u_B, v_B = Xm[0], Xm[1], Xm[2], Xm[3]

                W_u = 0.5 * (np.abs(u_A)**2 + np.abs(u_B)**2)
                W_v = 0.25 * (np.abs(v_A)**2 + np.abs(v_B)**2)

                k_m = k_phys + m_val * (2.0 * q_real)

                if W_u > 1e-5:
                    k_plot.append(fold_k(k_m))
                    w_plot.append(w_thz) 
                    weights_plot.append(W_u)

                if W_v > 1e-5:
                    k_plot.append(fold_k(k_m + q_real))
                    w_plot.append(w_thz) 
                    weights_plot.append(W_v)

                    k_plot.append(fold_k(k_m - q_real))
                    w_plot.append(w_thz) 
                    weights_plot.append(W_v)

k_plot = np.array(k_plot)
w_plot = np.array(w_plot)
weights_plot = np.array(weights_plot)
weights_norm = weights_plot / np.max(weights_plot)

# --- FILTRADO DE INTENSIDAD ---
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

point_sizes = 2.0

# --- GRÁFICO ---
plt.figure(figsize=(12, 7.5))
scatter = plt.scatter(
    k_plot_sorted, w_plot_sorted, c=W_plot_sorted, cmap='plasma',
    s=point_sizes, alpha=1.0, edgecolors='none', vmin=vmin_val, vmax=vmax_val,
)

plt.xlabel(r'Wave Vector $k$ (Dimer BZ, $d=2a$) [$-\pi/2, \pi/2$]', fontsize=18)
plt.ylabel(r'Frequency $\nu$ [THz]', fontsize=18)
plt.title(r'$\mathrm{MoI}_3$ LSWT M/L Dispersion (LLG Validated)', fontsize=22, pad=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

cbar = plt.colorbar(scatter)
cbar.set_label(r'$\log_{10} \mathcal{S}(k, \nu)$ Relative Intensity', fontsize=18)
cbar.ax.tick_params(labelsize=16)

plt.grid(True, alpha=0.3)
plt.xlim(-np.pi/2, np.pi/2)
plt.ylim(0, np.max(w_plot) * 1.05)
plt.gca().set_facecolor('#110022')
plt.tight_layout()
plt.show()