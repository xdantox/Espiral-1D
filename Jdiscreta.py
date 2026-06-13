import numpy as np
import matplotlib.pyplot as plt

norm = 0.05788 
Jnn = 46.812805 / norm      
dJnn = 44.873295 / norm
Jnnn= 2.60139 / norm
D = 0.76 / norm
K= 45.4866 / norm
S_mag = 1.0

# Anotaciones de celdas en el plot
ANNOTATE_CELLS = False

q_real = 1.0312752764
gamma = -1.286421

c = 2
th_vca = np.zeros(c)
for i in range(c):
    m = i // 2
    if i % 2 == 0:
        th_vca[i] = 2 * m * q_real + gamma
    else:
        th_vca[i] = (2 * m + 1) * q_real - gamma

def add_nn_block(M_k, i, j, phase, d_angle, J_link):
    cosd = np.cos(d_angle)
    sind2 = np.sin(d_angle)**2

    # Heisenberg
    delta_diag = -J_link * cosd
    M_k[2*i, 2*i]         += delta_diag
    M_k[2*i+1, 2*i+1]     += delta_diag
    M_k[2*j, 2*j]         += delta_diag
    M_k[2*j+1, 2*j+1]     += delta_diag
    M_k[2*i, 2*j]         +=  J_link * phase
    M_k[2*j, 2*i]         +=  J_link * np.conjugate(phase)
    M_k[2*i+1, 2*j+1]     +=  J_link * cosd * phase
    M_k[2*j+1, 2*i+1]     +=  J_link * cosd * np.conjugate(phase)

    # K term A
    K_A = 2 * K * S_mag**2 * cosd
    delta_KA = -K_A * cosd
    M_k[2*i, 2*i]         += delta_KA
    M_k[2*i+1, 2*i+1]     += delta_KA
    M_k[2*j, 2*j]         += delta_KA
    M_k[2*j+1, 2*j+1]     += delta_KA
    M_k[2*i, 2*j]         +=  K_A * phase
    M_k[2*j, 2*i]         +=  K_A * np.conjugate(phase)
    M_k[2*i+1, 2*j+1]     +=  K_A * cosd * phase
    M_k[2*j+1, 2*i+1]     +=  K_A * cosd * np.conjugate(phase)

    # K term B 
    K_B = 2 * K * S_mag**2 * sind2
    M_k[2*i+1, 2*i+1]     +=  K_B
    M_k[2*j+1, 2*j+1]     +=  K_B
    M_k[2*i+1, 2*j+1]     += -K_B * phase
    M_k[2*j+1, 2*i+1]     += -K_B * np.conjugate(phase)

def add_nnn_block(M_k, i, j, phase, d_angle, J_link):
    cosd = np.cos(d_angle)
    delta_diag = -J_link * cosd
    M_k[2*i, 2*i]         += delta_diag
    M_k[2*i+1, 2*i+1]     += delta_diag
    M_k[2*j, 2*j]         += delta_diag
    M_k[2*j+1, 2*j+1]     += delta_diag
    M_k[2*i, 2*j]         +=  J_link * phase
    M_k[2*j, 2*i]         +=  J_link * np.conjugate(phase)
    M_k[2*i+1, 2*j+1]     +=  J_link * cosd * phase
    M_k[2*j+1, 2*i+1]     +=  J_link * cosd * np.conjugate(phase)

Sigma = np.zeros((2*c, 2*c), dtype=complex)
for i in range(c):
    Sigma[2*i, 2*i+1] = 1.0
    Sigma[2*i+1, 2*i] = -1.0
Sigma /= S_mag

def get_bloch_matrix_cartesian(k_val):
    M_k = np.zeros((2*c, 2*c), dtype=complex)
    for i in range(c):
        th_i = th_vca[i]
        J_right = Jnn + (-1)**i * dJnn
        factor = 1
        j = (i + 1) % c
        d_ij = (th_vca[j] + (0 if i + 1 < c else c * q_real)) - th_i
        phase_nn = 1.0 + 0j if i + 1 < c else np.exp(1j * k_val * c/2 / factor)
        add_nn_block(M_k, i, j, phase_nn, d_ij, J_right)

        l = (i + 2) % c
        d_il = (th_vca[l] + (0 if i + 2 < c else c * q_real)) - th_i
        phase_nnn = 1.0 + 0j if i + 2 < c else np.exp(1j * k_val * c/2 / factor)
        add_nnn_block(M_k, i, l, phase_nnn, d_il, Jnnn)

        M_k[2*i, 2*i] += 2 * D
    return M_k

# ====================================================================
# PARCHE MINIMAL: M/L y Plegamiento
# ====================================================================

# Reducimos los puntos para que el scatter sea eficiente, pero ampliamos 
# el rango para capturar los cruces desde otras zonas.
q_vals = np.linspace(-10 * np.pi, 10 * np.pi, 8000)

k_plot = []
w_plot = []
weights_plot = []

def fold_k(k):
    """Fuerza a la zona del dímero [-pi/2, pi/2]"""
    return ((k + np.pi/2) % np.pi) - np.pi/2

for k in q_vals:
    M = get_bloch_matrix_cartesian(k)
    Dyn = Sigma @ M
    evals, evecs = np.linalg.eig(Dyn)
    
    # 2) Cambio de métrica k: Celda atómica vs Celda de dímero
    k_phys = k / 2.0 
    
    for idx in range(2*c):
        val = evals[idx]
        if np.imag(val) > 1e-6:
            w = np.imag(val) * 1.7e11
            X = evecs[:, idx]
            
            # Normalización robusta para evitar cortes en los cruces de bandas
            symp_norm = np.imag(np.vdot(X, Sigma @ X))
            if np.abs(symp_norm) > 1e-15:
                X = X / np.sqrt(np.abs(symp_norm))
            else:
                continue
                
            u_A, v_A, u_B, v_B = X[0], X[1], X[2], X[3]
            
            # 1) Base M/L: Suma incoherente (sin la fase phi_nu)
            W_u = np.abs(u_A)**2 + np.abs(u_B)**2
            W_v = 0.5 * (np.abs(v_A)**2 + np.abs(v_B)**2)
            
            # Guardamos banda u
            k_plot.append(fold_k(k_phys))
            w_plot.append(w)
            weights_plot.append(W_u)
            
            # Guardamos bandas v desplazadas por el inmensurado
            k_plot.append(fold_k(k_phys + q_real))
            w_plot.append(w)
            weights_plot.append(W_v)
            
            k_plot.append(fold_k(k_phys - q_real))
            w_plot.append(w)
            weights_plot.append(W_v)
k_plot = np.array(k_plot)
w_plot = np.array(w_plot)
weights_plot = np.array(weights_plot)

# ====================================================================
# LÓGICA LLG: Rango dinámico basado en percentiles
# ====================================================================
# 1. Calculamos el logaritmo base relativo al máximo absoluto matemático
W_raw_log = np.log10((weights_plot / np.max(weights_plot)) + 1e-15)

# 2. Aplicamos exactamente los mismos percentiles de tu código FFT
vmax_val = float(np.percentile(W_raw_log, 99.4))
W_raw_log = W_raw_log - vmax_val
vmin_val = float(np.percentile(W_raw_log, 4))
vmax_val = 0.0

# Filtro numérico básico para aliviar el ploteo (descartar lo que está 
# muy por debajo del percentil mínimo)
mask = W_raw_log > (vmin_val - 1.0)
k_plot = k_plot[mask]
w_plot = w_plot[mask]
W_plot = W_raw_log[mask]

plt.figure(figsize=(10, 6))

# 3. Graficamos inyectando el vmin y vmax directamente al scatter
scatter = plt.scatter(k_plot, w_plot, c=W_plot, cmap='plasma', 
                      s=2, alpha=0.8, edgecolors='none', 
                      vmin=vmin_val, vmax=vmax_val)

plt.xlabel('Wave Vector $k$ (Dimer BZ)')
plt.ylabel('Frequency $\\omega$')
plt.title(f'MoI3 LSWT M/L Dispersion and band structure for c=2')
plt.colorbar(scatter, label='$\\log_{10}$ Relative Intensity (shifted)')
plt.grid(True, alpha=0.3)
plt.xlim(-np.pi/2, np.pi/2)
plt.ylim(0, np.max(w_plot) * 1.05)

if ANNOTATE_CELLS:
    ax = plt.gca()
    omega_max = float(np.max(w_plot) * 1.05)
    y_base = min(2.0e14, 0.84 * omega_max)
    y_crys = min(y_base + 0.10 * omega_max, 0.95 * omega_max)
    y_mag = y_base
    y_pbc = max(y_base - 0.10 * omega_max, 0.08 * omega_max)

    # Celda cristalografica del dimero en k (d=2a): ancho completo de -pi/2 a pi/2.
    ax.annotate(
        "",
        xy=(-np.pi/2, y_crys),
        xytext=(np.pi/2, y_crys),
        arrowprops=dict(arrowstyle="<->", color="white", lw=1.4),
    )
    ax.text(
        0.0,
        y_crys - 0.035 * omega_max,
        r"Crystallographic Cell: (dimer, $d=2a$): $k\in[-\pi/2,\pi/2]$",
        color="white",
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="white"),
    )

    # Celda magnetica: modo incommensurado en +-q_real.
    ax.axvline(+q_real, color="cyan", lw=1.5, ls="--", alpha=0.95)
    ax.axvline(-q_real, color="cyan", lw=1.5, ls="--", alpha=0.95)

    ax.annotate(
        "",
        xy=(-q_real, y_mag),
        xytext=(q_real, y_mag),
        arrowprops=dict(arrowstyle="<->", color="cyan", lw=1.6),
    )
    ax.text(
        0.0,
        y_mag - 0.035 * omega_max,
        rf"Magnetic Cell: $k_{{mag}}\in[-q_{{inc}},q_{{inc}}],\ q_{{inc}}={q_real:.4f}$",
        color="cyan",
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="cyan"),
    )

plt.gca().set_facecolor('#110022')
plt.show()