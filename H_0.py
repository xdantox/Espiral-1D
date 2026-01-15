import numpy as np
import matplotlib.pyplot as plt

norm = 0.05788 
Jnn = 46.75 / norm  
dJnn = 44.85 / norm
Jnnn = 2.6 / norm
K = 45.4 / norm
D = 0.76 / norm
Din = D
S_mag = 1.0



N_sim   = 1198      # n de cadena0
M_sim   = 197       # M de cadena0
gamma   = -1.2867   # gamma de cadena0 (Phase Dimerization)

# Nota: alpha1 y phi1 SE IGNORAN aquí. 
# Su efecto se calcula analíticamente en las matrices Gamma, no en H0.

# ==========================================
# 2. INICIALIZACIÓN DE H_0 (Base Espiral)
# ==========================================
q_real = 2.0 * np.pi * M_sim / N_sim
c = 2

# Construimos los ángulos para la celda unidad fundamental
# Siguiendo la lógica: theta_n = n*q + gamma * (-1)^n
th_vca = np.zeros(c)
for i in range(c):
    parity = 1.0 if (i % 2 == 0) else -1.0
    th_vca[i] = i * q_real + gamma * parity

def add_nn_block(M_k, i, j, phase, d_angle, J_link):
    cosd = np.cos(d_angle)
    sind2 = np.sin(d_angle)**2

    # Heisenberg (add symmetric contributions to keep Mk Hermitian)
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

    # K term B (only affects v-sector)
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
# ...existing code...
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

        M_k[2*i+1, 2*i+1] += Din

    return M_k
q_vals = np.linspace(-5 * np.pi, 5 * np.pi, 120000)


bands = []
for k in q_vals:
    M = get_bloch_matrix_cartesian(k)
    Dyn = Sigma @ M
    evals = np.linalg.eigvals(Dyn)
    evals = np.sort(np.imag(evals))
    bands.append(evals[-c:])
bands = np.array(bands)

plt.figure(figsize=(10, 6))
for i in range(c):
    plt.plot(q_vals, 1.7e11 * bands[:, i], '-', linewidth=2, label=f'Banda {i+1}')
plt.xlabel('Vector de Onda $k$')
plt.ylabel('Frecuencia $\\omega$')
plt.title(f'Dispersión corregida (c={c})')
plt.grid(True, alpha=0.3)
plt.xlim(-np.pi,np.pi)
plt.axvline(q_real, color='r', linestyle='--', alpha=0.5, label='$q_{spiral}$')
plt.legend()
plt.show()
