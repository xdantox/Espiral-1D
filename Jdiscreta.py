import numpy as np
import matplotlib.pyplot as plt

norm = 0.05788 
Jnn = 48.891035 / norm      
dJnn = 48.620365 / norm
Jnnn= 1.26126 / norm
D = 0.31 / norm
K= 48.9119 / norm
S_mag = 1.0



q_real = 2.159648665
theta_A_offset = 1.265882
theta_B_offset = 1.875699

c = 2
th_vca = np.zeros(c)
for i in range(c):
    m = i // 2
    if i % 2 == 0:
        th_vca[i] = 2 * m * q_real + theta_A_offset
    else:
        th_vca[i] = (2 * m + 1) * q_real + theta_B_offset

# ...existing code...
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
