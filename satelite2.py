import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

# --- 1. PARÁMETROS FÍSICOS ---
norm = 0.05788 
Jnn  = 46.75 / norm
dJnn = 44.85 / norm
Jnnn = 2.6 / norm
D_xx = 0.76 / norm        # D (Hard Axis - Bare)
K    = -45.4 / norm
D_yy = -0.76 / norm        # D_plane (In-Plane - Modulado)
S_mag = 1.0

# --- PARÁMETROS DE GEOMETRÍA ---
q_real = 2.10838
theta_A_offset = np.pi/2-0.2841
theta_B_offset = np.pi/2 +0.2841
alpha = 0.0036

# --- ARGUMENTOS DE BESSEL ---
eps_1 = 2 * abs(alpha) * np.sin(q_real)
eps_2_vec = eps_1 * 2 * np.cos(q_real) 
eps_doble = 2 * eps_1 

# --- CONFIGURACIÓN CELDA c=2 ---
c = 2
th_vca = np.zeros(c)
for i in range(c):
    m = i // 2
    if i % 2 == 0:
        th_vca[i] = 2 * m * q_real + theta_A_offset
    else:
        th_vca[i] = (2 * m + 1) * q_real + theta_B_offset

# =============================================================================
# FUNCIONES DE BLOQUE CORREGIDAS
# =============================================================================
def add_nn_block(M_k, i, j, phase, d_angle, J_link, K_val, bessel_funcs, mode):
    
    # --- 1. DEFINICIÓN DE GEOMETRÍAS BASICAS ---
    
    # Geometría Simple: cos(theta)
    # Usada para: J (Transversal) y K (Hopping Longitudinal)
    if mode == 'cos':
        geo_simple = np.cos(d_angle) * bessel_funcs['J_eps']
    else:
        geo_simple = -np.sin(d_angle) * bessel_funcs['J_eps']

    # Geometría Doble: cos(2*theta)
    # Usada para: K (Todo el sector Transversal v) y K (Parte modulada de Masa u)
    if mode == 'cos':
        geo_double = np.cos(2 * d_angle) * bessel_funcs['J_2eps']
        term_bare = 1.0 # Constante aditiva solo para cos^2
    else:
        geo_double = -np.sin(2 * d_angle) * bessel_funcs['J_2eps']
        term_bare = 0.0

    # --- 2. FACTORES EFECTIVOS POR SECTOR (SEGÚN TU DERIVACIÓN) ---

    # Sector U (Longitudinal): Asimetría
    # Hopping: Proporcional a cos(theta) -> geo_simple
    # Masa: Proporcional a cos^2(theta) -> 0.5 * (1 + cos(2theta))
    k_hop_u  = geo_simple
    k_mass_u = 0.5 * (term_bare + geo_double)

    # Sector V (Transversal): SIMPLIFICACIÓN TOTAL
    # Tu fórmula: H = -K * cos(2theta) * (2vi vj - vi^2 - vj^2)
    # Tanto hopping como masa dependen puramente de cos(2theta) -> geo_double
    k_hop_v  = geo_double
    k_mass_v = geo_double

    # --- 3. INYECCIÓN EN MATRIZ ---

    # --- HEISENBERG J (Sin cambios) ---
    delta_diag = -J_link * geo_simple
    M_k[2*i, 2*i]     += delta_diag; M_k[2*i+1, 2*i+1] += delta_diag
    M_k[2*j, 2*j]     += delta_diag; M_k[2*j+1, 2*j+1] += delta_diag
    
    # Hopping U (Bare) - Solo en diagonal
    if mode == 'cos':
        M_k[2*i, 2*j] += J_link * phase
        M_k[2*j, 2*i] += J_link * np.conjugate(phase)
    
    # Hopping V (Modulado)
    M_k[2*i+1, 2*j+1] += J_link * geo_simple * phase
    M_k[2*j+1, 2*i+1] += J_link * geo_simple * np.conjugate(phase)

    # --- BICUADRÁTICO K (SIMPLIFICADO) ---
    
    # Prefactor común: -2 * K * S^2
    # Ojo con signos:
    # Masa tiene signo opuesto en el paréntesis (-u^2/2) -> Prefactor global se vuelve +
    # Hopping mantiene signo del prefactor
    
    # A. SECTOR U (Longitudinal) - Mezcla de geometrías
    
    # Masa U: + K * S^2 * (1 + cos 2theta)/2
    val_mass_u = +2.0 * K_val * S_mag**2 * k_mass_u
    M_k[2*i, 2*i] += val_mass_u
    M_k[2*j, 2*j] += val_mass_u
    
    # Hopping U: - 2 * K * S^2 * cos(theta)
    val_hop_u  = -2.0 * K_val * S_mag**2 * k_hop_u
    M_k[2*i, 2*j] += val_hop_u * phase
    M_k[2*j, 2*i] += val_hop_u * np.conjugate(phase)

    # B. SECTOR V (Transversal) - Geometría Pura Doble
    
    # Masa V: + K * S^2 * cos(2theta)
    # (El + viene de -K global * -1 interno)
    val_mass_v = +2.0 * K_val * S_mag**2 * k_mass_v
    M_k[2*i+1, 2*i+1] += val_mass_v
    M_k[2*j+1, 2*j+1] += val_mass_v
    
    # Hopping V: - 2 * K * S^2 * cos(2theta)
    val_hop_v  = -2.0 * K_val * S_mag**2 * k_hop_v
    M_k[2*i+1, 2*j+1] += val_hop_v * phase
    M_k[2*j+1, 2*i+1] += val_hop_v * np.conjugate(phase)

def add_nnn_block(M_k, i, j, phase, d_angle, J_link, bessel_val, mode):
    if mode == 'cos':
        geo = np.cos(d_angle) * bessel_val
    else:
        geo = -np.sin(d_angle) * bessel_val
    if mode == 'cos':
        M_k[2*i, 2*j]     +=  J_link * phase
        M_k[2*j, 2*i]     +=  J_link * np.conjugate(phase)

    delta_diag = -J_link * geo
    M_k[2*i, 2*i] += delta_diag; M_k[2*i+1, 2*i+1] += delta_diag
    M_k[2*j, 2*j] += delta_diag; M_k[2*j+1, 2*j+1] += delta_diag
    

    M_k[2*i+1, 2*j+1] += J_link * geo * phase
    M_k[2*j+1, 2*i+1] += J_link * geo * np.conjugate(phase)

# CORRECCIÓN 1: Flag include_bare
def add_anisotropy_plane(M_k, i, th_angle, D_val, bessel_val_2eps, mode, include_bare=False):
    if mode == 'cos':
        geo_2theta = np.cos(2 * th_angle) * bessel_val_2eps
    else:
        geo_2theta = -np.sin(2 * th_angle) * bessel_val_2eps

    # Modulado (v y u)
    term_v = -D_val * geo_2theta
    M_k[2*i+1, 2*i+1] += term_v

    term_u_mod = -(D_val / 2.0) * geo_2theta
    M_k[2*i, 2*i] += term_u_mod
    
    # Bare (Solo si se solicita explícitamente)
    if mode == 'cos' and include_bare:
        term_u_bare = +(D_val / 2.0) 
        M_k[2*i, 2*i] += term_u_bare

def get_block_4x4(k_val, block_type='M'):
    M_k = np.zeros((2*c, 2*c), dtype=complex)
    
    if block_type == 'M':
        mode_exchange = 'cos'
        bessel_exchange = {'J_eps': jv(0, eps_1), 'J_2eps': jv(0, eps_doble)}
        bes_J2 = jv(0, eps_2_vec)
        
        # CORRECCIÓN 2: Anisotropía nula en diagonal (bifurcación)
        mode_aniso = 'cos'
        bes_D_val = 2 * jv(1, eps_doble)
        include_bare_aniso = True # Incluir masa bare D/2
        
        D_hard_term = 2 * D_xx
        
    else: # 'V'
        mode_exchange = 'sin'
        bessel_exchange = {'J_eps': 2 * jv(1, eps_1), 'J_2eps': 2 * jv(1, eps_doble)}
        bes_J2 = 2 * jv(1, eps_2_vec)
        
        # Anisotropía usa J0 y Cos (estático)
        mode_aniso = 'cos'
        bes_D_val  = jv(0, eps_doble)
        include_bare_aniso = False # NO incluir masa bare en interacción
        
        D_hard_term = 0.0

    for i in range(c):
        th_i = th_vca[i]
        
        # Pasamos include_bare explícitamente
        add_anisotropy_plane(M_k, i, th_i, D_yy, bes_D_val, mode=mode_aniso, include_bare=include_bare_aniso)
        
        M_k[2*i, 2*i] += D_hard_term
        J_right = Jnn + (-1)**i * dJnn
        j = (i + 1) % c
        d_ij = (th_vca[j] + (0 if i + 1 < c else c * q_real)) - th_i
        phase_nn = 1.0 + 0j if i + 1 < c else np.exp(1j * k_val * c/2)
        
        add_nn_block(M_k, i, j, phase_nn, d_ij, J_right, K, bessel_exchange, mode=mode_exchange)

        l = (i + 2) % c
        d_il = (th_vca[l] + (0 if i + 2 < c else c * q_real)) - th_i
        phase_nnn = 1.0 + 0j if i + 2 < c else np.exp(1j * k_val * c/2)
        add_nnn_block(M_k, i, l, phase_nnn, d_il, Jnnn, bes_J2, mode=mode_exchange)

    return M_k

def get_floquet_matrix(k):
    # 1. Bloques Diagonales (Masas) - Sin cambios
    # Representan la energía interna de cada sector armónico
    M_minus = get_block_4x4(k - q_real, block_type='M') # m = -1
    M_zero  = get_block_4x4(k,          block_type='M') # m = 0
    M_plus  = get_block_4x4(k + q_real, block_type='M') # m = +1
    
    # 2. Bloques de Interacción (Satélites) - AHORA DEPENDIENTES DE K
    # Calculamos el bloque V específico para cada "puente" entre sectores.
    # Asumimos la convención: V_m conecta el sector m con m+1.
    
    # V_minus: Conecta el sector m=-1 (k-Q) hacia el m=0 (k)
    V_minus = get_block_4x4(k - q_real, block_type='V') 
    
    # V_zero: Conecta el sector m=0 (k) hacia el m=1 (k+Q)
    V_zero  = get_block_4x4(k, block_type='V')
    
    # 3. Ensamblaje de la Matriz 12x12
    H = np.zeros((12, 12), dtype=complex)
    
    # --- Diagonales ---
    H[0:4, 0:4]   = M_minus
    H[4:8, 4:8]   = M_zero
    H[8:12, 8:12] = M_plus
    
    # --- Off-Diagonales (Mezcla) ---
    
    # A. Interfaz entre m=-1 y m=0 (Usamos V_minus)
    # H[4:8, 0:4] es el bloque (1,0): Salta de -1 a 0
    H[4:8, 0:4]   = V_minus
    # H[0:4, 4:8] es el bloque (0,1): Salta de 0 a -1 (Hermítico)
    H[0:4, 4:8]   = V_minus.conj().T
    
    # B. Interfaz entre m=0 y m=1 (Usamos V_zero)
    # H[8:12, 4:8] es el bloque (2,1): Salta de 0 a 1
    H[8:12, 4:8]  = V_zero
    # H[4:8, 8:12] es el bloque (1,2): Salta de 1 a 0 (Hermítico)
    H[4:8, 8:12]  = V_zero.conj().T
    
    return H
# --- SOLVER ---
Sigma_small = np.zeros((2*c, 2*c), dtype=complex)
for i in range(c):
    Sigma_small[2*i, 2*i+1] = 1.0; Sigma_small[2*i+1, 2*i] = -1.0
Sigma_small /= S_mag
Sigma_big = np.kron(np.eye(3), Sigma_small)

q_vals = np.linspace(-6 * np.pi, 6 * np.pi, 4000)
bands = []

for k in q_vals:
    H_F = get_floquet_matrix(k)
    Dyn = 1j * Sigma_big @ H_F
    evals = np.linalg.eigvals(Dyn)
    # CORRECCIÓN 3: Usar np.imag para frecuencias
    bands.append(np.sort(np.real(evals)))

bands = 1.7e11 * np.array(bands)

# ... (Tu código anterior donde calculas 'bands' y 'q_vals') ...

# --- TRANSFORMACIÓN A MARCO LABORATORIO ---

plt.figure(figsize=(10, 6))

# q_vals es tu eje k en el marco rotado (centrado en 0)
# q_real es el vector de la espiral

# 1. Rama Shift-Left (k_lab = k_rot - q)
# Representa la componente S+ del laboratorio
k_lab_L = q_vals - q_real
plt.plot(k_lab_L, bands[:, 6:], 'b-', alpha=0.4, label='Lab $S^+$ ($k-q$)')

# 2. Rama Shift-Right (k_lab = k_rot + q)
# Representa la componente S- del laboratorio
k_lab_R = q_vals + q_real
plt.plot(k_lab_R, bands[:, 6:], 'r-', alpha=0.4, label='Lab $S^-$ ($k+q$)')

# Decoración
plt.xlabel(r'Vector de Onda Laboratorio $k_{lab}$')
plt.ylabel(r'Frecuencia $\omega$')
plt.title(r'Dispersión en Marco Laboratorio (Comparación LLG)')

# Líneas guía
plt.axvline(q_real, color='k', linestyle=':', alpha=0.3, label='Bragg Peak $+q$')
plt.axvline(-q_real, color='k', linestyle=':', alpha=0.3, label='Bragg Peak $-q$')
plt.axvline(0, color='k', linestyle='-', alpha=0.2)

# Ajuste de límites para ver la zona de interés
plt.xlim(-np.pi, np.pi) 
plt.ylim(bottom=0)

# Manejo de la leyenda para no repetir etiquetas
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.grid(True, alpha=0.3)
plt.show()