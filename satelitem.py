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
D_yy = -0.76 / norm        # D_plane (In-Plane - Modulado)
S_mag = 1.0
# --- PARÁMETROS DE GEOMETRÍA ---
q_real = 2.1083810463
gamma =2.8578
alpha = 0.0045

# --- TRUNCAMIENTO FLOQUET ---
# Incluye armónicos m = 0, ±1, ..., ±N_max
N_max = 1

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
        th_vca[i] = 2 * m * q_real + gamma
    else:
        th_vca[i] = (2 * m + 1) * q_real - gamma

# =============================================================================
# FUNCIONES DE BLOQUE CORREGIDAS
# =============================================================================
def add_nn_block(M_k, i, j, phase, d_angle, J_link, K_val, bessel_funcs, mode, include_bare=False):
    
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
        if include_bare:
            term_bare = 1.0 # Constante aditiva solo para cos^2
        else:
            term_bare = 0.0
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
    if mode == 'cos'and include_bare:
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
# Definición bloques de anisotropía in-plane modulados G+ y G-
def get_block_aniso_G_plus(m):
    """
    Calcula el bloque de anisotropía in-plane para el canal G+ (salto n -> n + m + 1).
    m: orden de la función de Bessel J_m(2*eps).
    """
    M_k = np.zeros((2*c, 2*c), dtype=complex)
    order = abs(m)   
    bes_val = jv(m, eps_doble)
    factor_m0 = 0.5 if m == 0 else 1.0
    
    if order % 2 == 0:
        # m par: proviene de cos(Theta) * cos(m*Phi)
        factor_parity = (-1)**(order // 2)
        for i in range(c):
            # val_v: -D/2 * parity * bessel * factor_m0 * cos(2th)
            val_v = -(D_yy ) * factor_parity * bes_val * factor_m0 * np.cos(2 * th_vca[i])
            M_k[2*i+1, 2*i+1] = val_v
            M_k[2*i, 2*i]     = val_v / 2.0 # Sector longitudinal u
    else:
        # m impar: proviene de -sin(Theta) * cos(m*Phi)
        factor_parity = (-1)**((order - 1) // 2)
        for i in range(c):
            # val_v: +D/2 * parity * bessel * factor_m0 * sin(2th)
            val_v = (D_yy ) * factor_parity * bes_val * factor_m0 * np.sin(2 * th_vca[i])
            M_k[2*i+1, 2*i+1] = val_v
            M_k[2*i, 2*i]     = val_v / 2.0
            
    return M_k

def get_block_aniso_G_minus(m):
    """
    Calcula el bloque de anisotropía in-plane para el canal G- (salto n -> n + m - 1).
    m: orden de la función de Bessel J_m(2*eps).
    """
    M_k = np.zeros((2*c, 2*c), dtype=complex)
    order = abs(m)   
    bes_val = jv(m, eps_doble)
    factor_m0 = 0.5 if m == 0 else 1.0
    
    if order % 2 == 0:
        factor_parity = (-1)**(order // 2)
        for i in range(c):
            val_v = -(D_yy ) * factor_parity * bes_val * factor_m0 * np.cos(2 * th_vca[i])
            M_k[2*i+1, 2*i+1] = val_v
            M_k[2*i, 2*i]     = val_v / 2.0
    else:
        factor_parity = (-1)**((order - 1) // 2)
        for i in range(c):
            val_v = (D_yy ) * factor_parity * bes_val * factor_m0 * np.sin(2 * th_vca[i])
            M_k[2*i+1, 2*i+1] = val_v
            M_k[2*i, 2*i]     = val_v / 2.0
            
    return M_k
def get_block_4x4(k_val, order=0, is_diagonal=False):
    """
    Calcula un bloque de la matriz de Floquet.
    
    Args:
        k_val: Momento evaluado (k + mQ).
        order: Distancia armónica (Delta m) para elegir la función de Bessel correcta.
        is_diagonal: (Bool) Si True, agrega los términos 'bare' (Energía cinética estática J y Anisotropía Hard Axis).
                     Si False, solo calcula los términos modulados por Bessel.
    """
    M_k = np.zeros((2*c, 2*c), dtype=complex)
    
    # -------------------------------------------------------------------------
    # 1. LÓGICA DE BESSEL
    # -------------------------------------------------------------------------


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

    # -------------------------------------------------------------------------
    # 2. TÉRMINOS BARE (Solo si es Diagonal explícitamente)
    # -------------------------------------------------------------------------
    if is_diagonal:
        # Solo en la diagonal principal se suma la energía base
        include_bare_hopping = True 
        D_hard_term = 2 * D_xx
        D_plane_bare = D_yy 
    else:
        # En los bloques de interacción (order > 0) NUNCA van estos términos
        include_bare_hopping = False
        D_hard_term = 0.0
        D_plane_bare = 0.0

    # -------------------------------------------------------------------------
    # 3. BUCLE DE LLENADO
    # -------------------------------------------------------------------------
    for i in range(c):
        th_i = th_vca[i]
        
        # A. Anisotropías BARE
        M_k[2*i, 2*i] += D_hard_term
        M_k[2*i, 2*i] += D_plane_bare

        # B. Primeros Vecinos (J1)
        J_right = Jnn + (-1)**i * dJnn
        j = (i + 1) % c
        d_ij = (th_vca[j] + (0 if i + 1 < c else c * q_real)) - th_i
        phase_nn = 1.0 + 0j if i + 1 < c else np.exp(1j * k_val * c)
        
        add_nn_block(M_k, i, j, phase_nn, d_ij, J_right, K, bessel_exchange, 
                     mode=mode_exchange, 
                     include_bare=include_bare_hopping) # <--- Flag Hopping

        # C. Segundos Vecinos (J2)
        l = (i + 2) % c
        offset_angle = 0 if i + 2 < c else c * q_real
        d_il = (th_vca[l] + offset_angle) - th_i
        phase_nnn = 1.0 + 0j if i + 2 < c else np.exp(1j * k_val * c)
        
        add_nnn_block(M_k, i, l, phase_nnn, d_il, Jnnn, bes_J2, 
                      mode=mode_exchange,
                      include_bare=include_bare_hopping) # <--- Flag Hopping

    return M_k

def get_floquet_matrix(k, N_max=1):
    block_size = 2 * c
    ms = np.arange(-N_max, N_max + 1)
    n_blocks = len(ms)
    H = np.zeros((block_size * n_blocks, block_size * n_blocks), dtype=complex)

    # --- 1. BLOQUES DIAGONALES ---
    for idx, m in enumerate(ms):
        r0, r1 = idx * block_size, (idx + 1) * block_size
        
        # Bloque base (Intercambio J0 y Hard Axis)
        # Nota: Asegúrate de que este get_block ya NO tenga la aniso in-plane modulada
        M_m = get_block_4x4(k + m * q_real, order=0, is_diagonal=True)
        
        # INYECCIÓN G-: El armónico m=1 de la anisotropía cae en la diagonal (1-1=0)
        M_m += 2.0 * get_block_aniso_G_minus(m=1)
        M_m += 2.0 * get_block_aniso_G_plus(m=-1)
            
        H[r0:r1, r0:r1] = M_m

    # --- 2. BLOQUES DE INTERACCIÓN ---
    max_possible_dist = n_blocks - 1


# --- 2. BLOQUES DE INTERACCIÓN ---
    max_possible_dist = n_blocks - 1

    for dist in range(1, max_possible_dist + 1):
        for idx in range(n_blocks - dist):
            # Índices de la submatriz
            r0, r1 = idx * block_size, (idx + 1) * block_size
            c0, c1 = (idx + dist) * block_size, (idx + dist + 1) * block_size

            dm = ms[idx + dist] - ms[idx]  # Salto positivo (+dist)

            # 1. Calcular SOLO UNA DIRECCIÓN (por ejemplo, el bloque superior/fuera de la diagonal)
            # Intercambio
            V_exch = get_block_4x4(k + ms[idx] * q_real, order=dm, is_diagonal=False)

            # Anisotropía: m>0 lógico para salto +dm
            V_aniso = (
                get_block_aniso_G_plus(m = dm - 1) +
                get_block_aniso_G_minus(m = dm + 1)
            )

            # Matriz de acoplamiento total para esta dirección
            V_total = V_exch + V_aniso

            # 2. INYECTAR USANDO SIMETRÍA HERMÍTICA
            # Si c0:c1 son las FILAS y r0:r1 las COLUMNAS:
            H[c0:c1, r0:r1] = V_total
            H[r0:r1, c0:c1] = V_total.conj().T  

    return H
            
# --- SOLVER ---
Sigma_small = np.zeros((2*c, 2*c), dtype=complex)
for i in range(c):
    Sigma_small[2*i, 2*i+1] = 1.0; Sigma_small[2*i+1, 2*i] = -1.0
Sigma_small /= S_mag
Sigma_big = np.kron(np.eye(2 * N_max + 1), Sigma_small)

q_vals = np.linspace(- 5*np.pi,   5*np.pi, 7000)
bands = []

for k in q_vals:
    H_F = get_floquet_matrix(k, N_max=N_max)
    Dyn = 1j * Sigma_big @ H_F
    evals = np.linalg.eigvals(Dyn)
    # CORRECCIÓN 3: Usar np.imag para frecuencias
    bands.append(np.sort(np.real(evals)))

bands = 1.7e11 * np.array(bands)
half_idx = (2 * c * (2 * N_max + 1)) // 2

# ... (Tu código anterior donde calculas 'bands' y 'q_vals') ...

# --- TRANSFORMACIÓN A MARCO LABORATORIO ---

plt.figure(figsize=(10, 6))

# q_vals es tu eje k en el marco rotado (centrado en 0)
# q_real es el vector de la espiral

# 1. Rama Shift-Left (k_lab = k_rot - q)
# Representa la componente S+ del laboratorio
k_lab_L = q_vals - 2 * q_real
plt.plot(k_lab_L, bands[:, half_idx:], 'b-', alpha=0.4, label='Lab $S^+$ ($k-q$)')

# 2. Rama Shift-Right (k_lab = k_rot + q)
# Representa la componente S- del laboratorio
k_lab_R = q_vals + 2 * q_real
plt.plot(k_lab_R, bands[:, half_idx:], 'r-', alpha=0.4, label='Lab $S^-$ ($k+q$)')

# Decoración
plt.xlabel(r'Vector de Onda Laboratorio $k_{lab}$')
plt.ylabel(r'Frecuencia $\omega$ [rad/s]' )
plt.title(r'Dispersión en Marco Laboratorio (Comparación LLG)')

# Líneas guía
plt.axvline(q_real, color='k', linestyle=':', alpha=0.3, label='Bragg Peak $+q$')
plt.axvline(-q_real, color='k', linestyle=':', alpha=0.3, label='Bragg Peak $-q$')
plt.axvline(0, color='k', linestyle='-', alpha=0.2)

# Ajuste de límites para ver la zona de interés
plt.xlim(-np.pi/2, np.pi/2) 
plt.ylim(bottom=0)

# Manejo de la leyenda para no repetir etiquetas
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))

plt.grid(True, alpha=0.3)
plt.show()