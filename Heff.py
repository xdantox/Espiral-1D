import numpy as np
from dot import dot, Mdot
# Parámetros
#Jnn = 48.891035
#dJnn = 48.620365
#Jnnn= 1.26126 
#De = 0.31
#Ka= 48.9119 
# Set 2 parameters
Jnn = 46.75
dJnn = 44.85
Jnnn = 2.6
Ka = 45.4
De = 0.76
#Jnn = 46.812805
#dJnn = 44.873295
#De = 0.76
#Jnnn = 2.60139
#Ka = 45.4866
#Jnn = 17.10185
#dJnn = 11.80055
#De = 0.33
#Jnnn = 0.0085
#Ka = 16.8117
muB_meV_T = 0.05788
# Definir J_perp (interacción entre ramas)
J_perp = 0.03  # Valor de ejemplo (meV), ajústalo según tu física (débil vs fuerte)
# Ve índices: 0:J_impar, 1:J_par, 2:Jnnn, 3:Ani, 4:Biquad, 5:J_perp
Ve = np.array([
    (Jnn - dJnn)/ muB_meV_T, 
    (Jnn + dJnn)/ muB_meV_T, 
    Jnnn/ muB_meV_T, 
    2*De/ muB_meV_T, 
    2*Ka/ muB_meV_T,
    J_perp/ muB_meV_T  
])
dt = 1e-15 # Paso de tiempo (s)
total_time = (2**17) * dt  # Tiempo total de simulación (s)
num_pasos = int(total_time / dt)  # Número total de pasos de tiempo
a1, a2 = 1.5 , 1  # Distancias alternantes
n = 1198   # Número de espines en la cadena


def H_eff_p_vectorized_PBC(cadena):
    n = len(cadena)
    H_total = np.zeros_like(cadena)
    
    # Índices de los vecinos
    i_minus_1 = np.roll(np.arange(n), 1)
    i_plus_1 = np.roll(np.arange(n), -1)
    i_minus_2 = np.roll(np.arange(n), 2)
    i_plus_2 = np.roll(np.arange(n), -2)

    # Interacciones para espines pares e impares
    indices_pares = np.arange(0, n, 2)
    indices_impares = np.arange(1, n, 2)

    # --- Interacciones de primeros vecinos (NN) ---
    # Para sitios PARES: H += Ve[0]*S_{i-1} + Ve[1]*S_{i+1}
    H_total[indices_pares] += Ve[0] * cadena[i_minus_1[indices_pares]]
    H_total[indices_pares] += Ve[1] * cadena[i_plus_1[indices_pares]]

    # Para sitios IMPARES: H += Ve[1]*S_{i-1} + Ve[0]*S_{i+1}
    H_total[indices_impares] += Ve[1] * cadena[i_minus_1[indices_impares]]
    H_total[indices_impares] += Ve[0] * cadena[i_plus_1[indices_impares]]

    # Jnnn (segundos vecinos)
    H_total += Ve[2] * (cadena[i_minus_2] + cadena[i_plus_2])

    # Anisotropía De
    H_total += Ve[3] * cadena[:, 0][:, np.newaxis] * np.array([1, 0, 0])

    # Término Bicuadrático Ka
    dot_minus_1 = np.einsum('ij,ij->i', cadena[i_minus_1], cadena)
    dot_plus_1 = np.einsum('ij,ij->i', cadena, cadena[i_plus_1])
    H_total += Ve[4] * (dot_minus_1[:, np.newaxis] * cadena[i_minus_1] + dot_plus_1[:, np.newaxis] * cadena[i_plus_1])
    
    return H_total

def H_eff_p_vectorized_PBC_folded(cadena):
    n = len(cadena)
    H_total = np.zeros_like(cadena)
    
    # Índices de los vecinos
    i_minus_1 = np.roll(np.arange(n), 1)
    i_plus_1 = np.roll(np.arange(n), -1)
    i_minus_2 = np.roll(np.arange(n), 2)
    i_plus_2 = np.roll(np.arange(n), -2)

    # --- NUEVO: Índices de la Topología Folded (Escalera) ---
    # Mapea el índice i al índice de la rama opuesta (N - 1 - i)
    # Esto crea la geometría de "horquilla"
    i_folded = (n - 1) - np.arange(n) 

    # Interacciones para espines pares e impares
    indices_pares = np.arange(0, n, 2)
    indices_impares = np.arange(1, n, 2)

    # --- Interacciones de primeros vecinos (NN) ---
    # Para sitios PARES
    H_total[indices_pares] += Ve[0] * cadena[i_minus_1[indices_pares]]
    H_total[indices_pares] += Ve[1] * cadena[i_plus_1[indices_pares]]

    # Para sitios IMPARES
    H_total[indices_impares] += Ve[1] * cadena[i_minus_1[indices_impares]]
    H_total[indices_impares] += Ve[0] * cadena[i_plus_1[indices_impares]]

    # Jnnn (segundos vecinos)
    H_total += Ve[2] * (cadena[i_minus_2] + cadena[i_plus_2])

    # Anisotropía De
    H_total += Ve[3] * cadena[:, 0][:, np.newaxis] * np.array([1, 0, 0])

    # Término Bicuadrático Ka
    dot_minus_1 = np.einsum('ij,ij->i', cadena[i_minus_1], cadena)
    dot_plus_1 = np.einsum('ij,ij->i', cadena, cadena[i_plus_1])
    H_total += Ve[4] * (dot_minus_1[:, np.newaxis] * cadena[i_minus_1] + dot_plus_1[:, np.newaxis] * cadena[i_plus_1])
    
    # --- NUEVO: Interacción Inter-Rama (Folded / Moiré) ---
    # Añade el campo efectivo proveniente de la rama opuesta.
    # H_eff += J_perp * S_{folded}
    H_total += Ve[5] * cadena[i_folded]

    return H_total


def H_eff_p_vectorized_FOBC(cadena):
    n = len(cadena)
    H_total = np.zeros_like(cadena)

    indices_pares = np.arange(0, n, 2)
    indices_impares = np.arange(1, n, 2)

    # Primeros vecinos (NN) sin wrap
    if n > 1:
        # aporte del vecino izquierdo (i-1), válido para i >= 1
        left = np.arange(1, n)
        left_pares = left[left % 2 == 0]
        left_impares = left[left % 2 == 1]
        H_total[left_pares] += Ve[0] * cadena[left_pares - 1]
        H_total[left_impares] += Ve[1] * cadena[left_impares - 1]

        # aporte del vecino derecho (i+1), válido para i <= n-2
        right = np.arange(0, n - 1)
        right_pares = right[right % 2 == 0]
        right_impares = right[right % 2 == 1]
        H_total[right_pares] += Ve[1] * cadena[right_pares + 1]
        H_total[right_impares] += Ve[0] * cadena[right_impares + 1]

    # Segundos vecinos (NNN) sin wrap: i-2 y i+2 válidos dentro de [0, n)
    if n > 2:
        mid = np.arange(2, n)        # para i-2
        H_total[mid] += Ve[2] * cadena[mid - 2]
        mid = np.arange(0, n - 2)    # para i+2
        H_total[mid] += Ve[2] * cadena[mid + 2]

    # Anisotropía De (afecta a todos)
    H_total += Ve[3] * cadena[:, 0][:, np.newaxis] * np.array([1, 0, 0])

    # Término bicuadrático Ka, solo donde existe vecino
    if n > 1:
        # con i-1
        i_left = np.arange(1, n)
        dot_left = np.einsum('ij,ij->i', cadena[i_left - 1], cadena[i_left])
        H_total[i_left] += Ve[4] * dot_left[:, np.newaxis] * cadena[i_left - 1]
        # con i+1
        i_right = np.arange(0, n - 1)
        dot_right = np.einsum('ij,ij->i', cadena[i_right], cadena[i_right + 1])
        H_total[i_right] += Ve[4] * dot_right[:, np.newaxis] * cadena[i_right + 1]

    return H_total



#Numba:

from numba import njit
import numpy as np

# @njit(cache=True) # Descomenta esto para guardar la compilación en disco
@njit
def H_eff_pbc_numba_in(cadena, J_pair_left, J_pair_right, J_nnn, D_aniso, K_biq):
    n = len(cadena)
    # empty_like es seguro porque sobrescribimos todo H_total abajo
    H_total = np.empty_like(cadena) 

    # NOTA: Borré las asignaciones 'J_pair_left = Ve[0]', etc.
    # Ahora usa los valores que realmente le pasas a la función.

    for i in range(n):
        idx_im1 = (i - 1) % n
        idx_ip1 = (i + 1) % n
        idx_im2 = (i - 2) % n
        idx_ip2 = (i + 2) % n

        # Acceso directo para evitar overhead de slicing
        S_i   = cadena[i]
        S_im1 = cadena[idx_im1]
        S_ip1 = cadena[idx_ip1]
        S_im2 = cadena[idx_im2]
        S_ip2 = cadena[idx_ip2]

        # --- 1. PRIMEROS VECINOS (NN) ---
        if i % 2 == 0:
            # Pares
            nn0 = J_pair_left * S_im1[0] + J_pair_right * S_ip1[0]
            nn1 = J_pair_left * S_im1[1] + J_pair_right * S_ip1[1]
            nn2 = J_pair_left * S_im1[2] + J_pair_right * S_ip1[2]
        else:
            # Impares
            nn0 = J_pair_right * S_im1[0] + J_pair_left * S_ip1[0]
            nn1 = J_pair_right * S_im1[1] + J_pair_left * S_ip1[1]
            nn2 = J_pair_right * S_im1[2] + J_pair_left * S_ip1[2]

        # --- 2. SEGUNDOS VECINOS (NNN) ---
        nnn0 = J_nnn * (S_im2[0] + S_ip2[0])
        nnn1 = J_nnn * (S_im2[1] + S_ip2[1])
        nnn2 = J_nnn * (S_im2[2] + S_ip2[2])

        # --- 3. TÉRMINO BICUADRÁTICO ---
        # Producto punto manual (más rápido que np.dot para vectores de 3D en loops)
        dot_left  = S_i[0]*S_im1[0] + S_i[1]*S_im1[1] + S_i[2]*S_im1[2]
        dot_right = S_i[0]*S_ip1[0] + S_i[1]*S_ip1[1] + S_i[2]*S_ip1[2]

        biq0 = K_biq * (dot_left * S_im1[0] + dot_right * S_ip1[0])
        biq1 = K_biq * (dot_left * S_im1[1] + dot_right * S_ip1[1])
        biq2 = K_biq * (dot_left * S_im1[2] + dot_right * S_ip1[2])

        # --- ASIGNACIÓN FINAL (Segura para empty_like) ---
        # Anisotropía en X (S_i[0]) e Y (S_i[1])
        H_total[i, 0] = nn0 + nnn0 + D_aniso * S_i[0] + biq0
        H_total[i, 1] = nn1 + nnn1 + D_aniso * S_i[1] + biq1
        H_total[i, 2] = nn2 + nnn2 + biq2

    return H_total

# Wrapper para facilitar el uso desde fuera
def H_eff_pbc_numba(cadena):
    # Desempaquetamos Ve aquí, en el nivel de Python, antes de llamar a Numba
    return H_eff_pbc_numba_in(
        cadena,
        Ve[0], Ve[1], Ve[2], Ve[3], Ve[4]
    )

    # ...existing code...

