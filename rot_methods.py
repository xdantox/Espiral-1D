import numpy as np
from Heff import H_eff_p_vectorized_PBC_folded,H_eff_p_vectorized_PBC,H_eff_p_vectorized_FOBC,H_eff_pbc_numba
from rot import omega



def implicit_midpoint_step_vectorized_PBC(c_old, dt, n_iter=14):
    """
    Versión TOTALMENTE VECTORIZADA del punto medio implícito.
    """
    # Estimación inicial: un paso de Euler explícito
    H0 = H_eff_p_vectorized_PBC(c_old)
    w0 = omega(H0, c_old)
    c_new = c_old + dt * np.cross(w0, c_old, axisa=1, axisb=1)
    c_new /= np.linalg.norm(c_new, axis=1)[:, None]

    for _ in range(n_iter):
        c_mid = 0.5 * (c_old + c_new)
        c_mid /= np.linalg.norm(c_mid, axis=1)[:, None] # Renormalizar punto medio
        
        H_mid = H_eff_p_vectorized_PBC(c_mid)
        w_mid = omega(H_mid, c_mid)
        
        # Aplicar rotación de forma vectorizada
        # 1. Calcular todas las matrices de rotación
        angle = np.linalg.norm(w_mid, axis=1) * dt
        axis = w_mid / np.linalg.norm(w_mid, axis=1)[:, None]
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        
        # Matriz de rotación para cada espín (n, 3, 3)
        R_matrices = np.array([
            [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
        ]).transpose(2, 0, 1)

        # 2. Aplicar todas las rotaciones a la vez con einsum
        c_new = np.einsum('nij,nj->ni', R_matrices, c_old)
        c_new /= np.linalg.norm(c_new, axis=1)[:, None]
        
    return c_new

def implicit_midpoint_step_vectorized_PBC_folded(c_old, dt, n_iter=14):
    """
    Versión TOTALMENTE VECTORIZADA del punto medio implícito.
    """
    # Estimación inicial: un paso de Euler explícito
    H0 = H_eff_p_vectorized_PBC_folded(c_old)
    w0 = omega(H0, c_old)
    c_new = c_old + dt * np.cross(w0, c_old, axisa=1, axisb=1)
    c_new /= np.linalg.norm(c_new, axis=1)[:, None]

    for _ in range(n_iter):
        c_mid = 0.5 * (c_old + c_new)
        c_mid /= np.linalg.norm(c_mid, axis=1)[:, None] # Renormalizar punto medio
        
        H_mid = H_eff_p_vectorized_PBC_folded(c_mid)
        w_mid = omega(H_mid, c_mid)
        
        # Aplicar rotación de forma vectorizada
        # 1. Calcular todas las matrices de rotación
        angle = np.linalg.norm(w_mid, axis=1) * dt
        axis = w_mid / np.linalg.norm(w_mid, axis=1)[:, None]
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        
        # Matriz de rotación para cada espín (n, 3, 3)
        R_matrices = np.array([
            [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
        ]).transpose(2, 0, 1)

        # 2. Aplicar todas las rotaciones a la vez con einsum
        c_new = np.einsum('nij,nj->ni', R_matrices, c_old)
        c_new /= np.linalg.norm(c_new, axis=1)[:, None]
        
    return c_new
def implicit_midpoint_step_vectorized_FOBC(c_old, dt, n_iter=14):
    """
    Versión TOTALMENTE VECTORIZADA del punto medio implícito con condiciones de frontera abiertas.
    """
    # Estimación inicial: un paso de Euler explícito
    H0 = H_eff_p_vectorized_FOBC(c_old)
    w0 = omega(H0, c_old)
    c_new = c_old + dt * np.cross(w0, c_old, axisa=1, axisb=1)
    c_new /= np.linalg.norm(c_new, axis=1)[:, None]

    for _ in range(n_iter):
        c_mid = 0.5 * (c_old + c_new)
        c_mid /= np.linalg.norm(c_mid, axis=1)[:, None] # Renormalizar punto medio
        
        H_mid = H_eff_p_vectorized_FOBC(c_mid)
        w_mid = omega(H_mid, c_mid)
        
        # Aplicar rotación de forma vectorizada
        # 1. Calcular todas las matrices de rotación
        angle = np.linalg.norm(w_mid, axis=1) * dt
        axis = w_mid / np.linalg.norm(w_mid, axis=1)[:, None]
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        
        # Matriz de rotación para cada espín (n, 3, 3)
        R_matrices = np.array([
            [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
        ]).transpose(2, 0, 1)

        # 2. Aplicar todas las rotaciones a la vez con einsum
        c_new = np.einsum('nij,nj->ni', R_matrices, c_old)
        c_new /= np.linalg.norm(c_new, axis=1)[:, None]
        
    return c_new

def implicit_midpoint_step_vectorized_PBC_numba(c_old, dt, n_iter=14):
    """
    Versión TOTALMENTE VECTORIZADA del punto medio implícito usando numba para acelerar.
    """
    # Estimación inicial: un paso de Euler explícito
    H0 = H_eff_pbc_numba(c_old)
    w0 = omega(H0, c_old)
    c_new = c_old + dt * np.cross(w0, c_old, axisa=1, axisb=1)
    c_new /= np.linalg.norm(c_new, axis=1)[:, None]

    for _ in range(n_iter):
        c_mid = 0.5 * (c_old + c_new)
        c_mid /= np.linalg.norm(c_mid, axis=1)[:, None] # Renormalizar punto medio
        
        H_mid = H_eff_pbc_numba(c_mid)
        w_mid = omega(H_mid, c_mid)
        
        # Aplicar rotación de forma vectorizada
        # 1. Calcular todas las matrices de rotación
        angle = np.linalg.norm(w_mid, axis=1) * dt
        axis = w_mid / np.linalg.norm(w_mid, axis=1)[:, None]
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        
        # Matriz de rotación para cada espín (n, 3, 3)
        R_matrices = np.array([
            [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
        ]).transpose(2, 0, 1)

        # 2. Aplicar todas las rotaciones a la vez con einsum
        c_new = np.einsum('nij,nj->ni', R_matrices, c_old)
        c_new /= np.linalg.norm(c_new, axis=1)[:, None]
        
    return c_new