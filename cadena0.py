import numpy as np

def cadena0spinhistory(n):
    Spin_history = np.load('D_plane = 1.05D  fluc.npy',mmap_mode='r')  # Carga del historial de spins
    num_pasos = Spin_history.shape[0]
    base = Spin_history[num_pasos-1]
    result = []
    pattern_len = len(base)
    noise_x = 0 * np.random.randn(n)
    noise_x -= noise_x.mean()  # centrar ruido (evita drift global)
    for i in range(n):
        vec = base[i % pattern_len].copy()  # PBC via índice modular
        vec[0] += noise_x[i]                # agregar ruido en eje x
        vec /= np.linalg.norm(vec)          # normalizar
        result.append(vec)
    return np.array(result)



def cadena0harmonic_PBC(
    n,
    M,
    gamma,
    mx=0.0,
    alpha1=0.0,
    phi1=0.0,
    phi_sign=1.0, # Ojo con el signo aquí también si definiste paridad
    noise_x=0.0,
):
    idx = np.arange(n, dtype=np.int64)
    
    # Cálculo de theta (Correcto)
    q = 2.0 * np.pi * M / n
    base = idx * q
    parity = np.where((idx & 1) == 0, 1.0, -1.0)
    theta = base + gamma * parity + alpha1 * np.sin(2.0 * q * idx + phi1)

    spins = np.zeros((n, 3), dtype=float)
    
    mx_val = np.clip(mx, -1.0, 1.0)
    plane_radius = np.sqrt(max(0.0, 1.0 - mx_val**2))
    
    spins[:, 0] = mx_val
    
    # --- CORRECCIÓN AQUÍ ---
    # Sy debe ser COSENO para coincidir con direct3_PBC
    # Sz debe ser SENO (o viceversa, pero Sy manda por la anisotropía)
    
    spins[:, 1] = plane_radius * np.cos(theta)  # Sy = cos(theta) (Eje Difícil)
    spins[:, 2] = plane_radius * np.sin(theta)  # Sz = sin(theta) (Eje Fácil)

    # Nota: phi_sign lo quité o aplícalo al seno, pero lo importante es Sy=Cos

    if noise_x != 0.0:
        noise = noise_x * np.random.randn(n)
        noise -= noise.mean()
        spins[:, 0] += noise 

    norms = np.linalg.norm(spins, axis=1, keepdims=True)
    np.divide(spins, norms, out=spins, where=norms > 0)
    return spins