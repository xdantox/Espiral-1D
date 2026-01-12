import numpy as np

def cadena0spinhistory(n):
    Spin_history = np.load('D_history.npy',mmap_mode='r')  # Carga del historial de spins
    num_pasos = Spin_history.shape[0]
    base = Spin_history[num_pasos-1]
    result = []
    pattern_len = len(base)
    noise_x = 0.0001 * np.random.randn(n)
    noise_x -= noise_x.mean()  # centrar ruido (evita drift global)
    for i in range(n):
        vec = base[i % pattern_len].copy()  # PBC via índice modular
        vec[0] += noise_x[i]                # agregar ruido en eje x
        vec /= np.linalg.norm(vec)          # normalizar
        result.append(vec)
    return np.array(result)

def cadena0ansatz(n, q, vartA, vartB, phi_sign=1.0, noise_x=0.0):
    """Genera el estado base helicoidal (θ-modulado) coherente con direct.py.

    Args:
        n: número total de espines en la cadena (se asume alternancia A/B).
        q: incremento helicoidal por sitio (mismo q usado en direct.py).
        vartA, vartB: offsets polares para las subredes A y B (en radianes).
        phi_sign: +1 fija φ=+π/2 (y>0); -1 aplica φ=-π/2 (y<0) en toda la cadena.
        noise_x: amplitud opcional de ruido aleatorio en el eje x (hard axis) para
                  romper degeneraciones numéricas; por defecto 0 → espiral ideal.

    Returns:
        np.ndarray de forma (n, 3) con los vectores de espín normalizados.
    """
    idx = np.arange(n, dtype=int)
    is_B = (idx % 2 == 1)

    # El ansatz en direct.py asigna θ_i = i·q + vartheta_{subred} para reproducir los saltos q±Δ.
    theta_offsets = np.where(is_B, vartB, vartA)
    theta = idx * q + theta_offsets

    spins = np.zeros((n, 3), dtype=float)
    spins[:, 1] = phi_sign * np.sin(theta)  # componente y
    spins[:, 2] = np.cos(theta)             # componente z

    if noise_x != 0.0:
        noise = noise_x * np.random.randn(n)
        noise -= noise.mean()
        spins[:, 0] = noise

    # Normalizar cada espín para asegurar |S|=1 (relevante si hay ruido en x).
    norms = np.linalg.norm(spins, axis=1, keepdims=True)
    np.divide(spins, norms, out=spins, where=norms > 0)
    return spins



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