import numpy as np

def cadena0spinhistory(
    n,
    source_file='D_plane = 1.65D relaxYZ n = 1188 noise=1e-3.npy',
    pulse=False,
    n0=None,
    eps=1e-4,
):
    Spin_history = np.load(source_file, mmap_mode='r')  # Carga del historial de spins
    num_pasos = Spin_history.shape[0]
    base = Spin_history[num_pasos-1]
    result = []
    pattern_len = len(base)
    np.random.seed(1) # Semilla para reproducibilidad
    noise_x = 0.00 * np.random.randn(n)
    noise_x -= noise_x.mean()  # centrar ruido (evita drift global)
    noise_y = 0.00 * np.random.randn(n)
    noise_y -= noise_y.mean()  # centrar ruido (evita drift global)
    for i in range(n):
        vec = base[i % pattern_len].copy()  # PBC via índice modular
        vec[0] += noise_x[i]                # agregar ruido en eje x
        vec[1] += noise_y[i]                # agregar ruido en eje y
        vec /= np.linalg.norm(vec)          # normalizar
        result.append(vec)
    spins = np.array(result)

    if pulse:
        if n0 is None:
            raise ValueError("n0 es obligatorio cuando pulse=True")
        if not 0 <= n0 < n:
            raise ValueError(f"n0 debe estar en [0, {n})")

        indices = np.arange(n)
        delta = (indices - n0 + n // 2) % n - n // 2
        weight = np.exp(-(delta.astype(float) ** 2) / 8.0)
        weight[np.abs(delta) > 4] = 0.0

        yhat = np.array([0.0, 1.0, 0.0])
        tangent = yhat - spins[:, 1, None] * spins
        rotation = np.cross(spins, yhat)
        spins += eps * weight[:, None] * (tangent + rotation)
        spins /= np.linalg.norm(spins, axis=1, keepdims=True)

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


def inicializar_espiral_XZ_monotona(N: int = 1198, S: float = 1.0, 
                                    incluir_peierls: bool = True) -> np.ndarray:
    """
    Inicializa una espiral coplanar monótona en el plano XZ para D_plane > D (o D_plane > 2D).
    
    - Plano fácil: XZ (Sy = 0.0 estrictamente para evitar nucleación de solitones)
    - Vector Q cerrado por PBC: M = 402 vueltas exactas en N = 1198
    - Dimerización de Peierls: gamma = -0.2845 rad en enlaces fuertes/débiles
    """
    idx = np.arange(N, dtype=float)
    
    # 1. Vector de onda cuantizado para PBC exacta en N = 1198
    M_vueltas = int(np.round(2.110643 * N / (2.0 * np.pi)))  # M = 402
    q_pbc = (2.0 * np.pi * M_vueltas) / N                   # = 2.108371 rad/sitio
    
    # 2. Fase angular planar theta_n en el plano (x, z)
    Theta_0 = 0.0
    theta = q_pbc * idx + Theta_0
    
    # 3. (Opcional) Dimerización intrínseca de Peierls del bulk
    if incluir_peierls:
        gamma_bulk = -0.284540  # rad
        staggered = (-1.0) ** np.arange(N)
        theta += gamma_bulk * staggered
    
    # 4. Construcción del arreglo 3D en el plano XZ
    spins = np.zeros((N, 3), dtype=float)
    spins[:, 0] = S * np.sin(theta)  # Sx (eje duro intermedio)
    spins[:, 1] = 0.0               # Sy (eje duro global -> CERO absoluto)
    spins[:, 2] = S * np.cos(theta)  # Sz (eje fácil)
    
    # Normalización estricta
    spins = S * (spins / np.linalg.norm(spins, axis=1, keepdims=True))
    
    return spins