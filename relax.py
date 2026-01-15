import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. TU FUNCIÓN DE CARGA (Tal cual la enviaste)
# ==============================================================================
def cadena0spinhistory(n):
    # Asegúrate de que este archivo esté en la misma carpeta o ajusta la ruta
    try:
        Spin_history = np.load('D_history.npy', mmap_mode='r')
    except FileNotFoundError:
        print("ERROR: No se encuentra el archivo .npy. Usando datos sintéticos para demo.")
        # Generar datos sintéticos si no hay archivo (SOLO PARA DEMOSTRACIÓN)
        q = 0.12 * np.pi
        x = np.arange(n)
        # Simulación de solitón suave (Jacobi-Anger válido)
        theta = q * x + 0.5 * np.sin(2 * q * x) 
        base_demo = np.zeros((n, 3))
        base_demo[:, 1] = np.sin(theta) # Y
        base_demo[:, 2] = np.cos(theta) # Z
        return base_demo

    num_pasos = Spin_history.shape[0]
    base = Spin_history[num_pasos-1]
    result = []
    pattern_len = len(base)
    
    # NOTA: Para el análisis estructural, es mejor poner el ruido en 0 
    # para ver la "anatomía" limpia del solitón.
    noise_magnitude = 0.0 # <--- Cambiado a 0 para análisis limpio (antes era 1.0 implícito)
    noise_x = noise_magnitude * np.random.randn(n)
    noise_x -= noise_x.mean()
    
    for i in range(n):
        vec = base[i % pattern_len].copy()
        vec[0] += noise_x[i]
        vec /= np.linalg.norm(vec)
        result.append(vec)
    return np.array(result)

# ==============================================================================
# 2. RUTINA DE ANÁLISIS (AUTOPSIA DEL SOLITÓN)
# ==============================================================================

# Configuración
N_SITES = 1198 # O el tamaño que desees analizar
spins = cadena0spinhistory(N_SITES)

# A. Calcular Ángulos en el plano de rotación (Asumiendo rotación en YZ)
# Ajusta los índices si tu plano es XZ o XY. 
# Aquí asumo: Eje cadena=X (idx 0), Plano rotación=YZ (idx 1, 2)
S_y = spins[:, 1]
S_z = spins[:, 2]
theta = np.arctan2(S_y, S_z)

# B. Calcular diferencia angular entre vecinos (Delta Theta)
# Usamos unwrap para evitar saltos de 2pi al calcular la diferencia
theta_unwrapped = np.unwrap(theta)
delta_theta = np.diff(theta_unwrapped)

# C. Análisis Espectral (FFT) de la estructura estática
# Usamos Sz porque lleva la modulación cos(theta)
fft_vals = np.fft.fft(S_z - np.mean(S_z)) 
fft_mag = np.abs(fft_vals) / N_SITES # Normalizado
k_vals = np.fft.fftfreq(N_SITES) * 2 * np.pi # Eje K

# ==============================================================================
# 3. GRAFICACIÓN E INTERPRETACIÓN
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# --- GRÁFICO 1: Perfil en Espacio Real (Prueba de Suavidad) ---
ax1.plot(delta_theta, '.-', color='teal', linewidth=1, markersize=3)
ax1.set_title(r'Prueba 1: Perfil de Variación Angular ($\Delta \theta_n$)')
ax1.set_ylabel(r'$\theta_{n+1} - \theta_n$ [rad]')
ax1.set_xlabel('Sitio n')
ax1.grid(True, alpha=0.3)
# Zoom opcional para ver el detalle de unos pocos periodos
ax1.set_xlim(0, 200) 

# --- GRÁFICO 2: Perfil Espectral (Prueba de Decaimiento) ---
# Filtramos solo frecuencias positivas y excluimos el DC
mask = (k_vals > 0) & (k_vals < np.pi) 
ax2.plot(k_vals[mask], fft_mag[mask], '.-', color='crimson', linewidth=1)
ax2.set_yscale('log') # ESCALA LOGARÍTMICA CRÍTICA
ax2.set_title(r'Prueba 2: Espectro de la Estructura Estática ($S_z$)')
ax2.set_ylabel(r'Log Amplitud FFT $|S_z(k)|$')
ax2.set_xlabel(r'$k$')
ax2.grid(True, which="both", alpha=0.3)
ax2.set_ylim(bottom=1e-10) # Límite inferior para limpiar ruido numérico

plt.tight_layout()
plt.show()