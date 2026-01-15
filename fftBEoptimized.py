import numpy as np
import matplotlib.pyplot as plt
import gc

from Heff import dt
from cadena0 import cadena0spinhistory

"""Espectro S(k,ω) usando el dímero como unidad (sin r_AB).

Si no conoces la geometría (distancias/posiciones reales) del dímero, no puedes fijar
la fase intradímero exp(-i k r_AB). Este script evita ese problema construyendo un
observable por dímero y haciendo FFT en el índice de dímero.

Canales del dímero:
    - M = (A + B)/2  (modo "bonding" / momento total del dímero)
    - L = (A - B)/2  (modo "antibonding" / estaggered intradímero)

El k que sale está en la BZ reducida del dímero: k ∈ [-π, π].
"""

# ===================================================================
# 0. CONFIGURACIÓN MÍNIMA
# ===================================================================

# Canal del dímero a analizar: "M", "L" o "ML" (suma de potencias de ambos)
DIMER_CHANNEL = "M"

# Componentes cartesianas incluidas en la potencia (0=x, 1=y, 2=z)
LAB_COMPONENTS = (0, 1, 2)

# Pre-procesado temporal
APPLY_DEMEAN = False
APPLY_HANN = True

# Si True: rFFT en tiempo (ω>=0) para ahorrar memoria
USE_RFFT_TIME = True

# Parámetros de ploteo
OMEGA_MAX = 5e14  # rad/s
SKIP = 1

# ===================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ===================================================================
print("Cargando datos...")
# Usamos mmap_mode='r' para no cargar todo el archivo en RAM de golpe si es gigante
Spin_history = np.load('spin_history.npy', mmap_mode='r')

num_pasos = Spin_history.shape[0]
n_spins = Spin_history.shape[1]
n_dimeros = n_spins // 2  # Número de celdas unidad (dímeros)

# Cargar configuración relajada (Ground State)
cadena0 = cadena0spinhistory(n_spins)

# Separar en subredes A (pares) y B (impares)
# Esto define nuestra celda unidad: Sitio A + Sitio B
spin_A = Spin_history[:, 0::2, :]
spin_B = Spin_history[:, 1::2, :]

gs_A = cadena0[0::2, :]
gs_B = cadena0[1::2, :]

print(f"Datos procesados. Sistema de {n_dimeros} celdas (dímeros).")


def compute_power_lab_dimer_from_spin(
    spin_A_lab,
    spin_B_lab,
    gs_A_lab,
    gs_B_lab,
    n_dimeros,
    channel="M",
    components=(0, 1, 2),
    demean=True,
    hann=True,
    use_rfft_time=True,
):
    """S(k,ω) del dímero como objeto (sin fase geométrica intradímero).

    Construye variables de celda (por dímero):
      M = (A + B)/2  y/o  L = (A - B)/2
    y hace FFT en tiempo y en el índice de dímero n.

    Esto define un k conjugado al índice de dímero (BZ reducida). Es correcto como
    observable de 'dímeros' y evita depender de r_AB. No reproduce la intensidad
    experimental a k absoluto si las posiciones reales no están definidas.
    """
    channel = channel.upper()
    if channel not in {"M", "L", "ML"}:
        raise ValueError("channel must be 'M', 'L', or 'ML'")

    num_pasos = spin_A_lab.shape[0]
    window_t = None
    if hann:
        window_t = np.hanning(num_pasos).astype(np.float32)[:, np.newaxis]

    n_omega = (num_pasos // 2 + 1) if use_rfft_time else num_pasos
    power = np.zeros((n_omega, n_dimeros), dtype=np.float64)

    def _accumulate_for_sign(sign):
        # sign=+1 for M, sign=-1 for L (since L ~ A - B)
        nonlocal power
        for comp in components:
            a = np.array(spin_A_lab[:, :, comp], dtype=np.float32, copy=True)
            b = np.array(spin_B_lab[:, :, comp], dtype=np.float32, copy=True)

            a -= gs_A_lab[np.newaxis, :, comp].astype(np.float32, copy=False)
            b -= gs_B_lab[np.newaxis, :, comp].astype(np.float32, copy=False)

            x = 0.5 * (a + sign * b)

            if demean:
                x -= np.mean(x, axis=0, keepdims=True)

            if window_t is not None:
                x *= window_t

            if use_rfft_time:
                fft_x = np.fft.rfft(x, axis=0)
                fft_x = np.fft.fft(fft_x, axis=1)
                fft_x = np.fft.fftshift(fft_x, axes=(1,))
            else:
                fft_x = np.fft.fftshift(np.fft.fftn(x, axes=(0, 1)), axes=(0, 1))

            power += (fft_x.real * fft_x.real + fft_x.imag * fft_x.imag)
            del a, b, x, fft_x
            gc.collect()

    if channel in {"M", "ML"}:
        _accumulate_for_sign(+1)
    if channel in {"L", "ML"}:
        _accumulate_for_sign(-1)

    return power
print("Calculando FFT (dímero)...")

total_power = compute_power_lab_dimer_from_spin(
    spin_A,
    spin_B,
    gs_A,
    gs_B,
    n_dimeros,
    channel=DIMER_CHANNEL,
    components=LAB_COMPONENTS,
    demean=APPLY_DEMEAN,
    hann=APPLY_HANN,
    use_rfft_time=USE_RFFT_TIME,
)

del Spin_history, spin_A, spin_B, gs_A, gs_B, cadena0
gc.collect()

# Escala logarítmica para visualización
log_mag = np.log10(total_power + 1e-12)
#log_mag = total_power.copy()
# ===================================================================
# 4. DEFINICIÓN DE EJES Y PLOT
# ===================================================================

# Eje K: Zona de Brillouin reducida [-pi, pi] (unidades inversas de celda dimerizada)
k_values = np.fft.fftshift(np.fft.fftfreq(n_dimeros, d=1.0) * 2 * np.pi)

# Eje Omega: Frecuencia
if USE_RFFT_TIME:
    omega_values = np.fft.rfftfreq(num_pasos, d=dt) * 2 * np.pi
else:
    omega_values = np.fft.fftshift(np.fft.fftfreq(num_pasos, d=dt) * 2 * np.pi)

# --- FILTRADO Y REDUCCIÓN PARA PLOTEO ---
omega_max = OMEGA_MAX
mask_w = (omega_values <= omega_max) if USE_RFFT_TIME else (np.abs(omega_values) <= omega_max)

# Recortamos arrays
omega_plot = omega_values[mask_w]
log_mag_plot = log_mag[mask_w, :] # Recortamos en eje de frecuencia

# Downsampling para que el plot sea ligero (opcional, skip=1 toma todos)
skip = SKIP
K_grid, W_grid = np.meshgrid(k_values[::skip], omega_plot[::skip])
Z_grid = log_mag_plot[::skip, ::skip]

# Rango dinámico de colores (percentiles para evitar ruido de fondo)
vmin = float(np.percentile(Z_grid, 4))
vmax = float(np.percentile(Z_grid, 99.4))

print("Generando gráfico...")
plt.figure(figsize=(10, 7))
mesh = plt.pcolormesh(K_grid, W_grid, Z_grid, 
                      cmap='plasma', 
                      vmin=vmin, vmax=vmax, 
                      shading='nearest')

plt.colorbar(mesh, label=r'$\log_{10} S(k, \omega)$')

# Decoración
plt.xlabel(r'$k$ (Reduced BZ) $[-\pi, \pi]$')
plt.ylabel(r'$\omega$ [rad/s]')
plt.title(fr'Dispersión (Dímero). Canal {DIMER_CHANNEL}')

# Ajustar límites visuales
plt.ylim(0, omega_max)
plt.xlim(-np.pi, np.pi)

plt.tight_layout()
plt.show()