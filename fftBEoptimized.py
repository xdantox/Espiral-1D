import numpy as np
import matplotlib.pyplot as plt
import gc  # Gestión de memoria
from Heff import dt
from cadena0 import cadena0spinhistory

# ===================================================================
# 0. OPCIONES DE CÁLCULO
# ===================================================================
# "local_incoherent": tu cálculo en marco local (dx,dy) con suma de potencias A+B
# "lab_coherent": suma coherente A+B en el marco del laboratorio (recomendado para espirales)
# "rotated_complex_coherent": marco local usando ψ = dx + i dy, con suma coherente A/B (útil si el GS es espiral plana)
METHOD = "lab_coherent"

# Posición relativa del sitio B dentro de la celda del dímero (unidades de celda dimerizada)
R_AB = 0.5

# Componentes del laboratorio a incluir en la potencia (0=x, 1=y, 2=z)
LAB_COMPONENTS = (0, 1, 2)

# Pre-procesado temporal
APPLY_DEMEAN = True
APPLY_HANN = True

# Convención de fase para el sitio B (depende de la convención de Fourier):
# usamos exp(-i k r_AB) para ser consistente con la FFT de NumPy.
PHASE_SIGN = -1

# ===================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ===================================================================
print("Cargando datos...")
# Usamos mmap_mode='r' para no cargar todo el archivo en RAM de golpe si es gigante
Spin_history = np.load('spin_history.npy')

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

# Calcular fluctuaciones respecto al estado base (Broadcasting)
# delta(t, celda) = S(t, celda) - S_0(celda)
delta_A = spin_A - gs_A[np.newaxis, :, :]
delta_B = spin_B - gs_B[np.newaxis, :, :]

print(f"Datos procesados. Sistema de {n_dimeros} celdas (dímeros).")


def compute_power_lab_coherent(delta_A_lab, delta_B_lab, n_dimeros, r_AB=0.5, components=(0, 1, 2), demean=True, hann=True):
    """S(k,ω) con suma coherente A+B en el marco del laboratorio.

    Implementa (convención FFT de NumPy):
        F_total(k,ω) = F_A(k,ω) + exp(-i k r_AB) F_B(k,ω)
        S = Σ_{α in components} |F_total^α|^2

    Nota: esto es la opción físicamente más limpia para comparar con experimento.
    """
    if demean:
        delta_A_lab = delta_A_lab - np.mean(delta_A_lab, axis=0, keepdims=True)
        delta_B_lab = delta_B_lab - np.mean(delta_B_lab, axis=0, keepdims=True)

    if hann:
        window_t = np.hanning(delta_A_lab.shape[0])[:, np.newaxis, np.newaxis]
        delta_A_lab = delta_A_lab * window_t
        delta_B_lab = delta_B_lab * window_t

    fft_A = np.fft.fftshift(np.fft.fftn(delta_A_lab, axes=(0, 1)), axes=(0, 1))
    fft_B = np.fft.fftshift(np.fft.fftn(delta_B_lab, axes=(0, 1)), axes=(0, 1))

    k_vals = np.fft.fftshift(np.fft.fftfreq(n_dimeros, d=1.0) * 2 * np.pi)
    phase = np.exp(-1j * k_vals[np.newaxis, :, np.newaxis] * r_AB)

    fft_tot = fft_A + fft_B * phase
    comp_idx = list(components)
    return np.sum(np.abs(fft_tot[:, :, comp_idx]) ** 2, axis=2)

# ===================================================================
# 2. DEFINICIÓN DEL SISTEMA DE REFERENCIA LOCAL (Vectorizado)
# ===================================================================

def get_transverse_projections(delta_fluctuations, ground_state_spins):
    """
    Proyecta las fluctuaciones sobre los ejes locales perpendiculares al espín estático.
    """
    # 1. Eje Z local = Dirección del espín en reposo
    norm_z = np.linalg.norm(ground_state_spins, axis=1, keepdims=True)
    z_loc = ground_state_spins / norm_z
    
    # 2. Ejes X e Y locales (Gram-Schmidt)
    # Usamos un vector de referencia arbitrario. [1,1,1] suele funcionar bien 
    # para espirales en el plano YZ.
    ref_vec = np.array([1., 0., 0.])
    
    # Proyección de ref sobre z_loc
    dot = np.sum(z_loc * ref_vec, axis=1, keepdims=True) * z_loc
    x_loc_raw = ref_vec - dot
    
    # Normalizar x_loc
    norm_x = np.linalg.norm(x_loc_raw, axis=1, keepdims=True)
    # Evitar división por cero si z_loc || ref_vec (muy improbable aquí)
    norm_x[norm_x < 1e-9] = 1.0 
    x_loc = x_loc_raw / norm_x
    
    # y_loc es el producto cruz
    y_loc = np.cross(z_loc, x_loc)
    
    # 3. Proyectar fluctuaciones (Einsum: Time, Cell, Vector -> Time, Cell)
    # 'ijk,jk->ij' significa: dot product a lo largo del eje k (x,y,z) para cada t y celda
    proj_x = np.einsum('ijk,jk->ij', delta_fluctuations, x_loc)
    proj_y = np.einsum('ijk,jk->ij', delta_fluctuations, y_loc)
    
    return proj_x, proj_y

# ===================================================================
# 3. TRANSFORMADA DE FOURIER Y ESPECTRO DE POTENCIA
# ===================================================================
print("Calculando FFT...")

def compute_power_local_periodogram(dx, dy, demean=True, hann=True):
    """Periodograma en el marco local: S ~ |FFT(dx)|^2 + |FFT(dy)|^2."""
    if demean:
        dx = dx - np.mean(dx, axis=0, keepdims=True)
        dy = dy - np.mean(dy, axis=0, keepdims=True)

    if hann:
        window_t = np.hanning(dx.shape[0])[:, np.newaxis]
        dx = dx * window_t
        dy = dy * window_t

    fft_x = np.fft.fftshift(np.fft.fftn(dx, axes=(0, 1)), axes=(0, 1))
    fft_y = np.fft.fftshift(np.fft.fftn(dy, axes=(0, 1)), axes=(0, 1))
    return np.abs(fft_x) ** 2 + np.abs(fft_y) ** 2


def compute_power_rotated_complex_coherent(dx_A, dy_A, dx_B, dy_B, n_dimeros, r_AB=0.5, demean=True, hann=True, phase_sign=-1):
    """Espectro en marco local usando variable compleja ψ = dx + i dy y suma coherente A/B.

    Nota: esto suele funcionar bien cuando el GS es (aprox) una espiral plana y el gauge
    de (x_loc, y_loc) queda fijado de forma suave (p.ej. ref_vec=[1,0,0]).
    """
    psi_A = dx_A + 1j * dy_A
    psi_B = dx_B + 1j * dy_B

    if demean:
        psi_A = psi_A - np.mean(psi_A, axis=0, keepdims=True)
        psi_B = psi_B - np.mean(psi_B, axis=0, keepdims=True)

    if hann:
        window_t = np.hanning(psi_A.shape[0])[:, np.newaxis]
        psi_A = psi_A * window_t
        psi_B = psi_B * window_t

    fft_A = np.fft.fftshift(np.fft.fftn(psi_A, axes=(0, 1)), axes=(0, 1))
    fft_B = np.fft.fftshift(np.fft.fftn(psi_B, axes=(0, 1)), axes=(0, 1))

    k_vals = np.fft.fftshift(np.fft.fftfreq(n_dimeros, d=1.0) * 2 * np.pi)
    phase = np.exp(phase_sign * 1j * k_vals[np.newaxis, :] * r_AB)

    fft_tot = fft_A + fft_B * phase
    return np.abs(fft_tot) ** 2


if METHOD == "lab_coherent":
    # Ya no necesitamos el historial completo, solo delta_A/delta_B
    del Spin_history, spin_A, spin_B
    gc.collect()

    total_power = compute_power_lab_coherent(
        delta_A,
        delta_B,
        n_dimeros,
        r_AB=R_AB,
        components=LAB_COMPONENTS,
        demean=APPLY_DEMEAN,
        hann=APPLY_HANN,
    )

    del delta_A, delta_B, gs_A, gs_B, cadena0
    gc.collect()

elif METHOD == "rotated_complex_coherent":
    print("Calculando proyecciones locales para subred A...")
    dx_A, dy_A = get_transverse_projections(delta_A, gs_A)

    print("Calculando proyecciones locales para subred B...")
    dx_B, dy_B = get_transverse_projections(delta_B, gs_B)

    del Spin_history, spin_A, spin_B, delta_A, delta_B
    gc.collect()

    total_power = compute_power_rotated_complex_coherent(
        dx_A,
        dy_A,
        dx_B,
        dy_B,
        n_dimeros,
        r_AB=R_AB,
        demean=APPLY_DEMEAN,
        hann=APPLY_HANN,
        phase_sign=PHASE_SIGN,
    )

    del dx_A, dy_A, dx_B, dy_B, gs_A, gs_B, cadena0
    gc.collect()

else:
    print("Calculando proyecciones locales para subred A...")
    dx_A, dy_A = get_transverse_projections(delta_A, gs_A)

    print("Calculando proyecciones locales para subred B...")
    dx_B, dy_B = get_transverse_projections(delta_B, gs_B)

    del Spin_history, spin_A, spin_B, gs_A, gs_B, delta_A, delta_B, cadena0
    gc.collect()

    power_A = compute_power_local_periodogram(dx_A, dy_A, demean=APPLY_DEMEAN, hann=APPLY_HANN)
    del dx_A, dy_A
    gc.collect()

    power_B = compute_power_local_periodogram(dx_B, dy_B, demean=APPLY_DEMEAN, hann=APPLY_HANN)
    del dx_B, dy_B
    gc.collect()

    total_power = power_A + power_B
    del power_A, power_B
    gc.collect()

# Escala logarítmica para visualización
log_mag = np.log10(total_power + 1e-12)

# ===================================================================
# 4. DEFINICIÓN DE EJES Y PLOT
# ===================================================================

# Eje K: Zona de Brillouin reducida [-pi, pi] (unidades inversas de celda dimerizada)
k_values = np.fft.fftshift(np.fft.fftfreq(n_dimeros, d=1.0) * 2 * np.pi)

# Eje Omega: Frecuencia
omega_values = np.fft.fftshift(np.fft.fftfreq(num_pasos, d=dt) * 2 * np.pi)

# --- FILTRADO Y REDUCCIÓN PARA PLOTEO ---
omega_max = 5e14 # Ajusta según tu sistema (Rad/s)
mask_w = np.abs(omega_values) <= omega_max

# Recortamos arrays
omega_plot = omega_values[mask_w]
log_mag_plot = log_mag[mask_w, :] # Recortamos en eje de frecuencia

# Downsampling para que el plot sea ligero (opcional, skip=1 toma todos)
skip = 1
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
plt.title(r'Dispersión de Ondas de Espín (Dímero Unit Cell)')

# Ajustar límites visuales
plt.ylim(0, omega_max)
plt.xlim(-np.pi, np.pi)

plt.tight_layout()
plt.show()