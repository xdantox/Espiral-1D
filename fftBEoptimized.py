import numpy as np
import matplotlib.pyplot as plt
import gc
from matplotlib.lines import Line2D

from Heff import dt
from cadena0 import cadena0spinhistory

"""Espectro S(k,ω) usando el dímero como unidad (sin r_AB).

Si no conoces la geometría (distancias/posiciones reales) del dímero, no puedes fijar
la fase intradímero exp(-i k r_AB). Este script evita ese problema construyendo un
observable por dímero y haciendo FFT en el índice de dímero.

Canales del dímero:
    - M = (A + B)/2  (modo "bonding" / momento total del dímero)
    - L = (A - B)/2  (modo "antibonding" / estaggered intradímero)

El k que sale está en la BZ del dímero con d=2a: k ∈ [-π/2, π/2].
"""

# ===================================================================
# 0. CONFIGURACIÓN MÍNIMA
# ===================================================================

# Canal del dímero a analizar: "M", "L" o "ML" (suma de potencias de ambos)
DIMER_CHANNEL = "M"

# Componentes cartesianas incluidas en la potencia (0=x, 1=y, 2=z)
LAB_COMPONENTS = (0, 1, 2)

# Pre-procesado temporal
APPLY_DEMEAN = True
APPLY_HANN = True

# Si True: rFFT en tiempo (ω>=0) para ahorrar memoria
USE_RFFT_TIME = True

# Parámetros de ploteo
OMEGA_MAX = 2.5e14  # rad/s
SKIP = 1

# Anotaciones de celdas en el plot FFT
ANNOTATE_CELLS = False
# Si se deja en None, se estima q_inc automáticamente desde el espectro integrado en omega.
Q_INCOMM_INPUT = 2.1083810463

# Marcado de satélites umklapp en el eje k: k + mQ
SHOW_UMKLAPP_SATELLITES = False
UMKLAPP_MAX_ORDER = 6          # m en [-M, M]
UMKLAPP_BASE_K = 0.0           # k de referencia en k + mQ
UMKLAPP_Q_INPUT = Q_INCOMM_INPUT         # Si None, usa q_incomm

# ===================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ===================================================================
print("Cargando datos...")
# Usamos mmap_mode='r' para no cargar todo el archivo en RAM de golpe si es gigante
Spin_history = np.load('D_plane = 0 fluc.npy', mmap_mode='r')

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


def estimate_q_incomm_from_power(k_vals, power_kw):
    """Estimación simple de q incommensurado desde S(k,omega) integrado en omega."""
    if k_vals.size < 3:
        return None

    dk = float(np.median(np.diff(np.sort(k_vals))))
    mask = np.abs(k_vals) > (1.5 * abs(dk))
    if not np.any(mask):
        return None

    idx_local = int(np.argmax(power_kw[mask]))
    k_peak = float(np.abs(k_vals[mask][idx_local]))
    return k_peak


def wrap_to_bz(k, kmin=-np.pi / 2, kmax=np.pi / 2):
    """Envuelve k periódicamente al intervalo [kmin, kmax]."""
    width = kmax - kmin
    if width <= 0:
        raise ValueError("Invalid Brillouin zone limits")
    kw = ((k - kmin) % width) + kmin
    if np.isclose(kw, kmin + width):
        kw = kmax
    return float(kw)


def build_umklapp_positions(k0, q, max_order, kmin=-np.pi / 2, kmax=np.pi / 2):
    """Genera posiciones únicas de k+mQ envueltas a la BZ."""
    if max_order < 0:
        return np.array([], dtype=float), np.array([], dtype=int)

    raw_vals = []
    orders = []
    for m in range(-max_order, max_order + 1):
        raw_vals.append(wrap_to_bz(k0 + m * q, kmin=kmin, kmax=kmax))
        orders.append(m)

    # Deduplicación numérica por redondeo para no repetir líneas colapsadas por wrapping.
    pos_to_order = {}
    for pos, m in zip(raw_vals, orders):
        key = round(float(pos), 10)
        if key not in pos_to_order or abs(m) < abs(pos_to_order[key]):
            pos_to_order[key] = m

    sorted_items = sorted(pos_to_order.items(), key=lambda t: t[0])
    positions = np.array([item[0] for item in sorted_items], dtype=float)
    m_orders = np.array([item[1] for item in sorted_items], dtype=int)
    return positions, m_orders


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

# Eje K: Zona de Brillouin del dímero con d=2a -> [-pi/2, pi/2]
k_values = np.fft.fftshift(np.fft.fftfreq(n_dimeros, d=1.0) * np.pi)

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

# Espectro integrado en omega para estimar q_inc si no se entrega input.
power_k = np.sum(total_power[mask_w, :], axis=0)
q_incomm = float(Q_INCOMM_INPUT) if Q_INCOMM_INPUT is not None else estimate_q_incomm_from_power(k_values, power_k)
if q_incomm is None:
    print("No se pudo estimar q_incomm automáticamente. Define Q_INCOMM_INPUT manualmente.")
else:
    src = "input" if Q_INCOMM_INPUT is not None else "estimado"
    print(f"q_incomm ({src}) = {q_incomm:.6f} rad")

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
plt.xlabel(r'$k$ (Dimer BZ, $d=2a$) $[-\pi/2, \pi/2]$')
plt.ylabel(r'$\omega$ [rad/s]')
plt.title(fr'Magnetic band structure of the dimer. {DIMER_CHANNEL} Channel')

# Ajustar límites visuales
plt.ylim(0, omega_max)
plt.xlim(-np.pi/2, np.pi/2)

# Satélites umklapp: líneas verticales rojas en k + mQ (replegado en la BZ)
if SHOW_UMKLAPP_SATELLITES:
    q_umklapp = float(UMKLAPP_Q_INPUT) if UMKLAPP_Q_INPUT is not None else q_incomm
    if q_umklapp is None:
        print("No se dibujan satélites umklapp: define UMKLAPP_Q_INPUT o q_incomm.")
    else:
        ax = plt.gca()
        sat_k, sat_m = build_umklapp_positions(
            UMKLAPP_BASE_K,
            q_umklapp,
            UMKLAPP_MAX_ORDER,
            kmin=-np.pi/2,
            kmax=np.pi/2,
        )
        if sat_k.size > 0:
            # Mismo estilo de "discrete k grid" en direct3_PBC: líneas + ticks menores etiquetados.
            ax.vlines(sat_k, 0.0, omega_max, colors="red", alpha=0.98, linewidth=2.2, zorder=2)
            ax.set_xticks(sat_k, minor=True)
            ax.set_xticklabels([f"{int(m)}" for m in sat_m], minor=True, rotation=0)

            major_size = plt.rcParams.get("xtick.labelsize", 10)
            if major_size == "medium":
                major_size = 10

            ax.tick_params(
                axis="x",
                which="minor",
                colors="red",
                labelsize=major_size,
                pad=14,
                length=5,
                width=1.2,
            )

            umklapp_proxy = Line2D(
                [0],
                [0],
                color="red",
                lw=2.2,
                label=fr"Umklapp grid: integer labels are order $n$ in $k_n=k_0+n q_{{inc}}$ (folded to BZ, $q_{{inc}}={q_umklapp:.4f}$)",            )
            handles, labels = ax.get_legend_handles_labels()
            handles.append(umklapp_proxy)
            labels.append(umklapp_proxy.get_label())
            ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.85)

if ANNOTATE_CELLS:
    ax = plt.gca()
    y_base = min(2.0e14, 0.84 * omega_max)
    y_crys = min(y_base + 0.10 * omega_max, 0.95 * omega_max)
    y_mag = y_base
    y_pbc = max(y_base - 0.10 * omega_max, 0.08 * omega_max)

    # Celda cristalográfica del dímero en k (d=2a): ancho completo de -pi/2 a pi/2.
    ax.annotate(
        "",
        xy=(-np.pi/2, y_crys),
        xytext=(np.pi/2, y_crys),
        arrowprops=dict(arrowstyle="<->", color="white", lw=1.4),
    )
    ax.text(
        0.0,
        y_crys - 0.035 * omega_max,
        r"Cristalographic Cell: (dimer, $d=2a$): $k\in[-\pi/2,\pi/2]$",
        color="white",
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="white"),
    )

    # 2) Celda magnética: modo incommensurado en ±q_inc.
    if q_incomm is not None:
        ax.axvline(+q_incomm, color="cyan", lw=1.5, ls="--", alpha=0.95)
        ax.axvline(-q_incomm, color="cyan", lw=1.5, ls="--", alpha=0.95)

        ax.annotate(
            "",
            xy=(-q_incomm, y_mag),
            xytext=(q_incomm, y_mag),
            arrowprops=dict(arrowstyle="<->", color="cyan", lw=1.6),
        )
        ax.text(
            0.0,
            y_mag - 0.035 * omega_max,
            rf"Magnetic Cell: $k_{{mag}}\in[-q_{{inc}},q_{{inc}}],\ q_{{inc}}={q_incomm:.4f}$",
            color="cyan",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="cyan"),
        )

    # 3) Celda PBC real: separación discreta Δk ~ 1/N (aquí en unidades del eje k usado).
    if k_values.size > 1:
        dk_pbc = float(np.median(np.diff(np.sort(k_values))))
        k_left = -0.5 * dk_pbc
        k_right = 0.5 * dk_pbc
        y_arrow = y_pbc

        ax.vlines([k_left, k_right], 0.0, omega_max, colors="red", linewidth=1.4, alpha=0.95)
        ax.annotate(
            "",
            xy=(k_left, y_arrow),
            xytext=(k_right, y_arrow),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.4),
        )
        ax.text(
            0.1 * dk_pbc,
            y_arrow - 0.035 * omega_max,
            rf"PBC Cell: $\Delta k_{{\mathrm{{PBC}}}}={dk_pbc:.2e}$",
            color="red",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="red"),
        )

plt.tight_layout()
plt.show()