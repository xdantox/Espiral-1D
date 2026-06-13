import numpy as np
import scipy.fft
import numexpr as ne  # IMPORTANTE: Librería para multithreading aritmético
import matplotlib.pyplot as plt
import gc
from matplotlib.lines import Line2D
import os

from Heff import dt
from cadena0 import cadena0spinhistory

# Forzar a NumExpr a detectar y usar TODOS los hilos lógicos del procesador
n_cores = ne.detect_number_of_cores()
ne.set_num_threads(n_cores)
print(f"NumExpr configurado para utilizar {n_cores} hilos de procesamiento.")

"""Espectro S(k, ν) usando el dímero como unidad (sin r_AB).

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

DIMER_CHANNEL = "ML"
LAB_COMPONENTS = (0, 1, 2)
APPLY_DEMEAN = True
APPLY_HANN = True
USE_RFFT_TIME = True
FREQ_MAX_THZ = 40.0  # THz (Aproximadamente equivalente a los antiguos 2.5e14 rad/s)
SKIP = 1
ANNOTATE_CELLS = False
Q_INCOMM_INPUT = 1.0332116073
SHOW_UMKLAPP_SATELLITES = False
UMKLAPP_MAX_ORDER = 6          
UMKLAPP_BASE_K = 0.0           
UMKLAPP_Q_INPUT = Q_INCOMM_INPUT         

# ===================================================================
# FUNCIONES DE APOYO (Notación Científica)
# ===================================================================
def format_sci_tex(value, decimals=2):
    """
    Convierte un número flotante de nomenclatura computacional (ej. 1.2e-04) 
    a nomenclatura científica para LaTeX (ej. 1.2 \times 10^{-4}).
    """
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coeff = value / (10**exponent)
    return fr"{coeff:.{decimals}f} \times 10^{{{exponent}}}"

# ===================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ===================================================================
print("Cargando datos en RAM...")
Spin_history = np.load('D_plane = 1.05D  fluc.npy')

num_pasos = Spin_history.shape[0]
n_spins = Spin_history.shape[1]
n_dimeros = n_spins // 2  

cadena0 = cadena0spinhistory(n_spins)

print("Reorganizando memoria física (C-Contiguous)...")
spin_A = np.ascontiguousarray(np.transpose(Spin_history[:, 0::2, :], (2, 0, 1)), dtype=np.float32)
spin_B = np.ascontiguousarray(np.transpose(Spin_history[:, 1::2, :], (2, 0, 1)), dtype=np.float32)

gs_A = np.ascontiguousarray(np.transpose(cadena0[0::2, :], (1, 0)), dtype=np.float32)
gs_B = np.ascontiguousarray(np.transpose(cadena0[1::2, :], (1, 0)), dtype=np.float32)

print(f"Datos procesados. Sistema de {n_dimeros} celdas (dímeros).")


def compute_power_lab_dimer_from_spin(
    spin_A_lab, spin_B_lab, gs_A_lab, gs_B_lab, n_dimeros,
    channel="M", components=(0, 1, 2), demean=True, hann=True, use_rfft_time=True,
):
    channel = channel.upper()
    if channel not in {"M", "L", "ML"}:
        raise ValueError("channel must be 'M', 'L', or 'ML'")

    num_pasos = spin_A_lab.shape[1] 
    window_t = None
    if hann:
        window_t = np.hanning(num_pasos).astype(np.float32)[:, np.newaxis]

    n_omega = (num_pasos // 2 + 1) if use_rfft_time else num_pasos
    power = np.zeros((n_omega, n_dimeros), dtype=np.float64)

    def _accumulate_for_sign(sign_val):
        nonlocal power
        for comp in components:
            a = spin_A_lab[comp, :, :]
            b = spin_B_lab[comp, :, :]
            gs_a = gs_A_lab[comp, np.newaxis, :]
            gs_b = gs_B_lab[comp, np.newaxis, :]

            x = ne.evaluate("0.5 * ((a - gs_a) + sign_val * (b - gs_b))")

            if demean:
                x_mean = np.mean(x, axis=0, keepdims=True)
                ne.evaluate("x - x_mean", out=x)

            if window_t is not None:
                ne.evaluate("x * window_t", out=x)

            if use_rfft_time:
                fft_x = scipy.fft.rfft(x, axis=0, workers=-1)
                fft_x = scipy.fft.fft(fft_x, axis=1, workers=-1)
                fft_x = np.fft.fftshift(fft_x, axes=(1,))
            else:
                fft_x = scipy.fft.fftn(x, axes=(0, 1), workers=-1)
                fft_x = np.fft.fftshift(fft_x, axes=(0, 1))

            fft_real = fft_x.real
            fft_imag = fft_x.imag
            
            power += ne.evaluate("fft_real**2 + fft_imag**2")
            
            del a, b, gs_a, gs_b, x, fft_x, fft_real, fft_imag

    if channel in {"M", "ML"}:
        _accumulate_for_sign(+1)
    if channel in {"L", "ML"}:
        _accumulate_for_sign(-1)

    return power

def estimate_q_incomm_from_power(k_vals, power_kw):
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
    width = kmax - kmin
    if width <= 0:
        raise ValueError("Invalid Brillouin zone limits")
    kw = ((k - kmin) % width) + kmin
    if np.isclose(kw, kmin + width):
        kw = kmax
    return float(kw)

def build_umklapp_positions(k0, q, max_order, kmin=-np.pi / 2, kmax=np.pi / 2):
    if max_order < 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    raw_vals = []
    orders = []
    for m in range(-max_order, max_order + 1):
        raw_vals.append(wrap_to_bz(k0 + m * q, kmin=kmin, kmax=kmax))
        orders.append(m)

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
    spin_A, spin_B, gs_A, gs_B, n_dimeros,
    channel=DIMER_CHANNEL, components=LAB_COMPONENTS,
    demean=APPLY_DEMEAN, hann=APPLY_HANN, use_rfft_time=USE_RFFT_TIME,
)

del Spin_history, spin_A, spin_B, gs_A, gs_B, cadena0
gc.collect()

log_mag = np.log10(total_power + 1e-12)

# ===================================================================
# 4. DEFINICIÓN DE EJES Y PLOT (AHORA EN THz)
# ===================================================================

k_values = np.fft.fftshift(np.fft.fftfreq(n_dimeros, d=1.0) * np.pi)

# CAMBIO 1: Cálculo de frecuencias en THz
if USE_RFFT_TIME:
    # np.fft.rfftfreq da el resultado en Hz (1/s). Dividimos por 1e12 para THz.
    freq_values = np.fft.rfftfreq(num_pasos, d=dt) / 1e12
else:
    freq_values = np.fft.fftshift(np.fft.fftfreq(num_pasos, d=dt)) / 1e12

# --- FILTRADO Y REDUCCIÓN PARA PLOTEO ---
f_max = FREQ_MAX_THZ
mask_w = (freq_values <= f_max) if USE_RFFT_TIME else (np.abs(freq_values) <= f_max)

freq_plot = freq_values[mask_w]
log_mag_plot = log_mag[mask_w, :] 

skip = SKIP
K_grid, F_grid = np.meshgrid(k_values[::skip], freq_plot[::skip])
Z_grid = log_mag_plot[::skip, ::skip]

power_k = np.sum(total_power[mask_w, :], axis=0)
q_incomm = float(Q_INCOMM_INPUT) if Q_INCOMM_INPUT is not None else estimate_q_incomm_from_power(k_values, power_k)
if q_incomm is None:
    print("No se pudo estimar q_incomm automáticamente. Define Q_INCOMM_INPUT manualmente.")
else:
    src = "input" if Q_INCOMM_INPUT is not None else "estimado"
    print(f"q_incomm ({src}) = {q_incomm:.6f} rad")

vmax = float(np.percentile(Z_grid, 99.4))
Z_grid = Z_grid - vmax
vmin = float(np.percentile(Z_grid, 4))
vmax = 0.0

# ====================================================================
# AJUSTES DE PRESENTACIÓN (LETRAS GRANDES)
# ====================================================================
TITLE_SIZE = 22
LABEL_SIZE = 18
TICK_SIZE = 16
TEXT_SIZE = 16
CBAR_LABEL_SIZE = 18

print("Generando gráfico...")
plt.figure(figsize=(12, 8))

mesh = plt.pcolormesh(K_grid, F_grid, Z_grid, 
                      cmap='plasma', 
                      vmin=vmin, vmax=vmax, 
                      shading='nearest')

cbar = plt.colorbar(mesh)
# CAMBIO 2: Actualización de la etiqueta S(k, w) a S(k, ν)
cbar.set_label(r'$\log_{10} S(k, \nu)$ (shifted)', fontsize=CBAR_LABEL_SIZE)
cbar.ax.tick_params(labelsize=TICK_SIZE)

plt.xlabel(r'$k$ (Dimer BZ, $d=2a$) $[-\pi/2, \pi/2]$', fontsize=LABEL_SIZE)
plt.ylabel(r'Frequency $\nu$ [THz]', fontsize=LABEL_SIZE) # CAMBIO 3: Etiqueta Eje Y
plt.title(fr'Magnetic band structure of the dimer. {DIMER_CHANNEL} Channel', fontsize=TITLE_SIZE, pad=15)

plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

plt.ylim(0, f_max)
plt.xlim(-np.pi/2, np.pi/2)

if SHOW_UMKLAPP_SATELLITES:
    q_umklapp = float(UMKLAPP_Q_INPUT) if UMKLAPP_Q_INPUT is not None else q_incomm
    if q_umklapp is not None:
        ax = plt.gca()
        sat_k, sat_m = build_umklapp_positions(
            UMKLAPP_BASE_K, q_umklapp, UMKLAPP_MAX_ORDER, kmin=-np.pi/2, kmax=np.pi/2,
        )
        if sat_k.size > 0:
            ax.vlines(sat_k, 0.0, f_max, colors="red", alpha=0.98, linewidth=2.2, zorder=2)
            ax.set_xticks(sat_k, minor=True)
            ax.set_xticklabels([f"{int(m)}" for m in sat_m], minor=True, rotation=0, fontsize=TICK_SIZE - 2)
            ax.tick_params(axis="x", which="minor", colors="red", labelsize=TICK_SIZE - 2, pad=14, length=5, width=1.2)
            umklapp_proxy = Line2D([0], [0], color="red", lw=2.2,
                label=fr"Umklapp grid: integer labels are order $n$ in $k_n=k_0+n q_{{inc}}$ (folded to BZ, $q_{{inc}}={q_umklapp:.4f}$)")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(umklapp_proxy)
            labels.append(umklapp_proxy.get_label())
            ax.legend(handles, labels, loc="upper right", fontsize=TEXT_SIZE - 2, framealpha=0.85)

if ANNOTATE_CELLS:
    ax = plt.gca()
    # CAMBIO 4: Reescalamos los topes de anotaciones usando el nuevo f_max en THz
    y_base = min(32.0, 0.84 * f_max) # 32 THz es aprox el equivalente a los antiguos 2.0e14 rad/s
    y_crys = min(y_base + 0.10 * f_max, 0.95 * f_max)
    y_mag = y_base
    y_pbc = max(y_base - 0.10 * f_max, 0.08 * f_max)

    ax.annotate("", xy=(-np.pi/2, y_crys), xytext=(np.pi/2, y_crys), arrowprops=dict(arrowstyle="<->", color="white", lw=1.4))
    ax.text(0.0, y_crys - 0.035 * f_max, r"Crystallographic Cell: (dimer, $d=2a$): $k\in[-\pi/2,\pi/2]$",
            color="white", ha="center", va="top", fontsize=TEXT_SIZE - 2, bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="white"))

    if q_incomm is not None:
        ax.axvline(+q_incomm, color="cyan", lw=1.5, ls="--", alpha=0.95)
        ax.axvline(-q_incomm, color="cyan", lw=1.5, ls="--", alpha=0.95)
        ax.annotate("", xy=(-q_incomm, y_mag), xytext=(q_incomm, y_mag), arrowprops=dict(arrowstyle="<->", color="cyan", lw=1.6))
        ax.text(0.0, y_mag - 0.035 * f_max, rf"Magnetic Cell: $k_{{mag}}\in[-q_{{inc}},q_{{inc}}],\ q_{{inc}}={q_incomm:.4f}$",
                color="cyan", ha="center", va="top", fontsize=TEXT_SIZE - 2, bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="cyan"))

    if k_values.size > 1:
        dk_pbc = float(np.median(np.diff(np.sort(k_values))))
        k_left = -0.5 * dk_pbc
        k_right = 0.5 * dk_pbc
        y_arrow = y_pbc
        ax.vlines([k_left, k_right], 0.0, f_max, colors="red", linewidth=1.4, alpha=0.95)
        ax.annotate("", xy=(k_left, y_arrow), xytext=(k_right, y_arrow), arrowprops=dict(arrowstyle="<->", color="red", lw=1.4))
        
        # CAMBIO 5: Aplicamos el formateador de notación científica LaTeX para dk_pbc
        ax.text(0.1 * dk_pbc, y_arrow - 0.035 * f_max, rf"PBC Cell: $\Delta k_{{PBC}}={format_sci_tex(dk_pbc)}$",
                color="red", ha="center", va="top", fontsize=TEXT_SIZE - 2, bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="red"))

plt.tight_layout()
plt.show()