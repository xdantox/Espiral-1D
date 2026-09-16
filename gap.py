import gc
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import scipy.fft
import scipy.signal

from Heff import dt
from cadena0 import cadena0spinhistory

# ===================================================================
# 0. CONFIGURACIÓN INICIAL Y MULTITHREADING
# ===================================================================
n_cores = ne.detect_number_of_cores()
ne.set_num_threads(n_cores)
print(f"NumExpr configurado para utilizar {n_cores} hilos de procesamiento.")

DIMER_CHANNEL = "ML"
LAB_COMPONENTS = (0, 1, 2)
APPLY_DEMEAN = True
APPLY_HANN = True
USE_RFFT_TIME = True
Q_INCOMM_INPUT = 1.0332116073

# Rango que abarca la banda inferior (~14 THz) y la superior (~15.5 THz)
F_SEARCH_MIN_THZ = 13.0
F_SEARCH_MAX_THZ = 15.0

# ===================================================================
# 1. CARGA DE DATOS Y REORGANIZACIÓN FÍSICA
# ===================================================================
print("Cargando datos en RAM...")
archivo_datos = 'D_plane = 1.0D fluc.npy'
Spin_history = np.load(archivo_datos)

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

# ===================================================================
# 2. CÁLCULO DEL ESPECTRO BASE (Dímero en marco de laboratorio)
# ===================================================================
def compute_power_lab_dimer_from_spin(
    spin_A_lab, spin_B_lab, gs_A_lab, gs_B_lab, n_dimeros,
    channel="M", components=(0, 1, 2), demean=True, hann=True, use_rfft_time=True,
):
    channel = channel.upper()
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

print("Calculando FFT (dímero)...")
total_power = compute_power_lab_dimer_from_spin(
    spin_A, spin_B, gs_A, gs_B, n_dimeros,
    channel=DIMER_CHANNEL, components=LAB_COMPONENTS,
    demean=APPLY_DEMEAN, hann=APPLY_HANN, use_rfft_time=USE_RFFT_TIME,
)

del Spin_history, spin_A, spin_B, gs_A, gs_B, cadena0
gc.collect()

# ===================================================================
# 3. DEFINICIÓN DE EJES FÍSICOS (k y THz)
# ===================================================================
k_values = np.fft.fftshift(np.fft.fftfreq(n_dimeros, d=1.0) * np.pi)

if USE_RFFT_TIME:
    freq_values = np.fft.rfftfreq(num_pasos, d=dt) / 1e12
else:
    freq_values = np.fft.fftshift(np.fft.fftfreq(num_pasos, d=dt)) / 1e12

# ===================================================================
# 4. ALGORITMO: SELECCIÓN DE PICOS FRONTERIZOS A TRAVÉS DEL GAP
# ===================================================================
def parabolic_peak_interpolation(y_array, x_array, peak_idx):
    """Resolución sub-bin para ubicar el vértice exacto del pico."""
    if peak_idx <= 0 or peak_idx >= len(y_array) - 1:
        return x_array[peak_idx]
    
    y1, y2, y3 = y_array[peak_idx-1], y_array[peak_idx], y_array[peak_idx+1]
    denominator = (y1 - 2*y2 + y3)
    if denominator == 0:
        return x_array[peak_idx]
        
    delta = 0.5 * (y1 - y3) / denominator
    df = x_array[peak_idx] - x_array[peak_idx-1]
    return x_array[peak_idx] + delta * df


def find_true_umklapp_minigap(
    total_power, k_values, freq_values, k_search_range, f_search_range, 
    f_split=14.7, delta_k_plot=0.06, delta_f_plot=1.5, plot_slice=True
):
    """
    Localiza el gap de forma inteligente:
    1. Rastrea f_inf y f_sup en los bines discretos de k.
    2. Aplica una interpolación parabólica (Taylor de la hipérbola) en el espacio k 
       para hallar el k_cruce sub-píxel y las frecuencias exactas del anticruce.
    """
    k_mask = (k_values >= k_search_range[0]) & (k_values <= k_search_range[1])
    valid_k_indices = np.where(k_mask)[0]
    
    df_thz = freq_values[1] - freq_values[0]
    min_dist_bins = max(1, int(0.04 / df_thz))
    
    # Listas para almacenar el perfil de las bandas a lo largo de k
    k_list = []
    f_lower_list = []
    f_upper_list = []
    
    for k_idx in valid_k_indices:
        actual_k = k_values[k_idx]
        power_1d = total_power[:, k_idx]
        
        f_mask = (freq_values >= f_search_range[0]) & (freq_values <= f_search_range[1])
        power_1d_search = power_1d[f_mask]
        freq_search = freq_values[f_mask]
        
        log_power_1d = np.log10(power_1d_search + 1e-15)
        umbral_ruido = max(np.max(log_power_1d) - 4.5, -4.0)
        
        peaks, _ = scipy.signal.find_peaks(
            log_power_1d, 
            height=umbral_ruido,
            prominence=0.35,
            distance=min_dist_bins
        )
        
        if len(peaks) < 2:
            continue
            
        freqs_picos = freq_search[peaks]
        mask_low = freqs_picos < f_split
        mask_high = freqs_picos > f_split
        
        if not np.any(mask_low) or not np.any(mask_high):
            continue
            
        p_low = peaks[mask_low][-1]
        p_high = peaks[mask_high][0]
        
        # Interpolación parabólica en el eje de Frecuencia (Resolución espectral)
        f_lower = parabolic_peak_interpolation(log_power_1d, freq_search, p_low)
        f_upper = parabolic_peak_interpolation(log_power_1d, freq_search, p_high)
        
        k_list.append(actual_k)
        f_lower_list.append(f_lower)
        f_upper_list.append(f_upper)

    if len(k_list) < 3:
        print("ADVERTENCIA: No hay suficientes puntos k para interpolar la parábola.")
        return None
        
    # Convertimos a arrays para el ajuste
    k_arr = np.array(k_list)
    f_inf_arr = np.array(f_lower_list)
    f_sup_arr = np.array(f_upper_list)
    gap_arr = f_sup_arr - f_inf_arr
    
    # 1. Encontrar el mínimo discreto como punto de partida
    idx_min = np.argmin(gap_arr)
    
    # 2. INTERPOLACIÓN PARABÓLICA EN EL ESPACIO K (Resolución espacial)
    if idx_min == 0 or idx_min == len(k_arr) - 1:
        print("ADVERTENCIA: El cruce está en el borde del k_search. Amplía la ventana.")
        best_k = k_arr[idx_min]
        best_f_lower = f_inf_arr[idx_min]
        best_f_upper = f_sup_arr[idx_min]
        min_gap = gap_arr[idx_min]
    else:
        # Tomamos el mínimo discreto y sus dos vecinos
        k_3 = k_arr[idx_min-1 : idx_min+2]
        f_inf_3 = f_inf_arr[idx_min-1 : idx_min+2]
        f_sup_3 = f_sup_arr[idx_min-1 : idx_min+2]
        
        # Ajustes polinomiales (ax^2 + bx + c) para las bandas y el gap
        coefs_inf = np.polyfit(k_3, f_inf_3, 2)
        coefs_sup = np.polyfit(k_3, f_sup_3, 2)
        coefs_gap = coefs_sup - coefs_inf
        
        # El vértice real de la parábola del gap (Derivada = 0 -> x = -b / 2a)
        a_gap, b_gap, _ = coefs_gap
        if a_gap > 0: # Confirmamos que el gap sea un valle (concavidad hacia arriba)
            best_k = -b_gap / (2.0 * a_gap)
            # Evaluamos las bandas individuales en este k analítico
            best_f_lower = np.polyval(coefs_inf, best_k)
            best_f_upper = np.polyval(coefs_sup, best_k)
            min_gap = best_f_upper - best_f_lower
        else:
            # Fallback al discreto en caso de ruido anómalo
            best_k = k_arr[idx_min]
            best_f_lower = f_inf_arr[idx_min]
            best_f_upper = f_sup_arr[idx_min]
            min_gap = gap_arr[idx_min]
            
    print("=" * 55)
    print("CRUCE DE BANDA ENCONTRADO (Interpolación Parabólica Sub-Píxel)")
    print("=" * 55)
    print(f"K del cruce (k_min) : {best_k:.5f} rad")
    print(f"Rama Inf. (Cima)    : {best_f_lower:.5f} THz")
    print(f"Rama Sup. (Valle)   : {best_f_upper:.5f} THz")
    print(f"MINIGAP (Δ)         : {min_gap:.5f} THz")
    print("=" * 55)
    
    # (El código de ploteo 2D y 1D que ya tenías va a continuación, 
    # usa best_k, best_f_lower, y best_f_upper para las líneas)
    
    if plot_slice:
        f_center = 0.5 * (best_f_lower + best_f_upper)
        
        # Subespacio centrado para capturar ambas ramas y el gap intermedio
        k_sub_mask = (k_values >= (best_k - delta_k_plot)) & (k_values <= (best_k + delta_k_plot))
        f_sub_mask = (freq_values >= (f_center - delta_f_plot)) & (freq_values <= (f_center + delta_f_plot))
        
        sub_k = k_values[k_sub_mask]
        sub_f = freq_values[f_sub_mask]
        sub_power = total_power[np.ix_(f_sub_mask, k_sub_mask)]
        log_sub_power = np.log10(sub_power + 1e-15)
        
        # Para el perfil 1D, tomamos el k discreto más cercano al k analítico
        k_idx_opt = np.argmin(np.abs(k_values - best_k))
        log_power_opt = np.log10(total_power[:, k_idx_opt][f_sub_mask] + 1e-15)

        fig, (ax2d, ax1d) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.15, 1]})
        
        # --- Panel 2D: Subespacio S(k, nu) ---
        c = ax2d.pcolormesh(
            sub_k, sub_f, log_sub_power,
            shading='auto', cmap='inferno', rasterized=True
        )
        cbar = fig.colorbar(c, ax=ax2d, pad=0.02)
        cbar.set_label(r'$\log_{10} S(k, \nu)$', fontsize=12)
        
        ax2d.axvline(best_k, color='deepskyblue', linestyle='--', linewidth=1.5, alpha=0.9,
                     label=fr'$k_{{\mathrm{{cruce}}}} = {best_k:.4f}$')
        ax2d.scatter([best_k, best_k], [best_f_lower, best_f_upper], 
                     color=['deepskyblue', 'lime'], edgecolor='black', s=60, zorder=5)
        
        ax2d.hlines(y=[best_f_lower, best_f_upper], xmin=sub_k.min(), xmax=sub_k.max(),
                    colors=['deepskyblue', 'lime'], linestyles=':', linewidth=1.2, alpha=0.9)
        
        arrow_k = best_k + 0.35 * (sub_k.max() - best_k)
        ax2d.annotate(
            '', xy=(arrow_k, best_f_lower), xytext=(arrow_k, best_f_upper),
            arrowprops=dict(arrowstyle='<->', color='white', lw=2.0)
        )
        ax2d.text(
            arrow_k + 0.003, f_center, fr'$\Delta = {min_gap:.4f}$ THz',
            color='white', fontsize=11, fontweight='bold', va='center', ha='left',
            bbox=dict(boxstyle="round,pad=0.25", facecolor='black', alpha=0.75, edgecolor='white')
        )
        
        ax2d.set_title(r'Subespacio $S(k, \nu)$ en el Cruce de Bandas', fontsize=13, pad=10)
        ax2d.set_xlabel(r'Vector de Onda $k$ [rad]', fontsize=12)
        ax2d.set_ylabel(r'Frecuencia $\nu$ [THz]', fontsize=12)
        ax2d.set_xlim(sub_k.min(), sub_k.max())
        ax2d.set_ylim(sub_f.min(), sub_f.max())
        ax2d.legend(loc='lower left', fontsize=10, facecolor='black', labelcolor='white')
        
        # --- Panel 1D: Frecuencia en X, Intensidad en Y ---
        # Graficamos el bin discreto más cercano, pero superponemos las líneas continuas
        ax1d.plot(sub_f, log_power_opt, 'o-', color='tab:blue', markersize=4, lw=1.5,
                  label=f'Perfil en $k_{{disc}} = {k_values[k_idx_opt]:.4f}$')
        
        ax1d.axvline(best_f_lower, color='deepskyblue', linestyle='--', linewidth=1.5,
                     label=fr'$\nu_{{\mathrm{{inf}}}}^{{\mathrm{{true}}}} = {best_f_lower:.3f}$ THz')
        ax1d.axvline(best_f_upper, color='lime', linestyle='--', linewidth=1.5,
                     label=fr'$\nu_{{\mathrm{{sup}}}}^{{\mathrm{{true}}}} = {best_f_upper:.3f}$ THz')
        
        y_range = np.max(log_power_opt) - np.min(log_power_opt)
        y_arrow = np.min(log_power_opt) + 0.45 * y_range
        
        ax1d.annotate(
            '', xy=(best_f_lower, y_arrow), xytext=(best_f_upper, y_arrow),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.8)
        )
        ax1d.text(
            f_center, y_arrow + 0.05 * y_range, fr'$\Delta = {min_gap:.4f}$ THz',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor='gray')
        )
        
        ax1d.set_title(fr'Perfil Analítico Transversal en $k = {best_k:.4f}$', fontsize=13, pad=10)
        ax1d.set_xlabel(r'Frecuencia $\nu$ [THz]', fontsize=12)
        ax1d.set_ylabel(r'Intensidad $\log_{10} S(k, \nu)$', fontsize=12)
        ax1d.set_xlim(sub_f.min(), sub_f.max())
        ax1d.grid(True, alpha=0.3)
        ax1d.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
    return min_gap

# ===================================================================
# 5. EJECUCIÓN DEL CÁLCULO
# ===================================================================
print("\nBuscando el minigap óptico...")

# Rango k centrado en el vértice del anticruce
k_target_expected = 0.54
k_window = 0.07
k_search = (k_target_expected - k_window, k_target_expected + k_window)

minigap_optico = find_true_umklapp_minigap(
    total_power=total_power, 
    k_values=k_values, 
    freq_values=freq_values, 
    k_search_range=k_search,          
    f_search_range=(F_SEARCH_MIN_THZ, F_SEARCH_MAX_THZ),
    f_split=20.7,         # Línea divisoria en la zona prohibida (entre 14.2 y 15.2 THz)
    delta_k_plot=0.06,
    delta_f_plot=1.5,     # Ventana vertical suficiente para encuadrar ambas bandas
    plot_slice=True
)