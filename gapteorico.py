import numpy as np
import scipy.linalg
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# ===================================================================
# 1. IMPORTACIÓN DESDE satelitem.py
# ===================================================================
try:
    from satelitem import get_floquet_matrix, c, q_real, N_max as N_max_default
except ImportError:
    import satelitem as st
    get_floquet_matrix = st.get_floquet_matrix
    c = st.c
    q_real = st.q_real
    N_max_default = 6

# Factor de conversión giromagnético a THz exacto de satelitem.py
FREQ_CONV_FACTOR = 1.7e11 / (2.0 * np.pi * 1e12)

# ===================================================================
# 2. DIAGONALIZADOR SIMPLÉCTICO
# ===================================================================
def compute_floquet_frequencies(k_phys, N_max=6):
    """
    Evalúa la supermatriz de Floquet convirtiendo el momento físico k_phys
    al momento de Bloch de la celda doble (k_bloch = 2 * k_phys).
    """
    k_bloch = 2.0 * k_phys
    H_F = get_floquet_matrix(k_bloch, N_max=N_max)
    total_dim = H_F.shape[0]
    
    sigma_small = np.zeros((2 * c, 2 * c), dtype=complex)
    for i in range(c):
        sigma_small[2 * i, 2 * i + 1] = 1.0
        sigma_small[2 * i + 1, 2 * i] = -1.0
        
    num_blocks = total_dim // (2 * c)
    Sigma = np.kron(np.eye(num_blocks, dtype=complex), sigma_small)
    
    D = 1j * (Sigma @ H_F)
    eigenvals = scipy.linalg.eigvals(D)
    
    real_freqs = np.sort(np.real(eigenvals[np.real(eigenvals) > 1e-6])) * FREQ_CONV_FACTOR
    return real_freqs

# ===================================================================
# 3. MEDIDOR DE GAP MEDIANTE f_split
# ===================================================================
def get_branch_gap_at_k(k_phys, f_split, f_window=(13.0, 15.5), N_max=6):
    """
    Calcula la distancia exacta a través del gap separando las bandas
    por debajo y por encima de f_split.
    """
    freqs = compute_floquet_frequencies(k_phys, N_max=N_max)
    
    # Enmascarar dentro de la ventana de análisis
    in_window = freqs[(freqs >= f_window[0]) & (freqs <= f_window[1])]
    
    lower_candidates = in_window[in_window < f_split]
    upper_candidates = in_window[in_window > f_split]
    
    if len(lower_candidates) == 0 or len(upper_candidates) == 0:
        return 999.0, None, None
        
    # Borde superior de la banda baja y borde inferior de la banda alta
    f_lower = np.max(lower_candidates)
    f_upper = np.min(upper_candidates)
    gap = f_upper - f_lower
    
    return gap, f_lower, f_upper

def find_theoretical_minigap(
    k_expected=-0.493, k_tolerance=0.06, 
    f_split=14.4, f_search_range=(13.2, 15.4), 
    N_max=6, n_k_points=140, plot_bands=True
):
    """
    Rastrea el anticruce minimizando la distancia entre las dos ramas
    que delimitan la zona prohibida alrededor de f_split.
    """
    k_min_range = k_expected - k_tolerance
    k_max_range = k_expected + k_tolerance
    
    print("=" * 60)
    print("BÚSQUEDA DEL MINIGAP TEÓRICO (DELIMITACIÓN POR f_split)")
    print("=" * 60)
    print(f"Rango k_phys       : [{k_min_range:.4f}, {k_max_range:.4f}] rad")
    print(f"Ventana Frecuencia : [{f_search_range[0]:.2f}, {f_search_range[1]:.2f}] THz")
    print(f"Línea f_split      : {f_split:.3f} THz")
    print(f"Truncamiento Floquet (N_max): {N_max}")
    
    # 1. Escaneo preliminar sobre grilla
    k_grid = np.linspace(k_min_range, k_max_range, n_k_points)
    gaps = []
    
    for k in k_grid:
        g, _, _ = get_branch_gap_at_k(k, f_split=f_split, f_window=f_search_range, N_max=N_max)
        gaps.append(g)
        
    gaps = np.array(gaps)
    valid_mask = gaps < 900.0
    
    if not np.any(valid_mask):
        print("ERROR: No se detectaron bandas a ambos lados de f_split en esa ventana.")
        return None, None, None, None
        
    best_idx = np.argmin(gaps)
    k_guess = k_grid[best_idx]
    
    # 2. Refinamiento continuo mediante optimización escalar
    def cost_func(k):
        g, _, _ = get_branch_gap_at_k(k, f_split=f_split, f_window=f_search_range, N_max=N_max)
        return g

    dk_bracket = (k_grid[1] - k_grid[0]) * 2.0
    opt_res = minimize_scalar(
        cost_func, 
        bracket=(k_guess - dk_bracket, k_guess, k_guess + dk_bracket),
        bounds=(k_min_range, k_max_range), 
        method='bounded'
    )
    
    best_k = opt_res.x
    min_gap, best_f_lower, best_f_upper = get_branch_gap_at_k(best_k, f_split=f_split, f_window=f_search_range, N_max=N_max)
    
    if best_f_lower is None or best_f_upper is None:
        best_k = k_guess
        min_gap, best_f_lower, best_f_upper = get_branch_gap_at_k(best_k, f_split=f_split, f_window=f_search_range, N_max=N_max)
    
    print("-" * 60)
    print("RESULTADOS DEL GAP TEÓRICO")
    print("-" * 60)
    print(f"k_phys de cruce (k_min)   : {best_k:.6f} rad")
    print(f"Rama Inferior             : {best_f_lower:.5f} THz")
    print(f"Rama Superior             : {best_f_upper:.5f} THz")
    print(f"MINIGAP TEÓRICO (Δ)       : {min_gap:.5f} THz")
    print("=" * 60)
    
    # 3. Graficación
    if plot_bands:
        k_dense = np.linspace(k_min_range, k_max_range, 180)
        bands_list = []
        
        for k in k_dense:
            f_all = compute_floquet_frequencies(k, N_max=N_max)
            mask = (f_all >= f_search_range[0]) & (f_all <= f_search_range[1])
            bands_list.append(f_all[mask])
            
        plt.figure(figsize=(9, 6.5))
        
        for k, freqs in zip(k_dense, bands_list):
            plt.scatter([k] * len(freqs), freqs, color='black', s=8, alpha=0.6)
            
        plt.axvline(best_k, color='deepskyblue', linestyle='--', linewidth=1.5,
                    label=fr'$k_{{\mathrm{{cruce}}}} = {best_k:.5f}$')
        
        # Puntos que delimitan el gap
        plt.scatter([best_k, best_k], [best_f_lower, best_f_upper], 
                    color=['deepskyblue', 'lime'], edgecolor='black', s=80, zorder=5)
        
        plt.axhline(best_f_lower, color='deepskyblue', linestyle=':', linewidth=1.2, alpha=0.8)
        plt.axhline(best_f_upper, color='lime', linestyle=':', linewidth=1.2, alpha=0.8)
        plt.axhline(f_split, color='gray', linestyle='--', linewidth=1.0, alpha=0.5, label=f'$f_{{\mathrm{{split}}}} = {f_split:.2f}$ THz')
        
        # Flecha anotadora del gap teórico
        arrow_k = best_k + (k_max_range - best_k) * 0.40
        plt.annotate(
            '', xy=(arrow_k, best_f_lower), xytext=(arrow_k, best_f_upper),
            arrowprops=dict(arrowstyle='<->', color='crimson', lw=2.0)
        )
        f_center = 0.5 * (best_f_lower + best_f_upper)
        plt.text(
            arrow_k + 0.002, f_center, fr'$\Delta_{{\mathrm{{teor}}}} = {min_gap:.4f}$ THz',
            color='crimson', fontsize=11, fontweight='bold', va='center', ha='left',
            bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.9, edgecolor='crimson')
        )
        
        plt.title(fr'Dispersión LSWT-Floquet ($N_{{\mathrm{{max}}}}={N_max}$): Minigap Óptico', fontsize=13, pad=12)
        plt.xlabel(r'Wave Vector $k$ (Dimer BZ) [rad]', fontsize=12)
        plt.ylabel(r'Frequency $\nu$ [THz]', fontsize=12)
        plt.xlim(k_min_range, k_max_range)
        plt.ylim(f_search_range[0], f_search_range[1])
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left', fontsize=10)
        plt.tight_layout()
        plt.show()
        
    return min_gap, best_k, best_f_lower, best_f_upper

# ===================================================================
# 4. EJECUCIÓN DIRECTA
# ===================================================================
if __name__ == "__main__":
    # Caso 1: Gap principal entre el grupo inferior (~13.8 THz) y superior (~15.0 THz)
    # f_split se ubica en el vacío espectral (14.4 THz)
    gap_val, k_c, f_low, f_up = find_theoretical_minigap(
        k_expected=-1.035,
        k_tolerance=0.04,
        f_split=20.6,                  # Línea de división en el vacío del gap
        f_search_range=(20.3, 21.0),
        N_max=N_max_default,
        plot_bands=True
    )