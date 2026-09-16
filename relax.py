import numpy as np
import matplotlib.pyplot as plt
from typing import cast
from mpl_toolkits.mplot3d import Axes3D
# Configuración
N_SITES = 1198 # O el tamaño que desees analizar

# Interruptores de ejecución (True/False)
RUN_THETA_DEFORMATION = True
RUN_DELTA_THETA_FFT = False
RUN_Q_DIMERIZED = False
RUN_ELLIPTICIDAD = False
RUN_LISSAJOUS_YZ = False
RUN_BASAL_AUTOPSY = False
RUN_STAGGERED_ORDER = False
RUN_HYBRID_TILT = True
RUN_BLOCH_3D = True

# Comparación opcional con alpha del minimizador de direct3_PBC.
# Reemplaza estos valores con los que importes de direct3_PBC.
IMPORTED_ANSATZ_PARAMS = {
    "q": 5.2514862797,        # Usando tu q_real más reciente (¡Ojo que antes decías 2.108!)
    "phi": 0.0,
    "gamma": 1.286581,
    "alpha": 0.002230,
    "phi_2q": -3.141587,      # Fase de alpha (phi_ind)
    "beta": -0.004351,        # Nuevo parámetro beta
    "psi_2q": -1.570795,      # Fase de beta (psi_ind)
}


# ==============================================================================
# 0. TU FUNCIÓN DE CARGA (Tal cual la enviaste)
# ==============================================================================
def cadena0spinhistory(n):
    # Asegúrate de que este archivo esté en la misma carpeta o ajusta la ruta
    try:
        Spin_history = np.load('D_plane = 1.0D relax.npy', mmap_mode='r')
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
# 1. CARGA DE DATOS (Tal cual la enviaste)
# ==============================================================================
spins = cadena0spinhistory(N_SITES)


# ==============================================================================
# 2. RUTINA DE ANÁLISIS (AUTOPSIA DEL SOLITÓN)
# ==============================================================================
def plot_theta_deformation(spins, plane_indices=(1, 2), n_range=(0, 400), title_suffix="", ansatz_params=None):
    """Main pictorial representation of theta_n in real space with enhanced legibility."""
    n_sites = spins.shape[0]
    idx = np.arange(n_sites)

    s_a = spins[:, plane_indices[0]]
    s_b = spins[:, plane_indices[1]]
    theta = np.arctan2(s_a, s_b)
    theta_unwrapped = np.unwrap(theta)

    staggered = (-1.0) ** idx

    if ansatz_params is None:
        raise ValueError("Debes pasar ansatz_params con q, phi, gamma, alpha, phi_2q, beta y psi_2q")

    q_ans = float(ansatz_params["q"])
    phi_ans = float(ansatz_params["phi"])
    gamma_ans = float(ansatz_params["gamma"])
    
    # --- PARÁMETROS DEL ANSATZ (CONTINUO + DIMERIZADO) ---
    alpha_ansatz = float(ansatz_params["alpha"])
    phi_2q = float(ansatz_params["phi_2q"])
    beta_ansatz = float(ansatz_params.get("beta", 0.0))
    psi_2q = float(ansatz_params.get("psi_2q", 0.0))

    # Extracción directa desde LLG exacto
    A_llg = np.vstack([idx, np.ones(n_sites), staggered]).T
    c_llg, _, _, _ = np.linalg.lstsq(A_llg, theta_unwrapped, rcond=None)
    q_llg_raw, phi_llg, gamma_llg = [float(v) for v in c_llg]

    # --- CORRECCIÓN DE RAMA 2pi (ALINEACIÓN DE LEYENDA) ---
    diff_direct = (q_llg_raw - q_ans) % (2.0 * np.pi)
    diff_direct = diff_direct - 2*np.pi if diff_direct > np.pi else diff_direct
    
    diff_conjugate = (-q_llg_raw - q_ans) % (2.0 * np.pi)
    diff_conjugate = diff_conjugate - 2*np.pi if diff_conjugate > np.pi else diff_conjugate

    if abs(diff_direct) < abs(diff_conjugate):
        q_llg_display = q_ans + diff_direct
    else:
        q_llg_display = q_ans + diff_conjugate

    # --- EL CÁLCULO CLAVE PARA LA GRÁFICA 1 ---
    theta_linear_llg = q_llg_raw * idx + phi_llg
    theta_linear_llg_staggered = theta_linear_llg + gamma_llg * staggered
    
    # NUEVO: Ansatz completo con alfa y beta
    theta_harmonic = (alpha_ansatz * np.sin(2.0 * q_ans * idx + phi_2q) + 
                      beta_ansatz * staggered * np.sin(2.0 * q_ans * idx + psi_2q))
    
    # CORREGIDO: Se suma sobre la base dimerizada (staggered) para que encaje visualmente
    theta_model_harm_only = theta_linear_llg_staggered + theta_harmonic

    # Residuo limpio (Contenido Armónico LLG real)
    delta_theta = theta_unwrapped - theta_linear_llg_staggered

    # ==========================================================================
    # GRÁFICO 1: FASE ESPACIAL
    # ==========================================================================
    fig_theta, ax_theta = plt.subplots(1, 1, figsize=(11, 5))

    ax_theta.plot(idx, theta_unwrapped, color="tab:blue", lw=1.5, label=r"$\theta_n$ (unwrapped)")
    ax_theta.plot(
        idx,
        theta_model_harm_only,
        color="black",
        lw=1.5,
        ls="--",
        label=(
            r"$\theta_n^{fit}=\theta_n^{(0)} + \alpha\sin(2qn+\phi_{2q}) + \beta(-1)^n\sin(2qn+\psi_{2q})$"
            + f"\n$\\alpha$={alpha_ansatz:.5f}, $\\beta$={beta_ansatz:.5f}"
        ),
    )
    ax_theta.set_ylabel(r"$\theta_n$ [rad]", fontsize=16)
    ax_theta.set_xlabel("Site $n$", fontsize=16)
    
    title_text = r"Spatial Phase $\theta_n$" + (f" - {title_suffix}" if title_suffix else "")
    ax_theta.set_title(title_text, fontsize=18, fontweight='bold', pad=15)
    
    ax_theta.tick_params(axis='both', which='major', labelsize=14)
    ax_theta.grid(True, alpha=0.3)
    
    ax_theta.legend(fontsize=13, loc='upper right')
    ax_theta.set_xlim(n_range)

    plt.tight_layout()
    plt.show()

    resid_ansatz = theta_harmonic
    
    # ==========================================================================
    # GRÁFICO 2: RESIDUO DE FASE (DOS PANELES)
    # ==========================================================================
    fig_res, (ax_res_top, ax_res_bottom) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True
    )

    ax_res_top.plot(
        idx,
        delta_theta,
        color="tab:purple",
        lw=1.5,
        label=(
            r"$\delta\theta_n^{LLG} = \theta_n - (q_{LLG}n + \phi_{LLG} + \gamma_{LLG}(-1)^n)$"
            + "\n(Exact harmonic content)"
            + f"\n$q_{{LLG}}$={q_llg_display:.6f}"
        ),
    )
    ax_res_top.axhline(0.0, color="black", ls="--", lw=1.0, alpha=0.5)
    ax_res_top.set_ylabel(r"$\delta\theta_n$ [rad]", fontsize=16)
    ax_res_top.tick_params(axis='y', labelsize=14)
    ax_res_top.grid(True, alpha=0.3)
    
    ymin, ymax = ax_res_top.get_ylim()
    ax_res_top.set_ylim(ymin, ymax + (ymax - ymin) * 0.45) 
    ax_res_top.legend(fontsize=13, loc='upper right')

    ax_res_bottom.plot(
        idx,
        resid_ansatz,
        color="black",
        lw=1.5,
        ls="--",
        label=(
            r"Ansatz: $\alpha\sin(2qn+\phi_{2q}) + \beta(-1)^n\sin(2qn+\psi_{2q})$"
            + f"\n$\\alpha$={alpha_ansatz:.5f}, $\\beta$={beta_ansatz:.5f}"
        ),
    )
    ax_res_bottom.axhline(0.0, color="black", ls="--", lw=1.0, alpha=0.5)
    ax_res_bottom.set_ylabel(r"$\delta\theta_n$ [rad]", fontsize=16)
    ax_res_bottom.set_xlabel("Site $n$", fontsize=16)
    ax_res_bottom.tick_params(axis='both', which='major', labelsize=14)
    ax_res_bottom.grid(True, alpha=0.3)
    
    ymin, ymax = ax_res_bottom.get_ylim()
    ax_res_bottom.set_ylim(ymin, ymax + (ymax - ymin) * 0.45) 
    ax_res_bottom.legend(fontsize=13, loc='upper right')
    ax_res_bottom.set_xlim(n_range)

    fig_res.suptitle(r"Phase Residual: Numerical (LLG) vs Analytical (Ansatz) Comparison", fontsize=18, fontweight='bold')
    plt.tight_layout()
    fig_res.subplots_adjust(top=0.92) 
    plt.show()

    # Retorno añadido para evitar errores de desempaquetado en la ejecución
    return q_llg_raw, gamma_llg, alpha_ansatz, beta_ansatz, phi_2q, psi_2q, theta_unwrapped, delta_theta


# Método principal recomendado para representar theta_n en la narrativa del manuscrito.
if RUN_THETA_DEFORMATION:
    q_fit_main, gamma_main, alpha_main, beta_main, phi2q_main, psi2q_main, theta_unwrap_main, dtheta_res_main = plot_theta_deformation(
        spins,
        plane_indices=(1, 2),
        n_range=(0, 300),
        title_suffix="Diagnóstico principal",
        ansatz_params=IMPORTED_ANSATZ_PARAMS,
    )

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
if RUN_DELTA_THETA_FFT:
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

def estimate_Q_dimerized(spins, plane_indices=(1, 2)):
    """
    Estima Q considerando la dimerización (zig-zag) de la espiral.
    Ajusta el modelo: theta(n) = Q*n + phi + gamma*(-1)^n
    
    Args:
        spins: Array (N, 3) con la configuración de espines.
        plane_indices: Índices del plano de rotación (por defecto YZ).
        
    Returns:
        Q_slope: Vector de onda promedio [rad/sitio].
        gamma_stagger: Amplitud de la oscilación por dimerización [rad].
        R_sq: Calidad del ajuste.
    """
    # 1. Obtener fase desenrollada
    S_a = spins[:, plane_indices[0]]
    S_b = spins[:, plane_indices[1]]
    theta = np.arctan2(S_a, S_b)
    theta_unwrapped = np.unwrap(theta)
    
    n_points = len(theta_unwrapped)
    x = np.arange(n_points)
    
    # 2. Construir la matriz de diseño para Mínimos Cuadrados
    # Queremos ajustar: y = c0*x + c1*1 + c2*(-1)^x
    # Columna 0: n (lineal) -> Su pendiente es Q
    # Columna 1: 1 (constante) -> Fase global
    # Columna 2: (-1)^n (staggered) -> Dimerización
    
    staggered_term = (-1)**x
    A = np.vstack([x, np.ones(n_points), staggered_term]).T
    
    # 3. Resolver el sistema lineal (A * c = theta)
    # c = [Q, phi, gamma]
    c, residuals, rank, s = np.linalg.lstsq(A, theta_unwrapped, rcond=None)
    
    Q_slope = c[0]
    phi_0 = c[1]
    gamma_stagger = c[2]
    
    # 4. Calcular R^2
    theta_pred = Q_slope * x + phi_0 + gamma_stagger * staggered_term
    ss_res = np.sum((theta_unwrapped - theta_pred) ** 2)
    ss_tot = np.sum((theta_unwrapped - np.mean(theta_unwrapped)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return Q_slope, gamma_stagger, r_squared, theta_unwrapped, theta_pred

# ==============================================================================
# INTEGRACIÓN EN TU SCRIPT
# ==============================================================================
# (Sustituye la llamada anterior por esta)

if RUN_Q_DIMERIZED:
    Q_est, gamma_val, R2, th_raw, th_fit = estimate_Q_dimerized(spins)

    print("="*45)
    print(f"ANÁLISIS DE VECTOR DE ONDA (CON DIMERIZACIÓN)")
    print("="*45)
    print(f"Q promedio (rad/sitio)   : {Q_est:.6f}")
    print(f"Staggering (gamma)       : {gamma_val:.6f} rad")
    print(f"Calidad de ajuste (R^2)  : {R2:.8f}")
    print("-" * 45)
    # Esto detecta si hay un problema de 'aliasing' de orden 3
    delta_3Q = (3 * abs(Q_est)) % (2*np.pi)
    print(f"Desajuste en k=0 (3Q-2pi): {delta_3Q - 2*np.pi if delta_3Q > np.pi else delta_3Q:.6f}")
    print("="*45)

# ==============================================================================
# 3. ANÁLISIS DE ELIPTICIDAD Y DEFORMACIÓN (NUEVA SECCIÓN)
# ==============================================================================

def analizar_elipticidad(spins):
    Sy = spins[:, 1]
    Sz = spins[:, 2]
    
    # 1. Ratio de Amplitudes (A_y / A_z)
    # En una espiral circular perfecta, amp_ratio = 1.0
    amp_y = np.std(Sy) 
    amp_z = np.std(Sz)
    ratio = amp_y / amp_z
    excentricidad = np.sqrt(1 - ratio**2) if ratio < 1 else 0

    # 2. Análisis de Armónicos en el espacio recíproco
    # La elipticidad genera armónicos impares (3Q, 5Q...)
    fft_z = np.fft.fft(Sz - np.mean(Sz))
    fft_mag = np.abs(fft_z)[:len(Sz)//2]
    peaks = np.argsort(fft_mag)[-2:] # Tomamos los dos picos más altos
    
    return ratio, excentricidad, fft_mag

if RUN_ELLIPTICIDAD:
    ratio, ecc, mag_z = analizar_elipticidad(spins)

    print("\n" + "="*45)
    print(f"DIAGNÓSTICO DE ELIPTICIDAD (D_plane)")
    print("="*45)
    print(f"Ratio de Amplitud (Ay/Az) : {ratio:.6f}")
    print(f"Excentricidad de la elipse: {ecc:.6f}")
    print(f"Estado: {'ELÍPTICO' if ratio < 0.98 else 'CIRCULAR'}")
    print("-" * 45)
    print(f"Interpretación: A menor ratio, mayor achatamiento")
    print(f"hacia el eje fácil (Z) debido a D_plane.")
    print("="*45)

# ==============================================================================
# 4. VISUALIZACIÓN DE LA TRAYECTORIA (Lissajous)
# ==============================================================================
if RUN_LISSAJOUS_YZ:
    if not RUN_ELLIPTICIDAD:
        ratio, _, _ = analizar_elipticidad(spins)
    plt.figure(figsize=(6, 6))
    plt.plot(spins[:200, 1], spins[:200, 2], 'o-', markersize=2, alpha=0.5, color='darkorange')
    plt.title(f'Trayectoria del Espín (Plano YZ)\nRatio Ay/Az = {ratio:.3f}')
    plt.xlabel('$S_y$ (Eje Difícil)')
    plt.ylabel('$S_z$ (Eje Fácil)')
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    plt.show()

def caracterizar_estado_basal(spins):
    """
    Analiza el orden AFM, el Canting y la Magnetización Neta.
    """
    N = len(spins)
    Sx, Sy, Sz = spins[:, 0], spins[:, 1], spins[:, 2]
    
    # 1. Magnetización Alternante (Staggered Magnetization)
    # Si es un AFM perfecto, m_staggered ~ 1.0
    staggered_z = Sz * ((-1)**np.arange(N))
    m_staggered = np.abs(np.mean(staggered_z))
    
    # 2. Canting y Magnetización Neta
    # El Canting suele aparecer en Sx si hay DMI o anisotropías cruzadas
    m_net_x = np.mean(Sx)
    m_net_y = np.mean(Sy)
    m_net_z = np.mean(Sz)
    
    # 3. Análisis de la "Pureza" del Solitón
    # Medimos qué tanto se desvía el orden alternante de ser constante
    staggered_fluctuation = np.std(staggered_z)

    return {
        "m_staggered": m_staggered,
        "m_net": [m_net_x, m_net_y, m_net_z],
        "std_staggered": staggered_fluctuation
    }

# ==============================================================================
# EJECUCIÓN DEL ANÁLISIS PROFUNDO
# ==============================================================================
if RUN_BASAL_AUTOPSY:
    stats = caracterizar_estado_basal(spins)

    print("\n" + "X"*45)
    print(f"AUTOPSIA DEL ESTADO BASAL (SOLITÓN AFM)")
    print("X"*45)
    print(f"Orden AFM (Staggered Mz)    : {stats['m_staggered']:.6f}")
    print(f"Fluctuación del Orden AFM   : {stats['std_staggered']:.6f}")
    print(f"Magnetización Neta (Canting):")
    print(f"   Mx (Out-of-plane): {stats['m_net'][0]:.6e}")
    print(f"   My (In-plane H)  : {stats['m_net'][1]:.6e}")
    print(f"   Mz (Easy Axis)   : {stats['m_net'][2]:.6e}")
    print("-" * 45)

    if stats['std_staggered'] > 0.05:
        print("ESTADO: Red de Solitones Modulada (No es AFM puro)")
    else:
        print("ESTADO: AFM Colineal Anclado")
    print("X"*45)

def plot_staggered_order(spins, n_range=(0, 400)):
    """
    Visualiza la Magnetización Alternante Local y detecta solitones.
    M_s^i = S_z,i * (-1)^i
    """
    N = len(spins)
    indices = np.arange(N)
    Sz = spins[:, 2]
    
    # 1. Calcular Magnetización Alternante Local
    # Esto 'endereza' el zigzag AFM
    m_staggered = Sz * ((-1)**indices)
    
    # 2. Configuración de la figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # --- Gráfico 1: El perfil de orden staggered ---
    ax1.plot(indices, m_staggered, 'o-', color='indigo', markersize=3, linewidth=1, alpha=0.7)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax1.set_title(r'Perfil de Magnetización Alternante Local ($m_s^i = S_{z,i} \cdot (-1)^i$)')
    ax1.set_ylabel(r'$m_s^i$')
    ax1.grid(True, alpha=0.2)
    
    # --- Gráfico 2: Zoom y detección de solitones ---
    # Los solitones aparecen como 'cruces' por el cero o cambios de signo
    ax2.fill_between(indices, m_staggered, 0, where=(m_staggered >= 0), color='blue', alpha=0.2)
    ax2.fill_between(indices, m_staggered, 0, where=(m_staggered < 0), color='red', alpha=0.2)
    ax2.set_ylabel('Regiones de Fase')
    ax2.set_xlabel('Sitio n')
    
    # Aplicar zoom
    ax1.set_xlim(n_range)
    plt.tight_layout()
    plt.show()

if RUN_STAGGERED_ORDER:
    # Ejecutar con tus datos cargados
    plot_staggered_order(spins, n_range=(0, 600))

# ==============================================================================
# 5. ANÁLISIS DEL "TILT" HÍBRIDO (Fuga al Eje X)
# ==============================================================================
def analizar_hybrid_tilt(spins):
    """
    Analiza si el sistema está usando el eje X (Tilt) para bypassear
    la barrera de potencial en el eje Y.
    """
    Sx = spins[:, 0]
    Sy = spins[:, 1]
    Sz = spins[:, 2]
    N = len(Sx)
    
    # 1. Cuantificación de la Fuga (Tilt)
    # Medimos cuánta "masa magnética" se ha ido al eje X
    avg_abs_Sx = np.mean(np.abs(Sx))
    max_Sx = np.max(np.abs(Sx))
    energy_fraction_X = np.sum(Sx**2) / np.sum(Sx**2 + Sy**2 + Sz**2)
    
    # 2. Correlación de Fuga: ¿Ocurre el tilt cuando Sy debería ser máximo?
    # Si el sistema evita Y, esperamos que |Sx| sea máximo cuando |Sz| es mínimo (en el ecuador)
    # y |Sy| sea menor de lo esperado.
    
    return avg_abs_Sx, max_Sx, energy_fraction_X

if RUN_HYBRID_TILT:
    # Ejecutar métricas
    avg_tilt, max_tilt, frac_x = analizar_hybrid_tilt(spins)

    print("\n" + "Z"*45)
    print(f"DIAGNÓSTICO DE TILT HÍBRIDO (Ruta de Escape X)")
    print("Z"*45)
    print(f"Magnitud media en X (|Sx|)    : {avg_tilt:.6f}")
    print(f"Tilt Máximo (Pico de Sx)     : {max_tilt:.6f}")
    print(f"Fracción de Energía en X     : {frac_x*100:.2f} %")
    print("-" * 45)
    if max_tilt > 0.1:
        print("CONCLUSIÓN: ¡HÍBRIDO DETECTADO!")
        print("El espín se está inclinando hacia X para cruzar el ecuador.")
        print("El Ansatz plano YZ ha colapsado.")
    else:
        print("CONCLUSIÓN: Rotación Plana YZ (Sin Tilt significativo)")
    print("Z"*45)
# ==============================================================================
# 6. VISUALIZACIÓN 3D (CORREGIDA)
# ==============================================================================
def plot_esfera_bloch_3d(spins, n_limit=300):
    # Crear figura
    fig = plt.figure(figsize=(10, 8))
    
    # FORZAR la proyección 3D explícitamente
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))
    
    # Datos limitados para no saturar
    subset = spins[:n_limit]
    xs, ys, zs = subset[:, 0], subset[:, 1], subset[:, 2]
    
    # CORRECCIÓN COLORMAP: Usar get_cmap en lugar de acceder como atributo directo
    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0, 1, len(xs)))
    
    # Plotear trayectoria
    # CORRECCIÓN SCATTER: Pylance a veces confunde zs con 's' (tamaño) en 2D.
    # Al correrlo funcionará bien, aunque el linter se queje.
    ax.scatter(xs, ys, zs, c=colors, s=10, depthshade=True)
    ax.plot(xs, ys, zs, color='gray', alpha=0.3, linewidth=0.5)
    
    # Dibujar Esfera de referencia
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = np.cos(u)*np.sin(v)
    y_sphere = np.sin(u)*np.sin(v)
    z_sphere = np.cos(v)
    
    # NOTA: Si VS Code marca en rojo 'plot_wireframe' o 'view_init', IGNÓRALO y ejecuta.
    # Es un falso positivo del editor, el código funcionará correctamente al ejecutarse.
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="k", alpha=0.1)

    # Ejes y Etiquetas
    ax.set_xlabel('$S_x$ (Eje X - Escape)')
    ax.set_ylabel('$S_y$ (Eje Y - Barrera)')
    ax.set_zlabel('$S_z$ (Eje Z - Fácil)')
    ax.set_title(f"Trayectoria 3D del Espín\n¿Evasión de Barrera? Max($S_x$)={np.max(np.abs(xs)):.3f}")
    
    # Vistas clave
    ax.view_init(elev=20, azim=45)
    plt.show()

    # --- GRÁFICO 2D DE CORTE TRANSVERSAL (X vs Y) ---
    plt.figure(figsize=(6,6))
    plt.plot(ys, xs, '.-', alpha=0.5, color='teal')
    plt.xlabel('$S_y$ (Barrera)')
    plt.ylabel('$S_x$ (Escape)')
    plt.title('Vista Superior: Competencia de Anisotropías (Plano XY)\nForma de "8" o "X" indica Tilt Híbrido.')
    plt.grid(True)
    plt.axis('equal')
    plt.axhline(0, color='k', alpha=0.3)
    plt.axvline(0, color='k', alpha=0.3)
    plt.show()

if RUN_BLOCH_3D:
    # Ejecutar
    plot_esfera_bloch_3d(spins, n_limit=400)

# ==============================================================================
# 7. ESTIMADOR MULTI-ARMÓNICO ESPACIAL (BUNCHING & INTERACCIÓN DE RED)
# ==============================================================================
def estimate_multiple_harmonics(spins, plane_indices=(1, 2), max_harmonics=10):
    """
    Extrae la deformación de la fase (delta_theta) y calcula matemáticamente 
    la amplitud pura (alpha_m, en 2mQ) y la amplitud dimerizada (beta_m, en 2mQ - pi).
    """
    Q_slope, gamma_stagger, _, theta_unwrapped, theta_pred = estimate_Q_dimerized(spins, plane_indices)
    delta_theta = theta_unwrapped - theta_pred
    
    n_points = len(theta_unwrapped)
    x = np.arange(n_points)
    staggered = (-1)**x  # Equivalente a e^{i*pi*x}, traslada la frecuencia en pi
    
    # Construir Matriz de Diseño para Mínimos Cuadrados
    basis_funcs = []
    for m in range(1, max_harmonics + 1):
        arg = 2 * m * Q_slope * x
        
        # 1. Base continua (2mQ)
        basis_funcs.append(np.sin(arg))
        basis_funcs.append(np.cos(arg))
        
        # 2. Base acoplada a la red / dimerizada (2mQ ± pi)
        basis_funcs.append(staggered * np.sin(arg))
        basis_funcs.append(staggered * np.cos(arg))
        
    A_mat = np.column_stack(basis_funcs)
    c_harm, _, _, _ = np.linalg.lstsq(A_mat, delta_theta, rcond=None)
    
    harmonics_data = []
    for m in range(1, max_harmonics + 1):
        idx = 4 * (m - 1)
        
        # Amplitud de la deformación inmensurable pura
        A_reg, B_reg = c_harm[idx], c_harm[idx+1]
        alpha_m = np.sqrt(A_reg**2 + B_reg**2)
        phi_m = np.arctan2(B_reg, A_reg)
        
        # Amplitud de la deformación anclada a la celda (Umklapp local)
        A_stag, B_stag = c_harm[idx+2], c_harm[idx+3]
        beta_m = np.sqrt(A_stag**2 + B_stag**2)
        psi_m = np.arctan2(B_stag, A_stag)
        
        harmonics_data.append({
            "m": m,
            "alpha": alpha_m, "phi": phi_m,
            "beta": beta_m, "psi": psi_m
        })
        
    return harmonics_data, delta_theta, Q_slope, gamma_stagger

# --- Interruptor de ejecución ---
RUN_HARMONIC_ESTIMATOR = True

if RUN_HARMONIC_ESTIMATOR:
    harmonics, d_theta, q_est, gamma_est = estimate_multiple_harmonics(spins, max_harmonics=10)
    
    print("\n" + "="*80)
    print("ESTIMACIÓN MULTI-ARMÓNICA (MODULACIÓN CONTINUA VS DISCRETA)")
    print("="*80)
    print(f"Vector Q base             : {q_est:.6f} rad/sitio")
    print(f"Dimerización (gamma)      : {gamma_est:.6f} rad")
    print("-" * 80)
    print(f"{'Orden':<7} | {'Amplitud Pura 2mQ (α)':<25} | {'Amplitud Dimerizada 2mQ-π (β)':<25}")
    print("-" * 80)
    
    for h in harmonics:
        marker = "->" if h['m'] == 1 else "  "
        print(f"{marker} m={h['m']:<2} | {h['alpha']:.6f} rad                 | {h['beta']:.6f} rad")
    print("="*80)
    
    if 'IMPORTED_ANSATZ_PARAMS' in globals():
        alpha_1 = harmonics[0]['alpha']
        beta_1 = harmonics[0]['beta']
        ansatz_alpha = IMPORTED_ANSATZ_PARAMS['alpha']
        
        # Volumen RMS total combinando las dos componentes dominantes
        volumen_real = np.sqrt(alpha_1**2 + beta_1**2)
        
        print(f"Amplitud 2q (Abre el gap en LSWT) : {alpha_1:.6f} rad")
        print(f"Amplitud 2q-pi (Fuga a la red)    : {beta_1:.6f} rad")
        print(f"Volumen de Deformación Real (RMS) : {volumen_real:.6f} rad")
        print(f"Alpha Teórico Forzado (Ansatz)    : {ansatz_alpha:.6f} rad")
        print(f"Discrepancia Volumen-Ansatz       : {abs(volumen_real - ansatz_alpha):.6e}")
    print("="*80)