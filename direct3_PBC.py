
"""
Herramienta para evaluar y optimizar el ansatz modulado en una cadena finita con PBC.
Incluye Anisotropía en el Plano (D_plane) para generar bunching y satélites.
"""

import math
import os
import time
from typing import Iterable, Sequence, Tuple, Union

from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# --- CONSTANTES GLOBALES ---
q_c = 2.0 * math.pi / 3.0

# Formato: (Name, Jbar, dJ, J2, K, D_axis, N)
# D_axis: Anisotropía longitudinal (Canting)

DEFAULT_SETS = [
	#("Set 1", 48.891035, 48.620365, 1.26126, 48.9119, 0.31,1198),
	("Set 2", 46.75, 44.85, 2.6, 45.4, 0.76, 1198),
	#("Set 3", 17.10185, 11.80055, 0.0085, 16.8117, 0.33, 1198),
	#("Set 4", 17.10185, 1.80055, 8.5, 18.1212, 3.3, 1198),

]
PARAM_NAMES = (
    "mx",
    "gamma",
    "alpha_ind",
    "phi_ind",
)
MX_BOUNDS = (-0.999, 0.999)
DEFAULT_BOUNDS = (MX_BOUNDS,) + tuple((-math.pi, math.pi) for _ in PARAM_NAMES[1:])
REPORT_PARAMS = PARAM_NAMES


# --- FUNCIONES AUXILIARES ---

def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _q_from_winding(M: Union[float, np.ndarray], chain_length: int) -> Union[float, np.ndarray]:
    if chain_length <= 0:
        raise ValueError("chain_length must be positive")
    factor = (2.0 * math.pi) / float(chain_length)
    q = factor * np.asarray(M, dtype=float)
    if q.ndim == 0:
        return float(q)
    return q


def theta_n(
    n: Iterable[int],
    q: float,
    gamma: float,
    alpha_ind: float,
    phi_ind: float,
) -> np.ndarray:
    """Evalúa el perfil angular modulado."""
    idx = np.asarray(n, dtype=np.int64)
    base = idx * q
    parity = np.where((idx & 1) == 0, 1.0, -1.0)
    
    # Perfil base + Dimerización + Modulación armónica (Soliton Lattice)
    profile = base + gamma * parity
    profile += alpha_ind * np.sin(2.0 * q * idx + phi_ind)
    return profile


def _canting_weights(mx: float) -> Tuple[float, float]:
    mx_sq = mx * mx
    plane_weight = max(0.0, 1.0 - mx_sq)
    return mx_sq, plane_weight


# --- NÚCLEO FÍSICO ---

def energy_components_modulated_finite(
    q: float,
    params: np.ndarray,
    Jbar: float,
    dJ: float,
    J2: float,
    K: float,
    D_axis: float,   # Anisotropía Axial (Eje X)
    D_plane: float,  # NUEVO: Anisotropía en el Plano de rotación (Eje Y)
    sites: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    
    if sites.size == 0:
        raise ValueError("sites array must be non-empty")
        
    mx = float(params[0])
    theta = theta_n(sites, q, *params[1:])
    mx_sq, plane_weight = _canting_weights(mx)

    # Corrección PBC
    N = sites.size
    winding_shift = q * float(N)

    # 1. Intercambio NN (J +/- dJ) y Bicuadrático (K)
    delta1 = np.roll(theta, -1) - theta
    delta1[-1] += winding_shift
    cos1 = np.cos(delta1)
    
    dot1 = mx_sq + plane_weight * cos1
    bonds = np.where((sites & 1) == 0, Jbar + dJ, Jbar - dJ)
    
    exch_nn = float(np.mean(bonds * dot1))
    biquad = float(K * np.mean(dot1**2))

    # 2. Intercambio NNN (J2)
    delta2 = np.roll(theta, -2) - theta
    delta2[-2:] += winding_shift
    dot2 = mx_sq + plane_weight * np.cos(delta2)
    exch_nnn = float(J2 * np.mean(dot2))

    # 3. Anisotropía Axial (D_axis * Sx^2)
    anis_axis = float(D_axis * mx_sq)

    # 4. NUEVO: Anisotropía en el Plano (D_plane * Sy^2)
    # Asumiendo parametrización: Sy = sqrt(1-mx^2) * cos(theta)
    # Esto genera el potencial cos^2(theta) que rompe la simetría rotacional
    Sy_sq_profile = np.cos(theta)**2
    anis_plane = float(D_plane * plane_weight * np.mean(Sy_sq_profile))

    total = float(exch_nn + biquad + exch_nnn + anis_axis + anis_plane)
    
    return total, exch_nn, biquad, exch_nnn, anis_axis, anis_plane


# --- OPTIMIZACIÓN ---

def minimize_modulated_parameters(
    M: float,
    chain_length: int,
    Jbar: float,
    dJ: float,
    J2: float,
    K: float,
    D_axis: float,
    D_plane: float,  # Pasamos el nuevo parámetro
    x0: Sequence[float] | np.ndarray | None = None,
    bounds=DEFAULT_BOUNDS,
    method: str = "L-BFGS-B",
    options=None,
) -> Tuple[float, np.ndarray, bool]:
    
    q = float(_q_from_winding(M, chain_length))
    init = np.zeros(len(PARAM_NAMES)) if x0 is None else np.array(x0, dtype=float)
    sites = np.arange(int(chain_length), dtype=np.int64)

    def objective(vec: np.ndarray) -> float:
        # Retornamos solo la energía total [0]
        return energy_components_modulated_finite(
            q, vec, Jbar, dJ, J2, K, D_axis, D_plane, sites
        )[0]

    res = minimize(
        objective,
        x0=init,
        method=method,
        bounds=bounds,
        options=options or {"maxiter": 400},
    )
    
    if not res.success:
        return float(objective(init)), init, False
    
    params_opt = np.asarray(res.x, dtype=float)
    # Normalizar fases
    for idx, name in enumerate(PARAM_NAMES):
        if name.startswith("phi"):
            params_opt[idx] = _wrap_pi(params_opt[idx])
            
    return float(res.fun), params_opt, True


def e_min_vs_winding_modulated(
    Jbar: float,
    dJ: float,
    J2: float,
    K: float,
    D_axis: float,
    D_plane: float, # Argumento añadido
    chain_length: int,
    M_values: Sequence[int] | np.ndarray | ArrayLike | None = None,
    init_guess = np.array([0.0, -0.3, 0.0, 0.0]),
    bounds=DEFAULT_BOUNDS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    if chain_length <= 0:
        raise ValueError("chain_length must be positive")
    
    if M_values is None:
        M_arr = np.arange(chain_length, dtype=int)
    else:
        M_arr = np.asarray(M_values, dtype=float)
        
    q_arr = np.asarray(_q_from_winding(M_arr, chain_length), dtype=float)
    energies = np.empty_like(q_arr, dtype=float)
    params_hist = np.empty((M_arr.size, len(PARAM_NAMES)), dtype=float)
    success = np.zeros(M_arr.size, dtype=bool)
    
    guess = np.zeros(len(PARAM_NAMES)) if init_guess is None else np.array(init_guess, dtype=float)
    
    for i, (M_val, q_val) in enumerate(zip(M_arr, q_arr)):
        e_val, opt_params, ok = minimize_modulated_parameters(
            M_val, chain_length, Jbar, dJ, J2, K, D_axis, D_plane,
            x0=guess, bounds=bounds,
        )
        energies[i] = e_val
        params_hist[i] = opt_params
        success[i] = ok
        guess = opt_params # Warm start para el siguiente q
        
    return M_arr, q_arr, energies, params_hist, success


# --- ANÁLISIS Y PLOTTING ---

def find_local_minima(q_arr, e_arr, window=1, tol_factor=0.5):
    q = np.asarray(q_arr)
    e = np.asarray(e_arr)
    n = q.size
    dq = 2.0 * math.pi / n
    idxs = []
    
    # Búsqueda básica de mínimos locales
    for i in range(n):
        lefts = [(i - k) % n for k in range(1, window + 1)]
        rights = [(i + k) % n for k in range(1, window + 1)]
        if all(e[i] < e[j] for j in lefts + rights):
            idxs.append(i)
            
    # Fallback
    if len(idxs) == 0 and window == 1:
        for i in range(n):
            if e[i] <= e[(i-1)%n] and e[i] <= e[(i+1)%n]:
                idxs.append(i)
                
    idxs = np.array(idxs, dtype=int)
    if idxs.size == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)
    
    phis_min = q[idxs]
    Umins = e[idxs]
    Emin = np.min(e)
    
    # Filtrado por tolerancia (solo mínimos profundos)
    tol = max(1e-12, tol_factor * abs(Emin) * dq)
    keep = np.where(Umins <= Emin + tol)[0]
    
    if keep.size == 0:
        return phis_min, Umins, idxs
    
    return q[idxs[keep]], e[idxs[keep]], idxs[keep]


def plot_profile_modulated(name, q, e, params=None, param_names=PARAM_NAMES, report=REPORT_PARAMS, q_c=q_c, outdir="profiles", windings=None):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    
    ax.plot(q, e, lw=1.0, label="E_min(q)")
    ax.axvline(q_c, color="k", ls="--", label="q_c")
    
    phis_min, Umins, idxs = find_local_minima(q, e)
    
    if idxs.size > 0:
        ax.scatter(q[idxs], Umins, c="C3", label="min (local)")
        if params is not None:
            report_idx = [param_names.index(p) for p in report if p in param_names]
            for idx in idxs:
                w_str = f"M={windings[idx]:.1f}, " if windings is not None else ""
                msg = f"MIN: {w_str}q={q[idx]:.5f}, E={e[idx]:.6f}"
                
                if report_idx:
                    # Imprimir Alpha_ind para ver si hubo bunching
                    pairs = [f"{param_names[j]}={params[idx, j]:.4f}" for j in report_idx]
                    msg += " | " + ", ".join(pairs)
                print(msg)
                
    ax.set_xlabel("q (rad)")
    ax.set_ylabel("Energy / site")
    ax.set_title(name)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return q[idxs] if idxs.size > 0 else np.array([])


def analyze_sets_modulated(
    sets=DEFAULT_SETS,
    Mq=18001,
    initial_guess: Sequence[float] | np.ndarray | None = None,
    D_plane_val: float = 0.0, # <-- NUEVO ARGUMENTO (Por defecto 0 apaga el efecto)
):
    all_results = []
    minima_q_lists = []
    guess = np.zeros(len(PARAM_NAMES)) if initial_guess is None else np.array(initial_guess, dtype=float)

    print(f"--- Iniciando Análisis con Anisotropía en Plano D_plane = {D_plane_val} ---")

    for raw in sets:
        name, Jbar, dJ, J2, K, D_axis, n_spins = raw # Desempaquetado original
        
        chain_length = int(n_spins)
        print(f"\nProcesando: {name} (N={chain_length})")
        
        t0 = time.time()
        M_max = min(Mq, chain_length)
        
        # Llamada con D_plane_val propagado
        Mvals, qvals, evals, params, success = e_min_vs_winding_modulated(
            Jbar, dJ, J2, K, D_axis, D_plane_val,
            chain_length=chain_length,
            M_values=np.arange(M_max, dtype=int),
            init_guess=guess,
        )
        
        t1 = time.time()
        print(f"Tiempo optimización: {t1 - t0:.1f}s")
        
        # Minimización final en el punto crítico q_c (para reporte)
        best_idx = int(np.argmin(evals))
        guess_next = params[best_idx]
        
        M_qc = chain_length * q_c / (2.0 * math.pi)
        e_qc, params_qc, ok_qc = minimize_modulated_parameters(
            M_qc, chain_length, Jbar, dJ, J2, K, D_axis, D_plane_val,
            x0=guess_next,
        )
        
        # Reporte de valores en q_c
        report_idx = [PARAM_NAMES.index(p) for p in REPORT_PARAMS if p in PARAM_NAMES]
        report_vals = ", ".join(f"{PARAM_NAMES[j]}={params_qc[j]:.4f}" for j in report_idx)
        print(f"En q_c (2pi/3): E={e_qc:.6f} | {report_vals}")
        
        # Plotting
        qmins = plot_profile_modulated(
            f"{name} (D_plane={D_plane_val})", 
            qvals, evals, params=params, windings=Mvals
        )
        
        minima_q_lists.append(qmins)
        all_results.append((name, qvals, evals, params, success, chain_length))
        
        guess = params[best_idx] # Warm start para el siguiente set
        
    return all_results, minima_q_lists


if __name__ == "__main__":
    # Prueba con un valor de anisotropía en el plano para activar el bunching
    # D_plane = 0.5 es un valor razonable para empezar a ver efectos fuertes.
    analyze_sets_modulated(DEFAULT_SETS, Mq=3001, D_plane_val=0)