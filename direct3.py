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
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter

# --- CONSTANTES GLOBALES ---
q_c = 2.0 * math.pi / 3.0

# Formato: (Name, Jbar, dJ, J2, K, D_axis, N)
# D_axis: Anisotropía longitudinal (Canting)

DEFAULT_SETS = [
    #("Set 1", 48.891035, 48.620365, 1.26126, 48.9119, 0.31,1198),
    ("MoI3", 46.812805, 44.873295, 2.60139, 45.4866, 0.76, 10000),
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
    
    # Perfil base + Dimerización + Armónicos m*q y 2*m*q (Soliton Lattice extendido)
    profile = base + gamma * parity
    profile += alpha_ind * np.sin(2.0 * q * idx + phi_ind)
    return profile


def _canting_weights(mx: float) -> Tuple[float, float]:
    mx_sq = mx * mx
    plane_weight = max(0.0, 1.0 - mx_sq)
    return mx_sq, plane_weight


def _format_fraction(numer: int, denom: int) -> str:
    if denom == 0:
        return "0"
    return f"{numer}*2π/{denom}"


def _apply_scientific_axes(ax: Axes, axes: str = "y", power_limits: tuple[int, int] = (-2, 2)) -> None:
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits(power_limits)
    fmt.set_useOffset(False)

    if "x" in axes:
        ax.xaxis.set_major_formatter(fmt)
    if "y" in axes:
        ax.yaxis.set_major_formatter(fmt)


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

def global_objective(vec: np.ndarray, q, Jbar, dJ, J2, K, D_axis, D_plane, sites) -> float:
    """
    Función objetivo global requerida para evitar errores de serialización (pickling)
    en Windows al paralelizar con SciPy (workers=-1).
    """
    return energy_components_modulated_finite(
        q, vec, Jbar, dJ, J2, K, D_axis, D_plane, sites
    )[0]


def minimize_modulated_parameters(
    M: float,
    chain_length: int,
    Jbar: float,
    dJ: float,
    J2: float,
    K: float,
    D_axis: float,
    D_plane: float,
    x0: Sequence[float] | np.ndarray | None = None,
    bounds=DEFAULT_BOUNDS,
    method: str = "L-BFGS-B",
    options=None,
) -> Tuple[float, np.ndarray, bool]:
    
    q = float(_q_from_winding(M, chain_length))
    init = np.zeros(len(PARAM_NAMES)) if x0 is None else np.array(x0, dtype=float)
    sites = np.arange(int(chain_length), dtype=np.int64)

    # Empaquetamos los argumentos de forma segura
    extra_args = (q, Jbar, dJ, J2, K, D_axis, D_plane, sites)

    res = minimize(
        global_objective,
        x0=init,
        args=extra_args,
        method=method,
        bounds=bounds,
        options=options or {"maxiter": 400, "workers": -1}, # <- Activado paralelismo de CPU
    )
    
    if not res.success:
        return float(global_objective(init, *extra_args)), init, False
    
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
    D_plane: float,
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
    keep = np.where(Umins < Emin + tol)[0]
    
    return phis_min[keep], Umins[keep], idxs[keep]


if __name__ == "__main__":
    # Prueba con un valor de anisotropía en el plano para activar el bunching
    # D_plane = 0.5 es un valor razonable para empezar a ver efectos fuertes.
    analyze_sets_modulated(DEFAULT_SETS, D_plane_val=0.76 * 0.5 )