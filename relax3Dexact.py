#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
relax3Dexact.py
===============
Caracterización analítica y numérica del estado fundamental 3D en cadenas 
cuasi-1D de MoI3 para el régimen extremo D_plane > D (con S = 1.0).

Implementa de forma analítica cerrada (SIN sumatorias sobre paredes k) el
Ansatz Genérico en el Límite Termodinámico (N -> infty) y lo compara contra
el Ansatz Exacto a N Fijo (N = 1198):
  - Fases periódicas:
      u_n       = pi * (n - v_0^asymp) / L_dom^(inf)
      u_tilde_n = pi * (n - v_0^bond)  / L_dom^(inf)
  - Esqueleto planar y quiralidad cerrados:
      theta_n^(0,inf) = Theta_bar_inf + sigma_0 * (q_bulk * L_dom^(inf) / pi) * arcsin(cos(u_n))
      sigma_inf(n)    = -sigma_0 * tanh((L_dom^(inf) / (pi * w_theta_teo)) * sin(u_n))
  - Coordenadas plegadas periódicas:
      d_inf(n) = (L_dom^(inf) / pi) * arcsin(sin(u_n))
      x_inf(n) = (L_dom^(inf) / pi) * arcsin(sin(u_tilde_n))
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, Any

# ==============================================================================
# 1. PARÁMETROS FÍSICOS Y CONFIGURACIÓN (S = 1.0 consistente con LLG)
# ==============================================================================
CONFIG = {
    "N_SITES": 1198,            # Longitud de la cadena
    "S": 1.0,                   # Norma de espín utilizada en la simulación LLG
    "J_NN_1": 91.6861,          # Intercambio dimerizado fuerte (A-B, enlaces pares)
    "J_NN_2": 1.93951,          # Intercambio dimerizado débil (B-A, enlaces impares)
    "J_NNN": 2.60139,           # Intercambio AFM segundo vecino
    "K": 45.4866,               # Intercambio bicuadrático
    "D": 0.76,                  # Anisotropía eje duro longitudinal (Sx^2)
    "D_PLANE_RATIO": 1.5,       # D_plane = 1.5 * D (Régimen no coplanar con flat bands)
    "LOAD_FROM_FILE": True,
    "INPUT_FILE": "D_plane = 0.5D relaxXZ.npy",
    "OUTPUT_TETRAD_FILE": "tetrad_ground_state_1.5D.npz",
    "OUTPUT_ANSATZ_PARAMS": "ansatz_params_infinite_4plus1_1.5D.npz",
    "RUN_RELAXATION": True,
    "RUN_PCA": True,
    "RUN_TORQUE": True,
    "RUN_CHIRALITY": True,
    "RUN_ENERGY": True,
    "RUN_ANSATZ_GLOBAL": True,
    "RUN_TETRAD": True,
    "RUN_PLOTS": True
}

# ==============================================================================
# 2. HAMILTONIANO Y CAMPO EFECTIVO LOCAL (PBC)
# ==============================================================================
def calcular_campo_efectivo(spins: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Calcula el campo efectivo B_eff_i = -dH / dS_i para cada sitio bajo PBC."""
    N = spins.shape[0]
    D = params["D"]
    D_plane = params["D_PLANE_RATIO"] * D
    K = params["K"]
    J_nnn = params["J_NNN"]
    
    J_fwd = np.where(np.arange(N) % 2 == 0, params["J_NN_1"], params["J_NN_2"])
    J_bwd = np.roll(J_fwd, 1)

    sp_1 = np.roll(spins, -1, axis=0)
    sm_1 = np.roll(spins, 1, axis=0)
    sp_2 = np.roll(spins, -2, axis=0)
    sm_2 = np.roll(spins, 2, axis=0)

    dot_fwd = np.sum(spins * sp_1, axis=1, keepdims=True)
    dot_bwd = np.sum(spins * sm_1, axis=1, keepdims=True)

    B_ex = -(J_fwd[:, None] * sp_1 + J_bwd[:, None] * sm_1)
    B_biq = -2.0 * K * (dot_fwd * sp_1 + dot_bwd * sm_1)
    B_nnn = -J_nnn * (sp_2 + sm_2)

    B_aniso = np.zeros_like(spins)
    B_aniso[:, 0] = -2.0 * D * spins[:, 0]
    B_aniso[:, 1] = -2.0 * D_plane * spins[:, 1]
    B_aniso[:, 2] = 0.0

    return B_ex + B_biq + B_nnn + B_aniso


def calcular_energia_total(spins: np.ndarray, params: Dict[str, Any]) -> float:
    """Calcula la energía total clásica del sistema bajo PBC."""
    N = spins.shape[0]
    D = params["D"]
    D_plane = params["D_PLANE_RATIO"] * D
    K = params["K"]
    J_nnn = params["J_NNN"]
    
    J_fwd = np.where(np.arange(N) % 2 == 0, params["J_NN_1"], params["J_NN_2"])
    sp_1 = np.roll(spins, -1, axis=0)
    sp_2 = np.roll(spins, -2, axis=0)

    dot_1 = np.sum(spins * sp_1, axis=1)
    dot_2 = np.sum(spins * sp_2, axis=1)

    E_heis = np.sum(J_fwd * dot_1)
    E_biq = np.sum(K * (dot_1**2))
    E_nnn = np.sum(J_nnn * dot_2)
    E_ani = np.sum(D * (spins[:, 0]**2) + D_plane * (spins[:, 1]**2))

    return float(E_heis + E_biq + E_nnn + E_ani)

# ==============================================================================
# 3. MOTOR DE RELAJACIÓN 3D
# ==============================================================================
def relajar_configuracion_3d(spins_init: np.ndarray, params: Dict[str, Any], 
                             max_iter: int = 40000, tol_tau: float = 1e-6) -> np.ndarray:
    S = params["S"]
    spins = spins_init.copy()
    spins = S * (spins / np.linalg.norm(spins, axis=1, keepdims=True))
    
    dt = 0.0002
    momentum = np.zeros_like(spins)
    damping = 0.94

    print(f"\n[RELAX] Relajando configuración 3D (N={len(spins)}, S={S}, D_plane/D={params['D_PLANE_RATIO']:.2f})...")
    for it in range(max_iter):
        B_eff = calcular_campo_efectivo(spins, params)
        tau = np.cross(spins, B_eff)
        max_tau = np.max(np.linalg.norm(tau, axis=1))

        if max_tau < tol_tau:
            print(f"[RELAX] Convergencia alcanzada en iteración {it} con Max(|Torque|) = {max_tau:.2e} meV")
            break

        proj_B = np.sum(B_eff * spins, axis=1, keepdims=True) / (S**2)
        force_perp = B_eff - proj_B * spins

        if np.sum(momentum * force_perp) < 0.0:
            momentum = np.zeros_like(spins)
            dt = max(dt * 0.75, 1e-5)
        else:
            dt = min(dt * 1.02, 0.0008)

        momentum = damping * momentum + dt * force_perp
        spins += momentum
        spins = S * (spins / np.linalg.norm(spins, axis=1, keepdims=True))

        if it % 5000 == 0 and it > 0:
            E_curr = calcular_energia_total(spins, params)
            print(f"  Iter {it:6d}: Max(|Torque|) = {max_tau:.2e} meV | E_total = {E_curr:.5f} meV")

    return spins

# ==============================================================================
# 4. MÉTODOS DE AUDITORÍA Y DIAGNÓSTICO
# ==============================================================================
def analizar_elipsoide_pca(spins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = len(spins)
    T = (spins.T @ spins) / N
    eigvals, eigvecs = np.linalg.eigh(T)
    idx_sort = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sort]
    eigvecs = eigvecs[:, idx_sort]

    print("\n" + "="*70)
    print("MÉTODO 1: ANÁLISIS DE EJES PRINCIPALES Y DIMENSIONALIDAD (PCA)")
    print("="*70)
    print(f"Momentos cuadráticos: λ_1={eigvals[0]:.6f}, λ_2={eigvals[1]:.6f}, λ_3={eigvals[2]:.6f}")
    print(f"Norma cuadrática media <||S_i||^2> : {np.sum(eigvals):.6f}")
    print(f"Eje normal de mínima varianza (v3) : [{eigvecs[0,2]:.4f}, {eigvecs[1,2]:.4f}, {eigvecs[2,2]:.4f}]")
    if eigvals[2] < 1e-4:
        print("-> CONCLUSIÓN PCA: La configuración es COPLANAR (2D).")
    else:
        print("-> CONCLUSIÓN PCA: La configuración es GENUINAMENTE NO-COPLANAR (3D).")
    print("="*70)
    return eigvals, eigvecs


def verificar_torque_equilibrio(spins: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    B_eff = calcular_campo_efectivo(spins, params)
    tau = np.cross(spins, B_eff)
    tau_norm = np.linalg.norm(tau, axis=1)
    max_tau = np.max(tau_norm)
    mean_tau = np.mean(tau_norm)

    print("\n" + "="*70)
    print("MÉTODO 2: AUDITORÍA DE CONVERGENCIA MECÁNICA (TORQUE RESIDUAL)")
    print("="*70)
    print(f"Torque máximo en la cadena : {max_tau:.4e} meV")
    print(f"Torque promedio RMS        : {mean_tau:.4e} meV")
    if max_tau < 1e-3:
        print("-> ESTADO ESTACIONARIO CONVERGIDO: Válido para teoría de ondas de espín.")
    else:
        print("-> ADVERTENCIA: Torque residual aún apreciable.")
    print("="*70)
    return tau_norm


def analizar_quiralidad_3d(spins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sp_1 = np.roll(spins, -1, axis=0)
    sm_1 = np.roll(spins, 1, axis=0)
    kappa = np.cross(spins, sp_1)
    chi = np.sum(sm_1 * kappa, axis=1)
    chi_rms = float(np.sqrt(np.mean(chi**2)))
    max_chi = float(np.max(np.abs(chi)))

    print("\n" + "="*70)
    print("MÉTODO 3: QUIRALIDAD MAGNÉTICA Y TOPOLOGÍA LOCAL")
    print("="*70)
    print(f"Quiralidad escalar RMS (χ_rms) : {chi_rms:.6e}")
    print(f"Quiralidad escalar Máxima      : {max_chi:.6e}")
    print("="*70)
    return kappa, chi


def calcular_densidad_energia(spins: np.ndarray, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    N = len(spins)
    D = params["D"]
    D_plane = params["D_PLANE_RATIO"] * D
    K = params["K"]
    J_nnn = params["J_NNN"]
    
    J_fwd = np.where(np.arange(N) % 2 == 0, params["J_NN_1"], params["J_NN_2"])
    sp_1 = np.roll(spins, -1, axis=0)
    sp_2 = np.roll(spins, -2, axis=0)

    dot_1 = np.sum(spins * sp_1, axis=1)
    dot_2 = np.sum(spins * sp_2, axis=1)

    E_heis = J_fwd * dot_1
    E_biq = K * (dot_1**2)
    E_nnn = J_nnn * dot_2
    E_ani_x = D * (spins[:, 0]**2)
    E_ani_y = D_plane * (spins[:, 1]**2)
    E_tot = E_heis + E_biq + E_nnn + E_ani_x + E_ani_y

    print("\n" + "="*70)
    print("MÉTODO 4: DESGLOSE DE DENSIDAD DE ENERGÍA LOCAL")
    print("="*70)
    print(f"Energía media por espín     : {np.mean(E_tot):.5f} meV")
    print(f"  Contribución Heisenberg   : {np.mean(E_heis):.5f} meV")
    print(f"  Contribución Bicuadrática : {np.mean(E_biq):.5f} meV")
    print(f"  Contribución Anisotropía X: {np.mean(E_ani_x):.5f} meV")
    print(f"  Contribución Anisotropía Y: {np.mean(E_ani_y):.5f} meV")
    print("="*70)
    return {
        "E_total": E_tot, "E_heis": E_heis, "E_biq": E_biq,
        "E_nnn": E_nnn, "E_ani_x": E_ani_x, "E_ani_y": E_ani_y
    }

# ==============================================================================
# 5. MÉTODO 5: ANSATZ CERRADO N -> INFTY (SIN SUMATORIAS) VS. ANSATZ FINITO N FIJO
# ==============================================================================
def extraer_y_validar_ansatz_global(spins: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    N = len(spins)
    S = params["S"]
    ratio_K_J1 = params["K"] / params["J_NN_1"]
    idx = np.arange(N, dtype=float)
    staggered = (-1.0) ** np.arange(N)

    Sx = spins[:, 0]
    Sy = spins[:, 1]
    Sz = spins[:, 2]

    theta_raw = np.arctan2(Sx, Sz)
    theta_unwrapped = np.unwrap(theta_raw)
    phi_raw = np.arcsin(np.clip(Sy / S, -1.0, 1.0))

    # --- A. EXTRACCIÓN DE LOS 4 + 1 PARÁMETROS Y POSICIONES DEL SISTEMA FINITO ---
    peaks_pos, _ = find_peaks(theta_unwrapped, distance=50, prominence=30)
    peaks_neg, _ = find_peaks(-theta_unwrapped, distance=50, prominence=30)
    v_init = np.sort(np.concatenate([peaks_pos, peaks_neg]))
    num_walls = len(v_init)

    wall_margin = 7
    bounds = np.concatenate([[0], v_init, [N - 1]])
    slopes_dom = np.zeros(num_walls + 1)
    intercepts_dom = np.zeros(num_walls + 1)
    gamma_dom = np.zeros(num_walls + 1)

    for k in range(num_walls + 1):
        n_start = bounds[k] if k == 0 else bounds[k] + wall_margin
        n_end = bounds[k + 1] if k == num_walls else bounds[k + 1] - wall_margin
        if n_end - n_start < 6:
            n_start, n_end = bounds[k], bounds[k + 1]
        sub_n = np.arange(n_start, n_end + 1)
        A_loc = np.column_stack([sub_n, np.ones_like(sub_n), staggered[sub_n]])
        c_loc, _, _, _ = np.linalg.lstsq(A_loc, theta_unwrapped[sub_n], rcond=None)
        slopes_dom[k] = c_loc[0]
        intercepts_dom[k] = c_loc[1]
        gamma_dom[k] = abs(c_loc[2])

    v_asymp_real = np.zeros(num_walls)
    for k in range(num_walls):
        v_asymp_real[k] = -(intercepts_dom[k + 1] - intercepts_dom[k]) / (slopes_dom[k + 1] - slopes_dom[k])

    v_bond_real = np.zeros(num_walls)
    for k, v_a in enumerate(v_asymp_real):
        i_min = max(0, int(np.floor(v_a)) - 5)
        i_max = min(N - 1, int(np.ceil(v_a)) + 5)
        win = np.arange(i_min, i_max + 1)
        w_e = phi_raw[win] ** 2
        v_cm = np.sum(win * w_e) / (np.sum(w_e) + 1e-15)
        v_bond_real[k] = np.floor(v_cm) + 0.5

    sigma_0 = 1.0 if slopes_dom[0] >= 0.0 else -1.0
    idx_l0 = max(0, int(round(v_bond_real[0] - 1.5)))
    idx_r0 = min(N - 1, int(round(v_bond_real[0] + 1.5)))
    eta_0 = 1.0 if (phi_raw[idx_l0] + phi_raw[idx_r0]) >= 0.0 else -1.0
    eta_k = np.array([eta_0 * ((-1.0) ** k) for k in range(num_walls)])
    tau_k = np.array([-sigma_0 * ((-1.0) ** k) for k in range(num_walls)])

    # Los 4 parámetros del Bulk Planar
    q_bulk = float(np.mean(np.abs(slopes_dom[1:-1])))
    gamma_bulk_est = float(np.mean(gamma_dom[1:-1]))
    Theta_0_finite = float(intercepts_dom[0])

    bulk_mask = np.ones(N, dtype=bool)
    for v_b in v_bond_real:
        bulk_mask &= (np.abs(idx - v_b) > 5.0)

    delta_q_real = np.diff(slopes_dom)
    theta_0_finite = Theta_0_finite + slopes_dom[0] * idx
    for k in range(num_walls):
        theta_0_finite += delta_q_real[k] * np.maximum(0.0, idx - v_asymp_real[k])

    w_th_init = 0.25 * ((2.0 * gamma_bulk_est * np.cos(gamma_bulk_est)) ** 2) * 2.65
    sigma_finite = np.full(N, slopes_dom[0] / q_bulk)
    for k in range(num_walls):
        dist_a = idx - v_asymp_real[k]
        sigma_finite += (delta_q_real[k] / (2.0 * q_bulk)) * (
            np.tanh(np.clip(dist_a / w_th_init, -40.0, 40.0)) + 1.0
        )

    arg_bunch_fin = 2.0 * theta_0_finite + 0.5 * sigma_finite * q_bulk
    M_th_fin = np.column_stack([
        sigma_finite * staggered,
        np.sin(arg_bunch_fin),
        sigma_finite * staggered * np.cos(arg_bunch_fin)
    ])
    coeffs_th, _, _, _ = np.linalg.lstsq(M_th_fin[bulk_mask], (theta_unwrapped - theta_0_finite)[bulk_mask], rcond=None)
    gamma_chiral = float(coeffs_th[0])
    alpha_bunch = float(coeffs_th[1])
    beta_bunch = float(coeffs_th[2])
    gamma_abs = abs(gamma_chiral)

    # Constantes esclavizadas de la pared (0 parámetros libres)
    Phi_1_teo = float(2.0 * gamma_abs * np.cos(gamma_abs))
    Phi_stag_teo = float(0.5 * np.sin(gamma_abs))
    Phi_3_teo = float(ratio_K_J1 * np.cos(gamma_abs) * Phi_1_teo)

    alpha_1_teo = -np.pi
    alpha_2_teo = -(0.5 * q_bulk - 0.25 * gamma_abs)
    alpha_3_teo = np.pi - (2.0 / 3.0) * q_bulk

    def construir_phi_finito(w_ph: float, v_bond_arr: np.ndarray) -> np.ndarray:
        ph_mod = np.zeros(N)
        for k in range(num_walls):
            pol = eta_k[k]
            dx = idx - v_bond_arr[k]
            abs_dx = np.abs(dx)
            sech_1 = 1.0 / np.cosh(np.clip(dx / w_ph, -40.0, 40.0))
            sech_3 = sech_1 ** 3
            s_bond = np.where(dx >= 0.0, 1.0, -1.0) * staggered

            ph_mod += pol * (
                sech_1 * (
                    Phi_1_teo * np.cos(q_bulk * abs_dx - alpha_1_teo)
                    + Phi_stag_teo * s_bond * np.cos(q_bulk * abs_dx - alpha_2_teo)
                )
                + sech_3 * Phi_3_teo * np.cos(3.0 * q_bulk * abs_dx - alpha_3_teo)
            )
        return ph_mod

    # Único parámetro libre de la pared: w_phi
    res_opt_ph = minimize_scalar(
        lambda w: float(np.sqrt(np.mean((phi_raw - construir_phi_finito(w, v_bond_real)) ** 2))),
        bounds=(1.5, 7.0),
        method="bounded"
    )
    w_phi_opt = float(res_opt_ph.x)

    w_theta_teo = 0.25 * (Phi_1_teo ** 2) * w_phi_opt
    W_wall_teo = 0.25 * (Phi_1_teo ** 2)
    A_piv_teo = 2.0 * gamma_abs - W_wall_teo

    # ==========================================================================
    # B. ANSATZ GENÉRICO CERRADO EN EL LÍMITE N -> INFTY (CERO BUCLES FOR)
    # ==========================================================================
    # 1. Período intrínseco L_dom^(inf) = 76 sitios y orígenes v_0^bond, v_0^asymp
    lambda_bunch = (2.0 * np.pi) / (2.0 * q_bulk - np.pi)
    P_moire = 13                                             # 13 ondas de Moiré = 75.65 sitios
    M_inf = int(np.round((P_moire * lambda_bunch) / 2.0))    # M_inf = 38 dímeros
    L_dom_inf = float(2 * M_inf)                             # L_dom^(inf) = 76.0 sitios
    K_inf = M_inf + P_moire                                  # K_inf = 51 semivueltas

    v_0_bond_inf = 0.5 * L_dom_inf - 1.5                     # = 36.5
    delta_v_piv_teo = np.pi / (2.0 * q_bulk)                 # = 0.74422 sitios
    v_0_asymp_inf = v_0_bond_inf + sigma_0 * delta_v_piv_teo # = 37.24422 sitios

    # Posiciones equiespaciadas de referencia (únicamente para graficar líneas verticales)
    k_arr = np.arange(num_walls, dtype=float)
    v_bond_inf = v_0_bond_inf + k_arr * L_dom_inf            # 36.5 + 76*k
    v_asymp_inf = v_0_asymp_inf + k_arr * L_dom_inf          # 37.244 + 76*k (100% equiespaciados)

    # 2. Fases periódicas reducidas u_n (vértice) y u_tilde_n (enlace)
    u_n = np.pi * (idx - v_0_asymp_inf) / L_dom_inf
    u_tilde_n = np.pi * (idx - v_0_bond_inf) / L_dom_inf

    # 3. Coordenadas plegadas periódicas y cargas topológicas (FÓRMULAS CERRADAS)
    d_inf = (L_dom_inf / np.pi) * np.arcsin(np.sin(u_n))
    x_inf = (L_dom_inf / np.pi) * np.arcsin(np.sin(u_tilde_n))
    abs_d_inf = np.abs(d_inf)
    abs_x_inf = np.abs(x_inf)

    tau_inf = -sigma_0 * np.where(np.cos(u_n) >= 0.0, 1.0, -1.0)
    eta_inf = eta_0 * np.where(np.cos(u_tilde_n) >= 0.0, 1.0, -1.0)
    s_inf = np.where(np.sin(u_tilde_n) >= 0.0, 1.0, -1.0) * staggered

    # 4. Esqueleto planar cerrado theta_n^(0,inf) y quiralidad cerrada sigma_inf(n)
    K_vert_0 = np.round((np.pi + sigma_0 * q_bulk * v_0_asymp_inf) / np.pi)  # K_vert_0 = 26
    Theta_vert_0_inf = K_vert_0 * np.pi + 0.5 * (gamma_abs - W_wall_teo)
    Theta_bar_inf = float(Theta_vert_0_inf - 0.5 * sigma_0 * q_bulk * L_dom_inf)

    theta_0_inf = Theta_bar_inf + sigma_0 * (q_bulk * L_dom_inf / np.pi) * np.arcsin(np.cos(u_n))
    sigma_inf = -sigma_0 * np.tanh((L_dom_inf / (np.pi * w_theta_teo)) * np.sin(u_n))
    Theta_0_inf = float(theta_0_inf[0])

    # 5. Corrección planar de pared cerrada delta_theta_n^(wall,inf) (SIN BUCLE FOR)
    dv_piv_int = abs(v_0_asymp_inf - np.round(v_0_asymp_inf))  # = 0.24422 sitios
    sech2_d_inf = (1.0 / np.cosh(np.clip(abs_d_inf / w_theta_teo, -40.0, 40.0))) ** 2
    sech2_piv_inf = (1.0 / np.cosh(np.clip(dv_piv_int / w_theta_teo, -40.0, 40.0))) ** 2
    sech2_x_inf = (1.0 / np.cosh(np.clip(abs_x_inf / w_phi_opt, -40.0, 40.0))) ** 2

    delta_theta_wall_inf = -tau_inf * (
        A_piv_teo * (sech2_d_inf / sech2_piv_inf)
        + W_wall_teo * sech2_x_inf * np.cos(2.0 * q_bulk * abs_x_inf + 0.5 * gamma_abs)
    )

    arg_bunch_inf = 2.0 * theta_0_inf + 0.5 * sigma_inf * q_bulk
    theta_bulk_corr_inf = (
        gamma_chiral * sigma_inf * staggered
        + alpha_bunch * np.sin(arg_bunch_inf)
        + beta_bunch * sigma_inf * staggered * np.cos(arg_bunch_inf)
    )
    theta_fit_inf = theta_0_inf + delta_theta_wall_inf + theta_bulk_corr_inf
    delta_theta_inf = theta_unwrapped - theta_fit_inf

    # 6. Solitón de Bloch cerrado phi_n^(inf) (SIN BUCLE FOR)
    sech_1_inf = 1.0 / np.cosh(np.clip(x_inf / w_phi_opt, -40.0, 40.0))
    sech_3_inf = sech_1_inf ** 3

    phi_fit_inf = eta_inf * (
        sech_1_inf * (
            -Phi_1_teo * np.cos(q_bulk * x_inf)
            + Phi_stag_teo * s_inf * np.cos(q_bulk * abs_x_inf - alpha_2_teo)
        )
        + Phi_3_teo * sech_3_inf * np.cos(3.0 * q_bulk * abs_x_inf - alpha_3_teo)
    )
    delta_phi_inf = phi_raw - phi_fit_inf

    spins_ans_inf = S * np.column_stack([
        np.sin(theta_fit_inf) * np.cos(phi_fit_inf),
        np.sin(phi_fit_inf),
        np.cos(theta_fit_inf) * np.cos(phi_fit_inf)
    ])
    diff_vec_inf = np.linalg.norm(spins - spins_ans_inf, axis=1)

    win_inf = slice(0, 250)
    rms_th_inf_win = float(np.sqrt(np.mean(delta_theta_inf[win_inf] ** 2)))
    rms_ph_inf_win = float(np.sqrt(np.mean(delta_phi_inf[win_inf] ** 2)))
    rms_vec_inf_win = float(np.sqrt(np.mean(diff_vec_inf[win_inf] ** 2)))

    # ==========================================================================
    # C. MODELO 2: ANSATZ EXACTO A N FIJO (CAJA FINITA N = 1198 CON SUMATORIAS)
    # ==========================================================================
    phi_fit_fin = construir_phi_finito(w_phi_opt, v_bond_real)
    delta_phi_fin = phi_raw - phi_fit_fin

    delta_theta_wall_fin = np.zeros(N)
    sigma_fin_exact = np.full(N, slopes_dom[0] / q_bulk)
    for k in range(num_walls):
        dist_a = idx - v_asymp_real[k]
        abs_da = np.abs(dist_a)
        sigma_fin_exact += (delta_q_real[k] / (2.0 * q_bulk)) * (
            np.tanh(np.clip(dist_a / w_theta_teo, -40.0, 40.0)) + 1.0
        )
        tk = tau_k[k]
        dv_piv_k = np.min(abs_da)
        sech2_d = (1.0 / np.cosh(np.clip(abs_da / w_theta_teo, -40.0, 40.0))) ** 2
        sech2_piv = (1.0 / np.cosh(np.clip(dv_piv_k / w_theta_teo, -40.0, 40.0))) ** 2

        dx = idx - v_bond_real[k]
        abs_dx = np.abs(dx)
        sech2_x = (1.0 / np.cosh(np.clip(dx / w_phi_opt, -40.0, 40.0))) ** 2

        delta_theta_wall_fin += -tk * (
            A_piv_teo * (sech2_d / sech2_piv)
            + W_wall_teo * sech2_x * np.cos(2.0 * q_bulk * abs_dx + 0.5 * gamma_abs)
        )

    M_th_final = np.column_stack([
        sigma_fin_exact * staggered,
        np.sin(2.0 * theta_0_finite + 0.5 * sigma_fin_exact * q_bulk),
        sigma_fin_exact * staggered * np.cos(2.0 * theta_0_finite + 0.5 * sigma_fin_exact * q_bulk)
    ])
    theta_fit_fin = theta_0_finite + delta_theta_wall_fin + (M_th_final @ coeffs_th)
    delta_theta_fin = theta_unwrapped - theta_fit_fin

    spins_ans_fin = S * np.column_stack([
        np.sin(theta_fit_fin) * np.cos(phi_fit_fin),
        np.sin(phi_fit_fin),
        np.cos(theta_fit_fin) * np.cos(phi_fit_fin)
    ])
    rms_err_fin = float(np.sqrt(np.mean(np.linalg.norm(spins - spins_ans_fin, axis=1) ** 2)))

    pinning_shift_bond = v_bond_real - v_bond_inf
    pinning_shift_asymp = v_asymp_real - v_asymp_inf

    # --- D. REPORTE COMPARATIVO EN TERMINAL ---
    print("\n" + "=" * 86)
    print("MÉTODO 5: ANSATZ CERRADO N -> INFTY (SIN SUMATORIAS) vs. ANSATZ FINITO (N = 1198)")
    print("=" * 86)
    print("1. PARÁMETROS GEOMÉTRICOS DEL ANSATZ CERRADO N -> INFTY (Ecuaciones periódicas):")
    print(f"   - Longitud de dominio intrínseca L_dom^(inf): 2 * {M_inf} = {L_dom_inf:.2f} sitios")
    print(f"   - Origen de enlace v_0^bond y vértice v_0^a : {v_0_bond_inf:.2f} y {v_0_asymp_inf:.4f} sitios")
    print(f"   - Nivel medio planar Theta_bar_inf          : {Theta_bar_inf:.6f} rad (theta_0^(inf)(0) = {Theta_0_inf:.6f} rad)")
    print(f"   - Vértices equiespaciados v_asymp^inf (1-4) : {[float(round(v, 2)) for v in v_asymp_inf[:4]]}")
    print(f"   - Vértices reales caja    v_asymp^real (1-4): {[float(round(v, 2)) for v in v_asymp_real[:4]]}")
    print(f"   - Salto sub-dímero en asíntotas (real - inf): {[float(round(d, 2)) for d in pinning_shift_asymp[:4]]} sitios")
    print(f"   - Centros de enlace v_bond^inf (primeros 6) : {[float(round(v, 2)) for v in v_bond_inf[:6]]}")
    print(f"   - Centros de enlace v_bond^real (primeros 6): {[float(round(v, 2)) for v in v_bond_real[:6]]}")
    print(f"   - Salto inter-dímero en enlaces (real - inf): {[float(round(d, 2)) for d in pinning_shift_bond[:6]]} sitios")
    print("-" * 86)
    print("2. LOS 4 + 1 PARÁMETROS FÍSICOS GLOBALES DEL SISTEMA:")
    print(f"   [Bulk 1] Vector Q helicoidal (q_bulk)       : {q_bulk:.6f} rad/sitio")
    print(f"   [Bulk 2] Dimerización quiral (gamma)        : {gamma_chiral:.6f} rad (|gamma| = {gamma_abs:.6f} rad)")
    print(f"   [Bulk 3] Spin-Bunching uniforme (alpha)     : {alpha_bunch:.6f} rad")
    print(f"   [Bulk 4] Spin-Bunching Moiré (beta)         : {beta_bunch:.6f} rad")
    print(f"   [Wall 1] Ancho del solitón 3D (w_phi)       : {w_phi_opt:.4f} sitios")
    print(f"   [Esclavizados] Phi_1 = {Phi_1_teo:.5f} | Phi_stag = {Phi_stag_teo:.5f} | Phi_3 = {Phi_3_teo:.5f} rad")
    print("-" * 86)
    print("3. COMPARACIÓN DE RESIDUOS: FÓRMULA CERRADA N -> INFTY vs. SUMATORIA A N FIJO:")
    print(f"   * FÓRMULA CERRADA N -> INFTY (arcsin/tanh periódicos sin bucles, n in [0, 250]):")
    print(f"       -> RMS(delta_theta^inf) : {rms_th_inf_win:.5e} rad (refleja salto de 1.49 sitios en k=1,2)")
    print(f"       -> RMS(delta_phi^inf)   : {rms_ph_inf_win:.5e} rad")
    print(f"       -> Error Vectorial 3D   : {rms_vec_inf_win:.5e} ({100.0 * rms_vec_inf_win / S:.2f}%)")
    print(f"   * ANSATZ EXACTO A N FIJO (Toda la cadena N = 1198 con 9x74 + 7x76 sitios):")
    print(f"       -> RMS(delta_theta^fin) : {np.sqrt(np.mean(delta_theta_fin**2)):.5e} rad")
    print(f"       -> RMS(delta_phi^fin)   : {np.sqrt(np.mean(delta_phi_fin**2)):.5e} rad")
    print(f"       -> Error Vectorial 3D   : {rms_err_fin:.5e} ({100.0 * rms_err_fin / S:.2f}%)")
    print("=" * 86)

    return {
        "q": q_bulk,
        "Theta_0_inf": Theta_0_inf,
        "Theta_0_fin": Theta_0_finite,
        "L_dom_inf": L_dom_inf,
        "w_theta": w_theta_teo,
        "gamma_chiral": gamma_chiral,
        "alpha_bunch": alpha_bunch,
        "beta_bunch": beta_bunch,
        "W_wall_amp": W_wall_teo,
        "A_piv_amp": A_piv_teo,
        "w_phi": w_phi_opt,
        "Phi_1": Phi_1_teo,
        "Phi_stag": Phi_stag_teo,
        "Phi_3": Phi_3_teo,
        "eta_k": eta_k,
        "theta_unwrapped": theta_unwrapped,
        "theta_fit_inf": theta_fit_inf,
        "delta_theta_inf": delta_theta_inf,
        "theta_fit_fin": theta_fit_fin,
        "delta_theta_fin": delta_theta_fin,
        "phi_raw": phi_raw,
        "phi_fit_inf": phi_fit_inf,
        "delta_phi_inf": delta_phi_inf,
        "phi_fit_fin": phi_fit_fin,
        "delta_phi_fin": delta_phi_fin,
        "v_bond_inf": v_bond_inf,
        "v_bond_real": v_bond_real,
        "v_asymp_inf": v_asymp_inf,
        "v_asymp_real": v_asymp_real,
        "pinning_shift": pinning_shift_bond,
        "rms_error_inf": rms_vec_inf_win,
        "rms_error_fin": rms_err_fin
    }

# ==============================================================================
# 6. MÉTODO 6: TÉTRADA LOCAL ORTONORMAL PARA LSWT
# ==============================================================================
def construir_tetrada_local(spins: np.ndarray, S: float, filename_npz: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(spins)
    e3 = spins / np.linalg.norm(spins, axis=1, keepdims=True)
    e1 = np.zeros_like(spins)
    e2 = np.zeros_like(spins)

    aux = np.array([0.0, 1.0, 0.0])
    for i in range(N):
        if np.abs(np.dot(e3[i], aux)) > 0.90:
            v_ref = np.array([1.0, 0.0, 0.0])
        else:
            v_ref = aux

        v1 = v_ref - np.dot(v_ref, e3[i]) * e3[i]
        e1[i] = v1 / np.linalg.norm(v1)
        e2[i] = np.cross(e3[i], e1[i])

    if filename_npz:
        np.savez_compressed(filename_npz, spins=spins, e1=e1, e2=e2, e3=e3, S=S)
        print(f"\n[EXPORT] Tétrada local guardada con éxito en: '{filename_npz}'")

    return e1, e2, e3

# ==============================================================================
# 7. GRÁFICOS DIAGNÓSTICOS: DIFERENCIACIÓN FRENTE A FÓRMULA CERRADA N -> INFTY
# ==============================================================================
def graficar_ansatz_infinito(ansatz_res: Dict[str, Any], n_range: Tuple[int, int] = (0, 360)):
    idx = np.arange(len(ansatz_res["theta_unwrapped"]))
    r_min, r_max = n_range
    sub_idx = idx[r_min:r_max]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11.5), sharex=True)

    # Panel 1: Fase planar theta_n vs Ansatz Cerrado Infinito
    axes[0].plot(sub_idx, ansatz_res["theta_unwrapped"][r_min:r_max], label=r"$\theta_n$ numérico (LLG, $S=1$)", color="tab:blue", lw=1.6)
    axes[0].plot(sub_idx, ansatz_res["theta_fit_inf"][r_min:r_max], label=r"Fórmula Cerrada $N\to\infty$ ($\arcsin(\cos u_n)$, $L_{\mathrm{dom}}^{(\infty)}=76$)", color="black", ls="--", lw=1.4)
    for k, (v_r, v_i) in enumerate(zip(ansatz_res["v_bond_real"], ansatz_res["v_bond_inf"])):
        if r_min <= v_r <= r_max or r_min <= v_i <= r_max:
            axes[0].axvline(v_r, color="red", ls=":", alpha=0.6, label=r"Pared real ($v_k^{\mathrm{real}}$)" if k == 0 else "")
            axes[0].axvline(v_i, color="darkorange", ls="-.", alpha=0.6, label=r"Pared equiespaciada ($\infty$)" if k == 0 else "")
    axes[0].set_ylabel(r"$\theta_n$ [rad]", fontsize=12)
    axes[0].set_title(r"Diferenciación entre la Fórmula Cerrada Periódica ($N\to\infty$) y el Ansatz a $N$ Fijo ($N=1198$)", fontsize=13.5, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9.5)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Diferencia delta_theta_n (Fórmula Cerrada Infinita vs Anclaje Finito)
    axes[1].plot(sub_idx, ansatz_res["delta_theta_inf"][r_min:r_max], label=r"$\delta\theta_n^{(\infty)} = \theta_n - \theta_n^{(\infty)}$ (Fórmula Cerrada Equiespaciada $N\to\infty$)", color="tab:orange", lw=1.3, alpha=0.9)
    axes[1].plot(sub_idx, ansatz_res["delta_theta_fin"][r_min:r_max], label=r"$\delta\theta_n^{(\mathrm{finito})} = \theta_n - \theta_n^{(\mathrm{finito})}$ (Ansatz a $N$ fijo)", color="tab:purple", lw=1.1)
    axes[1].axhline(0, color="k", ls=":", alpha=0.5)
    axes[1].set_ylabel(r"$\delta\theta_n$ [rad]", fontsize=12)
    axes[1].legend(loc="upper right", fontsize=9.5)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Ángulo de escape phi_n vs Fórmula Cerrada Infinita
    axes[2].plot(sub_idx, ansatz_res["phi_raw"][r_min:r_max], label=r"$\phi_n$ numérico (LLG, $S=1$)", color="tab:red", lw=1.3)
    axes[2].plot(sub_idx, ansatz_res["phi_fit_inf"][r_min:r_max], label=r"Fórmula Cerrada $\phi_n^{(\infty)}$ ($x_\infty(n) = \frac{L}{\pi}\arcsin(\sin\tilde{u}_n)$)", color="black", ls="--", lw=1.2)
    axes[2].set_ylabel(r"$\phi_n$ [rad]", fontsize=12)
    axes[2].legend(loc="upper right", fontsize=9.5)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Diferencia delta_phi_n (Fórmula Cerrada Infinita vs Anclaje Finito)
    axes[3].plot(sub_idx, ansatz_res["delta_phi_inf"][r_min:r_max], label=r"$\delta\phi_n^{(\infty)} = \phi_n - \phi_n^{(\infty)}$ (Fórmula Cerrada Equiespaciada $N\to\infty$)", color="darkorange", lw=1.4, alpha=0.85)
    axes[3].plot(sub_idx, ansatz_res["delta_phi_fin"][r_min:r_max], label=r"$\delta\phi_n^{(\mathrm{finito})} = \phi_n - \phi_n^{(\mathrm{finito})}$ (Ansatz a $N$ fijo)", color="darkgreen", lw=1.1, ls="--")
    axes[3].axhline(0, color="k", ls=":", alpha=0.5)
    axes[3].set_ylabel(r"$\delta\phi_n$ [rad]", fontsize=12)
    axes[3].set_xlabel(r"Sitio atómico $n$", fontsize=12)
    axes[3].legend(loc="upper right", fontsize=9.5)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 8. FLUJO DE EJECUCIÓN PRINCIPAL
# ==============================================================================
def main():
    print("="*70)
    print("   AUDITORÍA DEL ANSATZ CERRADO N -> INFTY vs. N FIJO (S=1.0)")
    print("="*70)

    N = CONFIG["N_SITES"]
    S = CONFIG["S"]
    spins = None

    if CONFIG["LOAD_FROM_FILE"] and os.path.exists(CONFIG["INPUT_FILE"]):
        try:
            print(f"[CARGA] Leyendo archivo: '{CONFIG['INPUT_FILE']}'...")
            raw_data = np.load(CONFIG["INPUT_FILE"])
            spins = raw_data[-1].copy() if raw_data.ndim == 3 else raw_data.copy()
            spins = S * (spins / np.linalg.norm(spins, axis=1, keepdims=True))
            print(f"[CARGA] Arreglo cargado exitosamente con norma S={S:.1f}. Shape: {spins.shape}")
        except Exception as e:
            print(f"[CARGA] Error al leer archivo ({e}). Se procederá a relajar.")

    if spins is None:
        print("[INIT] Inicializando espiral con perturbación out-of-plane...")
        q_init = 2.11031
        idx = np.arange(N)
        spins = np.zeros((N, 3))
        spins[:, 0] = S * np.sin(q_init * idx)
        spins[:, 2] = S * np.cos(q_init * idx)
        spins[:, 1] = S * 0.08 * np.random.randn(N)
        spins = S * (spins / np.linalg.norm(spins, axis=1, keepdims=True))

        if CONFIG["RUN_RELAXATION"]:
            spins = relajar_configuracion_3d(spins, CONFIG)
            np.save(CONFIG["INPUT_FILE"], spins)
            print(f"[GUARDAR] Configuración guardada en: '{CONFIG['INPUT_FILE']}'")

    if CONFIG["RUN_PCA"]:
        analizar_elipsoide_pca(spins)

    if CONFIG["RUN_TORQUE"]:
        verificar_torque_equilibrio(spins, CONFIG)

    if CONFIG["RUN_CHIRALITY"]:
        analizar_quiralidad_3d(spins)

    if CONFIG["RUN_ENERGY"]:
        calcular_densidad_energia(spins, CONFIG)

    ansatz_res = None
    if CONFIG["RUN_ANSATZ_GLOBAL"]:
        ansatz_res = extraer_y_validar_ansatz_global(spins, CONFIG)
        if CONFIG["OUTPUT_ANSATZ_PARAMS"]:
            np.savez_compressed(
                CONFIG["OUTPUT_ANSATZ_PARAMS"],
                q=ansatz_res["q"],
                Theta_0_inf=ansatz_res["Theta_0_inf"],
                L_dom_inf=ansatz_res["L_dom_inf"],
                w_theta=ansatz_res["w_theta"],
                gamma_chiral=ansatz_res["gamma_chiral"],
                alpha_bunch=ansatz_res["alpha_bunch"],
                beta_bunch=ansatz_res["beta_bunch"],
                W_wall_amp=ansatz_res["W_wall_amp"],
                A_piv_amp=ansatz_res["A_piv_amp"],
                w_phi=ansatz_res["w_phi"],
                Phi_1=ansatz_res["Phi_1"],
                Phi_stag=ansatz_res["Phi_stag"],
                Phi_3=ansatz_res["Phi_3"],
                v_bond_inf=ansatz_res["v_bond_inf"],
                v_bond_real=ansatz_res["v_bond_real"],
                pinning_shift=ansatz_res["pinning_shift"]
            )
            print(f"[EXPORT] Parámetros guardados en: '{CONFIG['OUTPUT_ANSATZ_PARAMS']}'")

    if CONFIG["RUN_TETRAD"]:
        construir_tetrada_local(spins, S, filename_npz=CONFIG["OUTPUT_TETRAD_FILE"])

    if CONFIG["RUN_PLOTS"] and (ansatz_res is not None):
        graficar_ansatz_infinito(ansatz_res, n_range=(0, 360))


if __name__ == "__main__":
    main()