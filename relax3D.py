#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Analisis de estados LLG relajados (MoX3, N = 1188) -- version limpia.

Calculo directo de la barrera de movimiento de una pared con el Hamiltoniano COMPLETO H(1):
  - se toma la pared del estado LLG tal cual (nucleo escapado incluido),
  - se desplaza su posicion con una coordenada colectiva
        A = sum_{n en ventana} sin(theta_{n+1} - theta_n)   (lineal en la posicion del pliegue),
  - en cada paso se relaja TODO el anillo (L-BFGS-B, gradiente analitico) con penalizacion
    sobre A, y se reporta la energia H(1) verdadera (sin la penalizacion).
No usa Floquet, ni funcional FK, ni constantes de calibracion.

Salidas: dE(d) para d en [-DMAX, DMAX] sitios, barrera B+ (hacia el vecino derecho) y B-
(hacia el vecino izquierdo), phi_max del estado, y la figura Figure_Barrier.pdf.
Tambien se conservan las figuras de perfiles (Fig. 1) y asimetria YZ/XZ (Fig. 3).
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

CONFIG = {
    "S": 1.0,
    "J_NN_1": 91.6861,
    "J_NN_2": 1.93951,
    "J_NNN": 2.60139,
    "K": 45.4866,
    "D": 0.76,
    "Q_BULK": 2.11031,

    # --- interruptores ---
    "RUN_BARRIER": False,
    "PLOT_FIG1_PROFILES": False,
    "PLOT_PHI_COMPARISON": True,
    "PLOT_FIG3_ASYMMETRY": False,

    # --- barrera ---
    "BARRIER_INCLUDE_NOISE": False,
    "SCAN_STEP": 0.5,
    "SCAN_DMAX": 3.0,
    "KPEN": 2.0e4,
    "WIN_HALF": 30,

    # --- figura 1 ---
    "INPUT_FILE_1": "D_plane = 2.2D relaxYZ n = 1188 noise=0.04 seed3.npy",
    "INPUT_FILE_2": "D_plane = 2.2D relaxYZ n = 1188.npy",
    "D_PLANE_RATIO": 2.2,
    "PHI_ESCAPE_TO_Y": True,

    # --- comparación de phi_n ---
    "PHI_COMPARISON_PLANE": "XZ",
    "PHI_COMPARISON_FILES": [
        "D_plane = 2.2D relaxYZ n = 1188 noise=1e-3.npy",
        "D_plane = 1.65D relaxYZ n = 1188 noise=1e-3.npy"
    ],

    # --- figura 3 ---
    "ASYM_FILE_YZ": "D_plane = 0.9D relax.npy",
    "ASYM_FILE_XZ": "D_plane = 0.9D relax.npy",
}
# (archivo, D_plane/D, escape_y)   escape_y=True -> plano XZ (phi hacia y); False -> plano YZ (phi hacia x)
_RAT = [(1.1, True), (1.2, True), (1.3, True), (1.4, True), (1.5, True), (1.55, True), (1.6, True),
        (1.65, True), (1.7, True), (1.8, True), (2.2, True), (2.8, True), (3.5, True), (5.0, True)]
_RAT_YZ = [0.3, 0.5, 0.7]


def _nombre(r, tipo):
    s = ("%gD" % r)
    return "D_plane = %s %s n = 1188.npy" % (s, tipo)


MUESTRAS = [(_nombre(r, "relaxXZ"), r, False) for r in _RAT_YZ]
MUESTRAS += [(_nombre(r, "relaxYZ"), r, esc) for r, esc in _RAT]
MUESTRAS_RUIDO = [(f.replace(".npy", " noise=1e-3.npy"), r, e) for f, r, e in MUESTRAS]

def resolver_ruta(nombre):
    if os.path.exists(nombre):
        return nombre
    for orig, alt in [("relax n = 1188.npy", "relax.npy"), ("relax.npy", "relax n = 1188.npy")]:
        cand = nombre.replace(orig, alt)
        if os.path.exists(cand):
            return cand
    return None


def cargar_estado_unico(ruta, spin_norm=1.0):
    raw = np.load(ruta)
    spins = raw[-1].copy() if raw.ndim == 3 else raw.copy()
    return spin_norm * spins / np.linalg.norm(spins, axis=1, keepdims=True)


def cargar_par_archivos(f1, f2, spin_norm=1.0):
    return cargar_estado_unico(f1, spin_norm), cargar_estado_unico(f2, spin_norm)


def extraer_angulos(spins, spin_norm, escape_y):
    """escape_y=True: plano XZ (phi hacia y). escape_y=False: plano YZ (phi hacia x)."""
    theta = np.unwrap(np.arctan2(spins[:, 0] if escape_y else spins[:, 1], spins[:, 2]))
    phi_c = spins[:, 1] if escape_y else spins[:, 0]
    phi = np.arcsin(np.clip(phi_c / spin_norm, -1.0, 1.0))
    return theta, phi


def calcular_energia_por_sitio(spins, ratio_d, cfg):
    d_plane = ratio_d * cfg["D"]
    j_fwd = np.where(np.arange(len(spins)) % 2 == 0, cfg["J_NN_1"], cfg["J_NN_2"])
    dot_1 = np.sum(spins * np.roll(spins, -1, axis=0), axis=1)
    dot_2 = np.sum(spins * np.roll(spins, -2, axis=0), axis=1)
    return (
        j_fwd * dot_1
        + cfg["K"] * (dot_1**2)
        + cfg["J_NNN"] * dot_2
        + cfg["D"] * (spins[:, 0] ** 2)
        + d_plane * (spins[:, 1] ** 2)
    )


def detectar_solitones(theta, phi):
    n = len(theta)
    dth = np.angle(np.exp(1j * (np.roll(theta, -1) - theta)))
    pendiente = np.convolve(np.pad(dth, 4, mode="wrap"), np.ones(4) / 4, mode="same")[4:-4]

    idx = np.arange(n)
    q_lw = 2.0 * np.pi / n
    phi_med = np.median([np.roll(phi, -o) for o in range(-7, 8)], axis=0)
    X = np.column_stack([np.ones(n), np.cos(q_lw * idx), np.sin(q_lw * idx)])
    phi_lw = X @ np.linalg.lstsq(X, phi_med, rcond=None)[0]
    dphi = phi - phi_lw

    cruces = []
    for i in range(n):
        if pendiente[i] * pendiente[(i + 1) % n] <= 0:
            if abs(pendiente[(i + 3) % n] - pendiente[(i - 2) % n]) > 0.8:
                if not cruces or min(abs(i - cruces[-1]), n - abs(i - cruces[-1])) > 3:
                    cruces.append(i)

    solitones = []
    usados = set()
    for c in cruces:
        candidatos = [off for off in range(-3, 4) if ((c + off) % n) % 2 == 1 and ((c + off) % n) not in usados]
        if not candidatos:
            candidatos = [off for off in range(-3, 4) if ((c + off) % n) not in usados]
        best_b = max(
            candidatos,
            key=lambda off: sum(
                abs(dphi[(c + off - m) % n] * dphi[(c + off + 1 + m) % n]) for m in range(3)
            ),
        )
        b = (c + best_b) % n
        usados.add(b)
        win = [(b + k) % n for k in range(-3, 5)]
        signo = float(np.sign(dphi[b] + dphi[(b + 1) % n])) or 1.0
        solitones.append({"core_bond": b, "x0": b + 0.5, "phi0": float(np.max(np.abs(dphi[win]))), "signo": signo})

    solitones.sort(key=lambda s: s["x0"])
    return solitones, phi_lw


# ----------------------------------------------------------------------------
# Hessiano exacto del modo phi y mu_dim medido
# ----------------------------------------------------------------------------


def graficar_figura_1_perfiles(cfg):
    f1_m = resolver_ruta(cfg["INPUT_FILE_1"])
    f2_m = resolver_ruta(cfg["INPUT_FILE_2"])
    if not f1_m or not f2_m:
        print(f"[Warning] Profile files not found: {cfg['INPUT_FILE_1']} or {cfg['INPUT_FILE_2']}")
        return

    s1, s2 = cargar_par_archivos(f1_m, f2_m, cfg["S"])
    ratio_d = cfg["D_PLANE_RATIO"]
    escape_y = cfg["PHI_ESCAPE_TO_Y"]

    n = len(s1)
    idx = np.arange(n)
    theta, phi = extraer_angulos(s1, cfg["S"], escape_y)
    solitones, phi_lw = detectar_solitones(theta, phi)
    dphi = phi - phi_lw

    e1 = calcular_energia_por_sitio(s1, ratio_d, cfg)
    e2 = calcular_energia_por_sitio(s2, ratio_d, cfg)
    diff_energia = np.abs(e1 - e2)

    w_wind = int(np.round(cfg["Q_BULK"] * n / (2.0 * np.pi)))
    ciclos_moire = abs(3 * w_wind - n)
    n_sol_moire = 2 * ciclos_moire
    delta_q = (ciclos_moire * 2.0 * np.pi) / n
    q_lw = 2.0 * np.pi / n
    l0 = n / n_sol_moire

    x0_arr = np.array([s["x0"] for s in solitones])
    if len(x0_arr) > 0:
        k_near = np.round((x0_arr - x0_arr[0]) / l0)
        x0_ref = np.mean(x0_arr - k_near * l0)
        if len(x0_arr) == n_sol_moire:
            u_k = x0_arr - (x0_ref + np.arange(n_sol_moire) * l0)
            X_u = np.column_stack([np.ones_like(x0_arr), np.cos(q_lw * x0_arr), np.sin(q_lw * x0_arr)])
            c_u = np.linalg.lstsq(X_u, u_k, rcond=None)[0]
            u_lw = c_u[0] + c_u[1] * np.cos(q_lw * idx) + c_u[2] * np.sin(q_lw * idx)
        else:
            u_lw = 0.0
        signo_env = solitones[0]["signo"]
    else:
        x0_ref, u_lw, signo_env = 0.0, 0.0, 1.0

    amp_env = np.max(np.abs(dphi))
    envolvente = phi_lw + signo_env * amp_env * np.sin(delta_q * (idx - u_lw - x0_ref))

    fig, axes = plt.subplots(3, 1, figsize=(13, 10.5), clear=True)

    axes[0].plot(idx, theta, color="tab:blue", lw=1.5, label=rf"Numerical $\theta_n$ (LLG, $N_s = {len(solitones)}$)")
    axes[0].set_ylabel(r"$\theta_n$ [rad]", fontsize=11)
    axes[0].set_title(rf"Chirality Inversion and Polar Soliton ($D_{{\mathrm{{plane}}}} = {ratio_d}D$)", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9.5)

    axes[1].plot(idx, phi, color="tab:red", lw=1.0, alpha=0.85, label=r"Numerical $\phi_n$ (LLG)")
    axes[1].plot(idx, phi_lw, color="tab:cyan", lw=1.4, linestyle="-.", label=r"Long-Wavelength Background $\phi_{\mathrm{LW}}(n)$ ($Q_{\mathrm{LW}} = 2\pi/N$)")
    axes[1].plot(
        idx,
        envolvente,
        color="tab:purple",
        lw=1.8,
        linestyle="--",
        label=r"Moiré Envelope $\phi_{\mathrm{LW}}(n) + \mathcal{A}\sin(\Delta q[n - u_{\mathrm{LW}}(n)])$",
    )
    for s in solitones:
        axes[1].axvline(s["x0"], color="black", linestyle=":", alpha=0.55)
    axes[1].set_ylabel(r"$\phi_n$ [rad]", fontsize=11)
    axes[1].set_title(r"Polar Profile $\phi_n$ and Long-Wavelength Modulated Moiré Envelope ($\Delta q$)", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9.5)

    axes[2].plot(idx, diff_energia, color="tab:green", lw=0.9, label=r"$|E_{i,1} - E_{i,2}|$")
    axes[2].set_ylabel(r"$|E_{i,1} - E_{i,2}|$ [meV]", fontsize=11)
    axes[2].set_xlabel(r"Atomic site $n$", fontsize=11)
    axes[2].set_title(r"Absolute Local Energy Difference: $|E_{i,1} - E_{i,2}|$", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=9.5)

    fig.tight_layout()
    fig.savefig("Figure1_Profiles_Moire.pdf", dpi=300)


def graficar_comparacion_phi(cfg):
    mode = cfg["PHI_COMPARISON_PLANE"].upper()
    if mode not in {"XZ", "YZ"}:
        raise ValueError("PHI_COMPARISON_PLANE debe ser 'XZ' o 'YZ'")

    archivos = cfg["PHI_COMPARISON_FILES"]
    if not 1 <= len(archivos) <= 3:
        raise ValueError("PHI_COMPARISON_FILES debe contener entre 1 y 3 archivos")

    escape_y = mode == "XZ"
    fig, ax = plt.subplots(figsize=(13, 5.5), clear=True)
    colores = ("tab:blue", "tab:orange", "tab:green")
    procesados = 0

    for nombre in archivos:
        ruta = resolver_ruta(nombre)
        if not ruta:
            print(f"[Warning] Comparison file not found: {nombre}")
            continue

        spins = cargar_estado_unico(ruta, cfg["S"])
        idx = np.arange(len(spins))
        _, phi = extraer_angulos(spins, cfg["S"], escape_y)
        ax.plot(idx, phi, color=colores[procesados], lw=1.1, label=os.path.basename(nombre))
        procesados += 1

    if not procesados:
        print("[Warning] No comparison files were processed.")
        plt.close(fig)
        return

    ax.axhline(0.0, color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"Atomic site $n$", fontsize=11)
    ax.set_ylabel(rf"$\phi_n^{{({mode})}}$ [rad]", fontsize=11)
    ax.set_title(rf"Comparison of $\phi_n$ profiles in the ${mode}$ plane", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"Figure_Compare_Phi_{mode}.pdf", dpi=300)


# ----------------------------------------------------------------------------
# Figura 2
# ----------------------------------------------------------------------------


def graficar_figura_3_asimetria(cfg):
    f_yz = resolver_ruta(cfg["ASYM_FILE_YZ"])
    f_xz = resolver_ruta(cfg["ASYM_FILE_XZ"])
    if not f_yz or not f_xz:
        print(f"[Warning] Asymmetry files not found: {cfg['ASYM_FILE_YZ']} or {cfg['ASYM_FILE_XZ']}")
        return

    s_yz = cargar_estado_unico(f_yz, cfg["S"])
    s_xz = cargar_estado_unico(f_xz, cfg["S"])
    n = len(s_yz)
    idx = np.arange(n)

    _, phi_yz = extraer_angulos(s_yz, cfg["S"], escape_y=False)
    _, phi_xz = extraer_angulos(s_xz, cfg["S"], escape_y=True)

    w_wind = int(np.round(cfg["Q_BULK"] * n / (2.0 * np.pi)))
    ciclos_moire = abs(3 * w_wind - n)
    n_sol_moire = 2 * ciclos_moire
    delta_q = (ciclos_moire * 2.0 * np.pi) / n
    l0 = n / n_sol_moire

    z_proj = -np.sum((phi_yz**2) * np.exp(1j * 2.0 * delta_q * idx))
    x0_ref = (np.angle(z_proj) / (2.0 * delta_q)) % l0
    nodos_moire = x0_ref + np.arange(n_sol_moire) * l0

    amp_yz = float(np.max(np.abs(phi_yz)))
    envolvente_pos = amp_yz * np.sin(delta_q * (idx - x0_ref))

    fig3, axes = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True, clear=True)

    axes[0].plot(idx, phi_yz, color="tab:red", lw=0.9, alpha=0.85, label=r"Remnant $\phi_n^{(YZ)}$ (LLG in $YZ$ plane)")
    axes[0].plot(
        idx,
        envolvente_pos,
        color="tab:purple",
        lw=1.8,
        linestyle="--",
        label=rf"Moiré Envelope $\pm\mathcal{{A}}\sin(\Delta q [n - x_0])$ ($N_s = {n_sol_moire}$ nodes)",
    )
    axes[0].plot(idx, -envolvente_pos, color="tab:purple", lw=1.8, linestyle="--", alpha=0.6)
    for x_node in nodos_moire:
        axes[0].axvline(x_node, color="black", linestyle=":", alpha=0.5, lw=1.0)

    axes[0].set_ylabel(r"$\phi_n^{(YZ)}$ [rad]", fontsize=11)
    axes[0].set_title(r"(a) Basal State in $YZ$ Plane", fontsize=16.5, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=15.5)

    axes[1].plot(idx, phi_xz, color="tab:blue", lw=0.9, alpha=0.85, label=r"Remnant $\phi_n^{(XZ)}$ (LLG in $XZ$ plane)")
    axes[1].axhline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.5)
    axes[1].set_ylabel(r"$\phi_n^{(XZ)}$ [rad]", fontsize=11)
    axes[1].set_xlabel(r"Atomic site $n$", fontsize=11)
    axes[1].set_title(r"(b) Basal State in $XZ$ Plane", fontsize=16.5, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=15.5)

    fig3.tight_layout()
    fig3.savefig("Figure_Asymmetry_Phi.pdf", dpi=300)


# ----------------------------------------------------------------------------
# Hamiltoniano completo: energia y gradiente analitico en (theta, phi)
# ----------------------------------------------------------------------------
def energia_y_gradiente(v, n, jb, d_plane, escape_y, cfg):
    """H(1) y dH/d(theta, phi). Parametrizacion (3) [XZ, escape_y=True] o (4) [YZ]."""
    jn, k_b, d_ax = cfg["J_NNN"], cfg["K"], cfg["D"]
    th, ph = v[:n], v[n:]
    c, s = np.cos(ph), np.sin(ph)
    if escape_y:
        S = np.stack([c * np.sin(th), s, c * np.cos(th)], 1)
        dth = np.stack([c * np.cos(th), 0 * th, -c * np.sin(th)], 1)
        dph = np.stack([-s * np.sin(th), c, -s * np.cos(th)], 1)
    else:
        S = np.stack([s, c * np.sin(th), c * np.cos(th)], 1)
        dth = np.stack([0 * th, c * np.cos(th), -c * np.sin(th)], 1)
        dph = np.stack([c, -s * np.sin(th), -s * np.cos(th)], 1)
    d1 = np.sum(S * np.roll(S, -1, 0), 1)
    d2 = np.sum(S * np.roll(S, -2, 0), 1)
    E = np.sum(jb * d1 + k_b * d1**2 + jn * d2 + d_ax * S[:, 0] ** 2 + d_plane * S[:, 1] ** 2)
    a = (jb + 2.0 * k_b * d1)[:, None]
    G = a * np.roll(S, -1, 0) + np.roll(a, 1, 0) * np.roll(S, 1, 0) + jn * (np.roll(S, -2, 0) + np.roll(S, 2, 0))
    G[:, 0] += 2.0 * d_ax * S[:, 0]
    G[:, 1] += 2.0 * d_plane * S[:, 1]
    return E, np.concatenate([np.sum(G * dth, 1), np.sum(G * dph, 1)])


def _relajar(f, v, args=()):
    r = minimize(f, v, args=args, jac=True, method="L-BFGS-B",
                 options={"maxiter": 30000, "maxfun": 60000, "gtol": 1e-10, "ftol": 1e-17})
    return r.x, r.fun, r.success


def barrido_barrera(spins, ratio_d, escape_y, cfg):
    """Barrera de movimiento de una pared medida con H(1) completo (ver docstring del modulo)."""
    n = len(spins)
    th, ph = extraer_angulos(spins, cfg["S"], escape_y)
    solitones, _ = detectar_solitones(th, ph)
    if len(solitones) < 3:
        return None
    centros = [s["core_bond"] for s in solitones]
    dist_der = [(centros[(i + 1) % len(centros)] - centros[i]) % n for i in range(len(centros))]
    dist_izq = [(centros[i] - centros[i - 1]) % n for i in range(len(centros))]
    med = float(np.median(dist_der))
    k = int(np.argmin([abs(dist_der[i] - med) + abs(dist_izq[i] - med) for i in range(len(centros))]))
    b = centros[k]

    jb = np.where(np.arange(n) % 2 == 0, cfg["J_NN_1"], cfg["J_NN_2"])
    d_plane = ratio_d * cfg["D"]
    f0 = lambda v: energia_y_gradiente(v, n, jb, d_plane, escape_y, cfg)
    v0, e0, ok0 = _relajar(f0, np.concatenate([th, ph]))

    m = cfg["WIN_HALF"]
    ii = np.arange(b - m, b + m) % n
    jj = (ii + 1) % n

    def A_of(v):
        return float(np.sum(np.sin(v[jj] - v[ii])))

    # pendiente del valle izquierdo (con signo): sitios b-20 .. b-8
    izq = np.arange(b - 20, b - 8) % n
    a_izq = float(np.mean(np.sin(v0[(izq + 1) % n] - v0[izq])))
    A0 = A_of(v0)
    kpen = cfg["KPEN"]

    def f_pen(v, At):
        E, g = energia_y_gradiente(v, n, jb, d_plane, escape_y, cfg)
        A = A_of(v)
        gA = np.zeros(2 * n)
        c = np.cos(v[jj] - v[ii])
        np.add.at(gA, jj, c)
        np.add.at(gA, ii, -c)
        return E + 0.5 * kpen * (A - At) ** 2, g + kpen * (A - At) * gA

    pasos = np.arange(cfg["SCAN_STEP"], cfg["SCAN_DMAX"] + 1e-9, cfg["SCAN_STEP"])
    curvas = {}
    for sgn in (+1, -1):
        v, d_list, de_list, conv = v0.copy(), [0.0], [0.0], True
        for d in pasos:
            At = A0 + sgn * 2.0 * a_izq * d
            v, _, ok = _relajar(f_pen, v, args=(At,))
            conv &= ok
            E, _ = energia_y_gradiente(v, n, jb, d_plane, escape_y, cfg)
            d_list.append(sgn * d)
            de_list.append(E - e0)
        curvas[sgn] = (np.array(d_list), np.array(de_list), conv)

    def barrera(de):
        i = int(np.argmax(de))
        return float(de[i]), bool(i < len(de) - 1)   # (altura, pico_resuelto)

    b_der, pico_der = barrera(curvas[+1][1])
    b_izq, pico_izq = barrera(curvas[-1][1])
    es_ruido = False
    return {
        "ratio": ratio_d, "x_aniso": abs(ratio_d - 1.0), "rama": "XZ" if escape_y else "YZ",
        "n_sol": len(solitones), "pared": int(b), "L_izq": int(dist_izq[k]), "L_der": int(dist_der[k]),
        "phi_max": float(np.max(np.abs(v0[n:]))), "conv": bool(ok0 and curvas[1][2] and curvas[-1][2]),
        "B_der": b_der, "pico_der": pico_der, "B_izq": b_izq, "pico_izq": pico_izq,
        "dE_der": curvas[+1], "dE_izq": curvas[-1],
        "pendiente_valle": a_izq,
    }


def calcular_barreras(cfg):
    lista = list(MUESTRAS) + (list(MUESTRAS_RUIDO) if cfg.get("BARRIER_INCLUDE_NOISE") else [])
    res = []
    for fname, ratio_d, esc_y in lista:
        ruta = resolver_ruta(fname)
        if not ruta:
            print("[Warning] File not found, skipping: %s" % fname)
            continue
        r = barrido_barrera(cargar_estado_unico(ruta, cfg["S"]), ratio_d, esc_y, cfg)
        if r is None:
            print("[Warning] <3 walls detected, skipping: %s" % fname)
            continue
        r["archivo"] = fname
        r["es_ruido"] = "noise" in fname.lower()
        res.append(r)
        print("  %s %-5g phi_max=%.2e  B_der=%.3e%s  B_izq=%.3e%s  conv=%s" % (
            r["rama"], ratio_d, r["phi_max"], r["B_der"], "" if r["pico_der"] else "+",
            r["B_izq"], "" if r["pico_izq"] else "+", r["conv"]), flush=True)
    if not res:
        print("[Warning] No samples processed.")
        return
    res.sort(key=lambda r: (r["rama"], r["x_aniso"], r["es_ruido"]))
    print("\n" + "=" * 112)
    print("  WALL-MOTION BARRIER FROM THE FULL HAMILTONIAN (energies in meV; '+' = peak not reached within DMAX)")
    print("=" * 112)
    print("%-6s | %-8s | %-6s | %-3s | %-9s | %-4s | %-4s | %-10s | %-13s | %-13s | %-5s" % (
        "Branch", "D/D", "x", "Ns", "phi_max", "L_l", "L_r", "slope", "B_left", "B_right", "conv"))
    print("-" * 112)
    for r in res:
        print("%-6s | %-8g | %-6.3f | %-3d | %-9.2e | %-4d | %-4d | %-10.4f | %-13s | %-13s | %-5s" % (
            r["rama"], r["ratio"], r["x_aniso"], r["n_sol"], r["phi_max"], r["L_izq"], r["L_der"],
            r["pendiente_valle"],
            "%.4e%s" % (r["B_izq"], "" if r["pico_izq"] else "+"),
            "%.4e%s" % (r["B_der"], "" if r["pico_der"] else "+"), str(r["conv"])))
    print("=" * 112)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), clear=True)
    for rama, mk in (("XZ", "o"), ("YZ", "s")):
        sub = [r for r in res if r["rama"] == rama and not r["es_ruido"]]
        if not sub:
            continue
        x = [r["x_aniso"] for r in sub]
        axes[0].plot(x, [max(r["B_der"], 1e-12) for r in sub], mk + "-", label="%s: B right" % rama)
        axes[0].plot(x, [max(r["B_izq"], 1e-12) for r in sub], mk + "--", label="%s: B left" % rama, alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$|D_{\mathrm{plane}}-D|/D$")
    axes[0].set_ylabel("Wall-motion barrier [meV]")
    axes[0].set_title("Barrier from full $H$ (collective coordinate)", fontweight="bold")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    for r in [q for q in res if q["rama"] == "XZ" and not q["es_ruido"]][::2]:
        d = np.concatenate([r["dE_izq"][0][::-1], r["dE_der"][0][1:]])
        e = np.concatenate([r["dE_izq"][1][::-1], r["dE_der"][1][1:]])
        axes[1].plot(d, e, "o-", ms=3, label="%gD" % r["ratio"])
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_xlabel("wall displacement [sites]")
    axes[1].set_ylabel(r"$H(d) - H(0)$ [meV]")
    axes[1].set_title("Energy along the displacement path", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("Figure_Barrier.pdf", dpi=300)


def main():
    if CONFIG.get("RUN_BARRIER", False):
        calcular_barreras(CONFIG)
    if CONFIG.get("PLOT_FIG1_PROFILES", False):
        graficar_figura_1_perfiles(CONFIG)
    if CONFIG.get("PLOT_PHI_COMPARISON", False):
        graficar_comparacion_phi(CONFIG)
    if CONFIG.get("PLOT_FIG3_ASYMMETRY", False):
        graficar_figura_3_asimetria(CONFIG)
    plt.show()


if __name__ == "__main__":
    main()