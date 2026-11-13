#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustez de la regularidad y test de fase de Moire (proxy LLG, sigma = 1e-3).

Mismo protocolo que flopproxy.py (estado base YZ relajado, ruido gaussiano en S_x con media restada, Dplane -> (1+x) D,
Euler explicito o punto medio implicito con renormalizacion, alpha = 1.3, 1 fs), pero guarda las POSICIONES de las paredes de cada corrida
(cache: ensemble_paredes.pkl; con REUSE = True no se vuelve a simular). Ademas de las N_SEEDS semillas con ruido, corre
una corrida limpia (sigma = 0) por cada x.

Salidas:
 [1] Tabla I completa por x: <Ns> +- std, P(Ns=18), mediana de delta_L, P(delta_L < 0.15).
 [2] Robustez de x_c^reg (ajuste logistico en ln x + bootstrap 68%) frente a: corte de delta_L (0.10, 0.15, 0.20, 0.25),
     delta_L recortado (se quitan los dos extremos de todo par de paredes a menos de LMIN_TRIM sitios: artefactos del
     detector), dispersion robusta (1.4826 MAD / mediana) y R2 = |<exp(2 i dq x_k)>| > 0.8.
 [3] Test de fase de Moire: la fuente coherente es ~ cos(s dq n + 3 theta0); theta0 se mide en el estado base, de modo que
     los nodos (zeros de la fuente, separados pi/dq = 66 sitios) quedan predichos sin ajustar. Para cada corrida se mide el
     desfase psi de la red de paredes respecto de los nodos y su dispersion rms alrededor de ese desfase. Un array
     "bloqueado a Moire" tiene rms pequeno (aleatorio: 19 sitios) y psi igual entre corridas y valores de x.
     Se reportan las dos variantes de signo (v = +1, -1); s es el sentido de giro de la helice base.
Requiere relax3D.py y el archivo base en la misma carpeta.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

from relax3D import CONFIG, detectar_solitones, extraer_angulos
from rot_methods import implicit_midpoint_step

BASE_FILE = "D_plane = D relaxYZ n = 1188.npy"
X_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 4.0]
SIGMA = 1e-3
N_SEEDS = 12
ALPHA = 1.3
DT = 0.003                       # eta * dt en meV^-1 (equivale a 1 fs)
USE_EXPLICIT_EULER = True       # False: punto medio implicito
N, W = 1188, 399
DQ = 2 * np.pi * (3 * W - N) / N
PER = np.pi / abs(DQ)            # separacion de nodos de Moire (66 sitios)
CACHE = ("ensemble_paredes.pkl" if USE_EXPLICIT_EULER
         else "ensemble_paredes_implicit_midpoint.pkl")
REUSE = True
N_PROC = max(1, (os.cpu_count() or 2) // 2)
CUTOFFS = (0.10, 0.15, 0.20, 0.25)
LMIN_TRIM = 8
X_LOCK_MIN = 1.2                 # x desde el cual se agregan las estadisticas de fase

J1, J2, JN, K, D = CONFIG["J_NN_1"], CONFIG["J_NN_2"], CONFIG["J_NNN"], CONFIG["K"], CONFIG["D"]


def n_pasos(x, sigma):
    L = min(20.0, max(7.0, np.log(1.5 / max(sigma, 1e-12))))
    return int(min(40000, max(6000, 1.5 * L / (2.0 * x * D * DT))))


def cargar_base(ruta):
    raw = np.load(ruta, mmap_mode="r")
    s = np.array(raw[-1] if raw.ndim == 3 else raw, dtype=float)
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def campo(S, Jb, Dp):
    d1 = (S * np.roll(S, -1, 0)).sum(1)
    a = (Jb + 2 * K * d1)[:, None]
    G = a * np.roll(S, -1, 0) + np.roll(a, 1, 0) * np.roll(S, 1, 0) + JN * (np.roll(S, -2, 0) + np.roll(S, 2, 0))
    G[:, 0] += 2 * D * S[:, 0]
    G[:, 1] += 2 * Dp * S[:, 1]
    return G


def correr(tarea):
    x, seed, sigma = tarea
    S = cargar_base(BASE_FILE)
    n = len(S)
    Jb = np.where(np.arange(n) % 2 == 0, J1, J2)
    if sigma > 0:
        rng = np.random.default_rng(seed)
        nz = sigma * rng.standard_normal(n)
        nz -= nz.mean()
        S[:, 0] += nz
        S /= np.linalg.norm(S, axis=1, keepdims=True)
    Dp = (1 + x) * D
    campo_actual = lambda spins: campo(spins, Jb, Dp)
    for _ in range(n_pasos(x, max(sigma, SIGMA))):
        if USE_EXPLICIT_EULER:
            G = campo_actual(S)
            SG = np.cross(S, G)
            S = S + DT * (-SG + ALPHA * np.cross(S, SG))
        else:
            S = implicit_midpoint_step(S, DT, campo_actual, ALPHA, True)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
    if not np.mean(S[:, 0] ** 2) > np.mean(S[:, 1] ** 2):
        return (x, seed, sigma, None)
    th, ph = extraer_angulos(S, 1.0, True)
    c = np.array(sorted(s["core_bond"] for s in detectar_solitones(th, ph)[0]), dtype=float)
    return (x, seed, sigma, c)


# ------------------------------- metricas ---------------------------------
def espaciados(c):
    return np.diff(np.r_[c, c[0] + N])


def dl(c):
    L = espaciados(c)
    return L.std() / L.mean()


def dl_trim(c):
    c = np.sort(np.asarray(c, float))
    while len(c) > 3:
        L = espaciados(c)
        i = int(np.argmin(L))
        if L[i] >= LMIN_TRIM:
            break
        c = np.delete(c, [i, (i + 1) % len(c)])
    return dl(c) if len(c) > 3 else np.nan


def dl_rob(c):
    L = espaciados(c)
    return 1.4826 * np.median(np.abs(L - np.median(L))) / np.median(L)


def r2(c):
    return abs(np.mean(np.exp(2j * DQ * c)))


CRITERIOS = {f"dL<{cut:.2f}": (lambda c, cut=cut: dl(c) < cut) for cut in CUTOFFS}
CRITERIOS["dL_recortado<0.15"] = lambda c: dl_trim(c) < 0.15
CRITERIOS["dL_robusto<0.15"] = lambda c: dl_rob(c) < 0.15
CRITERIOS["R2>0.8"] = lambda c: r2(c) > 0.8


def ajuste_logistico(x, k, n):
    np.seterr(over="ignore")
    x, k, n = np.asarray(x, float), np.asarray(k, float), np.asarray(n, float)

    def nll(p, kk):
        P = 1 / (1 + np.exp(-(np.log(x) - p[0]) / np.exp(p[1])))
        P = np.clip(P, 1e-9, 1 - 1e-9)
        return -np.sum(kk * np.log(P) + (n - kk) * np.log(1 - P))

    def fit(kk):
        return np.exp(minimize(nll, [np.log(1.0), 0.0], args=(kk,), method="Nelder-Mead").x[0])

    xc = fit(k)
    rng = np.random.default_rng(0)
    bs = np.array([fit(rng.binomial(n.astype(int), k / n)) for _ in range(500)])
    return xc, np.percentile(bs, [16, 84])


# ---------------------------- fase de los nodos -----------------------------
def fase_base():
    S = cargar_base(BASE_FILE)
    th = np.arctan2(S[:, 1], S[:, 2])                            # plano YZ
    s = 1 if np.mean(np.angle(np.exp(1j * (np.roll(th, -1) - th)))) > 0 else -1
    n = np.arange(N)
    th3 = np.angle(np.mean(np.exp(3j * (th - s * 2 * np.pi * W / N * n))))   # 3 theta0
    return s, th3


def desfase(c, n0):
    u = (c - n0) % PER
    psi = np.angle(np.mean(np.exp(2j * np.pi * u / PER))) * PER / (2 * np.pi)
    d = ((u - psi + PER / 2) % PER) - PER / 2
    return psi, float(np.sqrt(np.mean(d ** 2)))


def circ_media_std(psis):
    z = np.mean(np.exp(2j * np.pi * np.asarray(psis) / PER))
    media = np.angle(z) * PER / (2 * np.pi)
    std = PER / (2 * np.pi) * np.sqrt(-2 * np.log(max(abs(z), 1e-12)))
    return media, std


def main():
    print("INTEGRATOR=%s" % (
        "explicit Euler" if USE_EXPLICIT_EULER else "implicit midpoint"))
    if REUSE and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            datos = pickle.load(f)
        print(f"cache: {CACHE} ({len(datos)} corridas)")
    else:
        tareas = [(x, 0, 0.0) for x in X_LIST] + [(x, s, SIGMA) for x in X_LIST for s in range(N_SEEDS)]
        datos = []
        with Pool(N_PROC) as pool:
            for r in pool.imap_unordered(correr, tareas):
                datos.append(r)
                print(f"\r{len(datos)}/{len(tareas)}", end="", flush=True)
        print()
        with open(CACHE, "wb") as f:
            pickle.dump(datos, f)
    ruidosas = {x: [c for (xx, s, sg, c) in datos if xx == x and sg > 0 and c is not None] for x in X_LIST}
    limpias = {x: [c for (xx, s, sg, c) in datos if xx == x and sg == 0 and c is not None] for x in X_LIST}
        # ---------------- umbral de seleccion del numero de paredes ----------------
    for x in X_LIST:
        filas = [(s, c) for (xx, s, sg, c) in datos if xx == x and sg > 0]
        semillas = [s for s, c in filas]

        if (sorted(semillas) != list(range(N_SEEDS))
                or any(c is None for s, c in filas)):
            raise RuntimeError(
                f"Datos incompletos o semillas duplicadas para x={x}"
            )

    kM = np.array([
        sum(len(c) == 18 for c in ruidosas[x])
        for x in X_LIST
    ], dtype=float)

    nM = np.array([
        len(ruidosas[x]) for x in X_LIST
    ], dtype=float)

    xc, (lo, hi) = ajuste_logistico(X_LIST, kM, nM)

    print("\n[2a] UMBRAL DE SELECCION DEL NUMERO DE PAREDES")
    print("x       n    P(Ns=18)")
    for x, k, n in zip(X_LIST, kM, nM):
        print(f"{x:<6g}  {int(n):2d}   {k/n:.3f}")

    print(f"x_c = {xc:.3f}; IC bootstrap 68% = [{lo:.3f}, {hi:.3f}]")
    
    print("\n[1] TABLA I (sigma = %g)\n x     n  <Ns>+-std     P(18)  mediana dL  P(dL<0.15)" % SIGMA)
    for x in X_LIST:
        cs = ruidosas[x]
        ns = np.array([len(c) for c in cs])
        d = np.array([dl(c) for c in cs])
        print("%-5g %2d  %5.1f+-%-4.1f  %5.2f  %9.2f  %9.2f" % (x, len(cs), ns.mean(), ns.std(), np.mean(ns == 18),
                                                                  np.median(d), np.mean(d < 0.15)))

    print("\n[2] ROBUSTEZ DE x_c^reg: criterio | x_c [68%] | P(criterio) para x =", X_LIST)
    for nombre, f in CRITERIOS.items():
        k = np.array([sum(bool(f(c)) for c in ruidosas[x]) for x in X_LIST], float)
        n = np.array([len(ruidosas[x]) for x in X_LIST], float)
        if k.sum() == 0 or k.sum() == n.sum():
            texto = "sin cruce"
        else:
            xc, (lo, hi) = ajuste_logistico(X_LIST, k, n)
            texto = "%.2f [%.2f, %.2f]" % (xc, lo, hi)
        print("%-18s %-20s %s" % (nombre, texto, " ".join("%.2f" % v for v in k / n)))

    s, th3 = fase_base()
    print("\n[3] FASE DE MOIRE: giro de la helice base s = %+d; 3*theta0 = %.3f rad; aleatorio: rms = %.1f sitios" % (
        s, th3, PER / np.sqrt(12)))
    for v in (+1, -1):
        n0 = (v * (np.pi / 2 - th3) / DQ) % PER
        marca = "  <- variante de la derivacion (v = s)" if v == s else ""
        print(f"variante v = {v:+d}: primer nodo en n0 = {n0:.1f}{marca}")
        print("   x     psi_limpia  rms_limpia | psi ruido (circ.)  std_circ  rms mediano")
        psis_todas, rms_todas = [], []
        for x in X_LIST:
            pl, rl = (desfase(limpias[x][0], n0) if limpias[x] else (np.nan, np.nan))
            ps, rs = zip(*[desfase(c, n0) for c in ruidosas[x]]) if ruidosas[x] else ([np.nan], [np.nan])
            m, sd = circ_media_std(ps)
            if x >= X_LOCK_MIN:
                psis_todas += list(ps)
                rms_todas += list(rs)
            print("   %-5g %9.1f  %9.1f | %12.1f  %12.1f  %10.1f" % (x, pl, rl, m, sd, np.median(rs)))
        m, sd = circ_media_std(psis_todas)
        print("   x >= %g, todas las corridas con ruido: psi = %.1f, std_circ = %.1f, rms mediano = %.1f sitios" % (
            X_LOCK_MIN, m, sd, np.median(rms_todas)))


if __name__ == "__main__":
    main()