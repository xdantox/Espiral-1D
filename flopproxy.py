#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Proxy LLG ligero del spin-flop YZ -> XZ: estadistica de Ns, barrido de x, de sigma y de siembra coherente.

Mismo Hamiltoniano H(1), misma ecuacion LLG (precesion + amortiguamiento, alpha = 1.3) y mismo ruido que chain.py
(S_x gaussiano, media restada, renormalizado), con Euler explicito o punto medio implicito y solo hasta que el flop
termina. Estado base: ultimo fotograma de BASE_FILE (estado YZ relajado), como en cadena0.py.

Opciones (editar abajo):
  PRECESSION=False  -> control sobreamortiguado (sin precesion): Ns debe seguir ~ sqrt(x)
  USE_EXPLICIT_EULER -> True usa Euler explicito; False usa punto medio implicito de rot_methods.py
  SIGMA_LIST        -> amplitudes de ruido a barrer
  SEED_AMP          -> amplitud 'a' de una siembra coherente S_x += a sin(theta_n) sin(dq n) (0 = sin siembra)
Salida: <Ns>+-std, P(Ns=NTARGET), conteos por semilla y, por cada sigma, x_c (ajuste logistico en ln x + bootstrap).
Unidades de tiempo: 1/(eta*meV), eta = gamma/mu_B = 2.94e12 /(s meV); DT=0.003 equivale a 1 fs.
Requiere relax3D.py en la misma carpeta.
"""
import numpy as np
from scipy.optimize import minimize
from relax3D import detectar_solitones, extraer_angulos, CONFIG
from rot_methods import implicit_midpoint_step

BASE_FILE = "D_plane = D relaxYZ n = 1188.npy"
SIGMA_LIST=[0.04]
X_LIST=[0.2,0.4,0.65,1.0,1.5,2.0]
N_SEEDS = 12
SEED_AMP = 0.0
USE_EXPLICIT_EULER = True
PRECESSION = True
ALPHA = 1.3
DT = 0.003
NTARGET = 18
CV_LOCK = 0.15
W_WIND = 399                      # winding de la base; dq = 2 pi (3W - N)/N

J1, J2, JN, K, D = CONFIG["J_NN_1"], CONFIG["J_NN_2"], CONFIG["J_NNN"], CONFIG["K"], CONFIG["D"]


def n_pasos(x, sigma):
    """~1.5 veces el tiempo de flop estimado, t = L / (2 x D), L = ln(1.5/sigma) acotado a [7, 20]."""
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


def correr(S0, x, seed, sigma):
    n = len(S0)
    Jb = np.where(np.arange(n) % 2 == 0, J1, J2)
    rng = np.random.default_rng(seed)
    S = S0.copy()
    nz = sigma * rng.standard_normal(n)
    nz -= nz.mean()
    S[:, 0] += nz
    if SEED_AMP:
        dq = 2 * np.pi * (3 * W_WIND - n) / n
        S[:, 0] += SEED_AMP * S[:, 1] * np.sin(dq * np.arange(n))      # S_y = sin(theta) en el plano YZ
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    Dp = (1 + x) * D
    campo_actual = lambda spins: campo(spins, Jb, Dp)
    for _ in range(n_pasos(x, sigma)):
        if USE_EXPLICIT_EULER:
            G = campo_actual(S)
            if PRECESSION:
                SG = np.cross(S, G)
                S = S + DT * (-SG + ALPHA * np.cross(S, SG))
            else:
                S = S - DT * (G - (S * G).sum(1, keepdims=True) * S)
        else:
            S = implicit_midpoint_step(S, DT, campo_actual, ALPHA, PRECESSION)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
    xz = bool(np.mean(S[:, 0] ** 2) > np.mean(S[:, 1] ** 2))
    th, ph = extraer_angulos(S, 1.0, xz)
    sol, _ = detectar_solitones(th, ph)
    c = [s["core_bond"] for s in sol]
    cv = metricas(c, n) if len(c) > 2 else np.nan
    return xz, len(c), cv


def metricas(c, n):
    c = np.array(sorted(c))
    L = np.diff(np.r_[c, c[0] + n])
    return L.std() / L.mean()


def ajuste_logistico(x, k, n):
    np.seterr(over='ignore')
    def nll(p, kk):
        P = 1 / (1 + np.exp(-(np.log(x) - p[0]) / np.exp(p[1])))
        P = np.clip(P, 1e-9, 1 - 1e-9)
        return -np.sum(kk * np.log(P) + (n - kk) * np.log(1 - P))
    def fit(kk):
        return np.exp(minimize(nll, [np.log(0.6), 0.0], args=(kk,), method="Nelder-Mead").x[0])
    xc = fit(k)
    rng = np.random.default_rng(0)
    bs = np.array([fit(rng.binomial(n, k / n)) for _ in range(500)])
    return xc, np.percentile(bs, [16, 84])


def main():
    S0 = cargar_base(BASE_FILE)
    print("INTEGRATOR=%s PRECESSION=%s SEED_AMP=%g N_SEEDS=%d CV_LOCK=%.2f" % (
        "explicit Euler" if USE_EXPLICIT_EULER else "implicit midpoint",
        PRECESSION, SEED_AMP, N_SEEDS, CV_LOCK))
    for sigma in SIGMA_LIST:
        print("\n=== sigma = %g ===" % sigma)
        print("x | flop | <Ns>+-std | P(Ns=18) | medianCV | P(CV<c) | (Ns,CV) por semilla")
        xs, ks = [], []
        for x in X_LIST:
            r = [correr(S0, x, s, sigma) for s in range(N_SEEDS)]
            ok = [q for q in r if q[0]]
            if not ok:
                print(x, "sin flop")
                continue
            ns = np.array([q[1] for q in ok])
            cv = np.array([q[2] for q in ok])
            if len(ok) == N_SEEDS:
                xs.append(x)
                ks.append(int((cv < CV_LOCK).sum()))
            print("%g | %d/%d | %.1f+-%.1f | %.2f | %.2f | %.2f | %s" % (
                x, len(ok), N_SEEDS, ns.mean(), ns.std(), (ns == NTARGET).mean(),
                np.nanmedian(cv), (cv < CV_LOCK).mean(), [(q[1], round(q[2], 2)) for q in ok]), flush=True)
        if len(xs) >= 5 and 0 < sum(ks) < N_SEEDS * len(xs):
            xc, ci = ajuste_logistico(np.array(xs), np.array(ks), N_SEEDS)
            print("x_c(reg) = %.3f IC68 [%.3f, %.3f] -> D_crit = %.2f D" % (xc, ci[0], ci[1], 1 + xc))
        else:
            print("x_c(reg): sin transicion resuelta")


if __name__ == "__main__":
    main()