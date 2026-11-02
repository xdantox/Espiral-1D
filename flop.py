#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Proxy LLG LIGERO del spin-flop YZ -> XZ con ruido (Ns vs x) para obtener estadistica.

Usa el mismo Hamiltoniano H(1), el mismo ruido (S_x gaussiano, sigma = NOISE, media restada, renormalizado)
y la misma ecuacion LLG que chain.py (precesion + amortiguamiento, alpha = 1.3), pero con un integrador simple
(Euler + renormalizacion) y SOLO hasta que el flop termina (pocos ps). Cada corrida tarda segundos.
Estado base: ultimo fotograma de BASE_FILE (estado YZ relajado), igual que cadena0.py.

Unidades: tiempo en 1/(gamma/mu_B * meV) = 0.34 ps por unidad; dt = DT unidades (0.003 ~ 1 fs).
Requiere relax3D.py (detector de paredes) en la misma carpeta.
"""
import numpy as np
from relax3D import detectar_solitones, extraer_angulos, CONFIG

BASE_FILE = "D_plane = D relaxYZ n = 1188.npy"
X_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.2, 2.5, 4.0]
N_SEEDS = 12
ALPHA = 1.3
NOISE = 1e-3
DT = 0.003
STEPS_FOR_X = lambda x: int(min(30000, max(6000, 2.5 / (x * DT))))   # ~ varios e-folds de la inestabilidad
N_TARGET = 18

J1, J2, JN, K, D = CONFIG["J_NN_1"], CONFIG["J_NN_2"], CONFIG["J_NNN"], CONFIG["K"], CONFIG["D"]


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


def correr(S0, x, seed):
    n = len(S0)
    Jb = np.where(np.arange(n) % 2 == 0, J1, J2)
    rng = np.random.default_rng(seed)
    S = S0.copy()
    nz = NOISE * rng.standard_normal(n)
    nz -= nz.mean()
    S[:, 0] += nz
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    Dp = (1 + x) * D
    for _ in range(STEPS_FOR_X(x)):
        SG = np.cross(S, campo(S, Jb, Dp))
        S = S + DT * (-SG + ALPHA * np.cross(S, SG))
        S /= np.linalg.norm(S, axis=1, keepdims=True)
    xz = bool(np.mean(S[:, 0] ** 2) > np.mean(S[:, 1] ** 2))
    th, ph = extraer_angulos(S, 1.0, xz)
    return xz, len(detectar_solitones(th, ph)[0])


def main():
    S0 = cargar_base(BASE_FILE)
    print("%-6s | %-4s | %-9s | %-7s | %s" % ("x", "XZ", "<Ns>+-std", "P(Ns=%d)" % N_TARGET, "Ns por semilla"))
    filas = []
    for x in X_LIST:
        r = [correr(S0, x, s) for s in range(N_SEEDS)]
        ns = np.array([q[1] for q in r])
        filas.append((x, ns.mean(), (ns == N_TARGET).mean()))
        print("%-6g | %-4s | %4.1f+-%-4.1f | %-8.2f | %s" % (x, "ok" if all(q[0] for q in r) else "NO", ns.mean(), ns.std(), (ns == N_TARGET).mean(), ns.tolist()), flush=True)
    xc = next((f[0] for f in filas if f[2] >= 0.5), None)
    print("\nx donde P(Ns=%d) >= 0.5 por primera vez: %s" % (N_TARGET, xc))


if __name__ == "__main__":
    main()