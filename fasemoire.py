#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test que discrimina "nodos fijados por la fuente coherente" de "nodos fijados por la red".

La fuente coherente ~ cos(s dq n + 3 theta0) tiene nodos en n = v (pi/2 - 3 theta0)/dq + 66 k (v = +-1 es el signo
de la fase lenta). En el estado base 3 theta0 = pi/2, asi que los nodos caen en n = 0 (mod 66), que tambien es una
posicion simetrica de la red: con ese estado no se distingue la fuente coherente de un anclaje por la red.
Aqui se prepara una helice ideal en el plano YZ con fase inicial theta0 variable (y sentido s = -1, +1), se relaja
brevemente a Dplane = D (RELAX_STEPS pasos; la fase blanda casi no se mueve), se mide 3 theta0 del estado asi
obtenido, y se hace el quench (limpio y con ruido sigma). Para cada corrida se calcula el desfase psi de las paredes
respecto de: (a) los nodos predichos con v = +1, (b) con v = -1, (c) nodos fijos en n = 0 (anclaje por la red).
Si la fuente coherente fija los nodos, psi es ~constante (|psi| < 2 sitios) para la variante correcta en TODAS las
theta0, mientras (c) varia con 3 theta0 (los nodos predichos se desplazan en pi/3/dq = 22 sitios por paso de theta0).
Si las paredes siguen en n = 0 para todo theta0, manda la red. Requiere regular.py y relax3D.py en la carpeta.

USE_EXPLICIT_EULER -> True usa Euler explicito; False usa punto medio implicito de rot_methods.py.
"""
import numpy as np

from relax3D import detectar_solitones, extraer_angulos
from regular import (ALPHA, D, DQ, DT, J1, J2, N, PER, W, campo, circ_media_std, desfase, n_pasos)
from rot_methods import implicit_midpoint_step

THETA0 = np.linspace(0, 2 * np.pi / 3, 6, endpoint=False)     # 3 theta0 recorre 0..2 pi en pasos de pi/3
SENTIDOS = (-1, +1)
X_LIST = (1.2,)
SIGMA = 1e-3
SEEDS = range(4)
RELAX_STEPS = 2000
JB = np.where(np.arange(N) % 2 == 0, J1, J2)
USE_EXPLICIT_EULER = False


def helice(s, th0):
    th = s * 2 * np.pi * W / N * np.arange(N) + th0
    S = np.zeros((N, 3))
    S[:, 1], S[:, 2] = np.sin(th), np.cos(th)
    return S


def evolucionar(S, Dp, pasos):
    campo_actual = lambda spins: campo(spins, JB, Dp)
    for _ in range(pasos):
        if USE_EXPLICIT_EULER:
            G = campo_actual(S)
            SG = np.cross(S, G)
            S = S + DT * (-SG + ALPHA * np.cross(S, SG))
        else:
            S = implicit_midpoint_step(S, DT, campo_actual, ALPHA, True)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
    return S


def fase_estado(S):
    th = np.arctan2(S[:, 1], S[:, 2])
    s = 1 if np.mean(np.angle(np.exp(1j * (np.roll(th, -1) - th)))) > 0 else -1
    return s, np.angle(np.mean(np.exp(3j * (th - s * 2 * np.pi * W / N * np.arange(N)))))


def quench(S0, x, sigma, seed):
    S = S0.copy()
    if sigma > 0:
        nz = sigma * np.random.default_rng(seed).standard_normal(N)
        S[:, 0] += nz - nz.mean()
        S /= np.linalg.norm(S, axis=1, keepdims=True)
    S = evolucionar(S, (1 + x) * D, n_pasos(x, max(sigma, SIGMA)))
    if not np.mean(S[:, 0] ** 2) > np.mean(S[:, 1] ** 2):
        return None
    th, ph = extraer_angulos(S, 1.0, True)
    return np.array(sorted(s["core_bond"] for s in detectar_solitones(th, ph)[0]), dtype=float)


def main():
    resumen = {}
    for s_in in SENTIDOS:
        for th0 in THETA0:
            S0 = evolucionar(helice(s_in, th0), D, RELAX_STEPS)
            s, th3 = fase_estado(S0)
            n0 = {v: (v * (np.pi / 2 - th3) / DQ) % PER for v in (+1, -1)}
            n0[0] = 0.0
            for x in X_LIST:
                for sigma, seed in [(0.0, 0)] + [(SIGMA, k) for k in SEEDS]:
                    c = quench(S0, x, sigma, seed)
                    if c is None or len(c) < 4:
                        print(f"s={s:+d} 3th0={th3:+.2f} x={x:g} sigma={sigma:g}: sin flop XZ")
                        continue
                    r = {v: desfase(c, n0[v]) for v in (+1, -1, 0)}
                    print("s=%+d 3th0=%+.2f x=%-4g sig=%-5g Ns=%2d | psi,rms  v=+1: %+5.1f,%4.1f  v=-1: %+5.1f,%4.1f"
                          "  nodos en 0: %+5.1f,%4.1f" % (s, th3, x, sigma, len(c), *r[1], *r[-1], *r[0]), flush=True)
                    if sigma == 0:
                        for v in (+1, -1, 0):
                            resumen.setdefault((s, v), []).append(r[v][0])
    print("\nRESUMEN (corridas limpias): std circular de psi a traves de theta0 (sitios; ~0 = nodos fijados por esa hipotesis,")
    print("aleatorio ~ %.0f). v=+1 / v=-1: nodos de la fuente coherente; 'en 0': anclaje por la red." % (PER / np.sqrt(12)))
    for (s, v), psis in sorted(resumen.items()):
        nombre = {1: "v=+1", -1: "v=-1", 0: "nodos en 0"}[v]
        print(f"  s={s:+d} {nombre:11s}: std_circ = {circ_media_std(psis)[1]:.1f}")


if __name__ == "__main__":
    main()