#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura de paredes: ultimo fotograma de las corridas LLG de produccion para varios x.

Cada fila marca las posiciones de las paredes (0..N-1); las lineas grises son los nodos de Moire (separacion
N/Ns_M = 66 sitios) con la fase fijada por la corrida limpia de referencia. Imprime Ns, delta_L = std(L)/mean(L)
y R2 = |<exp(2 i dq x_k)>| de cada fila (para pegarlos en una tabla). Requiere relax3D.py en la misma carpeta.
"""
import numpy as np
import matplotlib.pyplot as plt
from relax3D import detectar_solitones, extraer_angulos

N, W = 1188, 399
DQ = 2 * np.pi * (3 * W - N) / N
PERIODO_NODOS = np.pi / DQ                                   # 66 sitios
REF = "D_plane = 2.2D relaxYZ n = 1188.npy"                  # limpio: fija la fase de los nodos


def nombre(r, sufijo=""):
    return f"D_plane = {r:g}D relaxYZ n = 1188{sufijo}.npy"


CASOS = [(REF, "unseeded, x = 1.2")]
CASOS += [(nombre(r, " noise=1e-3"), f"$\\sigma=10^{{-3}}$, x = {r - 1:.2g}") for r in (1.1, 1.2, 1.4, 1.65, 2.2, 5.0)]
CASOS += [(nombre(2.2, " noise=0.04 seed1"), "$\\sigma=0.04$, x = 1.2")]


def paredes(ruta):
    S = np.array(np.load(ruta, mmap_mode="r")[-1], dtype=float)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    theta, phi = extraer_angulos(S, 1.0, True)
    solitones, _ = detectar_solitones(theta, phi)
    return np.array(sorted(s["core_bond"] for s in solitones), dtype=float)


def metricas(c):
    L = np.diff(np.r_[c, c[0] + N])
    return len(c), L.std() / L.mean(), abs(np.mean(np.exp(2j * DQ * c)))


def main():
    c_ref = paredes(REF)
    x0 = (np.angle(np.mean(np.exp(2j * DQ * c_ref))) / (2 * DQ)) % PERIODO_NODOS
    nodos = x0 + PERIODO_NODOS * np.arange(int(round(N / PERIODO_NODOS)))
    filas = []
    for ruta, etiqueta in CASOS:
        try:
            filas.append((etiqueta, paredes(ruta)))
        except (FileNotFoundError, OSError):
            print("falta", ruta)
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(filas) + 0.9))
    for nodo in nodos:
        ax.axvline(nodo, color="0.85", lw=0.8, zorder=0)
    print("%-28s %4s %8s %6s" % ("caso", "Ns", "delta_L", "R2"))
    for i, (etiqueta, c) in enumerate(filas):
        y = len(filas) - 1 - i
        ax.vlines(c, y - 0.3, y + 0.3, color="C3" if "unseeded" not in etiqueta else "k", lw=1.6)
        ns, dl, r2 = metricas(c)
        ax.text(N + 12, y, f"$N_s$={ns}, $\\delta_L$={dl:.2f}", va="center", fontsize=8)
        print("%-28s %4d %8.2f %6.2f" % (etiqueta.replace("$", "").replace("\\", ""), ns, dl, r2))
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels([e for e, _ in filas][::-1], fontsize=8)
    ax.set_xlim(0, N)
    ax.set_xlabel("site $n$")
    ax.set_ylim(-0.7, len(filas) - 0.3)
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    fig.savefig("fig_paredes.pdf")
    fig.savefig("fig_paredes.png", dpi=180)


if __name__ == "__main__":
    main()