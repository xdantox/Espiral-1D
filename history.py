#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Evolucion temporal de Ns y phi_max en las HISTORIAS (T,N,3) guardadas por chain.py.

Lee los .npy con mmap (no carga los ~3.5 GB), toma unos pocos fotogramas y, para cada uno,
detecta paredes y reporta Ns(t), max|phi|(t), plano dominante y energia total.
Objetivo: decidir si las paredes se NUCLEAN (Ns se fija pronto) o se ANIQUILAN (Ns sube y luego baja).
Tiempo fisico del fotograma j: t = j * DT.  (chain.py: dt = 1e-15 s, T = 2**17 fotogramas)
Requiere relax3D.py en la misma carpeta.
"""
import os
import re
import numpy as np
from relax3D import CONFIG, detectar_solitones, extraer_angulos, calcular_energia_por_sitio, resolver_ruta

DT = 1e-15
FRAMES = [0, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 65536, 131071]
# Archivos a analizar: (nombre, incluir). Edita libremente.
ARCHIVOS = [
        "D_plane = 2.2D relaxYZ n = 1188 noise=0.04 seed1.npy",
        "D_plane = 2.2D relaxYZ n = 1188 noise=0.04 seed2.npy",
        "D_plane = 2.2D relaxYZ n = 1188 noise=0.04 seed3.npy"
]


def razon_desde_nombre(nombre):
    m = re.search(r"D_plane = ([0-9.]+)D", nombre)
    return float(m.group(1)) if m else 1.0


def analizar_historia(ruta, ratio, cfg):
    H = np.load(ruta, mmap_mode="r")
    if H.ndim != 3:
        print("  [Info] array estatico %s: sin historia temporal" % (H.shape,))
        return
    T = H.shape[0]
    print("  shape=%s dtype=%s" % (H.shape, H.dtype))
    print("  %-8s | %-9s | %-3s | %-3s | %-10s | %-14s" % ("frame", "t [ps]", "pl", "Ns", "phi_max", "E_total [meV]"))
    for j in [f for f in FRAMES if f < T] + ([T - 1] if (T - 1) not in FRAMES else []):
        s = np.array(H[j], dtype=float)
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        xz = bool(np.mean(s[:, 0] ** 2) > np.mean(s[:, 1] ** 2))   # plano dominante
        th, ph = extraer_angulos(s, cfg["S"], xz)
        try:
            ns = len(detectar_solitones(th, ph)[0])
        except Exception:
            ns = -1
        e = float(np.sum(calcular_energia_por_sitio(s, ratio, cfg)))
        print("  %-8d | %-9.3f | %-3s | %-3d | %-10.2e | %-14.6f" % (j, j * DT * 1e12, "XZ" if xz else "YZ", ns, float(np.max(np.abs(ph))), e), flush=True)


def main():
    for nombre in ARCHIVOS:
        ruta = resolver_ruta(nombre)
        if not ruta:
            print("[Warning] no encontrado: %s" % nombre)
            continue
        print("\n== %s ==" % nombre)
        analizar_historia(ruta, razon_desde_nombre(nombre), CONFIG)


if __name__ == "__main__":
    main()