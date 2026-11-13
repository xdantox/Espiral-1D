#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control del piso de la razon con puerta: mismo analisis de transmision.py, pero con ventana de frecuencia SUAVE.

No edita transmision.py: reemplaza su filtro rectangular (band_energy) por uno con bordes de coseno elevado de
ancho TAPER_THZ y ejecuta main(). Pega solo los bloques "[A] PUERTA" y "[C] PUERTA" (banda 1.7-2.5 THz).
Lectura: si con el filtro suave T_gate en d >= 198 baja de ~1e-4 (por ejemplo >= 1 decada) o empieza a decaer con d,
el piso era fuga espectral del filtro rectangular y los valores suaves son los que van al paper; si no cambia, el
piso no viene del filtro y se retira de la redaccion la explicacion por fuga.
"""
import importlib

import numpy as np
import transmision as T

T = importlib.reload(T)
TAPER_THZ = 0.3


def band_energy_suave(block, band):
    samples = block.shape[0]
    f = np.fft.rfftfreq(samples, T.DT_SAMPLE) / 1e12
    lo, hi = band
    up = np.clip((f - (lo - TAPER_THZ / 2)) / TAPER_THZ, 0.0, 1.0)
    down = np.clip(((hi + TAPER_THZ / 2) - f) / TAPER_THZ, 0.0, 1.0)
    w = (np.sin(0.5 * np.pi * up) * np.sin(0.5 * np.pi * down)) ** 2
    centered = block - block.mean(axis=0, keepdims=True)
    filtered = np.fft.irfft(np.fft.rfft(centered, axis=0) * w[:, None, None], n=samples, axis=0)
    return np.sum(filtered ** 2, axis=(1, 2))


if __name__ == "__main__":
    T.band_energy = band_energy_suave
    T.OUTPUT_FIGURE = "transmision_suave.png"
    print(f"filtro suave (coseno elevado, borde {TAPER_THZ} THz) en la banda de la puerta {T.GATE_BANDS}")
    T.main()