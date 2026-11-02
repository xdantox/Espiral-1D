#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analisis offline de transmision de magnones (version final).

No integra la LLG. Para cada estado (R, A, C) y cada pulso calcula
    delta = S_pulso - S_fondo
en detectores de periodo completo (67 sitios) a d = 132, 198, 264, 330 sitios de la fuente y mide la
transmision relativa a la helice uniforme R de dos maneras:

  (1) ESPECTRAL: T(nu, d) = P_estado / P_R, ventana de Hann sobre las primeras 60 y 80 ps.
  (2) CON PUERTA TEMPORAL: cociente de energias de banda en [t_R - 3 ps, t_R + W], con t_R el pico de R
      en esa banda y W = 5, 10, 20, 30 ps. Discrimina retraso (T_gate crece con W) de bloqueo (T_gate
      satura por debajo de 1).

Ademas: retraso del pico respecto de R, longitud de atenuacion xi (d = 132, 198, 264) y prueba de linealidad.
Requiere relax3D.py en la misma carpeta.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from relax3D import CONFIG, detectar_solitones, extraer_angulos


N = 1188
DT_SAMPLE = 10e-15
STEP_PS = DT_SAMPLE / 1e-12
ENERGY_EVERY = 10
DRIFT_LIMIT = 1e-10

MAX_SAMPLES = 8000                      # 80 ps
WINDOW_SAMPLES = (6000, 8000)           # 60 y 80 ps
NS_MAIN = 8000

DISTANCES = (132, 198, 264, 330)
HALF_WIDTH = 33                         # bloque de 67 sitios
XI_DISTANCES = (132, 198, 264)

SUMMARY_WINDOWS = ((1, 3), (3, 5), (5, 7), (16, 19), (19, 22), (22, 24))
GATE_BANDS = ((1.7, 2.5),)
GATE_BEFORE_PS = 3.0
GATE_WIDTHS_PS = (5.0, 10.0, 20.0, 30.0)
DELAY_SEARCH_PS = (5.0, 25.0)           # (antes, despues) del pico de R
EDGE_GUARD_PS = 5.0

POWER_FLOOR_REL = 1e-8
FLOOR_BANDS = ((0.3, 7.2), (13.0, 24.4))

OUTPUT_FIGURE = "transmision_final.png"
ORDER = ("R", "A", "C")

RUNS = {
    "A": {
        "ratio": 2.2,
        "background": "D_plane = 2.2D backgroundYZ n = 1188.npy",
        "pulses": (
            (0, "D_plane = 2.2D pulseYZ n = 1188 eps = 1e-4 n0 = 0.npy"),
            (100, "D_plane = 2.2D pulseYZ n = 1188 eps = 1e-4 n0 = 100.npy"),
        ),
        "linear": (0, "D_plane = 2.2D pulseYZ n = 1188 eps = 1e-3 n0 = 0.npy", 1e-3),
        "eps": 1e-4,
    },
    "C": {
        "ratio": 2.2,
        "background": "D_plane = 2.2D backgroundYZ n = 1188 noise=0.04 seed1.npy",
        "pulses": (
            (0, "D_plane = 2.2D pulseYZ n = 1188 noise=0.04 seed1 eps = 1e-4 n0 = 0.npy"),
            (100, "D_plane = 2.2D pulseYZ n = 1188 noise=0.04 seed1 eps = 1e-4 n0 = 100.npy"),
        ),
        "eps": 1e-4,
    },
    "R": {
        "ratio": 2.2,
        "background": "D_plane = 2.2D background n = 1188.npy",
        "pulses": (
            (0, "D_plane = 2.2D pulse n = 1188 eps = 1e-4 n0 = 0.npy"),
            (100, "D_plane = 2.2D pulse n = 1188 eps = 1e-4 n0 = 100.npy"),
        ),
        "eps": 1e-4,
    },
}


# ----------------------------------------------------------------------------
# Utilidades basicas
# ----------------------------------------------------------------------------
def load_history(path, max_samples=MAX_SAMPLES):
    raw = np.load(path, mmap_mode="r")
    if raw.ndim != 3 or raw.shape[1:] != (N, 3):
        raise ValueError(f"{path}: se esperaba (n, {N}, 3), obtenido {raw.shape}")
    return np.asarray(raw[:max_samples], dtype=np.float64)


def normalized_final_state(history, path):
    state = np.asarray(history[-1], dtype=np.float64)
    norms = np.linalg.norm(state, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{path}: contiene spins de norma cero")
    return state / norms


def soliton_positions(state):
    theta, phi = extraer_angulos(state, 1.0, True)
    solitons, _ = detectar_solitones(theta, phi)
    return np.array([s["core_bond"] for s in solitons], dtype=int)


def nearest_wall_distance(walls, site):
    if walls.size == 0:
        return np.nan
    d = np.abs(walls - site) % N
    return float(np.min(np.minimum(d, N - d)))


def relative_energy(history, ratio):
    d_plane = ratio * CONFIG["D"]
    right = np.roll(history, -1, axis=1)
    second = np.roll(history, -2, axis=1)
    dot1 = np.sum(history * right, axis=2)
    dot2 = np.sum(history * second, axis=2)
    j_forward = np.where(np.arange(N) % 2 == 0, CONFIG["J_NN_1"], CONFIG["J_NN_2"])
    return (
        np.sum(j_forward[None, :] * dot1, axis=1)
        + CONFIG["K"] * np.sum(dot1 ** 2, axis=1)
        + CONFIG["J_NNN"] * np.sum(dot2, axis=1)
        + CONFIG["D"] * np.sum(history[:, :, 0] ** 2, axis=1)
        + d_plane * np.sum(history[:, :, 1] ** 2, axis=1)
    )


def relative_drift(history, ratio):
    energy = relative_energy(history[::ENERGY_EVERY], ratio)
    denominator = max(abs(energy[0]), np.finfo(float).tiny)
    return float(np.max(np.abs((energy - energy[0]) / denominator)))


def detector_indices(n0):
    offsets = np.arange(-HALF_WIDTH, HALF_WIDTH + 1)
    result = np.empty((2, len(DISTANCES), offsets.size), dtype=int)
    for side, sign in enumerate((-1, 1)):
        for di, distance in enumerate(DISTANCES):
            result[side, di] = ((n0 + sign * distance) % N + offsets) % N
    return result


def fmt(value):
    if value is None or not np.isfinite(value) or value <= 0:
        return "  n/a"
    return f"{np.log10(value):6.2f}"


# ----------------------------------------------------------------------------
# Espectro y puerta temporal
# ----------------------------------------------------------------------------
def power_spectrum(signal):
    samples = signal.shape[0]
    centered = signal - signal.mean(axis=0, keepdims=True)
    window = np.hanning(samples)[:, None, None]
    spectrum = np.fft.rfft(centered * window, axis=0)
    power = np.sum(np.abs(spectrum) ** 2, axis=(1, 2))
    frequency = np.fft.rfftfreq(samples, DT_SAMPLE) / 1e12
    return frequency, power


def band_energy(block, band):
    """Energia de banda vs tiempo. block: (samples, sitios, 3)."""
    samples = block.shape[0]
    frequency = np.fft.rfftfreq(samples, DT_SAMPLE) / 1e12
    mask = (frequency >= band[0]) & (frequency < band[1])
    centered = block - block.mean(axis=0, keepdims=True)
    filtered = np.fft.irfft(np.fft.rfft(centered, axis=0) * mask[:, None, None], n=samples, axis=0)
    return np.sum(filtered ** 2, axis=(1, 2))


def analyze_pulse(pulse, background, n0):
    detectors = detector_indices(n0)
    samples = pulse.shape[0]
    nd = len(DISTANCES)
    delta = np.empty((2, nd, samples, 2 * HALF_WIDTH + 1, 3), dtype=np.float64)
    for side in range(2):
        for di in range(nd):
            sites = detectors[side, di]
            delta[side, di] = pulse[:, sites, :] - background[:, sites, :]

    spectra = {}
    for ns in WINDOW_SAMPLES:
        if ns > samples:
            continue
        powers = []
        for di in range(nd):
            frequency, left = power_spectrum(delta[0, di, :ns])
            _, right = power_spectrum(delta[1, di, :ns])
            powers.append(0.5 * (left + right))
        spectra[ns] = (frequency, np.asarray(powers))

    energies = np.empty((2, nd, len(GATE_BANDS), samples))
    for side in range(2):
        for di in range(nd):
            for bi, band in enumerate(GATE_BANDS):
                energies[side, di, bi] = band_energy(delta[side, di], band)
    return spectra, energies


def spectral_floor(power, frequency):
    mask = np.zeros(frequency.size, dtype=bool)
    for low, high in FLOOR_BANDS:
        mask |= (frequency >= low) & (frequency <= high)
    floor = np.full(power.shape[0], np.nan)
    for di in range(power.shape[0]):
        values = power[di, mask & (power[di] > 0)]
        if values.size:
            floor[di] = POWER_FLOOR_REL * np.median(values)
    return np.maximum(floor[:, None], np.finfo(float).tiny)


def valid_transmission(power, reference, floor):
    transmission = np.divide(power, reference, out=np.full_like(power, np.nan), where=reference > 0)
    valid = (reference > floor) & (power > floor) & np.isfinite(transmission) & (transmission > 0)
    return transmission, valid


def window_logT(frequency, transmission, valid):
    out = np.full((len(SUMMARY_WINDOWS), transmission.shape[0]), np.nan)
    for wi, (low, high) in enumerate(SUMMARY_WINDOWS):
        band = (frequency >= low) & (frequency < high)
        for di in range(transmission.shape[0]):
            selected = transmission[di, band][valid[di, band]]
            if selected.size:
                out[wi, di] = np.mean(np.log10(selected))
    return out


def xi_from_log10(values, distances):
    """xi (sitios) a partir de log10 T en varias d. Longitud de amplitud: T ~ exp(-2 d / xi)."""
    values = np.asarray(values, dtype=float)
    if values.size != len(distances) or not np.all(np.isfinite(values)):
        return np.nan
    slope = np.polyfit(np.asarray(distances, dtype=float), values, 1)[0]
    if slope >= 0:
        return np.nan
    return -2.0 / (np.log(10.0) * slope)


def gated_sums(e_state, e_ref):
    """Sumas de energia en puertas alrededor del pico de R. Arrays (2, nd, nb, samples)."""
    samples = e_state.shape[-1]
    guard = int(round(EDGE_GUARD_PS / STEP_PS))
    before = int(round(GATE_BEFORE_PS / STEP_PS))
    nd, nb = e_state.shape[1], e_state.shape[2]
    nw = len(GATE_WIDTHS_PS)
    num = np.zeros((nw, nd, nb))
    den = np.zeros((nw, nd, nb))
    cnt = np.zeros((nw, nd, nb), dtype=int)
    delays = [[[] for _ in range(nb)] for _ in range(nd)]
    t_ref = [[[] for _ in range(nb)] for _ in range(nd)]
    for side in range(2):
        for di in range(nd):
            for bi in range(nb):
                er = e_ref[side, di, bi]
                es = e_state[side, di, bi]
                ipk = guard + int(np.argmax(er[guard:samples - guard]))
                t_ref[di][bi].append(ipk * STEP_PS)
                for wi, width in enumerate(GATE_WIDTHS_PS):
                    lo = max(guard, ipk - before)
                    hi = ipk + int(round(width / STEP_PS))
                    if hi > samples - guard:
                        continue
                    num[wi, di, bi] += es[lo:hi + 1].sum()
                    den[wi, di, bi] += er[lo:hi + 1].sum()
                    cnt[wi, di, bi] += 1
                lo = max(guard, ipk - int(round(DELAY_SEARCH_PS[0] / STEP_PS)))
                hi = min(samples - guard, ipk + int(round(DELAY_SEARCH_PS[1] / STEP_PS)))
                segment = es[lo:hi]
                if segment.size > 2:
                    j = int(np.argmax(segment))
                    delays[di][bi].append((((lo + j) - ipk) * STEP_PS, j == 0 or j == segment.size - 1))
    return num, den, cnt, delays, t_ref


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    data = {}

    for label in ORDER:
        config = RUNS[label]
        background_path = Path(config["background"])
        pulse_specs = [(n0, Path(path)) for n0, path in config["pulses"]]
        missing = [p for p in [background_path] + [p for _, p in pulse_specs] if not p.exists()]
        if missing:
            print(f"{label}: omitido; faltan " + ", ".join(map(str, missing)))
            continue

        background = load_history(background_path)
        state = normalized_final_state(background, background_path)
        walls = np.sort(soliton_positions(state))
        print(f"\n{label}: paredes={walls.size}; muestras fondo={background.shape[0]} "
              f"({background.shape[0] * STEP_PS:.1f} ps); deriva_fondo="
              f"{relative_drift(background, config['ratio']):.2e}")

        entry = {"eps": config["eps"], "walls": walls, "sources": {}}
        for n0, path in pulse_specs:
            pulse = load_history(path)
            samples = min(background.shape[0], pulse.shape[0])
            if samples < MAX_SAMPLES:
                print(f"{label}: AVISO n0={n0}: solo {samples} muestras (< {MAX_SAMPLES})")
            pulse, bg = pulse[:samples], background[:samples]
            drift = relative_drift(pulse, config["ratio"])
            if drift > DRIFT_LIMIT:
                print(f"{label}: n0={n0} EXCLUIDA (deriva {drift:.2e} > {DRIFT_LIMIT:.0e})")
                continue
            centers = [(n0 + sign * d) % N for sign in (-1, 1) for d in DISTANCES]
            dist_wall = [nearest_wall_distance(walls, c) for c in centers]
            print(f"{label}: n0={n0}; deriva={drift:.2e}; dist(n0, pared)={nearest_wall_distance(walls, n0):.0f}; "
                  f"dist(centros detectores, pared)={[int(x) if np.isfinite(x) else None for x in dist_wall]}")
            spectra, energies = analyze_pulse(pulse, bg, n0)
            entry["sources"][n0] = {"spectra": spectra, "energies": energies}

        if "linear" in config:
            n0l, name, eps_l = config["linear"]
            lpath = Path(name)
            if lpath.exists() and n0l in entry["sources"]:
                hist = load_history(lpath)
                samples = min(background.shape[0], hist.shape[0])
                hist, bg = hist[:samples], background[:samples]
                drift = relative_drift(hist, config["ratio"])
                if drift <= DRIFT_LIMIT:
                    spectra, energies = analyze_pulse(hist, bg, n0l)
                    entry["linear"] = {"n0": n0l, "eps": eps_l, "spectra": spectra, "energies": energies}
                else:
                    print(f"{label}: historia de linealidad excluida (deriva {drift:.2e})")
        data[label] = entry

    if "R" not in data or not data["R"]["sources"]:
        raise FileNotFoundError("Se necesita la referencia R")

    ref = data["R"]
    ref_power, ref_freq, ref_floor = {}, {}, {}
    for ns in WINDOW_SAMPLES:
        arrays = [s["spectra"][ns][1] / ref["eps"] ** 2 for s in ref["sources"].values() if ns in s["spectra"]]
        if arrays:
            ref_power[ns] = np.mean(arrays, axis=0)
            ref_freq[ns] = next(iter(ref["sources"].values()))["spectra"][ns][0]
            ref_floor[ns] = spectral_floor(ref_power[ns], ref_freq[ns])

    # ---------------- (1) T espectral ----------------
    figure_data = {}
    for label in ("A", "C"):
        if label not in data or not data[label]["sources"]:
            continue
        entry = data[label]
        results = {}
        for ns in WINDOW_SAMPLES:
            if ns not in ref_power:
                continue
            arrays = [s["spectra"][ns][1] / entry["eps"] ** 2 for s in entry["sources"].values() if ns in s["spectra"]]
            if not arrays:
                continue
            power = np.mean(arrays, axis=0)
            transmission, valid = valid_transmission(power, ref_power[ns], ref_floor[ns])
            results[ns] = window_logT(ref_freq[ns], transmission, valid)
            if ns == NS_MAIN:
                figure_data[label] = (ref_freq[ns], transmission, valid)
        if NS_MAIN not in results:
            continue
        main_table = results[NS_MAIN]
        other = results.get(WINDOW_SAMPLES[0])
        print(f"\n[{label}] T ESPECTRAL: log10 T(d={DISTANCES}) a {NS_MAIN * STEP_PS:.0f} ps | "
              f"cambio ({NS_MAIN * STEP_PS:.0f} - {WINDOW_SAMPLES[0] * STEP_PS:.0f} ps) | xi (sitios)")
        for wi, (low, high) in enumerate(SUMMARY_WINDOWS):
            row = " ".join(f"{v:6.2f}" if np.isfinite(v) else "   n/a" for v in main_table[wi])
            if other is not None:
                diff = main_table[wi] - other[wi]
                drow = " ".join(f"{v:+6.2f}" if np.isfinite(v) else "   n/a" for v in diff)
            else:
                drow = "n/a"
            xi = xi_from_log10(main_table[wi, :len(XI_DISTANCES)], XI_DISTANCES)
            print(f"{low:>2}-{high:<2} THz | {row} | {drow} | {xi:7.1f}" if np.isfinite(xi)
                  else f"{low:>2}-{high:<2} THz | {row} | {drow} |     n/a")

    # ---------------- (2) T con puerta ----------------
    gate_store = {}
    iw20 = GATE_WIDTHS_PS.index(20.0)
    for label in ("A", "C"):
        if label not in data or not data[label]["sources"]:
            continue
        entry = data[label]
        total = None
        used = 0
        delays_all = [[[] for _ in GATE_BANDS] for _ in DISTANCES]
        tref_all = [[[] for _ in GATE_BANDS] for _ in DISTANCES]
        for n0, src in entry["sources"].items():
            if n0 not in ref["sources"]:
                continue
            e_state = src["energies"] / entry["eps"] ** 2
            e_ref = ref["sources"][n0]["energies"] / ref["eps"] ** 2
            if e_state.shape != e_ref.shape:
                print(f"{label}: n0={n0} omitido en la puerta (longitudes distintas)")
                continue
            num, den, cnt, delays, tref = gated_sums(e_state, e_ref)
            if total is None:
                total = [num, den, cnt]
            else:
                total[0] += num
                total[1] += den
                total[2] += cnt
            used += 1
            for di in range(len(DISTANCES)):
                for bi in range(len(GATE_BANDS)):
                    delays_all[di][bi].extend(delays[di][bi])
                    tref_all[di][bi].extend(tref[di][bi])
        if total is None:
            continue
        ratio = np.where(total[2] == 2 * used, total[0] / np.maximum(total[1], np.finfo(float).tiny), np.nan)
        gate_store[label] = ratio
        print(f"\n[{label}] PUERTA: log10 T_gate para W = {GATE_WIDTHS_PS} ps | t_pico_R (ps) | retraso mediano (ps) [% en borde]")
        print("   (T_gate crece con W => retraso; satura < 1 => bloqueo)")
        for bi, (low, high) in enumerate(GATE_BANDS):
            for di, distance in enumerate(DISTANCES):
                row = " ".join(fmt(ratio[wi, di, bi]) for wi in range(len(GATE_WIDTHS_PS)))
                tr = np.mean(tref_all[di][bi])
                dl = delays_all[di][bi]
                if dl:
                    median = np.median([x[0] for x in dl])
                    edge = 100.0 * np.mean([x[1] for x in dl])
                    dtext = f"{median:+6.2f} [{edge:3.0f}%]"
                else:
                    dtext = "   n/a"
                print(f"{low:>2}-{high:<2} THz d={distance:3d} | {row} | {tr:5.1f} | {dtext}")
        for bi, (low, high) in enumerate(GATE_BANDS):
            values = [ratio[iw20, DISTANCES.index(d), bi] for d in XI_DISTANCES]
            logs = [np.log10(v) if np.isfinite(v) and v > 0 else np.nan for v in values]
            xi = xi_from_log10(logs, XI_DISTANCES)
            print(f"   xi_gate(W=20 ps, d={XI_DISTANCES}) {low}-{high} THz: "
                  + (f"{xi:.1f}" if np.isfinite(xi) else "n/a"))

    # ---------------- Linealidad ----------------
    if "A" in data and "linear" in data["A"]:
        entry = data["A"]
        lin = entry["linear"]
        n0 = lin["n0"]
        base = entry["sources"][n0]
        if NS_MAIN in lin["spectra"] and NS_MAIN in base["spectra"]:
            base_p = base["spectra"][NS_MAIN][1] / entry["eps"] ** 2
            lin_p = lin["spectra"][NS_MAIN][1] / lin["eps"] ** 2
            floor = ref_floor[NS_MAIN]
            valid = (ref_power[NS_MAIN] > floor) & (base_p > floor) & (lin_p > floor)
            diff = np.abs(lin_p[valid] / base_p[valid] - 1.0)
            if diff.size:
                print(f"\nLINEALIDAD espectral (eps 1e-3 vs 1e-4, n0={n0}): p95={np.percentile(diff, 95):.2%}; "
                      f"mediana={np.median(diff):.2%}; criterio p95<5% -> "
                      f"{'OK' if np.percentile(diff, 95) < 0.05 else 'NO CUMPLE'}")
        e_ref = ref["sources"][n0]["energies"] / ref["eps"] ** 2
        e_base = base["energies"] / entry["eps"] ** 2
        e_lin = lin["energies"] / lin["eps"] ** 2
        if e_base.shape == e_ref.shape == e_lin.shape:
            nb_, db_, cb_, _, _ = gated_sums(e_base, e_ref)
            nl_, dl_, cl_, _, _ = gated_sums(e_lin, e_ref)
            ok = (cb_[iw20] == 2) & (cl_[iw20] == 2) & (nb_[iw20] > 0)
            if np.any(ok):
                ratios = nl_[iw20][ok] / nb_[iw20][ok]
                print(f"LINEALIDAD con puerta (W=20 ps): mediana={np.median(ratios):.4f}; "
                      f"min={ratios.min():.4f}; max={ratios.max():.4f}")

    # ---------------- Figura ----------------
    if {"A", "C"} <= figure_data.keys():
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, sharey=True)
        for axis, distance in zip(axes, (198, 264)):
            di = DISTANCES.index(distance)
            for label in ("A", "C"):
                frequency, transmission, valid = figure_data[label]
                curve = np.full(frequency.size, np.nan)
                curve[valid[di]] = np.log10(transmission[di, valid[di]])
                axis.plot(frequency, curve, label=label)
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_ylabel(f"d={distance}\nlog10 T")
            axis.set_xlim(0, 25)
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("Frecuencia (THz)")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_FIGURE, dpi=160)
        print(f"\nFigura guardada en {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()