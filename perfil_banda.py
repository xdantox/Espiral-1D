#!/usr/bin/env python3
"""Perfiles espaciales de la energia espectral de los pulsos."""

from pathlib import Path

import numpy as np

from transmision import DRIFT_LIMIT, RUNS, load_history, relative_drift


N = 1188
WINDOWS = (6000, 8000)
BANDS = ((1.0, 1.7), (1.7, 2.5), (2.5, 3.0), (3.0, 5.0), (5.0, 7.0), (16.0, 24.0))
BAND_LABELS = ("1-1.7", "1.7-2.5", "2.5-3", "3-5", "5-7", "16-24")
DT_SAMPLE = 10e-15
BIN_CENTERS = np.arange(-528, 529, 66)
ABS_CENTERS = np.arange(0, 529, 66)
PROFILE_OUTPUT = np.arange(0, 397, 66)


def band_energy_profile(delta, eps, bands=BANDS):
    """Return E[site, band] from a delta history of shape (ns, N, 3)."""
    ns = delta.shape[0]
    window = np.hanning(ns)[:, None, None]
    spectrum = np.fft.rfft(delta * window, axis=0)
    frequency = np.fft.rfftfreq(ns, DT_SAMPLE) / 1e12
    result = np.empty((N, len(bands)), dtype=np.float64)
    power = np.abs(spectrum) ** 2
    for band_index, (low, high) in enumerate(bands):
        mask = (frequency >= low) & (frequency < high)
        result[:, band_index] = power[mask].sum(axis=(0, 2)) / eps**2
    return result


def spatial_profile(energy, n0):
    """Average opposite bins and return values indexed by abs distance."""
    site = np.arange(N)
    s = ((site - n0 + N // 2) % N) - N // 2
    bins = np.full((len(BIN_CENTERS), energy.shape[1]), np.nan)
    for index, center in enumerate(BIN_CENTERS):
        selected = (s >= center - 33) & (s < center + 33)
        bins[index] = energy[selected].mean(axis=0)
    profile = np.empty((len(ABS_CENTERS), energy.shape[1]))
    for index, distance in enumerate(ABS_CENTERS):
        positive = np.where(BIN_CENTERS == distance)[0][0]
        negative = np.where(BIN_CENTERS == -distance)[0][0]
        if distance == 0:
            profile[index] = bins[positive]
        else:
            profile[index] = 0.5 * (bins[negative] + bins[positive])
    return profile


def load_energies(label, n0, ns):
    config = RUNS[label]
    background = load_history(Path(config["background"]))
    pulse_specs = dict(config["pulses"])
    pulse = load_history(Path(pulse_specs[n0]))
    samples = min(ns, background.shape[0], pulse.shape[0])
    if samples < ns:
        return None
    pulse = pulse[:ns]
    background = background[:ns]
    drift = relative_drift(pulse, config["ratio"])
    if drift > DRIFT_LIMIT:
        return None
    delta = pulse - background
    return band_energy_profile(delta, config["eps"])


def accepted_n0(label):
    """Keep the same pulse acceptance rule as transmision.py."""
    config = RUNS[label]
    result = []
    for n0, filename in config["pulses"]:
        pulse = load_history(Path(filename))
        drift = relative_drift(pulse, config["ratio"])
        if drift <= DRIFT_LIMIT:
            result.append(n0)
    return result


def combine_energies(label, n0s, ns):
    profiles = []
    totals = []
    for n0 in n0s:
        energy = load_energies(label, n0, ns)
        if energy is None:
            continue
        profiles.append(spatial_profile(energy, n0))
        totals.append(energy.sum(axis=0))
    if not profiles:
        return None
    return np.mean(profiles, axis=0), np.mean(totals, axis=0)


def print_result(label, n0s, reference, values_60, values_80):
    profile_60, total_60 = values_60
    profile_80, total_80 = values_80
    ref_profile_60, ref_total_60 = reference[6000]
    ref_profile_80, ref_total_80 = reference[8000]
    transmission = profile_80 / ref_profile_80
    transmission_60 = profile_60 / ref_profile_60
    total_transmission = total_80 / ref_total_80
    logs = np.log10(np.maximum(transmission, np.finfo(float).tiny))
    change = np.log10(np.maximum(transmission, np.finfo(float).tiny)) - np.log10(
        np.maximum(transmission_60, np.finfo(float).tiny)
    )
    distances = ABS_CENTERS
    fit_mask = (distances >= 132) & (distances <= 396)
    print(f"\n{label}" + (f" n0={n0s[0]}" if len(n0s) == 1 else " n0=promedio"))
    for band_index, band_label in enumerate(BAND_LABELS):
        fit_values = np.log(np.maximum(transmission[fit_mask, band_index], np.finfo(float).tiny))
        slope = np.polyfit(distances[fit_mask], fit_values, 1)[0]
        xi = -2.0 / slope if slope < 0 else np.nan
        row = " ".join(f"{logs[i, band_index]:6.2f}" for i in range(len(PROFILE_OUTPUT)))
        delta_row = " ".join(
            f"{change[i, band_index]:+6.2f}" for i in (PROFILE_OUTPUT == 198).nonzero()[0]
        )
        delta_row += " " + " ".join(
            f"{change[i, band_index]:+6.2f}" for i in (PROFILE_OUTPUT == 330).nonzero()[0]
        )
        xi_text = f"{xi:6.1f}" if np.isfinite(xi) else "   n/a"
        print(
            f" {band_label:>6} | {row} | d198,d330 {delta_row} | "
            f"anillo {np.log10(max(total_transmission[band_index], np.finfo(float).tiny)):6.2f} | "
            f"xi {xi_text}"
        )


def main():
    accepted = {label: accepted_n0(label) for label in ("R", "A", "C")}
    for label, n0s in accepted.items():
        print(f"{label}: n0 aceptados={n0s}")
    reference = {}
    for ns in WINDOWS:
        result = combine_energies("R", accepted["R"], ns)
        if result is None:
            raise RuntimeError(f"No hay pulsos R aceptados para ns={ns}")
        reference[ns] = result
    print_result("R", accepted["R"], reference, reference[6000], reference[8000])

    for label in ("A", "C"):
        groups = [[n0] for n0 in accepted[label]] if label == "A" else [accepted[label]]
        for n0s in groups:
            reference = {}
            reference_n0s = n0s if label == "A" else accepted["R"]
            for ns in WINDOWS:
                result = combine_energies("R", reference_n0s, ns)
                if result is None:
                    raise RuntimeError(f"No hay pulsos R aceptados para ns={ns}, n0={reference_n0s}")
                reference[ns] = result
            values = {}
            for ns in WINDOWS:
                result = combine_energies(label, n0s, ns)
                if result is None:
                    raise RuntimeError(f"No hay datos aceptados para {label}, n0={n0s}, ns={ns}")
                values[ns] = result
            print_result(label, n0s, reference, values[6000], values[8000])


if __name__ == "__main__":
    main()
