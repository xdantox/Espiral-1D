#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estabilidad termica de los arreglos de paredes: LLG estocastica con baño de Langevin.

Estados de partida (ultimo fotograma, D_plane = 2.2 D, plano XZ):
  A) "D_plane = 2.2D relaxYZ n = 1188.npy"                    (regular, 18 paredes)
  C) "D_plane = 2.2D relaxYZ n = 1188 noise=0.04 seed1.npy"   (irregular, ~46 paredes)

Ecuacion (la misma forma Landau-Lifshitz que chain.py), con F = -dH/dS + xi(t):
    dS/dt = eta S x F - eta alpha S x (S x F)  =  eta S x F + eta alpha [F - (S.F) S]
    eta = gamma / mu_B = 2.94e12 s^-1 meV^-1,  alpha = 1.3.

AMPLITUD DEL RUIDO (la parte delicada). Para esta forma LL, el ruido blanco gaussiano
    <xi_a(t) xi_b(t')> = 2 Theta delta_ab delta(t - t'),   Theta = alpha k_B T / ( eta (1 + alpha^2) )
reproduce la distribucion de Boltzmann. (Con 2 alpha k_B T/(gamma S) a secas, valido para la forma de Gilbert,
la temperatura efectiva saldria (1 + alpha^2) = 2.69 veces mayor.) Se verifica de todos modos, antes de las
corridas, con la hélice uniforme sin paredes: <H - H0>/N debe ser k_B T (2 grados de libertad por espin,
k_B T/2 cada uno). Se reporta el cociente medido/esperado.

Integrador: Heun estocastico (consistente con Stratonovich), ruido fijo durante cada paso, renormalizacion de |S|.
Paredes: cada SNAP_PS ps se promedia (cada SAMPLE_EVERY_FS fs) la pendiente local de theta y el angulo phi,
y se aplica detectar_solitones de relax3D.py al promedio.

Uso:  python stab_termica.py            (verificacion + corridas)
      python stab_termica.py check      (solo verificacion de equiparticion)
Requiere relax3D.py en la misma carpeta.
"""

import sys
import time

import numpy as np

from relax3D import CONFIG, detectar_solitones

# ------------------------------- parametros ---------------------------------
N = 1188
KBOLT = 0.08617333262                 # meV / K
GAMMA = 1.7e11                        # rad s^-1 T^-1
MU_B = 0.05788                        # meV / T
ETA = GAMMA / MU_B                    # s^-1 meV^-1
ALPHA = 1.3
DT_FS = 1.0
DT = DT_FS * 1e-15                    # s
RATIO = 2.2                           # D_plane / D de la simulacion

T_LIST = (0.1, 1.0, 4.0)              # K
SEEDS = (1, 2, 3)
T_TOTAL_PS = 300.0
SNAP_PS = 5.0
SAMPLE_EVERY_FS = 50.0
JUMP_SITES = 2                        # salto de pared considerado: > JUMP_SITES sitios

CHECK_T_LIST = (0.1, 1.0)             # K, temperaturas de la verificacion de equiparticion
RELAX_PS = 5.0                        # relajacion sin ruido antes de medir
EQUIL_PS = 20.0
MEAS_PS = 40.0
CHECK_TOLERANCE = 0.15                # |cociente - 1| <= 0.15 para dar por buena la verificacion
FORCE = False                         # True: ejecuta las corridas aunque falle la verificacion

STATES = {
    "A": "D_plane = 2.2D relaxYZ n = 1188.npy",
    "C": "D_plane = 2.2D relaxYZ n = 1188 noise=0.04 seed1.npy",
}
UNIFORM_FILES = ("D_plane = 2.2D relax n = 1188.npy", "D_plane = 2.2D background n = 1188.npy")
OUTPUT_LOG = "stab_termica_resultados.txt"

J_BOND = np.where(np.arange(N) % 2 == 0, CONFIG["J_NN_1"], CONFIG["J_NN_2"])
J_NNN = CONFIG["J_NNN"]
K_BIQ = CONFIG["K"]
D_AX = CONFIG["D"]
D_PL = RATIO * CONFIG["D"]


# ------------------------------ fisica basica --------------------------------
def campo(S):
    """G = dH/dS (meV) para H(1) con PBC."""
    d1 = np.einsum("ij,ij->i", S, np.roll(S, -1, 0))
    a = (J_BOND + 2.0 * K_BIQ * d1)[:, None]
    G = (a * np.roll(S, -1, 0) + np.roll(a, 1, 0) * np.roll(S, 1, 0)
         + J_NNN * (np.roll(S, -2, 0) + np.roll(S, 2, 0)))
    G[:, 0] += 2.0 * D_AX * S[:, 0]
    G[:, 1] += 2.0 * D_PL * S[:, 1]
    return G


def energia(S):
    d1 = np.einsum("ij,ij->i", S, np.roll(S, -1, 0))
    d2 = np.einsum("ij,ij->i", S, np.roll(S, -2, 0))
    return float(np.sum(J_BOND * d1 + K_BIQ * d1 ** 2 + J_NNN * d2
                        + D_AX * S[:, 0] ** 2 + D_PL * S[:, 1] ** 2))


def deriv(S, F):
    SF = np.einsum("ij,ij->i", S, F)[:, None]
    return ETA * (np.cross(S, F) + ALPHA * (F - SF * S))


def sigma_xi(T_K):
    """Desviacion tipica (meV) del campo aleatorio por componente y por paso de tiempo DT."""
    if T_K <= 0:
        return 0.0
    theta = ALPHA * KBOLT * T_K / (ETA * (1.0 + ALPHA ** 2))          # meV^2 s
    return float(np.sqrt(2.0 * theta / DT))


def paso_heun(S, sig, rng):
    xi = sig * rng.standard_normal(S.shape) if sig > 0 else 0.0
    F1 = -campo(S) + xi
    k1 = deriv(S, F1)
    S1 = S + DT * k1
    S1 /= np.linalg.norm(S1, axis=1, keepdims=True)
    F2 = -campo(S1) + xi
    k2 = deriv(S1, F2)
    S = S + 0.5 * DT * (k1 + k2)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    return S


def cargar_estado(ruta):
    raw = np.load(ruta, mmap_mode="r")
    S = np.array(raw[-1] if raw.ndim == 3 else raw, dtype=np.float64)
    if S.shape != (N, 3):
        raise ValueError(f"{ruta}: forma inesperada {S.shape}")
    return S / np.linalg.norm(S, axis=1, keepdims=True)


# ---------------------------- deteccion de paredes ----------------------------
def acumular(S, acc_dth, acc_ph):
    th = np.arctan2(S[:, 0], S[:, 2])                                  # plano XZ
    acc_dth += np.angle(np.exp(1j * (np.roll(th, -1) - th)))
    acc_ph += np.arcsin(np.clip(S[:, 1], -1.0, 1.0))


def posiciones(acc_dth, acc_ph, count):
    theta = np.concatenate(([0.0], np.cumsum(acc_dth[:-1] / count)))
    phi = acc_ph / count
    solitones, _ = detectar_solitones(theta, phi)
    return np.array(sorted(s["core_bond"] for s in solitones), dtype=int)


def posiciones_estado(S):
    acc_dth, acc_ph = np.zeros(N), np.zeros(N)
    acumular(S, acc_dth, acc_ph)
    return posiciones(acc_dth, acc_ph, 1)


def desplazamientos(c0, c):
    """Distancia circular de cada pared inicial a la pared mas cercana del instante actual."""
    if c.size == 0:
        return np.full(c0.size, np.nan)
    d = np.abs(c0[:, None] - c[None, :]) % N
    return np.minimum(d, N - d).min(axis=1)


# --------------------------- verificacion de equiparticion --------------------
def verificar_equiparticion():
    S_ref = None
    for nombre in UNIFORM_FILES:
        try:
            S_ref = cargar_estado(nombre)
            print(f"[check] helice uniforme: {nombre}")
            break
        except (FileNotFoundError, OSError):
            continue
    if S_ref is None:
        raise FileNotFoundError("No se encontro ninguna helice uniforme: " + ", ".join(UNIFORM_FILES))

    resultados = []
    ms_por_paso = None
    for T_K in CHECK_T_LIST:
        rng = np.random.default_rng([12345, int(round(T_K * 1000))])
        S = S_ref.copy()
        for _ in range(int(RELAX_PS * 1000 / DT_FS)):
            S = paso_heun(S, 0.0, rng)
        H0 = energia(S)
        sig = sigma_xi(T_K)
        for _ in range(int(EQUIL_PS * 1000 / DT_FS)):
            S = paso_heun(S, sig, rng)
        cada = max(1, int(SAMPLE_EVERY_FS / DT_FS))
        pasos = int(MEAS_PS * 1000 / DT_FS)
        valores = []
        t0 = time.time()
        for i in range(pasos):
            S = paso_heun(S, sig, rng)
            if (i + 1) % cada == 0:
                valores.append((energia(S) - H0) / N)
        ms_por_paso = 1e3 * (time.time() - t0) / pasos
        medido = float(np.mean(valores))
        esperado = KBOLT * T_K
        cociente = medido / esperado
        resultados.append((T_K, cociente))
        print(f"[check] T={T_K:g} K: <H-H0>/N = {medido:.4e} meV; esperado k_B T = {esperado:.4e} meV; "
              f"cociente medido/esperado = {cociente:.3f}  (sigma_xi = {sig:.3f} meV)", flush=True)
    ok = all(abs(c - 1.0) <= CHECK_TOLERANCE for _, c in resultados)
    if ok:
        print("[check] OK: el ruido reproduce k_B T por espin dentro de la tolerancia.")
    else:
        print("[check] FALLO. Guia: cociente ~ 2.7 -> falta el factor 1/(1+alpha^2) en el ruido; "
              "cociente ~ 0.37 -> sobra; cociente algo distinto de 1 solo a 1 K y ~1 a 0.1 K -> anarmonicidad; "
              "un cociente que cambie al usar DT_FS = 0.5 -> error de integracion.")
    return ok, ms_por_paso


# ---------------------------------- corridas ------------------------------------
def correr(label, S0, T_K, seed):
    rng = np.random.default_rng([seed, int(round(T_K * 1000)), ord(label)])
    sig = sigma_xi(T_K)
    c0 = posiciones_estado(S0)
    ns0 = int(c0.size)
    total = int(T_TOTAL_PS * 1000 / DT_FS)
    snap = int(SNAP_PS * 1000 / DT_FS)
    cada = max(1, int(SAMPLE_EVERY_FS / DT_FS))
    S = S0.copy()
    acc_dth, acc_ph, cnt = np.zeros(N), np.zeros(N), 0
    registros = {}
    evento = None
    for paso in range(1, total + 1):
        S = paso_heun(S, sig, rng)
        if paso % cada == 0:
            acumular(S, acc_dth, acc_ph)
            cnt += 1
        if paso % snap == 0:
            t_ps = paso * DT_FS / 1000.0
            c = posiciones(acc_dth, acc_ph, max(cnt, 1))
            disp = desplazamientos(c0, c)
            rms = float(np.sqrt(np.nanmean(disp ** 2))) if np.any(np.isfinite(disp)) else float("nan")
            maxd = float(np.nanmax(disp)) if np.any(np.isfinite(disp)) else float("nan")
            registros[round(t_ps)] = (int(c.size), rms)
            if evento is None:
                if c.size < ns0:
                    evento = f"aniquilacion a {t_ps:.0f} ps"
                elif maxd > JUMP_SITES:
                    evento = f"salto >{JUMP_SITES} sitios a {t_ps:.0f} ps"
            acc_dth[:] = 0.0
            acc_ph[:] = 0.0
            cnt = 0
    def ns_en(t):
        return str(registros[t][0]) if t in registros else "n/a"
    def rms_en(t):
        return f"{registros[t][1]:.1f}" if t in registros else "n/a"
    linea = (f"{label} | T={T_K:g} K | semilla {seed} | Ns(0/100/200/300 ps)={ns0}/{ns_en(100)}/{ns_en(200)}/{ns_en(300)}"
             f" | rms(100/200/300)={rms_en(100)}/{rms_en(200)}/{rms_en(300)} sitios"
             f" | primer evento: {evento if evento else 'ninguno'}")
    return linea, ns0, registros, evento


def main():
    solo_check = len(sys.argv) > 1 and sys.argv[1].lower() == "check"
    print(f"eta = {ETA:.4e} s^-1 meV^-1; alpha = {ALPHA}; dt = {DT_FS} fs; "
          f"Theta = alpha k_B T /(eta (1 + alpha^2))")
    ok, ms = verificar_equiparticion()
    if solo_check:
        return
    if not ok and not FORCE:
        print("Se detienen las corridas: corrige el ruido o pon FORCE = True para ejecutarlas igualmente.")
        return
    n_corridas = len(STATES) * len(T_LIST) * len(SEEDS)
    if ms:
        pasos = int(T_TOTAL_PS * 1000 / DT_FS)
        print(f"Estimacion: {ms:.2f} ms/paso -> {ms * pasos / 1000 / 60:.1f} min por corrida, "
              f"{ms * pasos * n_corridas / 1000 / 3600:.1f} h en total ({n_corridas} corridas).", flush=True)

    resumen = {}
    with open(OUTPUT_LOG, "a", encoding="utf-8") as log:
        for label, nombre in STATES.items():
            try:
                S0 = cargar_estado(nombre)
            except (FileNotFoundError, OSError):
                print(f"{label}: falta {nombre}; omitido")
                continue
            for T_K in T_LIST:
                for seed in SEEDS:
                    linea, ns0, registros, evento = correr(label, S0, T_K, seed)
                    print(linea, flush=True)
                    log.write(linea + "\n")
                    log.flush()
                    ns_final = registros[round(T_TOTAL_PS)][0] if round(T_TOTAL_PS) in registros else None
                    resumen.setdefault((label, T_K), []).append((ns0, ns_final, evento))
    print("\nRESUMEN (por estado y temperatura): Ns(0) -> Ns(final) por semilla; corridas con algun evento")
    for (label, T_K), filas in resumen.items():
        finales = [f"{a}->{b}" for a, b, _ in filas]
        con_evento = sum(1 for _, _, e in filas if e)
        print(f"{label} T={T_K:g} K: {', '.join(finales)}; eventos: {con_evento}/{len(filas)}")


if __name__ == "__main__":
    main()