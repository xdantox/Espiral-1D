"""Folded PBC minimizer (self-contained, minimal).

Goal: a single-file module (like direct3_PBC.py) that:
- Defines the folded Fourier ansatz for theta on half-chain + mirror.
- Adds a flexible mirror mismatch field delta(i) via truncated Fourier series.
- Evaluates the folded Hamiltonian energy under PBC.
- Minimizes parameters for a given nu, and optionally scans nu.

Model conventions (per-site averages):
- NN: (Jbar +/- dJ) * dot(S_i, S_{i+1})
- NNN: J2 * dot(S_i, S_{i+2})
- anis: D * Sx_i^2
- biquad: K * dot(S_i, S_{i+1})^2
- folded: (Jperp/2) * < dot(S_i, S_{N-1-i}) >
  (factor 1/2 avoids double counting when averaging over all i)

Spin parameterization:
- Canting field mx(i) = Sx(i) in [-1,1], remaining weight in YZ plane.
- Theta is the angle in the YZ plane.

Parameter layout for the minimizer (length depends on K_mx, K_delta, M_harm):
    p = [mx0,
             A_mx(1..K_mx), B_mx(1..K_mx),
             Q_nat, gamma, delta0,
             A_delta(1..K_delta), B_delta(1..K_delta),
             A_theta_cos(1..M_harm), B_theta_sin(1..M_harm)]

where:
- Q_sol = 4*pi*nu/N
- Q_sol = 4*pi*nu/N
- half-theta(i) = Q_nat*i + gamma*(-1)^i + Σ_m [A_cos[m]*cos(m*Q_sol*i) + B_sin[m]*sin(m*Q_sol*i)]
- delta(i) = delta0 + Σ_k A_delta[k]*cos(2π k i/N) + B_delta[k]*sin(2π k i/N)
- theta(second branch) is mirrored from the first branch plus delta(i).

"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


# (name, Jbar, dJ, J2, K, D, N, J_perp)
DEFAULT_SETS = [
    ("Set 1 (MoBr3 Dimerized)", 48.891035, 48.620365, 1.26126, 48.9119, 0.31, 1198, 0.03),
]


def q_sol_from_domains(nu: float, N: int) -> float:
    return float((4.0 * math.pi * float(nu)) / float(N))


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def fourier_field(N: int, offset: float, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float).reshape(-1)
    B = np.asarray(B, dtype=float).reshape(-1)
    if A.size != B.size:
        raise ValueError("A and B must have same length")

    K = int(A.size)
    out = np.full(int(N), float(offset), dtype=float)
    if K == 0:
        return out

    i = np.arange(int(N), dtype=float)
    for k in range(1, K + 1):
        ang = (2.0 * math.pi * float(k) * i) / float(N)
        out = out + float(A[k - 1]) * np.cos(ang) + float(B[k - 1]) * np.sin(ang)
    return out


def _fold_theta_from_half(N: int, half_theta: np.ndarray, delta_full: np.ndarray) -> np.ndarray:
    N = int(N)
    half_theta = np.asarray(half_theta, dtype=float).reshape(-1)
    delta_full = np.asarray(delta_full, dtype=float).reshape(-1)
    if delta_full.size != N:
        raise ValueError("delta_full must have length N")

    mid = N // 2
    half_len = mid if (N % 2 == 0) else (mid + 1)
    if half_theta.size != half_len:
        raise ValueError("half_theta has wrong length")

    second_len = N - half_len
    second_idx = np.arange(second_len, dtype=np.int64) + half_len

    # Mirror mapping:
    # - if N even: second branch mirrors indices 0..half_len-1
    # - if N odd: the middle site is unpaired; second branch mirrors 0..half_len-2
    if N % 2 == 0:
        mirror_idx = (half_len - 1) - np.arange(second_len, dtype=np.int64)
    else:
        mirror_idx = (half_len - 2) - np.arange(second_len, dtype=np.int64)

    theta = np.empty(N, dtype=float)
    theta[:half_len] = half_theta
    theta[second_idx] = half_theta[mirror_idx] + delta_full[second_idx]
    return theta


def theta_n_folded_fourier_deltafield(
    sites: Iterable[int] | np.ndarray,
    nu: float,
    Q_nat: float,
    gamma: float,
    delta0: float,
    A_delta: np.ndarray,
    B_delta: np.ndarray,
    A_theta_cos: np.ndarray,
    B_theta_sin: np.ndarray,
) -> np.ndarray:
    idx = np.asarray(sites, dtype=np.int64)
    N = int(idx.size)
    if N <= 0:
        raise ValueError("sites must be non-empty")

    A_theta_cos = np.asarray(A_theta_cos, dtype=float).reshape(-1)
    B_theta_sin = np.asarray(B_theta_sin, dtype=float).reshape(-1)
    if A_theta_cos.size != B_theta_sin.size:
        raise ValueError("A_theta_cos and B_theta_sin must have same length")

    mid = N // 2
    half_len = mid if (N % 2 == 0) else (mid + 1)
    half_idx = np.arange(half_len, dtype=np.int64)

    base = float(Q_nat) * half_idx.astype(float)
    parity = np.where((half_idx & 1) == 0, 1.0, -1.0)
    half_theta = base + float(gamma) * parity

    if A_theta_cos.size:
        m = np.arange(1, A_theta_cos.size + 1, dtype=float)
        Q_sol = q_sol_from_domains(nu, N)
        arg = (m[:, None] * Q_sol * half_idx[None, :].astype(float))
        half_theta = half_theta + np.sum(
            (A_theta_cos[:, None] * np.cos(arg)) + (B_theta_sin[:, None] * np.sin(arg)),
            axis=0,
        )

    delta_full = fourier_field(N, float(delta0), np.asarray(A_delta), np.asarray(B_delta))
    return _fold_theta_from_half(N, half_theta, delta_full)


def mx_field_fourier(N: int, mx0: float, A_mx: np.ndarray, B_mx: np.ndarray) -> np.ndarray:
    mx = fourier_field(int(N), float(mx0), np.asarray(A_mx), np.asarray(B_mx))
    return np.clip(mx, -0.999, 0.999)


def energy_components_folded_fourier_deltafield_finite(
    nu: float,
    params: np.ndarray,
    Jbar: float,
    dJ: float,
    J2: float,
    K: float,
    D: float,
    Jperp: float,
    sites: np.ndarray,
    M_harm: int,
    K_delta: int,
    K_mx: int,
) -> Tuple[float, float, float, float, float, float]:
    sites = np.asarray(sites, dtype=np.int64)
    N = int(sites.size)
    if N <= 0:
        raise ValueError("sites must be non-empty")

    M_harm = int(M_harm)
    K_delta = int(K_delta)
    K_mx = int(K_mx)

    expected = 4 + 2 * K_mx + 2 * K_delta + 2 * M_harm
    p = np.asarray(params, dtype=float).reshape(-1)
    if p.size != expected:
        raise ValueError(
            f"params length must be {expected} for K_mx={K_mx}, K_delta={K_delta}, M_harm={M_harm}"
        )

    mx0 = float(p[0])
    off = 1
    A_mx = p[off : off + K_mx]; off += K_mx
    B_mx = p[off : off + K_mx]; off += K_mx

    Q_nat = float(p[off]); off += 1
    gamma = float(p[off]); off += 1
    delta0 = float(p[off]); off += 1

    A_delta = p[off : off + K_delta]; off += K_delta
    B_delta = p[off : off + K_delta]; off += K_delta
    A_theta_cos = p[off : off + M_harm]; off += M_harm
    B_theta_sin = p[off : off + M_harm]

    theta = theta_n_folded_fourier_deltafield(
        sites,
        nu,
        Q_nat,
        gamma,
        delta0,
        A_delta,
        B_delta,
        A_theta_cos,
        B_theta_sin,
    )

    mx = mx_field_fourier(N, mx0, A_mx, B_mx)
    plane = np.sqrt(np.maximum(0.0, 1.0 - mx**2))

    # NN
    d1 = np.roll(theta, -1) - theta
    mx1 = np.roll(mx, -1)
    plane1 = np.roll(plane, -1)
    dot1 = (mx * mx1) + (plane * plane1) * np.cos(d1)
    bonds = np.where((sites & 1) == 0, Jbar + dJ, Jbar - dJ)
    exch_nn = float(np.mean(bonds * dot1))
    biquad = float(K * np.mean(dot1**2))

    # NNN
    d2 = np.roll(theta, -2) - theta
    mx2 = np.roll(mx, -2)
    plane2 = np.roll(plane, -2)
    dot2 = (mx * mx2) + (plane * plane2) * np.cos(d2)
    exch_nnn = float(J2 * np.mean(dot2))

    # anis
    anis = float(D * np.mean(mx**2))

    # folded coupling (mirror)
    dotp = (mx * mx[::-1]) + (plane * plane[::-1]) * np.cos(theta - theta[::-1])
    exch_perp = float(0.5 * Jperp * np.mean(dotp))

    total = float(exch_nn + biquad + exch_nnn + anis + exch_perp)
    return total, exch_nn, biquad, exch_nnn, anis, exch_perp


def minimize_folded_fourier_deltafield_parameters(
    nu: float,
    chain_length: int,
    Jbar: float,
    dJ: float,
    J2: float,
    K: float,
    D: float,
    Jperp: float,
    M_harm: int = 20,
    K_delta: int = 8,
    K_mx: int = 10,
    x0: Sequence[float] | np.ndarray | None = None,
    method: str = "L-BFGS-B",
    options=None,
) -> Tuple[float, np.ndarray, bool]:
    N = int(chain_length)
    if N <= 0:
        raise ValueError("chain_length must be positive")

    M_harm = int(M_harm)
    K_delta = int(K_delta)
    K_mx = int(K_mx)

    n_params = 4 + 2 * K_mx + 2 * K_delta + 2 * M_harm
    init = np.zeros(n_params, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
    if init.size != n_params:
        raise ValueError(f"x0 must have length {n_params}")

    # A reasonable default seed (avoid the all-zero stationary point).
    if x0 is None:
        init[0] = 0.0                 # mx0
        # A_mx and B_mx default to 0
        off0 = 1 + 2 * K_mx
        init[off0 + 0] = 2.0 * math.pi / 3.0  # Q_nat
        init[off0 + 1] = -0.30                # gamma
        init[off0 + 2] = math.pi              # delta0

    # Bounds
    b = []
    b += [(-0.999, 0.999)]  # mx0
    # mx harmonics: allow local peaks ~0.2 while keeping mean near 0
    b += [(-0.6, 0.6)] * (2 * K_mx)
    b += [(0.0, 2.0 * math.pi)]  # Q_nat
    b += [(-math.pi, math.pi)]   # gamma
    b += [(-math.pi, math.pi)]   # delta0
    b += [(-8.0, 8.0)] * (2 * K_delta)  # delta harmonics
    b += [(-math.pi, math.pi)] * (2 * M_harm)  # theta A and phi

    sites = np.arange(N, dtype=np.int64)

    def objective(vec: np.ndarray) -> float:
        return energy_components_folded_fourier_deltafield_finite(
            nu, vec, Jbar, dJ, J2, K, D, Jperp, sites, M_harm, K_delta, K_mx
        )[0]

    res = minimize(
        objective,
        x0=init,
        method=method,
        bounds=b,
        options=options or {"maxiter": 50000, "ftol": 1e-14},
    )

    # Return best known point even if success=False (SciPy can be conservative).
    xbest = np.asarray(res.x if getattr(res, "x", None) is not None else init, dtype=float)
    fbest = float(getattr(res, "fun", objective(xbest)))

    # Layout: [mx0, A_mx, B_mx, Q_nat, gamma, delta0, A_delta, B_delta, A_theta_cos, B_theta_sin]

    return fbest, xbest, bool(getattr(res, "success", False))


def minimize_folded_best_nu(
    nu_values: Iterable[int] | range,
    chain_length: int,
    Jbar: float,
    dJ: float,
    J2: float,
    K: float,
    D: float,
    Jperp: float,
    M_harm: int = 40,
    K_delta: int = 10,
    K_mx: int = 10,
) -> Tuple[int, float, np.ndarray, bool]:
    best = None
    best_nu = None
    best_p = None
    best_ok = False

    for nu in nu_values:
        e, p, ok = minimize_folded_fourier_deltafield_parameters(
            float(nu),
            int(chain_length),
            Jbar,
            dJ,
            J2,
            K,
            D,
            Jperp,
            M_harm=int(M_harm),
            K_delta=int(K_delta),
            K_mx=int(K_mx),
            x0=None,
        )
        if best is None or e < best:
            best = float(e)
            best_nu = int(nu)
            best_p = np.asarray(p, dtype=float)
            best_ok = bool(ok)
        print(f"nu={nu:g} | E={e:.12f} | ok={ok}")

    if best is None or best_nu is None or best_p is None:
        raise RuntimeError("No nu values evaluated")
    return best_nu, float(best), best_p, bool(best_ok)


def _main() -> np.ndarray:
    p = argparse.ArgumentParser(description="Folded PBC minimizer (theta harmonics + delta harmonics).")
    p.add_argument("--set", type=int, default=0)
    p.add_argument("--nu", type=float, default=3, help="If set, minimize only this nu")
    p.add_argument("--nu-min", type=int, default=1)
    p.add_argument("--nu-max", type=int, default=6)
    p.add_argument("--M", type=int, default=3, help="theta harmonics")
    p.add_argument("--Kdelta", type=int, default=3, help="delta harmonics")
    p.add_argument("--Kmx", type=int, default=3, help="mx(i) harmonics")
    args = p.parse_args()

    name, Jbar, dJ, J2, K, D, N, Jperp = DEFAULT_SETS[int(args.set)]
    N = int(N)

    print(f"Set: {name} | N={N} | Jperp={Jperp}")
    print(f"M={args.M} | Kdelta={args.Kdelta} | Kmx={args.Kmx}")

    if args.nu is not None:
        e, params, ok = minimize_folded_fourier_deltafield_parameters(
            float(args.nu),
            N,
            Jbar,
            dJ,
            J2,
            K,
            D,
            Jperp,
            M_harm=int(args.M),
            K_delta=int(args.Kdelta),
            K_mx=int(args.Kmx),
        )
        print(f"nu={float(args.nu):g} | E={e:.12f} | ok={ok}")
        print("out=", params)
        return np.asarray(params, dtype=float)
    else:
        nu_best, e_best, params_best, ok_best = minimize_folded_best_nu(
            range(int(args.nu_min), int(args.nu_max) + 1),
            N,
            Jbar,
            dJ,
            J2,
            K,
            D,
            Jperp,
            M_harm=int(args.M),
            K_delta=int(args.Kdelta),
            K_mx=int(args.Kmx),
        )
        print(f"\nBest nu={nu_best} | E={e_best:.12f} | ok={ok_best}")
        print("out=", params_best)
        return np.asarray(params_best, dtype=float)


if __name__ == "__main__":
    out = _main()
