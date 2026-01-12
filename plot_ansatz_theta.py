import argparse
import importlib
import os
import numpy as np
import matplotlib.pyplot as plt


def _wrap_pi(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def main():
    parser = argparse.ArgumentParser(description="Plot q_local per bond for the folded Fourier ansatz.")
    parser.add_argument("--nu", type=float, default=1.0, help="Domains per branch (half-chain).")
    parser.add_argument("--M", type=int, default=30, help="Number of harmonics (M_harm).")
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap q_local to (-pi, pi] for plotting (optional).",
    )
    parser.add_argument("--no-opt", action="store_true", help="Skip optimization and use x0 as-is.")
    parser.add_argument(
        "--x0",
        type=float,
        nargs="+",
        default=None,
        help="Initial parameter vector: [mx, Q_nat, gamma, delta, A_1..A_M, phi_1..phi_M]",
    )
    parser.add_argument("--out", type=str, default="q_local_ansatz.png", help="Output PNG filename.")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    import direct3_PBC_folded as f
    # En IPython, `run plot_ansatz_theta.py` reutiliza el mismo intérprete y
    # puede mantener un módulo cacheado. Forzamos reload para tomar los cambios.
    f = importlib.reload(f)

    name, Jbar, dJ, J2, K, D, N, Jperp = f.DEFAULT_SETS[0]
    N = int(N)

    n_params = 4 + 2 * int(args.M)
    if args.x0 is None:
        x0 = np.zeros(n_params, dtype=float)
        x0[0] = 0.0
        x0[1] = 2.0 * np.pi / 3.0
        x0[2] = -0.30
        x0[3] = np.pi
    else:
        x0 = np.asarray(args.x0, dtype=float)
        if x0.size != n_params:
            raise ValueError(f"--x0 must have length {n_params} for M={args.M}")

    if args.no_opt:
        params = x0
        ok = True
        e = float(
            f.energy_components_folded_fourier_finite(
                args.nu,
                params,
                Jbar,
                dJ,
                J2,
                K,
                D,
                Jperp,
                np.arange(N, dtype=np.int64),
                args.M,
            )[0]
        )
    else:
        e, params, ok = f.minimize_folded_fourier_parameters(
            args.nu,
            N,
            Jbar,
            dJ,
            J2,
            K,
            D,
            Jperp,
            M_harm=args.M,
            x0=x0,
        )

    Q_sol = f.q_sol_from_domains(args.nu, N)
    sites = np.arange(N, dtype=np.int64)
    A = params[4 : 4 + args.M]
    phi = params[4 + args.M : 4 + 2 * args.M]
    # IMPORTANTE: este script optimiza con minimize_folded_fourier_parameters,
    # por lo que debe construir theta con el MISMO ansatz (theta-Fourier).
    theta = f.theta_n_folded_fourier(sites, params[1], params[2], Q_sol, params[3], A, phi)

    # q_local definido de forma consistente con plot_q_local.py / analisis.py:
    # q_local = diff(unwrap(theta))
    theta_unwrapped = np.unwrap(theta)
    q_local = np.diff(theta_unwrapped)
    q_plot = _wrap_pi(q_local) if args.wrap else q_local

    plt.figure(figsize=(10, 4.2))
    x = np.arange(q_plot.size)
    plt.plot(x, q_plot, lw=1.0)
    plt.axhline(0.0, color="k", lw=0.7, alpha=0.6)
    plt.title(
        f"Ansatz q_local por enlace | {name} | nu={args.nu:g}, M={args.M} | E={e:.6f} | ok={ok}"
    )
    plt.xlabel("índice de enlace i (entre sitio i e i+1)")
    plt.ylabel("q_local [rad]" + (" (wrapped)" if args.wrap else ""))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(args.out, dpi=200)
    plt.show()
    print(f"Saved: {args.out}")
    print(f"E={e:.12f}, ok={ok}")
    print(
        f"mx={params[0]:.6f}, Q_nat={params[1]:.12f}, gamma={params[2]:.6f}, delta={params[3]:.6f}, Q_sol={Q_sol:.12f}"
    )


if __name__ == "__main__":
    main()
