
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fig. 1: competencia entre semilla coherente y ruido.
Usa los resultados ya calculados por regular.py. No realiza simulaciones.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Datos del ensamble ----------------

x = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8,
    1.0, 1.2, 1.5, 2.0, 2.5, 4.0
])

Ns_mean = np.array([
    8.3, 13.5, 15.5, 18.2, 19.5, 19.2, 19.0,
    18.5, 18.3, 18.3, 18.2, 18.3, 18.0
])

Ns_std = np.array([
    2.0, 3.0, 2.0, 2.1, 1.4, 1.3, 1.5,
    0.9, 0.7, 0.7, 0.6, 1.1, 0.0
])

P_M = np.array([
    0.000, 0.167, 0.250, 0.417, 0.417, 0.500,
    0.667, 0.750, 0.833, 0.833, 0.917, 0.917, 1.000
])

P_reg = np.array([
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.167, 0.500, 0.667, 0.833, 0.917, 0.917, 1.000
])

# Umbrales
xc = 0.562
xc_reg = 1.10

# Predicción teórica: solamente x <= 0.4
x_theory = np.linspace(0.1, 0.4, 150)
Ns_theory = 31.0 * np.sqrt(x_theory)

# Puntos teóricos en los cuatro valores del ensamble
x_theory_points = x[:4]
Ns_theory_points = 31.0 * np.sqrt(x_theory_points)

# ---------------- Estilo ----------------

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "axes.linewidth": 0.8,
})

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(7.1, 3.25),
    gridspec_kw={"width_ratios": [1.05, 1.0]}
)

# ---------------- Panel (a): número de paredes ----------------

ax1.axhline(
    18, color="C0", linestyle="--", linewidth=1.2,
    label=r"$N_s^{\mathrm{M}}=18$"
)

ax1.plot(
    x_theory, Ns_theory,
    "--", color="C1", linewidth=1.5,
    label=r"$N_s^{\mathrm{noise}}=31\sqrt{x}$"
)

ax1.plot(
    x_theory_points, Ns_theory_points,
    linestyle="none", marker="D", markersize=5,
    markerfacecolor="white", markeredgecolor="C2",
    markeredgewidth=1.2,
    label=r"$31\sqrt{x}$ at ensemble points"
)

ax1.errorbar(
    x, Ns_mean, yerr=Ns_std,
    fmt="o-", color="C0",
    linewidth=1.5, markersize=4.5,
    capsize=2.5, elinewidth=1.0,
    label=r"Ensemble, $\sigma=10^{-3}$"
)

ax1.set_xscale("log")
ax1.set_xlim(0.08, 4.5)
ax1.set_ylim(5.5, 22.5)
ax1.set_xlabel(r"$x=(D_{\mathrm{plane}}-D)/D$")
ax1.set_ylabel(r"Mean wall number $N_s$")
ax1.text(0.025, 0.96, "(a)", transform=ax1.transAxes,
         va="top", fontsize=13, fontweight="bold")

ax1.grid(alpha=0.2)
ax1.legend(loc="lower right", frameon=False)

# ---------------- Panel (b): probabilidades ----------------

ax2.plot(
    x, P_M, "o-", color="C0",
    markersize=4.5, linewidth=1.5,
    label=r"$P(N_s=18)$"
)

ax2.plot(
    x, P_reg, "s-", color="C1",
    markersize=4.5, linewidth=1.5,
    label=r"$P(\delta_L<0.15)$"
)

ax2.axvline(
    xc, color="C0", linestyle=":", linewidth=1.2
)
ax2.axvline(
    xc_reg, color="C1", linestyle=":", linewidth=1.2
)

ax2.text(
    xc * 0.98, 0.69, r"$x_c=0.56$",
    fontsize=10, ha="right"
)
ax2.text(
    xc_reg * 1.04, 0.30, r"$x_c^{\mathrm{reg}}=1.10$",
    fontsize=10, ha="left"
)

ax2.set_xscale("log")
ax2.set_xlim(0.08, 4.5)
ax2.set_ylim(-0.02, 1.05)
ax2.set_xlabel(r"$x=(D_{\mathrm{plane}}-D)/D$")
ax2.set_ylabel("Fraction of runs")
ax2.text(0.025, 0.96, "(b)", transform=ax2.transAxes,
         va="top", fontsize=13, fontweight="bold")

ax2.grid(alpha=0.2)
ax2.legend(loc="lower right", frameon=False)

# ---------------- Guardar ----------------

fig.tight_layout(w_pad=1.7)

out = Path(__file__).resolve().parent
fig.savefig(out / "Fig1.pdf", bbox_inches="tight")
fig.savefig(out / "Fig1.png", dpi=300, bbox_inches="tight")

# También se conserva la tabla usada para generar la figura.
np.savetxt(
    out / "Fig1_datos.csv",
    np.column_stack([x, Ns_mean, Ns_std, P_M, P_reg]),
    delimiter=",",
    header="x,Ns_mean,Ns_std,P_M,P_reg",
    comments=""
)

print("Fig1.pdf, Fig1.png y Fig1_datos.csv guardados.")
plt.show()
