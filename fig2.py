
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fig. 2: perfil espacial de atenuación de señal magnónica.
Utiliza los resultados numéricos ya obtenidos. No integra LLG.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Datos originales ----------------

distance = np.array([0, 66, 132, 198, 264, 330, 396])

# Logaritmo decimal de la razón de energía local normalizada
regular_wall = np.array([
    0.85, 0.49, -1.66, -3.68, -5.61, -7.48, -9.54
])

regular_valley = np.array([
    0.83, -0.93, -2.81, -4.84, -7.18, -8.91, -10.67
])

irregular = np.array([
    1.20, -0.52, -4.92, -7.68, -8.69, -9.36, -8.88
])

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

fig, ax = plt.subplots(figsize=(3.45, 2.85))

ax.plot(
    distance, regular_wall,
    "o-", color="C0", markersize=4.5,
    linewidth=1.5,
    label="A (18 walls), source on a wall"
)

ax.plot(
    distance, regular_valley,
    "s-", color="C1", markersize=4.5,
    linewidth=1.5,
    label="A, source in a valley"
)

ax.plot(
    distance, irregular,
    "^-", color="C2", markersize=5,
    linewidth=1.5,
    label="C (46 walls)"
)

ax.axhline(0, color="black", linewidth=0.8)

ax.set_xlim(-5, 405)
ax.set_ylim(-11.3, 1.8)

ax.set_xticks(np.arange(0, 401, 50))

ax.set_xlabel(r"Distance to the source $|s|$ (sites)")
ax.set_ylabel(
    r"$\log_{10}\mathcal{R}_{\Omega}(s)$"
    "\n(1.7--2.5 THz)"
)

ax.grid(alpha=0.2)
ax.legend(loc="lower left", frameon=True, framealpha=0.9)

fig.tight_layout()

out = Path(__file__).resolve().parent
fig.savefig(out / "Fig2.pdf", bbox_inches="tight")
fig.savefig(out / "Fig2.png", dpi=300, bbox_inches="tight")

# Datos tabulados para reproducibilidad.
np.savetxt(
    out / "Fig2_datos.csv",
    np.column_stack([
        distance, regular_wall, regular_valley, irregular
    ]),
    delimiter=",",
    header="distance,regular_wall,regular_valley,irregular",
    comments=""
)

print("Fig2.pdf, Fig2.png y Fig2_datos.csv guardados.")
plt.show()
