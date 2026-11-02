import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ==============================================================================
# CONFIGURACIÓN TIPOGRÁFICA LIMPIA (PRB/PRL)
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 150,
})

# Datos
d_ratio = np.linspace(0.1, 1.0, 10)
gap_llg = np.array([0.12858, 0.25046, 0.37662, 0.50400, 0.63289,
                    0.76198, 0.89281, 1.02379, 1.15528, 1.28750])
gap_lswt = np.array([0.11833, 0.23788, 0.35861, 0.48047, 0.60339,
                     0.72729, 0.85208, 0.97767, 1.10394, 1.23077])

# Barras de error
err_llg = np.full_like(gap_llg, 0.012)

# LSWT: Error asimétrico SOLO HACIA ARRIBA [inferior=0, superior=cota HF]
err_lswt_upper = 0.0567 * d_ratio
err_lswt_asym = np.array([np.zeros_like(gap_lswt), err_lswt_upper])

diff_gap = gap_llg - gap_lswt

# Regresiones lineales
res_llg = linregress(d_ratio, gap_llg)
res_lswt = linregress(d_ratio, gap_lswt)
x_line = np.linspace(0.04, 1.06, 100)

fft_floor = 0.012
x_crossover = fft_floor / 0.0567  # ~0.2116

# ==============================================================================
# FIGURA 1x2 DESPEJADA
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.8), layout="constrained")

# ------------------------------------------------------------------------------
# PANEL (a): EVOLUCIÓN CON COTA UNIDIRECCIONAL
# ------------------------------------------------------------------------------
ax1.plot(x_line, res_llg.slope * x_line + res_llg.intercept,
         color="#C0392B", lw=1.5, ls="-", zorder=2)
ax1.plot(x_line, res_lswt.slope * x_line + res_lswt.intercept,
         color="#16A085", lw=1.5, ls="--", zorder=2)

dx = 0.007
ax1.errorbar(
    d_ratio - dx, gap_llg, yerr=err_llg, fmt="o", color="#C0392B",
    ecolor="#C0392B", elinewidth=1.0, capsize=2.5, markersize=5.0,
    mfc="#E74C3C", mec="black", mew=0.5, label="LLG (Soliton Lattice)", zorder=3
)
# Solo barra superior
ax1.errorbar(
    d_ratio + dx, gap_lswt, yerr=err_lswt_asym, fmt="s", color="#16A085",
    ecolor="#16A085", elinewidth=1.0, capsize=2.5, markersize=4.8,
    mfc="#2ECC71", mec="black", mew=0.5, label=r"LSWT + Linear error ($\delta\Delta\nu \geq 0$)", zorder=3
)

# Cuadro de ajustes (esquina superior izquierda)
fit_text = (
    r"$\mathbf{LLG:}\ \ \Delta\nu = 1.291\,x - 0.009$" + "\n"
    r"$\mathbf{LSWT:}\ \Delta\nu = 1.237\,x - 0.011$"
)
ax1.text(0.05, 0.94, fit_text, transform=ax1.transAxes, va="top",
         fontsize=8.5, bbox=dict(boxstyle="round,pad=0.35", fc="#fafafa", ec="#d0d0d0", lw=0.7))

ax1.set_xlabel(r"Anisotropy Ratio $D_{\mathrm{plane}} / D$")
ax1.set_ylabel(r"Primary Minigap $\Delta\nu$ [THz]")
ax1.set_xlim(0.04, 1.06)
ax1.set_ylim(0.0, 1.45)
ax1.grid(True, ls=":", lw=0.5, alpha=0.6)
ax1.legend(loc="lower right", frameon=True, edgecolor="#d0d0d0", framealpha=0.92, fontsize=8.2)
ax1.text(0.05, 0.05, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold")

# ------------------------------------------------------------------------------
# PANEL (b): RESIDUO Y RÉGIMEN DE RESOLUCIÓN FFT
# ------------------------------------------------------------------------------
# Zona sombreada de resolución limitada (x <= 0.21)
ax2.axvspan(0.04, x_crossover, color="#f2f2f2", zorder=1)
ax2.axvline(x_crossover, color="#aaaaaa", ls=":", lw=1.0, zorder=2)

# Línea del piso FFT
ax2.axhline(fft_floor, color="#888888", ls=":", lw=1.2,
            label=r"FFT Floor ($\sigma_{\mathrm{FFT}} \approx 0.012$ THz)", zorder=2)

# Recta de Hellmann-Feynman
ax2.plot(x_line, 0.0567 * x_line, color="#16A085", ls="--", lw=1.6,
         label=r"Linear error ($\delta(\Delta\nu)$)", zorder=2)

# Datos de discrepancia
ax2.plot(d_ratio, diff_gap, color="black", marker="D", markersize=4.8,
         mfc="white", mec="black", mew=0.9, lw=1.2, label=r"Residual (LLG $-$ LSWT)", zorder=3)

# Texto discreto indicando la zona de baja anisotropía
ax2.text(0.12, 0.003, r"$x \leq 0.21$", ha="center", fontsize=8.0, color="#693A3A", style="italic")

ax2.set_xlabel(r"Anisotropy Ratio $D_{\mathrm{plane}} / D$")
ax2.set_ylabel(r"Discrepancy $\delta(\Delta\nu)$ [THz]")
ax2.set_xlim(0.04, 1.06)
ax2.set_ylim(-0.005, 0.075)
ax2.grid(True, ls=":", lw=0.5, alpha=0.6)
ax2.legend(loc="upper left", frameon=True, edgecolor="#d0d0d0", framealpha=0.92, fontsize=8.2)
ax2.text(0.90, 0.05, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

plt.savefig("fig_minigap_corrected.pdf", dpi=300)
plt.savefig("fig_minigap_corrected.png", dpi=300)
plt.show()