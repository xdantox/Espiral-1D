import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. DATOS DE ENTRADA REFINADOS (LSWT, m = 2, k ≈ 0.4928 rad)
# =====================================================================
# Datos en orden decreciente (1.0D -> 0.1D)
x_raw = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
y_raw = np.array(
    [
        0.0628,
        0.0518,
        0.0417,
        0.0326,
        0.0244,
        0.01725,
        0.0112,
        0.0065,
        0.0030,
        0.0011,
    ]
)

# Ordenamos de menor a mayor (0.1D -> 1.0D)
sort_idx = np.argsort(x_raw)
x = x_raw[sort_idx]
y = y_raw[sort_idx]

# =====================================================================
# 2. AJUSTE PARABÓLICO (GRADO 2) E INTERPOLACIÓN CONTINUA
# =====================================================================
# Ajuste cuadrático: y = a*x^2 + b*x + c
poly2_coeffs = np.polyfit(x, y, deg=2)
a, b, c = poly2_coeffs
p2 = np.poly1d(poly2_coeffs)

# Malla densa para interpolación suave continua
x_dense = np.linspace(0.0, 1.0, 400)
y_dense = p2(x_dense)

# Residuos y métricas estadísticas
residuals = y - p2(x)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1.0 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean(residuals**2))
max_residual_ghz = np.max(np.abs(residuals)) * 1e3

# =====================================================================
# 3. IMPRESIÓN DE RESULTADOS
# =====================================================================
print("=" * 65)
print("AJUSTE PARABÓLICO: SEGUNDO MINIGAP (m = 2, k ≈ 0.4928 rad)")
print("=" * 65)
print(f"Ecuación ajustada : Δν(x) = {a:.5f} x² + {b:.5f} x + {c:.5f}  [THz]")
print(f"Coeficiente R²    : {r2:.6f}")
print(f"Error RMSE        : {rmse:.6f} THz ({rmse * 1e3:.3f} GHz)")
print(f"Residuo máximo    : {max_residual_ghz:.3f} GHz")
print("=" * 65)

# =====================================================================
# 4. GRÁFICA DE CALIDAD EDITORIAL (Physical Review Style)
# =====================================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True,
    }
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(6.0, 5.5),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
)

# --- Panel 1: Datos LSWT y Curva Parabólica ---
ax1.plot(
    x_dense,
    y_dense,
    "-",
    color="#1f77b4",
    lw=2.0,
    label=(
        rf"Parabolic fit: $\Delta\nu = {a:.4f}x^2 + {b:.4f}x"
        + (rf" + {c:.4f}$" if c >= 0 else rf" - {abs(c):.4f}$")
        + f"\n($R^2 = {r2:.5f}$)"
    ),
)
ax1.scatter(
    x,
    y,
    color="#d62728",
    edgecolor="black",
    s=55,
    zorder=5,
    label=r"LSWT ($m=2$, $k \approx 0.4928$\,rad)",
)

ax1.set_ylabel(r"Minigap $\Delta\nu^{(2)}$ [THz]", fontsize=18)
ax1.tick_params(axis="y", labelsize=15)
ax1.set_ylim(-0.003, 0.070)
ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=15)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.set_title(
    r"Secondary Umklapp Minigap Scaling vs. $D_{\mathrm{plane}}/D$",
    fontsize=21,
    pad=10,
)

# --- Panel 2: Residuos (LSWT - Fit) en GHz ---
ax2.axhline(0, color="black", lw=1.0, linestyle="-")
ax2.plot(
    x,
    residuals * 1e3,
    "o-",
    color="#1f77b4",
    lw=1.5,
    ms=4.5,
    label="Fit Residuals",
)

ax2.set_xlabel(r"Transverse Anisotropy Ratio $x = D_{\mathrm{plane}}/D$")
ax2.set_ylabel(r"Res. [GHz]", fontsize=18)
ax2.tick_params(axis="y", labelsize=15)
ax2.set_xlim(0.05, 1.05)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=15)

plt.show()