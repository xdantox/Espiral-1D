import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ==============================================================================
# 1. PARÁMETROS FÍSICOS DE FRENKEL-KONTOROVA
# ==============================================================================
N = 36                # Número de partículas
b = np.pi/4.0         # Periodo del sustrato (distancia entre valles)
a = 1.0           # Longitud natural del resorte libre
kappa = 1.0           # Constante elástica del resorte
V_MAX = 0.35          # Barrera máxima del sustrato

# Coordenadas iniciales: cadena homogénea sin deformación
x = (np.arange(N) - N / 2.0) * a
v = np.zeros(N)

dt = 0.15
damping = 0.15
two_pi_b = 2.0 * np.pi / b

def relax_chain(current_V, substeps=14):
    """Dinámica sobreamortiguada 1D en espacio continuo."""
    global x, v
    for _ in range(substeps):
        f = np.zeros(N)
        stretch = (x[1:] - x[:-1]) - a
        f[:-1] += kappa * stretch
        f[1:]  -= kappa * stretch
        f += - current_V * two_pi_b * np.sin(two_pi_b * x)
        
        v = (v + f * dt) * (1.0 - damping)
        x += v * dt

# ==============================================================================
# 2. PROTOCOLO TEMPORAL DE LA ANIMACIÓN
# ==============================================================================
n_hold1 = 15          # Pausa en V = 0
n_ramp  = 50          # Crecimiento adiabático de V
n_hold2 = 25          # Pausa en V_max
total_frames = n_hold1 + n_ramp + n_hold2

V_schedule = np.concatenate([
    np.zeros(n_hold1),
    np.linspace(0.0, V_MAX, n_ramp),
    np.full(n_hold2, V_MAX)
])

# ==============================================================================
# 3. FIGURA: PANELES CONTINUOS (1 y 2) Y PANEL DISCRETO (3)
# ==============================================================================
fig = plt.figure(figsize=(10, 6.4), dpi=130)
# GridSpec: Paneles 1 y 2 comparten eje continuo; Panel 3 es independiente
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.40, 0.95], hspace=0.28)

ax_pot = fig.add_subplot(gs[0])
ax_chain = fig.add_subplot(gs[1], sharex=ax_pot)
ax_strain = fig.add_subplot(gs[2])

fig.subplots_adjust(top=0.88, bottom=0.10, left=0.10, right=0.95)

x_window = 16.0
x_grid = np.linspace(-x_window, x_window, 800)

# --- PANEL 1: ENERGÍA POTENCIAL V(x) (CONTINUO) ---
line_pot, = ax_pot.plot([], [], color='#334155', lw=2.0, label=r"$V(x) = -V\cos(2\pi x / b)$")
ax_pot.set_ylabel(r"Energía $V(x)$", fontsize=9.5, fontweight='bold')
ax_pot.set_xlim(-x_window, x_window)
ax_pot.set_ylim(-0.52, 0.58)

# COTAS INTUITIVAS DE a Y b (Medidores directos en el potencial)
# 1. Cota del periodo del sustrato b
ax_pot.annotate('', xy=(0, 0.44), xytext=(b, 0.44),
                arrowprops=dict(arrowstyle='<->', color='#0284c7', lw=1.8))
ax_pot.text(b / 2.0, 0.47, r"Periodo potencial $b = \pi/4$",
            color='#0284c7', fontsize=8.5, ha='center', fontweight='bold')

# 2. Cota de la longitud natural del resorte a
ax_pot.annotate('', xy=(0, 0.26), xytext=(a, 0.26),
                arrowprops=dict(arrowstyle='<->', color='#16a34a', lw=1.8))
ax_pot.text(a / 2.0, 0.16, r"Resorte libre $a = 1.0$",
            color='#16a34a', fontsize=8.5, ha='center', fontweight='bold')

ax_pot.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

# --- PANEL 2: CADENA 1D EN ESPACIO REAL (CONTINUO) ---
ax_chain.axhline(0, color='#94a3b8', lw=1.5, zorder=1)
scatter_particles = ax_chain.scatter([], [], s=70, c='#38bdf8', edgecolors='#0f172a', lw=1.5, zorder=5, label=r"Partículas en $x_n$")
ax_chain.set_ylim(-0.5, 0.5)
ax_chain.set_yticks([])
ax_chain.set_ylabel("Cadena 1D", fontsize=9.5, fontweight='bold')
ax_chain.set_xlabel("Coordenada espacial continua ($x$)", fontsize=9.5, fontweight='bold')
ax_chain.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

# Guías verticales desde los pozos de potencial (Panel 1 -> Panel 2)
valleys = np.arange(-6, 7) * b
for vx in valleys:
    if -x_window <= vx <= x_window:
        ax_pot.axvline(vx, color='#0284c7', ls=':', alpha=0.35, lw=1.0)
        ax_chain.axvline(vx, color='#0284c7', ls=':', alpha=0.35, lw=1.0)

# --- PANEL 3: DEFORMACIÓN EN LA RED DISCRETA (DISCRETO) ---
bond_indices = np.arange(-N//2 + 1, N//2)  # Coordenada discreta entera n
line_strain, = ax_strain.plot([], [], 'o-', color='#dc2626', lw=2.0, ms=5, mfc='#ffffff', mew=1.6, label=r"Deformación $\Delta x_n - a$")
ax_strain.axhline(0, color='#64748b', ls='--', lw=1.2, label=r"Resorte relajado ($\Delta x_n = a$)")

ax_strain.set_xlim(bond_indices[0] - 0.5, bond_indices[-1] + 0.5)
ax_strain.set_ylim(-0.30, 0.70)
ax_strain.set_ylabel(r"Deformación elástica", fontsize=9.5, fontweight='bold')
ax_strain.set_xlabel(r"Índice discreto del enlace ($n \in \mathbb{Z}$)", fontsize=9.5, fontweight='bold')
ax_strain.set_xticks(np.arange(-15, 16, 5))
ax_strain.grid(True, linestyle=':', alpha=0.6)
ax_strain.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

spring_lines = []

# ==============================================================================
# 4. ACTUALIZACIÓN DE CUADROS
# ==============================================================================
def update(frame):
    current_V = V_schedule[frame]
    relax_chain(current_V)
    
    # 1. Potencial
    U_grid = - current_V * np.cos(two_pi_b * x_grid)
    line_pot.set_data(x_grid, U_grid)
    
    # 2. Cadena física continua en x
    mask = (x >= -x_window - 1.5) & (x <= x_window + 1.5)
    scatter_particles.set_offsets(np.column_stack([x[mask], np.zeros(np.sum(mask))]))
    
    global spring_lines
    for sl in spring_lines:
        sl.remove()
    spring_lines.clear()
    
    strains = (x[1:] - x[:-1]) - a
    
    for i in range(N - 1):
        if x[i] < -x_window - 2 or x[i+1] > x_window + 2:
            continue
        tens = np.clip(np.abs(strains[i]) / 0.45, 0.0, 1.0)
        col = (0.13 + 0.75 * tens, 0.77 * (1.0 - tens) + 0.15 * tens, 0.35 * (1.0 - tens) + 0.15 * tens)
        sl, = ax_chain.plot([x[i], x[i+1]], [0, 0], color=col, lw=5.5, solid_capstyle='round', zorder=2)
        spring_lines.append(sl)
    
    # 3. Deformación sobre la red DISCRETA de enlaces n
    line_strain.set_data(bond_indices, strains)
    
    # 4. Título dinámico
    if current_V < 0.01:
        txt = "RÉGIMEN HOMOGÉNEO (V = 0.00): Sin deformación — Todos los enlaces miden a"
        col_t = "#0f172a"
    elif frame < n_hold1 + n_ramp:
        txt = f"COMPETICIÓN ESPACIAL (V = {current_V:.2f}): El resorte (a) compite con los pozos (b)"
        col_t = "#0284c7"
    else:
        txt = f"RED DE SOLITONES ESTABILIZADA (V = {current_V:.2f}): Picos discretos de deformación"
        col_t = "#dc2626"
        
    fig.suptitle(txt, fontsize=10.5, fontweight='bold', color=col_t, y=0.96)
    return line_pot, scatter_particles, line_strain

# Generación
print("Generando animación corregida...")
ani = FuncAnimation(fig, update, frames=total_frames, interval=60)
ani.save("soliton_transition.gif", writer=PillowWriter(fps=16))
print("¡Archivo generado!")