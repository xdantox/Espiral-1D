import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
# from scipy.interpolate import PchipInterpolator  # Opcional: más estable que spline cúbico
from scipy.optimize import minimize_scalar, OptimizeResult

# --- tus imports del modelo ---
from Heff import dt, total_time
from cadena0 import cadena_0_MoBr3,cadena_0_BE_vectorized,cadena_0_MoCl3,cadena0MoI3
from rot_methods import implicit_midpoint_step_vectorized
from E import ET

# ---------- utilidades ----------
num_pasos = int(total_time / dt)

def energia_final_por_spin(n: int) -> float:
    """Evoluciona la cadena de longitud n (par) y devuelve E/N final."""
    if n % 2 != 0:
        raise ValueError("n debe ser par")
    c = cadena0MoI3(n)
    # Ejecutar hasta ~total_time (no total_time - dt)
    for _ in range(num_pasos):
        c = implicit_midpoint_step_vectorized(c, dt)
    # ET puede aceptar historial de 1 paso
    E_tot = ET(c[np.newaxis, :, :])
    return float(E_tot) / n

def round_to_even(x: float) -> int:
    """Redondea x al entero par más cercano."""
    return int(round(x / 2) * 2)

# ---------- 1) Malla moderada ----------
n_values = list(range(4, 604 , 2))  # puedes ajustar
E_per_spin: list[float] = []
for n in n_values:
    E = energia_final_por_spin(n)
    E_per_spin.append(E)
    print(f"n={n:4d} -> E/N = {E:.6f}")



xs = np.linspace(min(n_values), max(n_values), 200)
plt.plot(n_values, E_per_spin, 'o', label='Datos')
plt.xlabel('n')
plt.ylabel('E/N')
plt.title('Mínimo continuo y candidatos pares')
plt.legend()
plt.grid(True, ls='--', alpha=0.5)
plt.show()



