import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from dot import Mdot 
from Heff import  dt, total_time, n
from rand import generar_cadena_spines
from cadena0 import cadena0ansatz,cadena0spinhistory,cadena0harmonic_PBC
from rot_methods import implicit_midpoint_step_vectorized_PBC_folded,implicit_midpoint_step_vectorized_PBC, implicit_midpoint_step_vectorized_FOBC, implicit_midpoint_step_vectorized_PBC_numba
from animation import animation 
from E import ET_PBC
import direct3_PBC_folded as f
# Parámetros
gamma = 1.7e11  # Frecuencia de Larmor (rad/s/T)
num_pasos = int(total_time / dt)  # Número total de pasos de tiempo
# Ejecutar el método y almacenar los resultados para graficar

cadena0 = cadena0spinhistory(n) # Generar la cadena de espines
Spin_history = np.zeros((num_pasos, len(cadena0), 3))
Spin_history[0] = cadena0
c_0 = cadena0.copy()

# Metodo de integración
for j in range(num_pasos-1):
    c_0 = implicit_midpoint_step_vectorized_PBC_numba(c_0, dt)
    Spin_history[j+1] = c_0

Energy = ET_PBC(c_0[np.newaxis,:,:])  # Energía total de la cadena en el tiempo
print(f"Energy/N = {float(Energy)/n:.6f}")
np.save("spin_history.npy", Spin_history)
animation(Spin_history,dt)  # Animación de la evolución de los espines
