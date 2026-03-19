import matplotlib.pyplot as plt
import numpy as np

# Datos extraídos
N_list = [400, 630, 780, 998, 1198]
N_arr = np.array(N_list)
inv_N = 1.0 / N_arr

# Rama 1 y 2
q_arr_1 = np.array([1.03673, 1.02725, 1.03109, 1.03251, 1.03321])
q_arr_2 = np.array([2.10487, 2.11434, 2.11051, 2.10909, 2.10838])
e_arr = np.array([-23.987083, -23.987556, -23.987872, -23.987815, -23.987751])

# --- NUEVO: Ajuste de Escalado de Tamaño Finito (Finite-Size Scaling) ---
# Ajustamos un polinomio de grado 2 a la tendencia vs 1/N
fit_q2 = np.polyfit(inv_N, q_arr_2, 2)
fit_e  = np.polyfit(inv_N, e_arr, 2)

# Creamos un eje X continuo desde el cero exacto hasta el máximo 1/N
inv_N_continua = np.linspace(0, max(inv_N), 100)

# Evaluamos las curvas de ajuste
q2_fit_curve = np.polyval(fit_q2, inv_N_continua)
e_fit_curve  = np.polyval(fit_e, inv_N_continua)

# El valor extrapolado al límite termodinámico (1/N = 0) es el término independiente
q_inf = fit_q2[2]
e_inf = fit_e[2]

# --- CONFIGURACIÓN DE GRÁFICOS ---
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# --- Gráfico 1: Convergencia del pitch (q) ---
# Datos originales
axes[0].plot(inv_N, q_arr_1, 's', color='tab:blue', alpha=0.5, label='Rama 1 data ($q \\approx 1.03$)')
axes[0].plot(inv_N, q_arr_2, 'o', color='tab:red', label='Rama 2 data')

# Curva de extrapolación
axes[0].plot(inv_N_continua, q2_fit_curve, '--', color='tab:red', alpha=0.7, label='Ajuste polinomial')

# Marca del límite termodinámico (Estrella)
axes[0].scatter([0], [q_inf], marker='*', s=150, color='gold', edgecolor='black', zorder=5, 
                label=f'$q_{{\\infty}} = {q_inf:.5f}$')

axes[0].set_xlim(max(inv_N) + 0.0005, -0.0002) # El cero a la izquierda
axes[0].set_xlabel('$1 / N$')
axes[0].set_ylabel('Pitch de la hélice, $q$ (rad)')
axes[0].set_title('Extrapolación del vector de onda al límite continuo')
axes[0].grid(alpha=0.4, ls=':')
axes[0].legend()

# --- Gráfico 2: Convergencia de la Energía ---
# Datos originales
axes[1].plot(inv_N, e_arr, 'd', color='tab:green', markersize=7, label='Energía data')

# Curva de extrapolación
axes[1].plot(inv_N_continua, e_fit_curve, '--', color='tab:green', alpha=0.7, label='Ajuste polinomial')

# Marca del límite termodinámico (Estrella)
axes[1].scatter([0], [e_inf], marker='*', s=150, color='gold', edgecolor='black', zorder=5, 
                label=f'$E_{{\\infty}} = {e_inf:.5f}$')

axes[1].set_xlim(max(inv_N) + 0.0005, -0.0002)
axes[1].set_xlabel('$1 / N$')
axes[1].set_ylabel('Energía del estado base por sitio')
axes[1].set_title('Extrapolación de la energía al límite continuo')
axes[1].grid(alpha=0.4, ls=':')
axes[1].legend()

plt.show()