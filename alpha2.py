import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Definimos la Dimensión Anómala W(l, V)
def W(l, V):
    # La solución de la cuadrática que asegura W(0,0) = -1
    discriminante = (2 + V)**2 - 4 * np.exp(-l)
    # Evitar pequeños errores numéricos negativos en el discriminante
    discriminante = np.maximum(discriminante, 0) 
    return (-(2 + V) + np.sqrt(discriminante)) / 2.0

# 2. Definimos el sistema de EDOs
# y[0] = V(l)
# y[1] = I(l) (la integral de W)
def sistema_EDO(l, y):
    V = y[0]
    W_val = W(l, V)
    
    dV_dl = 1 - V * W_val
    dI_dl = W_val
    return [dV_dl, dI_dl]

# 3. Condiciones Iniciales y Rango de Integración
l_inicial = 0.0
l_final = 20.0 # Aproximación asintótica al infinito (UV)
y0 = [0.0, 0.0] # V(0) = 0, I(0) = 0

# 4. Resolvemos el sistema
print("Iniciando integración del flujo del vacío...")
solucion = solve_ivp(
    sistema_EDO, 
    [l_inicial, l_final], 
    y0, 
    method='Radau', # Método robusto para este tipo de asintóticas
    dense_output=True,
    rtol=1e-9, 
    atol=1e-12
)

# 5. Extracción de Resultados
l_vals = np.linspace(l_inicial, l_final, 500)
V_vals = solucion.sol(l_vals)[0]
I_vals = solucion.sol(l_vals)[1]

# El valor asintótico de la integral
I_infinito = I_vals[-1]

# Calculamos f'(0) y el valor de alpha
f_prime_0 = np.exp(-I_infinito)
alpha = f_prime_0 / (4 * np.pi)

print("-" * 40)
print(f"RESULTADOS DEL MODELO")
print("-" * 40)
print(f"Integral total de W (I_inf): {I_infinito:.6f}")
print(f"Ancho de banda IR f'(0):    {f_prime_0:.6f}")
print(f"Constante derivada (alpha): {alpha:.6f}")
print("-" * 40)

# 6. Gráficos de Ontología Cósmica
plt.figure(figsize=(12, 5))

# Gráfico 1: Evolución de V(l)
plt.subplot(1, 2, 1)
plt.plot(l_vals, V_vals, 'b-', label='Eficiencia V(l)')
plt.plot(l_vals, l_vals, 'r--', alpha=0.5, label='Asíntota UV (V=l)')
plt.title('Proporción Termodinámica')
plt.xlabel('Escala l (ln p)')
plt.ylabel('V(l)')
plt.grid(True); plt.legend()

# Gráfico 2: Evolución de la Dimensión Efectiva D_eff
D_eff = f_prime_0 * np.exp(I_vals) # f'(l)
plt.subplot(1, 2, 2)
plt.plot(l_vals, D_eff, 'g-', label="D_eff (Ancho de banda)")
plt.axhline(1.0, color='k', linestyle='--', label="Origen UV (d=1)")
plt.title('Confinamiento de Información')
plt.xlabel('Escala l (ln p)')
plt.ylabel("D_eff (f')")
plt.grid(True); plt.legend()

plt.tight_layout()
plt.show()