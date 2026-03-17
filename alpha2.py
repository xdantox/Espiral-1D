import numpy as np
import math

# El Vacío Platónico o Estado Ideal
Z0 = 4 * math.pi**3 + math.pi**2 + math.pi

# Usamos la Función Beta de la EDO exacta: p * (dZ/dp)
def p_Z_prime_exacto(Z):
    numerador = (Z**3) * (Z0 - Z)**2
    denominador = Z * (Z0 - Z) - 2
    return numerador / denominador

def c1_exacto(Z): 
    # Rigidez Geométrica exacta: Z^2 / (p * Z')
    return (Z**2) / p_Z_prime_exacto(Z)

def C_Euler_exacto(Z): 
    # Carga Topológica
    return 2 / Z

def Evaluador_de_Transicion(Z_test):
    c = c1_exacto(Z_test)
    C_val = C_Euler_exacto(Z_test)
    
    discriminante = 1 - 4 * c * C_val
    
    # Presión fractal total (tu fórmula exacta con el signo negativo)
    P_star_IR = (1 - np.sqrt(discriminante)) / (2 * c)
    
    # Deformación Total respecto al vacío platónico
    Deformacion = Z0 - Z_test
    
    return Deformacion, P_star_IR

print("--- RESOLUCIÓN CON EL RG FLOW EXACTO ---")
# Evaluamos en el rango infrarrojo de nuestro universo, incluyendo CODATA
valores_Z = [137.025, 137.030, 137.035, 137.035999084]

for Z in valores_Z:
    Def, P_star = Evaluador_de_Transicion(Z)
    diferencia = Def - P_star
    print(f"Z: {Z:.6f} | Deformación: {Def:.6f} | Presión Fractal: {P_star:.6f} | Diferencia: {diferencia:.2e}")