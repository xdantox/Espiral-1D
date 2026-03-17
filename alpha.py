import math
from scipy.optimize import root_scalar

def Z_Hopf(p): return 4*p**3 + p**2 + p
def Z_prime(p): return 12*p**2 + 2*p + 1

# El estado platónico inicial (Gauge)
Z_ideal = Z_Hopf(math.pi)

def c1(p): 
    # Rigidez Geométrica
    return (Z_Hopf(p)**2)/(p * Z_prime(p))

def C_Euler(p): 
    # Carga Topológica Incompresible (Autoadaptativa)
    return 2 / Z_Hopf(p)

def Ecuacion_de_Accion(p):
    """
    Se iguala la Deformación Macroscópica a la Presión Fractal
    """
    c = c1(p)
    C_val = C_Euler(p)
    
    # Presión fractal total
    P_star = (1 - math.sqrt(1 - 4*C_val)) / (2*c)
    
    # Deformación Total
    Deformacion = Z_ideal - Z_Hopf(p)
    
    return Deformacion - P_star

print("--- RESOLUCIÓN DEL UNIVERSO AUTOADAPTATIVO ---")
# Buscamos la intersección entre 3.1415 y pi
sol = root_scalar(Ecuacion_de_Accion, bracket=[3.14150, math.pi], xtol=1e-14)

if sol.converged:
    p_star = sol.root
    z_star = Z_Hopf(p_star)
    print(f"pi* de equilibrio: {p_star:.15f}")
    print(f"Z* observable:     {z_star:.15f}")
    print(f"CODATA 2022:       137.035999084")
else:
    print("No convergió.")