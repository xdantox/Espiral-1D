import numpy as np
# Parámetros
gamma = 1.7e11
alpha = 1.2

def omega(Heff, m):
    return gamma * (Heff - alpha * np.cross(m, Heff))



