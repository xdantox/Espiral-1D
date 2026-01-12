import numpy as np
# Parámetros
gamma = 1.7e11
alpha = 0

def omega(Heff, m):
    return gamma * (Heff - alpha * np.cross(m, Heff))



