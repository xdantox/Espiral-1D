import numpy as np  
# Generar la cadena inicial de espines
def generar_cadena_spines(n):
    direcciones = np.random.uniform(-1,1,(n,3))  # Valores aleatorios entre -1 y 1
    direcciones /= np.linalg.norm(direcciones, axis=1)[:, np.newaxis]
    return direcciones