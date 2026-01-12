import numpy as np
from Heff import Jnn, dJnn, Jnnn, De, Ka,J_perp

# Energía para una cadena, por componentes de los spines, por sitio.
def Ei_PBC(cadena, i) -> float:
    if i == (len(cadena) - 1):
        if i % 2 == 0:
            return (Jnn + dJnn) * np.dot(cadena[i],cadena[0]) + Jnnn * np.dot(cadena[i],cadena[1]) + De* np.dot(cadena[i], np.array([1,0,0]))**2 + De * np.dot(cadena[i], np.array([0,1,0]))**2 + Ka * np.dot(cadena[i],cadena[0])**2
        else:
            return (Jnn - dJnn) * np.dot(cadena[i],cadena[0]) + Jnnn * np.dot(cadena[i],cadena[1]) + De* np.dot(cadena[i], np.array([1,0,0]))**2 + De * np.dot(cadena[i], np.array([0,1,0]))**2 + Ka * np.dot(cadena[i],cadena[0])**2
    if i == (len(cadena) - 2):
        if i % 2 == 0:
            return (Jnn + dJnn) * np.dot(cadena[i],cadena[i+1]) + Jnnn * np.dot(cadena[i],cadena[0]) + De* np.dot(cadena[i], np.array([1,0,0]))**2 + De * np.dot(cadena[i], np.array([0,1,0]))**2 + Ka * np.dot(cadena[i],cadena[i+1])**2
        else:
            return (Jnn - dJnn) * np.dot(cadena[i],cadena[i+1]) + Jnnn * np.dot(cadena[i],cadena[0]) + De* np.dot(cadena[i], np.array([1,0,0]))**2 + De * np.dot(cadena[i], np.array([0,1,0]))**2 + Ka * np.dot(cadena[i],cadena[i+1])**2
    if i % 2 == 0:
        return (Jnn + dJnn) * np.dot(cadena[i],cadena[i+1]) + Jnnn * np.dot(cadena[i],cadena[i+2]) + De* np.dot(cadena[i], np.array([1,0,0]))**2 + De * np.dot(cadena[i], np.array([0,1,0]))**2 + Ka * np.dot(cadena[i],cadena[i+1])**2
    else:
        return (Jnn - dJnn) * np.dot(cadena[i],cadena[i+1]) + Jnnn * np.dot(cadena[i],cadena[i+2]) + De* np.dot(cadena[i], np.array([1,0,0]))**2 + De * np.dot(cadena[i], np.array([0,1,0]))**2 + Ka * np.dot(cadena[i],cadena[i+1])**2


def ET_PBC(Spin_history):
    # Energía total de la cadena en el tiempo
    num_pasos = Spin_history.shape[0]
    cadena0 = Spin_history[0]
    Energy = np.zeros(num_pasos)
    for j in range(num_pasos):
        for i in range(len(cadena0)):
            Energy[j] += Ei_PBC(Spin_history[j],i)
    return Energy



def E0(cadena0):
    # Energía total de la cadena en el tiempo
    Energy: float = 0.0
    for i in range(len(cadena0)):
        Energy += Ei_PBC(cadena0,i)
    return Energy/len(cadena0)


