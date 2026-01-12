import numpy as np
from Heff import dt, n, total_time
# ——————————————————————————————————————————
# Parámetros extra para los pulsos alternativos
L_sinc_space = 1.01        # ancho de la sinc espacial (en sitios)
tau_sinc_time = 1.01*dt      # ancho de la sinc temporal (~1 paso)
# ——————————————————————————————————————————

def B_ext_sinc(t, i):
    """
    Pulso tipo sinc en espacio y en tiempo:
    • sinc espacial → excitación plana en k (hasta ~1/L_sinc_space)
    • sinc temporal  → excitación plana en ω (hasta ~1/tau_sinc_time)
    """
    # posición periódica del centro del pulso
    pulse_pos = n//2

    dx = ((i - pulse_pos + n//2) % n) - n//2
    x = dx / L_sinc_space
    G = np.sinc(x)

    # sinc temporal centrada en t0_g
    y = (t - tau_g) / tau_sinc_time
    E_t = np.sinc(y)

    return np.array([G * E_t, 0, 0])

# Parámetros globales para el pulso gaussiano
sigma_g     = 1            # ancho espacial (en sitios)
amplitude_g = 0.2           # amplitud del pulso
v_g         = 0.1 / dt           # velocidad de desplazamiento (sitios/s)
tau_g       = total_time//2           # tiempo característico
t0_g        = 3 * tau_g        # centro temporal

# Campo magnético externo de forzamiento en el eje de la cadena
def B_ext_g_real(t, i):
    """
    Pulso gaussiano puro en espacio sin restricción en k.
    Retorna [B_x,0,0].
    """
    # centro del pulso (periódico)
    pulse_pos = (v_g * t) % n
    
    # distancia mínima periódica
    Δ = ((i - pulse_pos + n/2) % n) - n/2
    
    # envolvente espacial gaussiana
    G = np.exp(- (Δ**2) / (2 * sigma_g**2))
    
    # envolvente temporal gaussiana
    dt_loc = (t % (2*t0_g)) - t0_g
    E_t = np.exp(- (dt_loc**2) / (2 * tau_g**2))
    
    return np.array([amplitude_g * G * E_t, 0, 0])

# Modo de Fourier centrado en [-π, π)
k_modes = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n))  # n debe estar definido
# Coeficientes gaussianos en el espacio de k (centrados en k=0)
coef_modes = (amplitude_g / n) * np.exp(-0.5 * (k_modes * sigma_g)**2)

def B_ext_g(t, i):
    """
    Pulso gaussiano espacial con restricción a modos de Fourier permitidos.
    Retorna el campo externo en el sitio i y tiempo t como [B_x, 0, 0].
    """
    # posición periódica del centro del pulso (mod n)
    pulse_pos = n//2

    # Producto escalar entre coeficientes y cosenos: superposición de modos válidos
    # i puede ser escalar o vector: np.cos se vectoriza automáticamente
    G = np.sum(coef_modes * np.cos(k_modes * (i - pulse_pos)))

    # Envolvente temporal gaussiana (centrada en t0_g, con duración tau_g)
    dt = (t % (2 * t0_g)) - t0_g
    E_t = np.exp(- (dt**2) / (2 * tau_g**2))

    return np.array([E_t * G, 0, 0])


def B_ext_s(t, i,f):
    """
    Campo magnético externo senoidal que:
      - Sintetiza una señal summando los 6 modos espaciales permitidos por la periodicidad (k = 2π·n/6).
      - Incorpora una oscilación temporal senoidal con frecuencia f.
      - Se desplaza periódicamente a lo largo de la cadena.
    
    Parámetros:
      t: tiempo global (s)
      i: índice del spin (0, 1, ..., 5)
      f: frecuencia de oscilación temporal (Hz)
      
    Retorna:
      Vector de campo (aplicado en x).
    """
    # Definición de los 6 modos permitidos
    n_modes = n
    k_modes = np.array([2 * np.pi * n / n_modes for n in range(n_modes)])
    
    # Síntesis espacial: se suma una señal senoidal para cada modo permitido.  
    # Cada término tiene la forma sin( k_mode * desplazamiento espacial + fase temporal )
    # donde la fase temporal es 2π·f·t.
    # El término (i - pulse_center) permite desplazar el patrón a lo largo de la cadena.
    field_spatial = np.sum(np.sin(k_modes * i + 2 * np.pi * f * t))
    
    amplitude = 1e-2  # Escala del campo
    
    # Campo aplicado en la dirección x
    field = amplitude * field_spatial
    return np.array([field, 0, 0])

def B_ext_static_delta(t, i):
    """
    Pulso espacial fijo en i0, delta‐like en el tiempo.
    Excita planamente el rango de k permitido por la forma espacial,
    sin ningún v_g ≠ 0.
    """
    i0 = n//2
    # distancia periódica mínima
    dx = ((i - i0 + n/2) % n) - n/2
    # envolvente espacial (e.g. gaussiana fija)
    G = np.exp(-dx**2/(2*sigma_g**2))
    # delta‐like en el tiempo
    if abs(t - total_time/2) <= dt/2:
        E_t = 100      
    else:
        E_t = 0.0
    return np.array([G * E_t, 0, 0])