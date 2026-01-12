from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from typing import cast
from mpl_toolkits.mplot3d.axes3d import Axes3D
from E import ET_PBC

a1, a2 = 1.5 , 1  # Distancias alternantes


def animation(Spin_history,dt):
    num_pasos = Spin_history.shape[0]  # Número de pasos temporales
    n = Spin_history.shape[1]  # Número total de spins
    total_time = num_pasos * dt  # Tiempo total de la simulación
    cadena0 = Spin_history[0,:,:]  # Estado inicial de la cadena
    # Visualización de la evolución de la magnetización en una cadena 1D
    fig = plt.figure()
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))

    # Generar posiciones de los espines en 1D con distancias alternantes a1 y a2
    positions = np.zeros((n, 3))
    current_position = 0
    for i in range(n):
        positions[i, 0] = current_position
        if i % 2 == 0:
            current_position += a1
        else:
            current_position += a2
        
    def update(frame: int):
        ax.clear()
        quiv = ax.quiver(
            positions[:, 0], positions[:, 1], positions[:, 2],
            Spin_history[frame, :, 0], Spin_history[frame, :, 1], Spin_history[frame, :, 2],
            color='r'
        )
        x_min = float(positions[:, 0].min() - 1)
        x_max = float(positions[:, 0].max() + 1)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(f'Tiempo: {frame * dt:.2e} s')
        return (quiv,)

    ani = FuncAnimation(fig, update, frames=num_pasos, interval=20)
    plt.show()
    Energy = ET_PBC(Spin_history)  # Energía total de la cadena en el tiempo
    #plt de la energía por spin

    time_axis = np.linspace(0, total_time, num_pasos)
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, Energy/n)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Energía total por spin')
    plt.title('Energía por spin versus tiempo')
    plt.grid(True)
    plt.show()
    

    # Realizar la transformada de Fourier de la energía
    p= num_pasos-num_pasos//4
    n_samples = num_pasos-p
    Energy_fft = np.fft.rfft(Energy[n_samples:])
    frequencies = np.fft.rfftfreq(p,dt)

    # Eliminar la componente DC y su entorno cercano
    dc_threshold = 1 # Umbral para eliminar frecuencias cercanas a la DC
    Energy_fft[np.abs(frequencies) < dc_threshold] = 0

    # Graficar la energía E(w) vs w excluyendo el componente DC
    plt.figure()
    plt.plot(frequencies, np.abs(Energy_fft))
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Energía (Transformada de Fourier)')
    plt.title('Energía E(w) vs Frecuencia w (sin componente DC)')
    plt.show()

    # Analizar la evolucion temporal de la componente x de cada spin
    fft_freqs = np.fft.rfftfreq(p, dt)
    fft_magnitudes = []
    for i in range(n):
        fft_val = np.fft.rfft(Spin_history[n_samples:, i, 0])
        fft_magnitudes.append(np.abs(fft_val))
    fft_magnitudes = np.array(fft_magnitudes)

    # Sumar o promediar sobre spines
    total_spectrum = np.sum(fft_magnitudes, axis=0)

    # Eliminar la componente DC y su entorno cercano
    total_spectrum[np.abs(fft_freqs) < dc_threshold] = 0

    # Graficar el espectro global
    plt.figure()
    plt.plot(fft_freqs, total_spectrum)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud FFT')
    plt.title('Espectro global de las oscilaciones')
    plt.show()

    return 
    

