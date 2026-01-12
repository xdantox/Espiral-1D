import numpy as np
import matplotlib.pyplot as plt
from rot import Rotacion_exacta, skew_symmetric_matrix
from matplotlib.animation import FuncAnimation # Importar FuncAnimation
from mpl_toolkits.mplot3d import Axes3D # Importar Axes3D

# Parámetros
gamma = 1.7e11
dt = 1e-14
ttotal_time = 1e-12  # Tiempo total de simulación (s)
n_steps = int(ttotal_time / dt)  # Número total de pasos de tiempo

# Campo fijo
H = np.array([0, 100, 100])
# Estado inicial (normalizado)
m = np.array([1.0, 0.0, 0.0])
m /= np.linalg.norm(m)

# Integración y monitoreo de E(t) y m(t)
E = np.empty(n_steps)
mnorm_history = np.zeros_like(E)
m2_history = np.zeros_like(E)
Ortogonal = np.zeros((n_steps,3,3,3))
m_history = np.zeros((n_steps, 3)) # Array para guardar el historial de m

for j in range(n_steps):
    # 1) calculo ω en el inicio
    ω0 = gamma * H
    # 2) rotación half‐step
    R_half = Rotacion_exacta(ω0, dt/2)
    m_half = R_half @ m
    # 3) calculo ω en el punto medio
    ω_mid  = gamma * H
    # 4) paso completo usando ω_mid
    R_full = Rotacion_exacta(ω_mid, dt)
    m_full = R_full @ m_half
    # 5) paso simétrico
    ω_final = gamma * H
    R_s = Rotacion_exacta(ω_final, dt/2)
    m = R_s @ m_full
    # 6) grabo estado y energía
    m_history[j] = m # Guardar el vector m actual
    E[j] = -np.dot(m, H)
    mnorm_history[j] = np.linalg.norm(m)
    m2_history[j] = m[2]
    # print(m) # Descomentar si necesitas ver el vector m en cada paso
    Ortogonal[j] = np.array([R_half @ R_half.T, R_full @ R_full.T, R_s @ R_s.T])

# --- Gráficas estáticas ---
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, dt*n_steps, n_steps), E)
plt.xlabel('Tiempo (s)')
plt.ylabel('Energía E(t)')
plt.title('E(t) para un solo espín en campo fijo (α=0)')
plt.grid(True)
# plt.show() # Mostrar esta figura inmediatamente si se desea

plt.figure(figsize=(10, 5))
#graficar la normalización y otros diagnósticos
plt.plot(np.linspace(0, dt*n_steps, n_steps), mnorm_history, label='mnorm(t)')
# Graficar solo un elemento de las matrices de ortogonalidad para evitar sobrecargar
plt.plot(np.linspace(0, dt*n_steps, n_steps), Ortogonal[:,0,0,0], label='R_half Diag[0]-1')
plt.plot(np.linspace(0, dt*n_steps, n_steps), Ortogonal[:,1,0,0], label='R_full OffDiag[0]')
plt.plot(np.linspace(0, dt*n_steps, n_steps), Ortogonal[:,2,0,0], label='R_s OffDiag[0]')
plt.plot(np.linspace(0, dt*n_steps, n_steps), m2_history, label='m[2]')
plt.legend()
plt.xlabel('Tiempo (s)')
plt.ylabel('Valor')
plt.title('Diagnósticos: Norma, Ortogonalidad y m[2]')
plt.grid(True)
# plt.show() # Mostrar esta figura inmediatamente si se desea

# --- Animación del Spin ---
fig_anim = plt.figure(figsize=(6, 6))
ax_anim = fig_anim.add_subplot(111, projection='3d')
ax_anim.set_xlim([-1.1, 1.1])
ax_anim.set_ylim([-1.1, 1.1])
ax_anim.set_zlim([-1.1, 1.1])
ax_anim.set_xlabel('X')
ax_anim.set_ylabel('Y')
ax_anim.set_zlabel('Z')
ax_anim.set_title('Rotación del Spin')
ax_anim.grid(True)

# Dibujar una esfera unitaria como referencia
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sphere = np.cos(u)*np.sin(v)
y_sphere = np.sin(u)*np.sin(v)
z_sphere = np.cos(v)
ax_anim.plot_wireframe(x_sphere, y_sphere, z_sphere, color="grey", alpha=0.3)

# Inicializar la flecha (quiver) que representa el spin
# Usamos quiver para dibujar una flecha desde el origen (0,0,0) hasta el punto m
quiver = ax_anim.quiver(0, 0, 0, m_history[0,0], m_history[0,1], m_history[0,2], color='r', length=1.0, normalize=False)

# Función que actualiza la animación en cada frame
def update(frame):
    global quiver
    # Obtener el vector m para el frame actual
    mx, my, mz = m_history[frame]
    # Actualizar los datos del quiver (la flecha)
    # quiver necesita los segmentos [x, y, z, u, v, w] donde (x,y,z) es el inicio y (u,v,w) el vector
    # Como la flecha siempre empieza en el origen, solo actualizamos el vector (u,v,w)
    # Necesitamos eliminar el quiver anterior y crear uno nuevo para actualizarlo correctamente en 3D
    quiver.remove()
    quiver = ax_anim.quiver(0, 0, 0, mx, my, mz, color='r', length=np.linalg.norm(m_history[frame]), normalize=False) # Usar norma actual
    ax_anim.set_title(f'Rotación del Spin (t={frame*dt:.2e} s)')
    return quiver,

# Crear la animación
# frames: número de frames (pasos de tiempo)
# interval: tiempo entre frames en milisegundos
# blit=True: optimización para redibujar solo lo que cambia (puede dar problemas a veces)
# repeat=False: no repetir la animación al terminar
ani = FuncAnimation(fig_anim, update, frames=n_steps, interval=20, blit=False, repeat=False)

# Mostrar todas las figuras
plt.show()

# Opcional: Guardar la animación (requiere ffmpeg u otro backend instalado)
# print("Guardando animación...")
# ani.save('spin_rotation.mp4', writer='ffmpeg', fps=30)
# print("Animación guardada.")