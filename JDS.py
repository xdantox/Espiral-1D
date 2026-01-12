import numpy as np
import matplotlib.pyplot as plt
from cadena0 import cadena_0, cadena_0_BE, cadena_0_Jnnn_dimer, cadena_0_MoBr3,cadena_0_BE_vectorized
# Spins data with non-collinear ground state



n = 1200
spins = cadena_0_BE_vectorized(n)

# Convert spins to spherical angles
theta0 = []
phi0 = []
for x, y, z in spins:
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    theta0.append(theta)
    phi0.append(phi)

# Physical parameters
N = len(theta0)  # Number of spins
c = 6 # Spins per unit cell
n_cells = N // c

# Physical parameter (nearest-neighbor exchange)
Jnn = 46.75 / 0.05788
dJnn = 44.85 /0.05788
Jnnn = 2.6 / 0.05788
D = 0.76 / 0.05788
K = 45.4 / 0.05788
#Jnn = 48.891035 / 0.05788
#dJnn = 48.620365 / 0.05788
#Jnnn= 1.26126 / 0.05788
#D = 0.31 / 0.05788
#K= 48.9119 / 0.05788
# Initialize dynamical matrix (2N x 2N)
M = np.zeros((2 * N, 2 * N))

# Build the dynamical matrix
for i in range(N):
    #Dimerized constant:
    Ji = Jnn + (-1)**i * dJnn  # Dimerized nearest-neighbor exchange constant
    Jk = Jnn + (-1)**(i-1) * dJnn

    # Current spin
    theta_i = theta0[i]
    phi_i = phi0[i]
    ct_i = np.cos(theta_i)
    st_i = np.sin(theta_i)
    cp_i = np.cos(phi_i)
    sp_i = np.sin(phi_i)
    
    # Right neighbor (i+1)
    j = (i + 1) % N
    theta_j = theta0[j]
    phi_j = phi0[j]
    ct_j = np.cos(theta_j)
    st_j = np.sin(theta_j)
    cp_j = np.cos(phi_j)
    sp_j = np.sin(phi_j)
    
    # Left neighbor (i-1)
    k = (i - 1) % N
    theta_k = theta0[k]
    phi_k = phi0[k]
    ct_k = np.cos(theta_k)
    st_k = np.sin(theta_k)
    cp_k = np.cos(phi_k)
    sp_k = np.sin(phi_k)

    # Next-nearest neighbors
    # Right next-nearest neighbor (i+2)
    l = (i + 2) % N
    theta_l = theta0[l]
    phi_l = phi0[l]
    ct_l = np.cos(theta_l)
    st_l = np.sin(theta_l)
    cp_l = np.cos(phi_l)
    sp_l = np.sin(phi_l)

    # Left next-nearest neighbor (i-2)
    m = (i - 2) % N
    theta_m = theta0[m]
    phi_m = phi0[m]
    ct_m = np.cos(theta_m)
    st_m = np.sin(theta_m)
    cp_m = np.cos(phi_m)
    sp_m = np.sin(phi_m)
    
    # Precompute trigonometric differences
    dphi_ji = phi_j - phi_i
    cdphi_ji = np.cos(dphi_ji)
    sdphi_ji = np.sin(dphi_ji)
    
    dphi_ik = phi_i - phi_k
    cdphi_ik = np.cos(dphi_ik)
    sdphi_ik = np.sin(dphi_ik)

    # Precompute trigonometric differences for next-nearest neighbors
    dphi_li = phi_l - phi_i
    cdphi_li = np.cos(dphi_li)
    sdphi_li = np.sin(dphi_li)

    dphi_im = phi_i - phi_m
    cdphi_im = np.cos(dphi_im)
    sdphi_im = np.sin(dphi_im)
    
    #Nearest-neighbor exchange interactions
    # Diagonal B
    M[2*i, 2*i+1] += Ji * (ct_i * st_j * sdphi_ji)  # δθ_iδφ_i (S_i * S_j)
    M[2*i, 2*i+1] += -Jk * (ct_i * st_k * sdphi_ik) # δθ_iδφ_i (S_k * S_i)

    # Diagonal C
    M[2*i+1, 2*i] += Ji * (ct_i * st_j * sdphi_ji)   # δφ_iδθ_i (S_i * S_j)
    M[2*i+1, 2*i] += -Jk * (ct_i * st_k * sdphi_ik)    # δφ_iδθ_i (S_k * S_i)
    
    # Diagonal A
    M[2*i, 2*i] += -Ji * (ct_i * ct_j + st_i * st_j * cdphi_ji)    #δθ_iδθ_i (S_i * S_j)
    M[2*i, 2*i] += -Jk * (ct_k * ct_i + st_k * st_i * cdphi_ik)    #δθ_iδθ_i (S_k * S_i)

    # Diagonal D
    M[2*i+1, 2*i+1] += -Ji * (st_i * st_j * cdphi_ji)   #δφ_iδφ_i (S_i * S_j)
    M[2*i+1, 2*i+1] += -Jk * (st_k * st_i * cdphi_ik)   #δφ_iδφ_i (S_k * S_i)

    # Off-diagonal B
    M[2*i, (2*i + 3) % (2*N)] += -Ji * (ct_i * st_j * (sdphi_ji) )   # δθ_iδφ_j (S_i * S_j)
    M[2*i, (2*i - 1) % (2*N)] += Jk *  (ct_i * st_k * (sdphi_ik) )   # δθ_iδφ_k (S_k * S_i)
    
    # Off-diagonal C
    M[2*i+1, (2*i + 2) % (2*N)] += Ji * (st_i * ct_j * (sdphi_ji) )   # δφ_iδθ_j (S_i * S_j)
    M[2*i+1, (2*i - 2) % (2*N)] += -Jk * (st_i * ct_k * (sdphi_ik) )    # δφ_iδθ_k (S_k * S_i)

    # Off-diagonal A
    M[2*i, (2*i + 2) % (2*N)] += Ji * (st_i * st_j + ct_i * ct_j * cdphi_ji)   #δθ_iδθ_j (S_i * S_j)
    M[2*i, (2*i - 2) % (2*N)] += Jk * (st_k * st_i + ct_k * ct_i * cdphi_ik)   #δθ_iδθ_k (S_k * S_i)
   
    #Off-diagonal D
    M[2*i+1, (2*i + 3) % (2*N)] += Ji * (st_i * st_j * cdphi_ji)  #δφ_iδφ_j (S_i * S_j)
    M[2*i+1, (2*i - 1) % (2*N)] += Jk * (st_k * st_i * cdphi_ik)  #δφ_iδφ_k (S_k * S_i)

    # Next-nearest neighbor interactions
    # Diagonal B
    M[2*i, 2*i+1] += Jnnn * (ct_i * st_l * sdphi_li)  # δθ_iδφ_i (S_i * S_l)
    M[2*i, 2*i+1] += -Jnnn * (ct_i * st_m * sdphi_im)  # δθ_iδφ_i (S_i * S_m)    

    # Diagonal C
    M[2*i+1, 2*i] += Jnnn * (ct_i * st_l * sdphi_li)  # δφ_iδθ_i (S_i * S_l)
    M[2*i+1, 2*i] += -Jnnn * (ct_i * st_m * sdphi_im)  # δφ_iδθ_i (S_i * S_m)

    # Diagonal A
    M[2*i, 2*i] += -Jnnn * (ct_i * ct_l + st_i * st_l * cdphi_li)  # δθ_iδθ_i (S_i * S_l)
    M[2*i, 2*i] += -Jnnn * (ct_i * ct_m + st_i * st_m * cdphi_im)  # δθ_iδθ_i (S_i * S_m)

    # Diagonal D
    M[2*i + 1, 2*i + 1] += -Jnnn * (st_i * st_l * cdphi_li)  # δφ_iδφ_i (S_i * S_l)
    M[2*i + 1, 2*i + 1] += -Jnnn * (st_i * st_m * cdphi_im)  # δφ_iδφ_i (S_i * S_m)  

    # Off-diagonal B
    M[2*i, (2*i + 5) % (2*N)] += -Jnnn * (ct_i * st_l * sdphi_li)  # δθ_iδφ_l (S_i * S_l)
    M[2*i, (2*i - 3) % (2*N)] += Jnnn * (ct_i * st_m * sdphi_im)  # δθ_iδφ_m (S_i * S_m)
    
    # Off-diagonal C
    M[2*i+1, (2*i + 4) % (2*N)] += Jnnn * (st_i * ct_l * sdphi_li)  # δφ_iδθ_l (S_i * S_l)
    M[2*i+1, (2*i - 4) % (2*N)] += -Jnnn * (st_i * ct_m * sdphi_im)  # δφ_iδθ_m (S_i * S_m)      

    # Off-diagonal A
    M[2*i, (2*i + 4) % (2*N)] += Jnnn * (st_i * st_l + ct_i * ct_l * cdphi_li)  # δθ_iδθ_l (S_i * S_l)
    M[2*i, (2*i - 4) % (2*N)] += Jnnn * (st_i * st_m + ct_i * ct_m * cdphi_im)  # δθ_iδθ_m (S_i * S_m)

    # Off-diagonal D
    M[2*i+1, (2*i + 5) % (2*N)] += Jnnn * (st_i * st_l * cdphi_li)  # δφ_iδφ_l (S_i * S_l)
    M[2*i+1, (2*i - 3) % (2*N)] += Jnnn * (st_i * st_m * cdphi_im)  # δφ_iδφ_m (S_i * S_m)
    
    # Anisotropy

    # Diagonal B
    M[2*i, 2*i + 1] += -4*D * (ct_i * cp_i * st_i * sp_i)  # δθ_iδφ_i (S_i * S_i+1)

    # Diagonal C
    M[2*i+1, 2*i] += -4*D * (ct_i * cp_i * st_i * sp_i)    # δφ_iδθ_i (S_i * S_i+1)

    # Diagonal A
    M[2*i, 2*i] += 2 * D * (-(st_i * cp_i)**2 + (ct_i * cp_i)**2) # δθ_iδθ_i (S_i * S_i+1)

    # Diagonal D
    M[2*i + 1, 2*i + 1] += 2 * D * (-(st_i * cp_i)**2 + (st_i * sp_i)**2) # δφ_iδφ_i (S_i * S_i+1)

    #Biquadratic-exchange

    # Diagonal A
    M[2*i, 2*i] += 2 * K * ((ct_i * st_j * cdphi_ji)**2 + (ct_j * st_i)**2 - (st_i * st_j * cdphi_ji) * (st_i * st_j * cdphi_ji + ct_i * ct_j) - (st_i * st_j * cdphi_ji + ct_i * ct_j) * ct_i * ct_j - 2 * ct_i * st_j * cdphi_ji * ct_j * st_i)  # δθ_iδθ_i (S_i * S_j)
    M[2*i, 2*i] += 2 * K * ((st_k * ct_i * cdphi_ik)**2 + (ct_k * st_i)**2 - (st_k * st_i * cdphi_ik) * (st_k * st_i * cdphi_ik + ct_k * ct_i) - (st_k * st_i * cdphi_ik + ct_k * ct_i) * ct_k * ct_i - 2 * st_k * ct_i * cdphi_ik * ct_k * st_i)  # δθ_iδθ_i (S_k * S_i)

    #Diagonal B
    M[2*i, 2*i + 1] += 2 * K * (ct_i * st_j * cdphi_ji * sdphi_ji * st_i * st_j - sdphi_ji * st_i * st_j * ct_j * st_i + ct_i * sdphi_ji * st_j * (st_i * st_j * cdphi_ji + ct_i * ct_j)) # δθ_iδφ_i (S_i * S_j)
    M[2*i, 2*i + 1] += 2 * K * (-st_k * ct_i * cdphi_ik * sdphi_ik * st_k * st_i + sdphi_ik * st_k * st_i * ct_k * st_i - st_k * ct_i * sdphi_ik * (st_k * st_i * cdphi_ik + ct_k * ct_i))  # δθ_iδφ_i (S_k * S_i)
    
    #Diagonal C
    M[2*i + 1, 2*i] += 2 * K * (ct_i * st_j * cdphi_ji * sdphi_ji * st_i * st_j - sdphi_ji * st_i * st_j * ct_j * st_i + ct_i * sdphi_ji * st_j * (st_i * st_j * cdphi_ji + ct_i * ct_j)) # δφ_iδθ_i (S_i * S_j)
    M[2*i + 1, 2*i] += 2 * K * (-st_k * ct_i * cdphi_ik * sdphi_ik * st_k * st_i + sdphi_ik * st_k * st_i * ct_k * st_i - st_k * ct_i * sdphi_ik * (st_k * st_i * cdphi_ik + ct_k * ct_i))  # δφ_iδθ_i (S_k * S_i)

    # Diagonal D
    M[2*i + 1, 2*i + 1] += 2 * K * ((sdphi_ji * st_i * st_j)**2 - st_i * st_j * cdphi_ji * (st_i * st_j * cdphi_ji + ct_i * ct_j))  # δφ_iδφ_i (S_i * S_j)
    M[2*i + 1, 2*i + 1] += 2 * K * ((sdphi_ik * st_k * st_i)**2 - st_k * st_i * cdphi_ik * (st_k * st_i * cdphi_ik + ct_k * ct_i))  # δφ_iδφ_i (S_k * S_i)
    
    # Off-diagonal B
    M[2*i, (2*i + 3) % (2*N)] += 2 * K * (- ct_i * st_j * cdphi_ji * sdphi_ji * st_i * st_j + sdphi_ji * st_i * st_j * ct_j * st_i - ct_i * st_j * sdphi_ji * (st_i * st_j * cdphi_ji + ct_i * ct_j)) # δθ_iδφ_j (S_i * S_j)
    M[2*i, (2*i - 1) % (2*N)] += 2 * K * (st_k * ct_i * cdphi_ik * sdphi_ik * st_k * st_i - sdphi_ik * st_k * st_i * ct_k * st_i + st_k * ct_i * sdphi_ik * (st_k * st_i * cdphi_ik + ct_k * ct_i))  # δθ_iδφ_k (S_k * S_i)

    # Off-diagonal C
    M[2*i + 1, (2*i + 2) % (2*N)] += 2 * K * (st_i * ct_j * cdphi_ji * sdphi_ji * st_i * st_j - sdphi_ji * st_i * st_j * ct_i * st_j + st_i * ct_j * sdphi_ji * (st_i * st_j * cdphi_ji + ct_i * ct_j)) # δφ_iδθ_j (S_i * S_j)
    M[2*i + 1, (2*i - 2) % (2*N)] += 2 * K * (- ct_k * st_i * cdphi_ik * sdphi_ik * st_k * st_i + sdphi_ik * st_k * st_i * ct_i * st_k - ct_k * st_i * sdphi_ik * (st_k * st_i * cdphi_ik + ct_k * ct_i))  # δφ_iδθ_k (S_k * S_i)

    # Off-diagonal A
    M[2*i, (2*i + 2) % (2*N)] += 2 * K * (st_i * ct_j * cdphi_ji * (ct_i * st_j * cdphi_ji - ct_j * st_i) - ct_i * st_j * cdphi_ji * (ct_i * st_j) + ct_i * st_j * ct_j * st_i + ct_i * ct_j * cdphi_ji * (st_i * st_j * cdphi_ji + ct_i * ct_j) + st_i * st_j * (st_i * st_j * cdphi_ji + ct_i * ct_j)) # δθ_iδθ_j (S_i * S_j)
    M[2*i, (2*i - 2) % (2*N)] += 2 * K * (st_k * ct_i * cdphi_ik * (ct_k * st_i * cdphi_ik - ct_i * st_k) - ct_k * st_i * cdphi_ik * (ct_k * st_i) + ct_k * st_i * ct_i * st_k + ct_k * ct_i * cdphi_ik * (st_k * st_i * cdphi_ik + ct_k * ct_i) + st_k * st_i * (st_k * st_i * cdphi_ik + ct_k * ct_i)) # δθ_iδθ_k (S_k * S_i)

    # Off-diagonal D
    M[2*i+1, (2*i + 3) % (2*N)] += 2 * K * (-(sdphi_ji * st_i * st_j)**2 + st_i * st_j * cdphi_ji * (st_i * st_j * cdphi_ji + ct_i * ct_j))  # δφ_iδφ_j (S_i * S_j)
    M[2*i+1, (2*i - 1) % (2*N)] += 2 * K * (-(sdphi_ik * st_k * st_i)**2 + st_k * st_i * cdphi_ik * (st_k * st_i * cdphi_ik + ct_k * ct_i))  # δφ_iδφ_k (S_k * S_i)


 # Extract blocks for Bloch's theorem
q_vals = np.linspace(-2*np.pi, 2*np.pi, 1000, endpoint=False)
M_qvalues = np.zeros((len(q_vals),2*c,2*c), dtype=complex)

def M_qv(q, M=M, c=c):
    # Bloch sum over unit cells: M_q = sum_r M_r * exp(-iqr)
    Mq = np.zeros((2*c, 2*c), dtype=complex)
    for r in range(-1, 2):  # Solo bloques de -1, 0, 1
        for alpha in range(c):
            for beta in range(c):
                block = M[2*alpha:2*alpha+2, 2*(beta + r*c)% (2*N):2*(beta + r*c)% (2*N)+2]
                Mq[2*alpha:2*alpha+2, 2*beta:2*beta+2] += block * np.exp(-1j * q * r)
    return Mq
        
    
# Compute dispersion

eigs = np.zeros((len(q_vals), 2*c), dtype=complex)

for qi, q in enumerate(q_vals):
    M_qvalues[qi,:,:] = M_qv(q)
    # Initialize matrix G_q
    G_q = np.zeros((2*c, 2*c), dtype=complex)
    for i in range(c):
        st_i = np.sin(theta0[i])
        G_q[2*i, 2*i+1] = st_i
        G_q[2*i+1, 2*i] = -st_i
    M_qi = np.linalg.inv(G_q) @ M_qvalues[qi,:,:]
    eigs[qi,:] = np.linalg.eigvals(M_qi)
    
# Plot the dispersion (frequencies = |Im(eigenvalues)|)
plt.figure(figsize=(8, 5))
for band in range(2*c):
    freqs = 1.7e11 * np.imag(eigs[:, band])
    plt.plot(q_vals, freqs, '.', markersize=2)

plt.xlabel('Wavevector (q)')
plt.ylabel('Frequency (ω)')
plt.title('Spin Wave Dispersion of Antiferromagnetic Chain')
plt.grid(True)
plt.show()

