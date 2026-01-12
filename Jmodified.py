import numpy as np
import matplotlib.pyplot as plt


def magnetic_cell_configuration(n_cells, q_star, phase_offsets, phi_offsets=None, polar_angle=np.pi/2, spin_magnitude=1.0):
	"""Generate a helical configuration in a magnetic unit cell.

	The classical spin at site (n, a) is obtained from
	θ_{n, a} = n * m * q_star + theta_offsets[a]
	φ_{n, a} = phi_offsets[a] (default: 0)

	Parameters
	----------
	n_cells : int
		Number of magnetic unit cells.
	q_star : float
		Helical wave-vector of the underlying incommensurate state.
	theta_offsets : sequence of floats
		Intracell polar-angle offsets θ_a defining a(n) for each sublattice.
	phi_offsets : sequence of floats, optional
		Intracell azimuthal offsets φ_a (defaults to zeros).
	spin_magnitude : float, optional
		Spin magnitude |S|. Defaults to 1.

	Returns
	-------
	ndarray, shape (n_cells * m, 3)
		Cartesian spin directions for the commensurate magnetic cell.
	"""
	phase_offsets = np.asarray(phase_offsets, dtype=float)
	if phi_offsets is None:
		# phi_offsets are additional azimuthal offsets per sublattice (defaults zero)
		phi_offsets = np.zeros_like(phase_offsets)
	else:
		phi_offsets = np.asarray(phi_offsets, dtype=float)

	m = len(phase_offsets)
	spins = []
	for n in range(n_cells):
		base_phase = n * m * q_star
		for phase_off, phi_off in zip(phase_offsets, phi_offsets):
			# polar angle is fixed (helix in xy plane by default)
			theta = polar_angle
			# azimuthal phase encodes the helix
			phi = base_phase + phase_off + phi_off
			sx = spin_magnitude * np.sin(theta) * np.cos(phi)
			sy = spin_magnitude * np.sin(theta) * np.sin(phi)
			sz = spin_magnitude * np.cos(theta)
			spins.append((sx, sy, sz))
	return np.asarray(spins)


# Magnetic-cell parameters (m = 2 example)
q_star = 2.110556313  # helical wave-vector  (adjust to target configuration)
A = 0.561      # intracell deformation a(n)
phase_offsets = (0, q_star + A)
phi_offsets = (0.0, 0.0)  # helix around z; customize if needed

n_cells = 120
spins = magnetic_cell_configuration(n_cells, q_star, phase_offsets, phi_offsets)


def local_basis(theta, phi):
	e_theta = np.array([
		np.cos(theta) * np.cos(phi),
		np.cos(theta) * np.sin(phi),
		-np.sin(theta),
	])
	e_phi = np.array([
		-np.sin(phi),
		np.cos(phi),
		0.0,
	])
	e_spin = np.array([
		np.sin(theta) * np.cos(phi),
		np.sin(theta) * np.sin(phi),
		np.cos(theta),
	])
	return e_theta, e_phi, e_spin


def rotation_matrix_z(angle):
	cos_a, sin_a = np.cos(angle), np.sin(angle)
	return np.array([
		[cos_a, -sin_a, 0.0],
		[sin_a, cos_a, 0.0],
		[0.0, 0.0, 1.0],
	])


def rotate_basis(basis, angle):
	rot = rotation_matrix_z(angle)
	return tuple(rot @ vec for vec in basis)

# compatibility alias for patches using rotz
def rotz(angle):
	return rotation_matrix_z(angle)


def angle_wrap(angle):
	return (angle + np.pi) % (2 * np.pi) - np.pi


theta0 = []
phi0 = []
for x, y, z in spins:
	theta0.append(np.arccos(z))
	phi0.append(np.arctan2(y, x))

theta0 = np.asarray(theta0)
phi0 = np.asarray(phi0)

N = len(theta0)
c = len(phase_offsets)
n_cells = N // c

Jnn = 46.75 / 0.05788
dJnn = 44.85 / 0.05788
Jnnn = 2.6 / 0.05788
D = 0.76 / 0.05788
K = 45.4 / 0.05788

M = np.zeros((2 * N, 2 * N), dtype=complex)

local_bases = [local_basis(theta, phi) for theta, phi in zip(theta0, phi0)]
sin_theta = np.sin(theta0)
cos_theta = np.cos(theta0)

# --- compute effective magnetic bond lengths a_bond and real-space positions x ---
# use wrapped differences of the azimuthal phase to obtain a_bond = Δφ / q_star
dphi = np.angle(np.exp(1j * (np.roll(phi0, -1) - phi0)))
a_bond = dphi / q_star
x = np.zeros(N)
for ii in range(1, N):
	x[ii] = x[ii-1] + a_bond[ii-1]

for i in range(N):
	Ji = Jnn + (-1) ** i * dJnn
	Jk = Jnn + (-1) ** (i - 1) * dJnn

	j = (i + 1) % N
	k = (i - 1) % N
	l = (i + 2) % N
	m = (i - 2) % N

	st_i = sin_theta[i]
	ct_i = cos_theta[i]

	e_theta_i, e_phi_i, e_s_i = local_bases[i]

	# compute magnetic separations and rotate neighbor bases by delta = q_star * a_ij
	a_ij = x[j] - x[i]
	a_ik = x[k] - x[i]
	a_il = x[l] - x[i]
	a_im = x[m] - x[i]

	delta_j = q_star * a_ij
	delta_k = q_star * a_ik
	delta_l = q_star * a_il
	delta_m = q_star * a_im

	Rj = rotation_matrix_z(delta_j)
	Rk = rotation_matrix_z(delta_k)
	Rl = rotation_matrix_z(delta_l)
	Rm = rotation_matrix_z(delta_m)

	e_theta_j_rot, e_phi_j_rot, e_s_j_rot = tuple(Rj @ v for v in local_bases[j])
	e_theta_k_rot, e_phi_k_rot, e_s_k_rot = tuple(Rk @ v for v in local_bases[k])
	e_theta_l_rot, e_phi_l_rot, e_s_l_rot = tuple(Rl @ v for v in local_bases[l])
	e_theta_m_rot, e_phi_m_rot, e_s_m_rot = tuple(Rm @ v for v in local_bases[m])

	st_j = sin_theta[j]
	st_k = sin_theta[k]
	st_l = sin_theta[l]
	st_m = sin_theta[m]

	# Nearest-neighbor: right neighbor (j)
	Jnn_ti_ti = -np.dot(e_s_i, e_s_j_rot)
	Jnn_pi_pi = -st_i**2 * np.dot(e_s_i, e_s_j_rot) - st_i * ct_i * np.dot(e_theta_i, e_s_j_rot)
	Jnn_ti_pi = ct_i * np.dot(e_phi_i, e_s_j_rot)

	M[2 * i, 2 * i] += Ji * Jnn_ti_ti
	M[2 * i, 2 * i + 1] += Ji * Jnn_ti_pi
	M[2 * i + 1, 2 * i] += Ji * Jnn_ti_pi
	M[2 * i + 1, 2 * i + 1] += Ji * Jnn_pi_pi

	Jnn_ti_tj = np.dot(e_theta_i, e_theta_j_rot)
	Jnn_ti_pj = st_j * np.dot(e_theta_i, e_phi_j_rot)
	Jnn_pi_tj = st_i * np.dot(e_phi_i, e_theta_j_rot)
	Jnn_pi_pj = st_i * st_j * np.dot(e_phi_i, e_phi_j_rot)

	M[2 * i, (2 * i + 2) % (2 * N)] += Ji * Jnn_ti_tj
	M[2 * i, (2 * i + 3) % (2 * N)] += Ji * Jnn_ti_pj
	M[2 * i + 1, (2 * i + 2) % (2 * N)] += Ji * Jnn_pi_tj
	M[2 * i + 1, (2 * i + 3) % (2 * N)] += Ji * Jnn_pi_pj

	# Nearest-neighbor: left neighbor (k)
	Jnn_ti_ti = -np.dot(e_s_i, e_s_k_rot)
	Jnn_pi_pi = -st_i**2 * np.dot(e_s_i, e_s_k_rot) - st_i * ct_i * np.dot(e_theta_i, e_s_k_rot)
	Jnn_ti_pi = ct_i * np.dot(e_phi_i, e_s_k_rot)

	M[2 * i, 2 * i] += Jk * Jnn_ti_ti
	M[2 * i, 2 * i + 1] += Jk * Jnn_ti_pi
	M[2 * i + 1, 2 * i] += Jk * Jnn_ti_pi
	M[2 * i + 1, 2 * i + 1] += Jk * Jnn_pi_pi

	Jnn_ti_tk = np.dot(e_theta_i, e_theta_k_rot)
	Jnn_ti_pk = st_k * np.dot(e_theta_i, e_phi_k_rot)
	Jnn_pi_tk = st_i * np.dot(e_phi_i, e_theta_k_rot)
	Jnn_pi_pk = st_i * st_k * np.dot(e_phi_i, e_phi_k_rot)

	M[2 * i, (2 * i - 2) % (2 * N)] += Jk * Jnn_ti_tk
	M[2 * i, (2 * i - 1) % (2 * N)] += Jk * Jnn_ti_pk
	M[2 * i + 1, (2 * i - 2) % (2 * N)] += Jk * Jnn_pi_tk
	M[2 * i + 1, (2 * i - 1) % (2 * N)] += Jk * Jnn_pi_pk

	# Next-nearest-neighbor: right neighbor (l)
	Jnnn_ti_ti = -np.dot(e_s_i, e_s_l_rot)
	Jnnn_pi_pi = -st_i**2 * np.dot(e_s_i, e_s_l_rot) - st_i * ct_i * np.dot(e_theta_i, e_s_l_rot)
	Jnnn_ti_pi = ct_i * np.dot(e_phi_i, e_s_l_rot)

	M[2 * i, 2 * i] += Jnnn * Jnnn_ti_ti
	M[2 * i, 2 * i + 1] += Jnnn * Jnnn_ti_pi
	M[2 * i + 1, 2 * i] += Jnnn * Jnnn_ti_pi
	M[2 * i + 1, 2 * i + 1] += Jnnn * Jnnn_pi_pi

	Jnnn_ti_tj = np.dot(e_theta_i, e_theta_l_rot)
	Jnnn_ti_pj = st_l * np.dot(e_theta_i, e_phi_l_rot)
	Jnnn_pi_tj = st_i * np.dot(e_phi_i, e_theta_l_rot)
	Jnnn_pi_pj = st_i * st_l * np.dot(e_phi_i, e_phi_l_rot)

	M[2 * i, (2 * i + 4) % (2 * N)] += Jnnn * Jnnn_ti_tj
	M[2 * i, (2 * i + 5) % (2 * N)] += Jnnn * Jnnn_ti_pj
	M[2 * i + 1, (2 * i + 4) % (2 * N)] += Jnnn * Jnnn_pi_tj
	M[2 * i + 1, (2 * i + 5) % (2 * N)] += Jnnn * Jnnn_pi_pj

	# Next-nearest-neighbor: left neighbor (m)
	Jnnn_ti_ti = -np.dot(e_s_i, e_s_m_rot)
	Jnnn_pi_pi = -st_i**2 * np.dot(e_s_i, e_s_m_rot) - st_i * ct_i * np.dot(e_theta_i, e_s_m_rot)
	Jnnn_ti_pi = ct_i * np.dot(e_phi_i, e_s_m_rot)

	M[2 * i, 2 * i] += Jnnn * Jnnn_ti_ti
	M[2 * i, 2 * i + 1] += Jnnn * Jnnn_ti_pi
	M[2 * i + 1, 2 * i] += Jnnn * Jnnn_ti_pi
	M[2 * i + 1, 2 * i + 1] += Jnnn * Jnnn_pi_pi

	Jnnn_ti_tj = np.dot(e_theta_i, e_theta_m_rot)
	Jnnn_ti_pj = st_m * np.dot(e_theta_i, e_phi_m_rot)
	Jnnn_pi_tj = st_i * np.dot(e_phi_i, e_theta_m_rot)
	Jnnn_pi_pj = st_i * st_m * np.dot(e_phi_i, e_phi_m_rot)

	M[2 * i, (2 * i - 4) % (2 * N)] += Jnnn * Jnnn_ti_tj
	M[2 * i, (2 * i - 3) % (2 * N)] += Jnnn * Jnnn_ti_pj
	M[2 * i + 1, (2 * i - 4) % (2 * N)] += Jnnn * Jnnn_pi_tj
	M[2 * i + 1, (2 * i - 3) % (2 * N)] += Jnnn * Jnnn_pi_pj

	# Anisotropy along global x
	D_tt = 2 * e_theta_i[0] ** 2 - 2 * e_s_i[0] ** 2
	D_pp = (
		2 * (st_i * e_phi_i[0]) ** 2
		- 2 * (st_i * e_s_i[0]) ** 2
		- 2 * (st_i * ct_i * (e_s_i[0] * e_theta_i[0]))
	)
	D_tp = 2 * st_i * e_theta_i[0] * e_phi_i[0] + 2 * ct_i * e_s_i[0] * e_phi_i[0]

	M[2 * i, 2 * i] += D * D_tt
	M[2 * i, 2 * i + 1] += D * D_tp
	M[2 * i + 1, 2 * i] += D * D_tp
	M[2 * i + 1, 2 * i + 1] += D * D_pp

	# Biquadratic exchange with right neighbor (j)
	K_ti_ti = -2 * np.dot(e_s_i, e_s_j_rot) ** 2 + 2 * np.dot(e_s_j_rot, e_theta_i) ** 2
	K_pi_pi = (
		2 * np.dot(e_s_i, e_s_j_rot)
		* (-st_i**2 * np.dot(e_s_i, e_s_j_rot) - st_i * ct_i * np.dot(e_s_j_rot, e_theta_i))
		+ (st_i * np.dot(e_s_j_rot, e_phi_i)) ** 2
	)
	K_ti_pi = (
		2 * np.dot(e_s_i, e_s_j_rot) * np.dot(e_s_j_rot, e_phi_i) * ct_i
		+ 2 * np.dot(e_s_j_rot, e_theta_i) * np.dot(e_s_j_rot, e_phi_i) * st_i
	)

	M[2 * i, 2 * i] += K * K_ti_ti
	M[2 * i, 2 * i + 1] += K * K_ti_pi
	M[2 * i + 1, 2 * i] += K * K_ti_pi
	M[2 * i + 1, 2 * i + 1] += K * K_pi_pi

	K_ti_tj = (
		2 * np.dot(e_s_i, e_s_j_rot) * np.dot(e_theta_i, e_theta_j_rot)
		+ 2 * np.dot(e_s_i, e_theta_j_rot) * np.dot(e_s_j_rot, e_theta_i)
	)
	K_ti_pj = (
		2 * np.dot(e_s_i, e_s_j_rot) * np.dot(e_theta_i, e_phi_j_rot) * st_j
		+ 2 * np.dot(e_s_i, e_phi_j_rot) * np.dot(e_s_j_rot, e_theta_i) * st_j
	)
	K_pi_tj = (
		2 * np.dot(e_s_i, e_s_j_rot) * np.dot(e_phi_i, e_theta_j_rot) * st_i
		+ 2 * np.dot(e_s_i, e_theta_j_rot) * np.dot(e_s_j_rot, e_phi_i) * st_i
	)
	K_pi_pj = (
		2 * np.dot(e_s_i, e_s_j_rot) * np.dot(e_phi_i, e_phi_j_rot) * st_i * st_j
		+ 2 * np.dot(e_s_i, e_phi_j_rot) * np.dot(e_s_j_rot, e_phi_i) * st_i * st_j
	)

	M[2 * i, (2 * i + 2) % (2 * N)] += K * K_ti_tj
	M[2 * i, (2 * i + 3) % (2 * N)] += K * K_ti_pj
	M[2 * i + 1, (2 * i + 2) % (2 * N)] += K * K_pi_tj
	M[2 * i + 1, (2 * i + 3) % (2 * N)] += K * K_pi_pj

	# Biquadratic exchange with left neighbor (k)
	K_ti_ti = -2 * np.dot(e_s_i, e_s_k_rot) ** 2 + 2 * np.dot(e_s_k_rot, e_theta_i) ** 2
	K_pi_pi = (
		2 * np.dot(e_s_i, e_s_k_rot)
		* (-st_i**2 * np.dot(e_s_i, e_s_k_rot) - st_i * ct_i * np.dot(e_s_k_rot, e_theta_i))
		+ (st_i * np.dot(e_s_k_rot, e_phi_i)) ** 2
	)
	K_ti_pi = (
		2 * np.dot(e_s_i, e_s_k_rot) * np.dot(e_s_k_rot, e_phi_i) * ct_i
		+ 2 * np.dot(e_s_k_rot, e_theta_i) * np.dot(e_s_k_rot, e_phi_i) * st_i
	)

	M[2 * i, 2 * i] += K * K_ti_ti
	M[2 * i, 2 * i + 1] += K * K_ti_pi
	M[2 * i + 1, 2 * i] += K * K_ti_pi
	M[2 * i + 1, 2 * i + 1] += K * K_pi_pi

	K_ti_tk = (
		2 * np.dot(e_s_i, e_s_k_rot) * np.dot(e_theta_i, e_theta_k_rot)
		+ 2 * np.dot(e_s_i, e_theta_k_rot) * np.dot(e_s_k_rot, e_theta_i)
	)
	K_ti_pk = (
		2 * np.dot(e_s_i, e_s_k_rot) * np.dot(e_theta_i, e_phi_k_rot) * st_k
		+ 2 * np.dot(e_s_i, e_phi_k_rot) * np.dot(e_s_k_rot, e_theta_i) * st_k
	)
	K_pi_tk = (
		2 * np.dot(e_s_i, e_s_k_rot) * np.dot(e_phi_i, e_theta_k_rot) * st_i
		+ 2 * np.dot(e_s_i, e_theta_k_rot) * np.dot(e_s_k_rot, e_phi_i) * st_i
	)
	K_pi_pk = (
		2 * np.dot(e_s_i, e_s_k_rot) * np.dot(e_phi_i, e_phi_k_rot) * st_i * st_k
		+ 2 * np.dot(e_s_i, e_phi_k_rot) * np.dot(e_s_k_rot, e_phi_i) * st_i * st_k
	)

	M[2 * i, (2 * i - 2) % (2 * N)] += K * K_ti_tk
	M[2 * i, (2 * i - 1) % (2 * N)] += K * K_ti_pk
	M[2 * i + 1, (2 * i - 2) % (2 * N)] += K * K_pi_tk
	M[2 * i + 1, (2 * i - 1) % (2 * N)] += K * K_pi_pk


q_vals = np.linspace(-2 * np.pi, 2 * np.pi, 1000, endpoint=False)
M_qvalues = np.zeros((len(q_vals), 2 * c, 2 * c), dtype=complex)


def M_qv(q, M, x, c):
	Mq = np.zeros((2 * c, 2 * c), dtype=complex)
	N = len(x)
	n_cells = N // c
	for alpha in range(c):
		for beta in range(c):
			sum_block = np.zeros((2, 2), dtype=complex)
			for ci in range(n_cells):
				i_global = ci * c + alpha
				for cj in range(n_cells):
					j_global = cj * c + beta
					block = M[2 * i_global : 2 * i_global + 2, 2 * j_global : 2 * j_global + 2]
					if np.allclose(block, 0):
						continue
					phase = np.exp(-1j * q * (x[j_global] - x[i_global]))
					sum_block += block * phase
			Mq[2 * alpha : 2 * alpha + 2, 2 * beta : 2 * beta + 2] = sum_block
	return Mq


eigs = np.zeros((len(q_vals), 2 * c), dtype=complex)

for qi, q in enumerate(q_vals):
	M_qvalues[qi, :, :] = M_qv(q, M, x, c)
	G_q = np.zeros((2 * c, 2 * c), dtype=complex)
	for unit_index in range(c):
		st_unit = np.sin(theta0[unit_index])
		G_q[2 * unit_index, 2 * unit_index + 1] = st_unit
		G_q[2 * unit_index + 1, 2 * unit_index] = -st_unit
	M_qi = np.linalg.inv(G_q) @ M_qvalues[qi, :, :]
	eigs[qi, :] = np.linalg.eigvals(M_qi)


plt.figure(figsize=(8, 5))
for band in range(2 * c):
	freqs = 1.7e11 * np.imag(eigs)
	plt.plot(q_vals, freqs[:, band], ".", markersize=2)

plt.xlabel("Wavevector (q)")
plt.ylabel("Frequency (ω)")
plt.title("Spin Wave Dispersion of Antiferromagnetic Chain")
plt.grid(True)
plt.show()
