from scipy import linalg
import numpy as np
from numpy import random as rnd
import control
import matplotlib.pyplot as plt
import torch

plt.rcParams["text.usetex"] = True

def compute_sequence_to_mat_difference_norm(sequence, mat):
    # Compute the difference in norm between two sequences of matrices
    norm_diff = []
    for i in range(len(sequence)):
        diff = np.linalg.norm(sequence[i] - mat,  ord='fro')
        norm_diff.append(diff)
    return norm_diff

def plot_eigenvalues(matrix):
    # Compute eigenvalues of the matrix
    eigenvalues, _ = np.linalg.eig(matrix)

    # Plot eigenvalues with respect to the unit circle
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='b', marker='x', s=20)
    unit_circle = plt.Circle((0, 0), 1, color='r', fill=False)
    ax = plt.gca()
    ax.add_patch(unit_circle)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Eigenvalues with Respect to the Unit Circle')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

N = 10
n_x = 4
n_u = 2
n_iterations = 100

A = [np.matrix(np.zeros((n_x, n_x))) for i in range(N)]
B = [np.matrix(np.zeros((n_x, n_u))) for i in range(N)]

Q = [ np.matrix(np.zeros((N*n_x, N*n_x))) for i in range(N)]
R = [ [np.matrix(np.zeros((n_u, n_u))) for i in range(N)] for j in range(N)]
K = [ np.matrix(np.zeros((n_u, N*n_x))) for i in range(N)]

I_N = np.matrix(np.eye(N))

# Only for visualization
K_t = [ [np.matrix(np.zeros((n_u, N*n_x)))  for t in range(n_iterations)] for i in range(N)]

for i in range(N):
    A[i][:,:] = -0.5 + 3 * rnd.random_sample(size=[n_x, n_x])
    B[i][:, :] = -0.5 + 3 * rnd.random_sample(size=[n_x, n_u])
    ctrb_mat = control.ctrb(A[i], B[i])
    while np.linalg.matrix_rank(ctrb_mat) != A[i].shape[0]:
        A[i][:, :] = -0.5 + 3 * rnd.random_sample(size=[n_x, n_x])
        B[i][:, :] = -0.5 + 3 * rnd.random_sample(size=[n_x, n_u])
        sys = control.StateSpace(A[i], B[i], np.eye(A[i].shape[0]), np.zeros(B[i].shape))
        ctrb_mat = control.ctrb(sys)

    Q_i = rnd.random_sample(size=[N*n_x, N*n_x])
    Q_i = Q_i + Q_i.T
    if min(np.linalg.eigvalsh(Q_i))<0:
        Q_i = Q_i - min(np.linalg.eigvalsh(Q_i)) * np.eye(N*n_x)
    Q[i][:, :] = Q_i

    R_i = rnd.random_sample(size=[N*n_u, N*n_u])
    R_i = R_i +  R_i.T
    if min(np.linalg.eigvalsh(R_i)) <= 0:
        R_i = R_i - min(np.linalg.eigvalsh(R_i)) * np.eye(N*n_u)
    for j in range(N):
        R[i][j][:, :] = R_i[n_u * i: n_u * (i + 1), n_u * j: n_u * (j + 1)]

    # Initialize controller to a local stabilizing one
    P_i = linalg.solve_discrete_are(A[i], B[i], np.eye(n_x), np.eye(n_u))
    K[i][:,i*n_x:(i+1)*n_x] = np.linalg.inv(np.eye(n_u) + B[i].T * P_i * B[i] ) * (B[i].T * P_i * A[i])


for k in range(n_iterations):
    for i in range(N):
        A_K = np.matrix(np.zeros((N*n_x, N*n_x)))
        for j in range(N):
            if j != i:
                A_K[j*n_x:(j+1)*n_x,:] = np.kron(I_N[j,:], A[j]) - (B[j]* K[j])
            else:
                A_K[i*n_x:(i+1)*n_x, i*n_x:(i+1)*n_x] = A[i]
        B_bar_i = np.kron(I_N[:, i], B[i])
        S = np.matrix(np.zeros((N * n_x, n_u)))
        for j in range(N):
            if j!=i:
                S = S + (R[i][j]* K[j]).T
        P_i = linalg.solve_discrete_are(A_K, B_bar_i, Q[i], R[i][i], s=S)
        K[i] = np.linalg.inv(R[i][i] + B_bar_i.T * P_i * B_bar_i ) * (B_bar_i.T * P_i * A_K + S.T)
        K_t[i][k] = K[i]


# Check stability
A_K = np.matrix(np.zeros((N*n_x, N*n_x)))
for i in range(N):
    A_K[j * n_x:(j + 1) * n_x, :] = np.kron(I_N[j, :], A[j]) - (B[j] * K[j])
    # Plot eigenvalues
plot_eigenvalues(A_K)


# Compute the difference in norm between the sequences
difference_norm = [compute_sequence_to_mat_difference_norm(K_t[i], K_t[i][-1]) for i in range(N)]

# Plotting the difference in norm

for i, lst in enumerate(difference_norm):
    plt.plot(lst, label=f'Agent {i+1}')

plt.xlabel('t')
plt.ylabel('$\|K_i(t) - K_i(T)\|$')
plt.legend()
plt.grid(True)
plt.show(block=True)

