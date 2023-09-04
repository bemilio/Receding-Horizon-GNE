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

def compute_prediction_model(A,B,t_pred=1000):
    n = A.shape[0]
    m = B.shape[1]
    T = np.zeros((n*t_pred, n))
    S = np.zeros((n*t_pred, m*t_pred))
    A_pow = np.matrix(np.eye(n))
    for t in range(t_pred):
        S[t*n:(t+1)*n, :m] = A_pow * B
        if t!=0:
            S[t * n:(t + 1) * n, m:] = S[(t-1) * n:(t) * n, :-m]
        A_pow = A_pow * A
        T[n*t:n*(t+1),:] = A_pow
    return T,S

def compute_extended_cost_matrices(Q,R, index_agent, t_pred=1000):
    i = index_agent
    N = len(Q)
    Q_i_extd = [np.matrix(np.kron(np.eye(t_pred), Q[i][i*n_x:(i+1)*n_x, j*n_x:(j+1)*n_x ])) for j in range(N) ]
    R_i_extd = [np.matrix(np.kron(np.eye(t_pred), R[i][j])) for j in range(N)]
    return Q_i_extd, R_i_extd

def computed_extended_control_mapping(A,B,K,t_pred=1000):
    # maps (stacked) initial state to (stacked) input sequences
    N = len(A)
    n_x = A[0].shape[0]
    n_u = B[0].shape[1]
    A_bar = np.matrix(np.zeros((N*n_x, N*n_x)))
    B_bar = np.matrix(np.zeros((N*n_x, N*n_u)))
    K_bar = np.matrix(np.zeros((N*n_u, N*n_x)))
    for i in range(N):
        A_bar[i*n_x:(i+1)*n_x, i*n_x:(i+1)*n_x] = A[i]
        B_bar[i * n_x:(i + 1) * n_x, i*n_u:(i+1)*n_u] = B[i]
        K_bar[i*n_u:(i+1)*n_u, :] = K[i]
    T_K = np.matrix(np.zeros((t_pred*N*n_x, N*n_x)))
    A_K_pow = np.matrix(np.eye(N*n_x))
    for t in range(t_pred):
        T_K[t*N*n_x: (t+1)*N*n_x] = A_K_pow
        A_K_pow = A_K_pow * (A_bar-B_bar * K_bar)
    K_extd = -np.kron(np.eye(t_pred), K_bar) * T_K
    return K_extd

if __name__ == "__main__":
    N = 10
    n_x = 1
    n_u = 1
    n_iterations = 50
    length_prediction_model = 100

    A = [np.matrix(np.zeros((n_x, n_x))) for i in range(N)]
    B = [np.matrix(np.zeros((n_x, n_u))) for i in range(N)]

    Q_i = [ np.matrix(np.zeros((n_x, n_x))) for i in range(N)]
    Q_j = [np.matrix(np.zeros((n_x, (N-1)*n_x))) for i in range(N)]
    R = [ [np.matrix(np.zeros((n_u, n_u))) for i in range(N)] for j in range(N) ]
    K   = [ np.matrix(np.zeros((n_u, N*n_x))) for i in range(N)]

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
            ctrb_mat = control.ctrb(A[i], B[i])

        Q_temp = rnd.random_sample(size=[n_x, n_x])
        Q_temp = Q_temp + Q_temp.T
        if min(np.linalg.eigvalsh(Q_temp))<0:
            Q_temp = Q_temp - min(np.linalg.eigvalsh(Q_temp)) * np.eye(n_x)
        Q_i[i][:, :] = Q_temp

        Q_temp = rnd.random_sample(size=[n_x, (N-1)*n_x])
        Q_j[i][:, :] = Q_temp

        R_temp = rnd.random_sample(size=[n_u, n_u])
        R_temp = R_temp +  R_temp.T
        if min(np.linalg.eigvalsh(R_temp)) <= 0:
            R_temp = R_temp - min(np.linalg.eigvalsh(R_temp)) * np.eye(n_u)
        R[i][i][:, :] = R_temp

        for j in range(N):
            if j!=i:
                R_temp = rnd.random_sample(size=[n_u, n_u])
                R[i][j][:, :] = R_temp

        # Initialize controller to a local stabilizing one
        P_i = linalg.solve_discrete_are(A[i], B[i], np.eye(n_x), np.eye(n_u))
        K[i][:,i*n_x:(i+1)*n_x] = np.linalg.inv(np.eye(n_u) + B[i].T * P_i * B[i] ) * (B[i].T * P_i * A[i])


    for k in range(n_iterations):
        for i in range(N):
            A_bar_i = np.matrix(np.zeros(((N+1)*n_x, (N+1)*n_x)))
            A_bar_i[0:n_x, 0:n_x] = A[i]
            for j in range(N):
                A_bar_i[(j+1)*n_x:(j+2)*n_x,n_x:] = np.kron(I_N[j,:], A[j]) - (B[j]* K[j])
            B_bar_i = np.matrix(np.zeros(((N+1)*n_x, n_u)))
            B_bar_i[0:n_x, :] = B[i]
            Q_bar_i = np.matrix(np.zeros(((N+1)*n_x, (N+1)*n_x)))
            Q_bar_i[0:n_x,:] = np.column_stack((Q_i[i], Q_j[i][:,:i*n_x], np.zeros((n_x, n_x)), Q_j[i][:,i*n_x:]))
            Q_bar_i[:, 0:n_x] = Q_bar_i[0:n_x,:].T
            Q_bar_i[n_x:, n_x:] = - min(np.linalg.eigvalsh(Q_bar_i))*np.eye(N*n_x)

            S = np.matrix(np.zeros(((N+1) * n_x, n_u)))
            for j in range(N):
                if j!=i:
                    S[n_x:, :] = S[n_x:, :]  + K[j].T * R[i][j]
            try:
                P_i = linalg.solve_discrete_are(A_bar_i, B_bar_i, Q_bar_i, R[i][i], s=S)
            except Exception as e:
                print("Error: ", str(e))
            K_temp = np.linalg.inv(R[i][i] + B_bar_i.T * P_i * B_bar_i ) * (B_bar_i.T * P_i * A_bar_i + S.T)
            # if np.linalg.norm(K_temp[:, (i+1)*n_x:(i+2)*n_x])> .00001:
            #     print("Warning: it looks like the controller depends on the auxiliary state")
            K[i][:, :] = K_temp[:, n_x:]
            K[i][:, i*n_x: (i+1)*n_x] =  K_temp[:,:n_x]  # K[i][:, i*n_x: (i+1)*n_x]
            K_t[i][k] = K[i]


    # Check stability
    A_K = np.matrix(np.zeros((N*n_x, N*n_x)))
    for i in range(N):
        A_K[j * n_x:(j + 1) * n_x, :] = np.kron(I_N[j, :], A[j]) - (B[j] * K[j])
        # Plot eigenvalues
    plot_eigenvalues(A_K)

    # Compute the difference in norm between the sequence of controller and controller at last iteration
    difference_norm = [compute_sequence_to_mat_difference_norm(K_t[i], K_t[i][-1]) for i in range(N)]

    # Plotting the difference in norm

    for i, lst in enumerate(difference_norm):
        plt.plot(lst, label=f'Agent {i+1}')

    plt.xlabel('t')
    plt.ylabel('$\|K_i(t) - K_i(T)\|$')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

    # Compute pseudogradient of truncated OCPs
    # PROBLEM: not reliable as S, T explode with A unstable
    # K_extd = computed_extended_control_mapping(A,B,K,t_pred=length_prediction_model)
    # T=[]
    # S=[]
    # for i in range(N):
    #     T_i, S_i = compute_prediction_model(A[i], B[i], t_pred=length_prediction_model)
    #     T.append(T_i)
    #     S.append(S_i)
    #
    # for i in range(N):
    #     Q_i_extd, R_i_extd = compute_extended_cost_matrices(Q, R, i, t_pred=length_prediction_model)
    #
    #     Lin_map_nabla_J_i = np.matrix(np.column_stack([S[i].T * Q_i_extd[j] * S[j] + R_i_extd[j] for j in range(N)])) * K_extd + \
    #                         np.matrix(np.column_stack([S[i].T * Q_i_extd[j] * T[j] for j in range(N)]))
    #     print("Norm of pseudogradient mapping: " + str(np.linalg.norm(Lin_map_nabla_J_i,  ord='fro')))

    # Check Bellman principle


