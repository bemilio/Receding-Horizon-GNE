import pickle
import numpy as np
import scipy
import control
from games.staticgames import multiagent_array
from numpy.linalg import norm, eigvals
def compute_P_tilde(A, B, K, Q, R, integration_length = 100):
    N_agents = B.shape[0]
    n_x = B.shape[1]
    n_u = B.shape[2]
    P_LQR = np.zeros((N_agents, n_x, n_x))
    K_LQR = np.zeros((N_agents, n_u, n_x))
    I_x = np.eye(n_x)
    A_cl_i = np.zeros((N_agents, n_x, n_x))
    for i in range(N_agents):
        P_LQR[i] = scipy.linalg.solve_discrete_are(A, B[i], Q[i], R[i])
        K_LQR[i] = - np.linalg.inv(B[i].T @ P_LQR[i] @ B[i] + R[i]) @ B[i].T @ P_LQR[i] @ A
        A_cl_i[i] = (A.T @ (I_x - B[i] @ np.linalg.inv(R[i] + B[i].T @ P_LQR[i] @ B[i]) @ B[i].T @ P_LQR[i]).T).T
    A_cl = A + np.sum(B @ K, axis=0)
    b = np.zeros((N_agents, integration_length * 2, n_x, n_x))
    w = np.zeros((N_agents, integration_length * 4, n_x, n_x))
    for i in range(N_agents):
        A_cl_power_k = np.eye(n_x)
        for k in range(w.shape[1]):
            w[i, k] = (np.sum(B @ K, axis=0) - B[i] @ K[i]) @ A_cl_power_k
            A_cl_power_k = A_cl_power_k @ A_cl
        for k in range(b.shape[1]):
            for h in range(k, w.shape[1]):
                b[i, k] = b[i, k] + np.linalg.matrix_power(A_cl_i[i].T, h - k + 1) @ P_LQR[i] @ w[i, h]
    P_tilde = b[:,0]
    return P_LQR, P_tilde


f = open(r'C:\Users\ebenenati\surfdrive - Emilio Benenati@surfdrive.surf.nl\TUDelft\Simulations\verify_sylvester_equation\12_10_23\verify_sylvester_equation_result.pkl', 'rb')

stacked_P_difference_after_reinitialization,\
stacked_K_difference_after_reinitialization,\
stacked_P_tilde_difference,\
stacked_systems_with_multiple_solutions,\
stacked_solutions_of_systems_with_mult_sol,\
stacked_systems_with_P_tilde_different = pickle.load(f)

N_solutions = []
for sol in stacked_solutions_of_systems_with_mult_sol:
    N_solutions.append(len(sol))

N_pos_def_sol = []
for sol in stacked_solutions_of_systems_with_mult_sol:
    n_pdef = 0
    for P in sol:
        if np.min(np.linalg.eigvals(P))>0:
            n_pdef = n_pdef +1
    N_pos_def_sol.append(n_pdef)

N_pos_def_symmetrized_sol = []
for sol in stacked_solutions_of_systems_with_mult_sol:
    n_pdef = 0
    for P in sol:
        P = multiagent_array(P)
        if np.min(np.linalg.eigvals(P + P.T3D())) > 0:
            n_pdef = n_pdef + 1
    N_pos_def_symmetrized_sol.append(n_pdef)

index_multiple_pos_def_sol = [index for index, value in enumerate(N_pos_def_sol) if value >1]
if len(index_multiple_pos_def_sol)>0:
    A, B, Q, R = stacked_systems_with_multiple_solutions[index_multiple_pos_def_sol[0]]
    n_x = A.shape[0]
    N_agents = B.shape[0]
    K_all = []
    A_one_all = []
    P_lqr_all = []
    P_tilde_all = []
    P_tilde_alt_all = []
    for P in stacked_solutions_of_systems_with_mult_sol[index_multiple_pos_def_sol[0]]:
        M = np.linalg.inv(np.eye(n_x) + np.sum(B @ np.linalg.inv(R) @ B.T3D() @ P, axis=0))
        K = - np.linalg.inv(R) @ B.T3D() @ P @ M @ A
        K_all.append(- K)
        A_one = A + np.sum(B @ K, axis=0)
        A_one_all.append(A_one)
        P_lqr, P_tilde = compute_P_tilde(A,B, K, Q, R)
        K_lqr = - np.linalg.inv(R + B.T3D() @ P_lqr @ B) @ (B.T3D() @ P_lqr @ A)
        P_lqr_all.append(P_lqr)
        P_tilde_all.append(P_tilde)
        P_tilde_alt = np.zeros((N_agents, n_x, n_x))
        A_cl_lqr = A + B @ K_lqr
        for i in range(N_agents):
            sum_j_BK = np.sum( B @ K, axis = 0) - B[i] @ K[i]
            try:
                P_tilde_alt[i] = control.dlyap(A_cl_lqr[i].T, A_one.T, A_cl_lqr[i].T @ P_lqr[i] @ sum_j_BK )
            except Exception:
                P_tilde_alt[i] = None * np.eye(n_x)
        P_tilde_alt_all.append(P_tilde_alt)


norms_critical_P_tilde = []
difference_critical_P_tilde = []
for sys in stacked_systems_with_P_tilde_different:
    A, B, Q, R, P, P_tilde = sys
    N_agents = B.shape[0]
    n_x = B.shape[1]
    P_LQR = np.zeros((N_agents, n_x, n_x))
    for i in range(N_agents):
        P_LQR[i] = scipy.linalg.solve_discrete_are(A, B[i], Q[i], R[i])
    difference_critical_P_tilde.append(norm(P - P_LQR - P_tilde))
    if norm(P - P_LQR - P_tilde)>1:
        print("pause..")
    if norm(P_tilde) < 100:
        print("pause...")
    norms_critical_P_tilde.append(norm(P_tilde))

print("max P difference after initialization change: " + str(max(stacked_P_difference_after_reinitialization)))
print("max K difference after initialization change: " + str(max(stacked_K_difference_after_reinitialization)))
print("max P difference with P tilde: " + str(max(stacked_P_tilde_difference)))
