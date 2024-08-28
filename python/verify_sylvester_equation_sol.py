import warnings
import numpy as np
import control
from games import dyngames
from numpy.linalg import norm
from games.staticgames import multiagent_array
import scipy
import pickle
import logging

'''
This script checks if:
 1) the non-stabilizing solutions of an O-NE change when restarting the O-NE seeking algorithm
 2) The P_tilde matrix obtained by numerical integration is equal to P - P_LQR where P solves sylvester
'''

logging.basicConfig(filename='log_verify_sylvester_equation_sol.txt', filemode='w', level=logging.DEBUG)

N = 3
n_x = 2
n_u = 2
n_iter = 1000
eps_error = 10**(-5)

stacked_P_difference_after_reinitialization = []
stacked_K_difference_after_reinitialization = []

stacked_systems_with_multiple_solutions = []
stacked_solutions_of_systems_with_mult_sol = []

stacked_P_tilde_difference = []
stacked_systems_with_P_tilde_different = []

n_attempts = 1000
n_reinitializations = 200
integration_length = 200

def compute_P_tilde(A, B, K, Q, R, method = "sylvester", integration_length = integration_length):
    N_agents = B.shape[0]
    n_x = B.shape[1]
    n_u = B.shape[2]
    P_LQR = np.zeros((N_agents, n_x, n_x))
    K_LQR = np.zeros((N_agents, n_u, n_x))
    I_x = np.eye(n_x)
    A_cl_i = np.zeros((N_agents, n_x, n_x))
    A_cl = A + np.sum(B @ K, axis=0)
    for i in range(N_agents):
        P_LQR[i] = scipy.linalg.solve_discrete_are(A, B[i], Q[i], R[i])
        K_LQR[i] = - np.linalg.inv(B[i].T @ P_LQR[i] @ B[i] + R[i]) @ B[i].T @ P_LQR[i] @ A
        A_cl_i[i] = (A.T @ (I_x - B[i] @ np.linalg.inv(R[i] + B[i].T @ P_LQR[i] @ B[i]) @ B[i].T @ P_LQR[i]).T).T
    if method == "summation":
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
    elif method == "sylvester":
        A_cl_lqr = A + B @ K_LQR
        P_tilde = np.zeros((N_agents, n_x, n_x))
        for i in range(N_agents):
            sum_j_BK = np.sum(B @ K, axis=0) - B[i] @ K[i]
            try:
                P_tilde[i] = control.dlyap(A_cl_lqr[i].T, A_cl.T, A_cl_lqr[i].T @ P_LQR[i] @ sum_j_BK)
            except Exception:
                P_tilde[i] = None * np.eye(n_x)
    else:
        ValueError("[compute_P_tilde] method must be 'silvester' or 'summation' ")

    err_test = 0
    for i in range(N_agents):
        err_test = max(err_test, norm(P_tilde[i] + A_cl_i[i].T @ P_LQR[i] @ A - A.T @ (P_tilde[i] + P_LQR[i]) @ A_cl))
    return P_LQR, P_tilde

for attempt in range(n_attempts):
    A, B, Q, R = dyngames.LQ.generate_random_game(N, n_x, n_u)
    A = multiagent_array(A)
    B = multiagent_array(B)
    Q = multiagent_array(Q)
    R = multiagent_array(R)

    test_number_new_initialization = 0
    for test_number_new_initialization in range(n_reinitializations):
        print("test " + str(attempt) + "; reinitialization" + str(test_number_new_initialization))
        logging.info("test " + str(attempt) + "; reinitialization" + str(test_number_new_initialization))
        K = np.zeros((N, n_u, n_x))
        P = np.zeros((N, n_x, n_x))
        is_initialization_stable = False
        while not is_initialization_stable:
            K_init = 10 * np.random.randn(N, n_u, n_x)
            is_initialization_stable = True
            # is_initialization_stable = max(np.abs(np.linalg.eigvals(A + np.sum(B @ K_init, axis=0)))) < 1
        K = K_init
        for k in range(n_iter):
            A_cl = A + np.sum(B @ K, axis=0)
            for i in range(N):
                try:
                    P[i][:, :] = control.dlyap(A.T, A_cl.T, Q[i])
                except Exception as e:
                    print("[solve_open_loop_inf_hor_problem] An error occurred while solving the Sylvester equation:")
                    print(str(e))
                    logging.info("[solve_open_loop_inf_hor_problem] An error occurred while solving the Sylvester equation:" + str(e))
            M = np.linalg.inv(np.eye(n_x) + np.sum(B @ np.linalg.inv(R) @ B.T3D() @ P, axis=0))
            K = - np.linalg.inv(R) @ B.T3D() @ P @ M @ A  # Batch multiplication
            if n_iter % 10 == 0:
                err = 0
                A_Tinv = np.linalg.inv(A.T)
                S = B @ np.linalg.inv(R) @ B.T3D()
                for i in range(N):
                    err = err + norm(A_Tinv @ (Q[i] - P[i]) + P[i] @ (A + np.sum(S @ A_Tinv @ (Q - P), axis=0)))
                if err < eps_error:
                    break
        is_P_posdef = True
        is_OL_stable = True
        is_P_symmetric = True
        if err > eps_error:
            print("Could not find solution")
            logging.info("Could not find solution")
            is_solved = False
            break
        else:
            is_solved = True
            for i in range(N):
                if min(np.linalg.eigvals(P[i])) < 0:
                    warnings.warn("The open loop P is non-positive definite")
                    is_P_posdef = False
                if np.linalg.norm(P[i] - P[i].T) > eps_error:
                    is_P_symmetric = False
            if max(np.abs(np.linalg.eigvals(A + np.sum(B @ K, axis=0)))) > 1.001:
                warnings.warn("The infinite horizon OL-GNE has an unstable dynamics")
                is_OL_stable = False
        new_sol_available = False
        if test_number_new_initialization == 0:
            all_P_solutions = [P]
            all_K_solutions = [K]
            new_sol_available = True
        if test_number_new_initialization >= 1:
            print("P difference after initialization change: " + str(norm(P-all_P_solutions[0])))
            logging.info("P difference after initialization change: " + str(norm(P-all_P_solutions[0])))
            print("K difference after initialization change: " + str(norm(K - all_K_solutions[0])))
            logging.info("K difference after initialization change: " + str(norm(K - all_K_solutions[0])))
            stacked_P_difference_after_reinitialization.append(norm(P-all_P_solutions[0]))
            stacked_K_difference_after_reinitialization.append(norm(K - all_K_solutions[0]))
            if all(norm(P-P_sol) > 0.001 for P_sol in all_P_solutions):
                all_P_solutions.append(P)
                new_sol_available = True
        if test_number_new_initialization==n_reinitializations-1:
            # save all solutions
            if len(all_P_solutions)>1:
                stacked_systems_with_multiple_solutions.append((A,B,Q,R))
                stacked_solutions_of_systems_with_mult_sol.append(all_P_solutions)
        if is_solved and new_sol_available:
            # If there is a new solution, check if P_tilde = P - P_LQR
            P_LQR, P_tilde = compute_P_tilde(A, B, K, Q, R, integration_length= integration_length, method="sylvester")
            stacked_P_tilde_difference.append(norm(P - P_LQR - P_tilde))
            if norm(P - P_LQR - P_tilde)>0.001:
                stacked_systems_with_P_tilde_different.append((A, B, Q, R, P, P_tilde))
            print("P tilde difference: " + str(norm(P - P_LQR - P_tilde)))

print("max P difference after initialization change: " + str(max(stacked_P_difference_after_reinitialization)))
logging.info("max P difference after initialization change: " + str(max(stacked_P_difference_after_reinitialization)))
print("max K difference after initialization change: " + str(max(stacked_K_difference_after_reinitialization)))
logging.info("max K difference after initialization change: " + str(max(stacked_K_difference_after_reinitialization)))
print("max P difference with P tilde: " + str(max(stacked_P_tilde_difference)))
logging.info("max P difference with P tilde: " + str(max(stacked_P_tilde_difference)))
f = open('verify_sylvester_equation_result.pkl', 'wb')
# pickle.dump([ x_store, u_store, residual_store, u_pred_traj_store, x_pred_traj_store, u_shifted_traj_store,\
#               K_CL_store, K_OL_store, A_store, B_store,
#               T_hor_to_test, N_agents_to_test,
#               cost_store, u_PMP_CL_store, u_PMP_OL_store, P_OL_store,
#               is_game_solved_store, is_ONE_solved_store, is_P_subspace_store,
#               is_subspace_stable_store, is_subspace_unique_store], f)
pickle.dump([ stacked_P_difference_after_reinitialization,
              stacked_K_difference_after_reinitialization,
              stacked_P_tilde_difference,
              stacked_systems_with_multiple_solutions,
              stacked_solutions_of_systems_with_mult_sol,
              stacked_systems_with_P_tilde_different], f)
f.close()
print("Saved")
logging.info("Saved, job done")