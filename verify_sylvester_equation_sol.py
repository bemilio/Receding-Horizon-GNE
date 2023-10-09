import warnings
import numpy as np
import control
from games import dyngames
from numpy.linalg import norm
from games.staticgames import multiagent_array


N = 6
n_x = 4
n_u = 3
n_iter = 1000
eps_error = 10**(-6)

stacked_P_difference_after_reinitialization = []
stacked_K_difference_after_reinitialization = []

for attempt in range(1000):
    print("attempt " + str(attempt))
    A, B, Q, R = dyngames.LQ.generate_random_game(N, n_x, n_u)
    A = multiagent_array(A)
    B = multiagent_array(B)
    Q = multiagent_array(Q)
    R = multiagent_array(R)

    test_number_new_initialization = 0
    for test_number_new_initialization in range(10):
        K = np.zeros((N, n_u, n_x))
        P = np.zeros((N, n_x, n_x))
        is_initialization_stable = False
        while not is_initialization_stable:
            K_init = np.random.randn(N, n_u, n_x)
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
            print("[solve_open_loop_inf_hor_problem] Could not find solution")
            is_solved = False
        else:
            is_solved = True
            for i in range(N):
                if min(np.linalg.eigvals(P[i])) < 0:
                    warnings.warn("[solve_open_loop_inf_hor_problem] The open loop P is non-positive definite")
                    is_P_posdef = False
                if np.linalg.norm(P[i] - P[i].T) > eps_error:
                    is_P_symmetric = False
            if max(np.abs(np.linalg.eigvals(A + np.sum(B @ K, axis=0)))) > 1.001:
                warnings.warn("The infinite horizon OL-GNE has an unstable dynamics")
                is_OL_stable = False
        if test_number_new_initialization == 0 and not is_OL_stable:
            P_test_0 = P
            K_test_0 = K
        if test_number_new_initialization>=1 and not is_OL_stable:
            print("P difference after initialization change: " + str(norm(P-P_test_0)))
            print("K difference after initialization change: " + str(norm(K - K_test_0)))
            stacked_P_difference_after_reinitialization.append(norm(P-P_test_0))
            stacked_K_difference_after_reinitialization.append(norm(K - K_test_0))

        if test_number_new_initialization == 0 and  is_OL_stable:
            # we don't care about stable solutions in this test
            break

print("max P difference after initialization change: " + str(max(stacked_P_difference_after_reinitialization)))
print("max K difference after initialization change: " + str(max(stacked_K_difference_after_reinitialization)))
