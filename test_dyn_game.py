import numpy as np
import networkx as nx
# import torch
import pickle
from games.staticgames import batch_mult_with_coulumn_stack

import numpy.linalg.linalg

from games.dyngames import LQ
from algorithms.GNE_centralized import pFB_algorithm
import matplotlib.pyplot as plt
from scipy.linalg import block_diag, norm

## This script creates a "game" with two unconstrained decoupled double integrators with terminal cost given by the solution
## of the DARE. The trajectory is compared with the one resulting by the LQR.

if __name__ == '__main__':
    N_it_per_residual_computation = 10
    # parameters
    N_iter = 5000
    T_sim = 20
    eps = 10**(-5) # convergence threshold
    test_threshold = eps *10
    ##########################################
    #             Game inizialization        #
    ##########################################
    dyn_game = LQ(None, None, None, None, None, None, None, None, None, None, None, None, test=True)
    x_0 = np.random.random_sample(size=(dyn_game.n_x, 1))
    x_0_LQR = x_0
    x_0_inf_hor = x_0
    # Define stacked LQR
    K = np.stack([-numpy.linalg.inv(dyn_game.R[i]+ dyn_game.B[i].T @ dyn_game.P[i] @ dyn_game.B[i]) @ \
                  dyn_game.B[i].T @  dyn_game.P[i] @ dyn_game.A for i in range(dyn_game.N_agents)])
    # P_inf_hor, K_inf_hor = dyn_game.solve_closed_loop_inf_hor_problem(method='riccati')
    P_inf_hor, K_inf_hor = dyn_game.solve_open_loop_inf_hor_problem()
    ##########################################
    #   Variables storage inizialization     #
    ##########################################
    x_store = np.zeros((dyn_game.n_x, T_sim))
    x_0_LQR_store = np.zeros((dyn_game.n_x, T_sim))
    x_0_inf_hor_store = np.zeros((dyn_game.n_x, T_sim))
    u_store = np.zeros((dyn_game.N_agents, dyn_game.n_u, T_sim))
    for t in range(T_sim):
        game_t = dyn_game.generate_game_from_initial_state(x_0)
        #######################################
        #          GNE seeking                #
        #######################################
        # alg. initialization
        alg = pFB_algorithm(game_t, primal_stepsize=0.01, dual_stepsize=0.01)
        index_storage = 0
        avg_time_per_it = 0
        for k in range(N_iter):
            if k % N_it_per_residual_computation == 0:
                u_all, d, r, c = alg.get_state()
                index_storage = index_storage + 1
                if r.item()<=eps:
                    break
            #  Algorithm run
            alg.run_once()
        # Convert optimization variable into state and input
        u_all, d, r, c = alg.get_state()
        u_0 = dyn_game.get_input_timestep_from_opt_var(u_all,0)
        u_store[:, :, t] = u_0.squeeze(2)
        # Evolve state
        x_store[:, t] = x_0.squeeze(1)
        x_0 = dyn_game.A @ x_0 + np.sum(dyn_game.B @ u_0, axis=0)
        x_0_LQR_store[ :, t] = x_0_LQR.squeeze(1)
        x_0_LQR = dyn_game.A @ x_0_LQR + np.sum(dyn_game.B @ K @ x_0_LQR, axis=0)
        # print("Timestep " + str(t) + " of " + str(T_sim))
        x_0_inf_hor_store[:,t] = x_0_inf_hor.squeeze(1)
        x_0_inf_hor = dyn_game.A @ x_0_inf_hor + np.sum(dyn_game.B @ K_inf_hor @ x_0_inf_hor, axis=0)

    plt.plot(x_store[:,:].T, color='blue')
    plt.plot(x_0_LQR_store[:, :].T, color='red')
    plt.plot(x_0_inf_hor_store[:, :].T, color='green')
    plt.show(block='False')
    if np.linalg.norm(x_store - x_0_LQR_store) < test_threshold \
            and np.linalg.norm(x_store - x_0_inf_hor_store)< test_threshold \
            and norm(P_inf_hor - dyn_game.P) < test_threshold:
        print("Dynamic game test PASSED")
    else:
        print("Dynamic game test FAILED")