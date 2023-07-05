import numpy as np
import networkx as nx
# import torch
import pickle

import numpy.linalg.linalg

from games.dyngames import LQ_decoupled
from algorithms.GNE_centralized import pFB_algorithm


if __name__ == '__main__':

    N_it_per_residual_computation = 10

    # parameters
    N_iter = 10000
    T_sim = 20
    eps = 10**(-5) # convergence threshold
    ##########################################
    #             Game inizialization        #
    ##########################################
    dyn_game = LQ_decoupled(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, test=True)
    x_0 = np.random.random_sample(size=(dyn_game.N_agents, dyn_game.n_x, 1))
    x_0_LQR = x_0
    K = - numpy.linalg.inv(dyn_game.R) @ np.transpose(dyn_game.B, axes=(0,2,1)) @  dyn_game.P
    ##########################################
    #   Variables storage inizialization     #
    ##########################################
    x_store = np.zeros((dyn_game.N_agents, dyn_game.n_x, T_sim))
    x_0_LQR_store = np.zeros((dyn_game.N_agents, dyn_game.n_x, T_sim))
    u_store = np.zeros((dyn_game.N_agents, dyn_game.n_x, T_sim))
    for t in range(T_sim):
        game_t = dyn_game.generate_game_from_initial_state(x_0)
        #######################################
        #          GNE seeking                #
        #######################################
        # alg. initialization
        alg = pFB_algorithm(game_t, primal_stepsize=0.1, dual_stepsize=0.1)
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
        u_0 = dyn_game.get_first_input_from_opt_var(u_all)
        u_store[:, :, t] = u_0.squeeze(2)
        # Evolve state
        x_0 = dyn_game.A @ x_0 + dyn_game.B @ u_0
        x_store[:, :, t] = x_0.squeeze(2)
        x_0_LQR = (dyn_game.A + K @ dyn_game.B) @ x_0_LQR
        x_0_LQR_store[:, :, t] = x_0_LQR.squeeze(2)
        print("Timestep " + str(t) + " of " + str(T_sim))


    if np.linalg.norm(x_store - x_0_LQR_store) < eps:
        print("Dynamic game test PASSED")
    else:
        print("Dynamic game test FAILED")