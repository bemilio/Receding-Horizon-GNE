import warnings

import numpy as np
import networkx as nx
# import torch
import pickle
from games.dyngames import LQ_decoupled
from algorithms.GNE_centralized import pFB_algorithm
import matplotlib.pyplot as plt
import time
import logging
import sys
from scipy.linalg import norm
import copy
import math

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    if len(sys.argv) < 2:
        seed = 1
        job_id=0
    else:
        seed=int(sys.argv[1])
        job_id = int(sys.argv[2])
    print("Random seed set to  " + str(seed))
    logging.info("Random seed set to  " + str(seed))
    np.random.seed(seed)
    N_it_per_residual_computation = 10
    N_agents = 2
    N_random_tests = 1

    # parameters
    N_iter = 10000
    n_x = 3
    n_u = 2
    T_hor = 3
    T_sim = 10
    eps = 10**(-5) # convergence threshold

    for test in range(N_random_tests):
        ##########################################
        #        Test case creation              #
        #########################################
        A, B, Q, R = LQ_decoupled.generate_random_game(N_agents, n_x, n_u)
        A_x_ineq_loc = np.stack([ np.vstack((np.eye(n_x), -np.eye(n_x))) for _ in range(N_agents)], axis=0)
        A_u_ineq_loc = np.stack([ np.vstack((np.eye(n_u), -np.eye(n_u))) for _ in range(N_agents)], axis=0)
        b_x_ineq_loc = 10*np.ones((N_agents, 2*n_x, 1))
        b_u_ineq_loc = 10*np.ones((N_agents, 2*n_u, 1))
        A_x_ineq_sh = np.zeros((N_agents, 1, n_x))
        A_u_ineq_sh = np.zeros((N_agents, 1, n_u))
        b_x_ineq_sh = np.zeros((N_agents, 1, 1))
        b_u_ineq_sh = np.zeros((N_agents, 1, 1))
        P = Q
        ##########################################
        #             Game inizialization        #
        ##########################################
        print("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        logging.info("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        dyn_game = LQ_decoupled(N_agents, A, B, Q, R, P,
                         A_x_ineq_loc, b_x_ineq_loc, A_x_ineq_sh, b_x_ineq_sh,
                         A_u_ineq_loc, b_u_ineq_loc, A_u_ineq_sh, b_u_ineq_sh,
                         T_hor)
        dyn_game.set_term_cost_to_inf_hor_sol()
        x_0 = np.random.random_sample(size=(N_agents, n_x, 1))
        x_last = np.zeros((N_agents, n_x, 1)) #stores last state of the sequence
        if test == 0:
            ##########################################
            #   Variables storage inizialization     #
            ##########################################
            x_store = np.zeros((N_random_tests, N_agents, n_x, T_sim))
            u_store = np.zeros((N_random_tests, N_agents, n_u, T_sim))
            u_pred_traj_store = np.zeros((N_random_tests, N_agents, T_hor, n_u, T_sim))
            residual_store = np.zeros((N_random_tests, (N_iter // N_it_per_residual_computation), T_sim))
            # local_constr_viol = np.zeros((N_random_tests, 1, T_sim))
            # shared_const_viol = np.zeros((N_random_tests, 1, T_sim))

        for t in range(T_sim):
            if t>=1:
                # set second-to-last state to be equal to the predicted last state of previous timestep
                dyn_game.A_eq_loc, dyn_game.b_eq_loc_from_x, dyn_game.b_eq_loc_affine = \
                    dyn_game.generate_state_equality_constr(np.stack([np.eye(n_x) for _ in range(N_agents)]), x_last, T_hor-1)
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
                    # Save performance metrics
                    u_all, d, r, c = alg.get_state()
                    residual_store[test, index_storage, t] = r
                    # print("Iteration " + str(k) + " Residual: " + str(r.item()))
                    # logging.info("Iteration " + str(k) + " Residual: " + str(r.item()))
                    index_storage = index_storage + 1
                    if r.item()<=eps:
                        break
                #  Algorithm run
                alg.run_once()
            # Convert optimization variable into state and input
            u_all, d, r, c = alg.get_state()
            u_0 = dyn_game.get_first_input_from_opt_var(u_all)
            u_store[test, :, :, t] = u_0.squeeze(2)
            residual_store[test, index_storage,t]  = r
            u_pred_traj_store[test, :, :, :, t] = dyn_game.get_predicted_input_trajectory_from_opt_var(u_all)
            # just for testing, check whether last state is as predicted
            if t >= 1:
                if norm(x_last - (dyn_game.S_single[:, -2 * n_x:-n_x, :] @ u_all + dyn_game.T_single[:, -2 * n_x:-n_x,
                                                                                   :] @ x_0)) > eps:
                    warnings.warn("The terminal constraint is not satisfied")
            # store last state
            x_last = dyn_game.S_single[:, -n_x:, :] @ u_all + dyn_game.T_single[:, -n_x:, :] @ x_0
            # Evolve state
            x_0 = A @ x_0 + B @ u_0
            x_store[test, :, :, t] = x_0.squeeze(2)


    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result_'+ str(job_id) + ".pkl", 'wb')
    pickle.dump([ x_store, u_store, residual_store, u_pred_traj_store], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")


