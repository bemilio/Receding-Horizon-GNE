import numpy as np
import networkx as nx
# import torch
import pickle
from games.dyngames import LQ_decoupled
import matplotlib.pyplot as plt
import time
import logging
import sys
import copy
import math

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    use_test_game = False  # trigger 2-players sample zero-sum monotone game
    if use_test_game:
        print("WARNING: test game will be used.")
        logging.info("WARNING: test game will be used.")
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
    N_agents = 6
    N_random_tests = 1

    # parameters
    N_iter = 100000

    for test in range(N_random_tests):
        ##########################################
        #        Test case creation              #
        ##########################################
        game_params = LQ_decoupled(N_agents, A, B, Q, R, P,
                         A_x_ineq_loc, b_x_ineq_loc, A_x_ineq_sh, b_x_ineq_sh,
                         A_u_ineq_loc, b_u_ineq_loc, A_u_ineq_sh, b_u_ineq_sh,
                         T_hor)
        print("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        logging.info("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        ##########################################
        #             Game inizialization        #
        ##########################################
        game = AggregativePartialInfo(N_agents, comm_graph, game_params.Q, game_params.q, game_params.C, game_params.D,\
                                      game_params.A_eq_local_const, game_params.b_eq_local_const, \
                                      game_params.A_eq_shared_const, game_params.b_eq_shared_const, game_params.A_sel_positive_vars)
        x_0 = torch.zeros(game.N_agents, game.n_opt_variables) + \
            torch.bmm(game_params.A_sel_positive_vars, torch.ones(game.N_agents, game.n_opt_variables, 1)).flatten(1)
        if test == 0:
            print("The game has " + str(game.N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + " local eq. constraints; " + str(game.n_shared_eq_constr) + " shared eq. constraints" )
            logging.info("The game has " + str(game.N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + str(game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(game.n_shared_eq_constr) + " shared eq. constraints" )
            ##########################################
            #   Variables storage inizialization     #
            ##########################################
            # pFB-Tichonov
            x_store = torch.zeros(N_random_tests, game.N_agents, game.n_opt_variables)
            dual_share_store = torch.zeros(N_random_tests, game.N_agents, game.n_shared_eq_constr)
            dual_loc_store = torch.zeros(N_random_tests, game.N_agents, game.n_loc_eq_constr)
            aux_store = torch.zeros(N_random_tests, game.N_agents, game.n_shared_eq_constr)
            res_est_store = torch.zeros(N_random_tests, game.N_agents, game.n_shared_eq_constr)
            sigma_est_store = torch.zeros(N_random_tests, game.N_agents, game.n_agg_variables)
            residual_store = torch.zeros(N_random_tests, (N_iter // N_it_per_residual_computation))
            local_constr_viol = torch.zeros(N_random_tests, 1)
            shared_const_viol = torch.zeros(N_random_tests, 1)

        #######################################
        #          GNE seeking                #
        #######################################
        # alg. initialization
        alg = primal_dual(game)
        # The theoretically-sound stepsize is too small!
        # alg.set_stepsize_using_Lip_const(safety_margin=.9)
        index_storage = 0
        avg_time_per_it = 0
        for k in range(N_iter):
            if k % N_it_per_residual_computation == 0:
                # Save performance metrics
                x, d, d_l, aux, agg, res_est, r, c, const_viol_sh, const_viol_loc, dist_ref  = alg.get_state()
                residual_store[test, index_storage] = r
                print("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(avg_time_per_it))
                logging.info("Iteration " + str(k) + " Residual: " + str(r.item()) +" Average time: " + str(avg_time_per_it))
                index_storage = index_storage + 1
            #  Algorithm run
            start_time = time.time()
            alg.run_once()
            end_time = time.time()
            avg_time_per_it = (avg_time_per_it * k + (end_time - start_time)) / (k + 1)

        # Store final variables
        x, d, d_l, aux, agg, res_est, r, c, const_viol_sh, const_viol_loc, dist_ref = alg.get_state()
        x_store[test, :, :] = x.flatten(1)
        dual_share_store[test, :, :] = d.flatten(1)
        dual_loc_store[test,:,:] = d_l.flatten(1)
        aux_store[test, :, :] = aux.flatten(1)
        sigma_est_store[test,:,:] = agg.flatten(1)
        res_est_store[test,:,:] = res_est.flatten(1)
        local_constr_viol[test] = const_viol_loc
        shared_const_viol[test] = const_viol_sh

    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result_'+ str(job_id) + ".pkl", 'wb')
    pickle.dump([ x_store, residual_store, dual_share_store, dual_loc_store,
                  local_constr_viol, shared_const_viol,
                  loc_const_viol_tvar, shared_const_viol_tvar,
                  distance_from_optimal_tvar, game_params.edge_to_index, N_iter_per_timestep ], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")


