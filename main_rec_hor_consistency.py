import warnings
import numpy as np
np.set_printoptions(precision=3)
# import torch
import pickle
import games.dyngames as dyngames
from algorithms.GNE_centralized import pFB_algorithm
import logging
import sys


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
    N_agents_to_test = [3]
    N_random_tests = 1

    # parameters
    N_iter = 10**7
    n_x = 3
    n_u = 2
    T_hor_to_test = [9]
    T_sim = 10
    eps = 10**(-5) # convergence threshold

    ##########################################
    #   Variables storage inizialization     #
    ##########################################
    x_store = [ [ np.zeros((N_random_tests, n_x, T_sim), dtype=np.float32) for N_agents in N_agents_to_test ] for _ in T_hor_to_test ]
    u_store = [ [ np.zeros((N_random_tests, N_agents, n_u, T_sim), dtype=np.float32) for N_agents in N_agents_to_test ] for _ in T_hor_to_test ]
    u_PMP_CL_store = [[np.zeros((N_random_tests, N_agents, n_u, T_sim)) for N_agents in N_agents_to_test] for _ in T_hor_to_test]
    u_PMP_OL_store = [[np.zeros((N_random_tests, N_agents, n_u, T_sim)) for N_agents in N_agents_to_test] for _ in T_hor_to_test]
    u_pred_traj_store = [ [ np.zeros((N_random_tests, N_agents, T_hor * n_u, T_sim), dtype=np.float32) for N_agents in N_agents_to_test ] for T_hor in T_hor_to_test ]
    x_pred_traj_store = [ [ np.zeros((N_random_tests, T_hor * n_x, T_sim), dtype=np.float32) for N_agents in N_agents_to_test ] for T_hor in T_hor_to_test ]
    u_shifted_traj_store = [ [ np.zeros((N_random_tests, N_agents, T_hor * n_u, T_sim), dtype=np.float32) for N_agents in N_agents_to_test ] for T_hor in T_hor_to_test ]
    residual_store = [ [np.zeros((N_random_tests, (N_iter // N_it_per_residual_computation), T_sim), dtype=np.float32) for _ in N_agents_to_test ] for _ in T_hor_to_test ]
    cost_store = [[np.zeros((N_random_tests, N_agents, T_sim), dtype=np.float32) for N_agents in N_agents_to_test] for _ in T_hor_to_test]
    K_CL_store = [ [ np.zeros((N_random_tests, N_agents, n_u, n_x)) for N_agents in N_agents_to_test ] for _ in T_hor_to_test ]
    K_OL_store = [ [ np.zeros((N_random_tests, N_agents, n_u, n_x), dtype=np.float32) for N_agents in N_agents_to_test ] for _ in T_hor_to_test ]
    P_OL_store = [[np.zeros((N_random_tests, N_agents, n_x, n_x), dtype=np.float32) for N_agents in N_agents_to_test] for _ in T_hor_to_test]
    A_store = [ [ np.zeros((N_random_tests, n_x, n_x)) for N_agents in N_agents_to_test ] for _ in T_hor_to_test ]
    B_store = [ [ np.zeros((N_random_tests, N_agents, n_x, n_u)) for N_agents in N_agents_to_test ] for _ in T_hor_to_test ]
    is_P_symmetric_store = [[ [False for _ in range(N_random_tests)]  for _ in N_agents_to_test] for _ in T_hor_to_test]
    is_P_posdef_store = [[ [False for _ in range(N_random_tests)]  for _ in N_agents_to_test] for _ in T_hor_to_test]
    is_K_stable_store = [[ [False for _ in range(N_random_tests)]  for _ in N_agents_to_test] for _ in T_hor_to_test]
    is_ONE_and_affine_cost_to_go_equal_store = [[ [False for _ in range(N_random_tests)]  for _ in N_agents_to_test] for _ in T_hor_to_test]
    # is_fin_hor_input_equal_to_inf_hor_store = [[ [False for _ in range(N_random_tests)]  for _ in N_agents_to_test] for _ in T_hor_to_test]
    is_P_subspace_store = [[ [False for _ in range(N_random_tests)]  for _ in N_agents_to_test] for _ in T_hor_to_test]
    is_subspace_stable_store = [[ [False for _ in range(N_random_tests)]  for _ in N_agents_to_test] for _ in T_hor_to_test]
    is_subspace_unique_store = [[ [False for _ in range(N_random_tests)]  for _ in N_agents_to_test] for _ in T_hor_to_test]
    is_game_solved_store = [[ [ False for _ in range(N_random_tests)] for _ in N_agents_to_test] for _ in T_hor_to_test]
    is_ONE_solved_store = [[ [ False for _ in range(N_random_tests)] for _ in N_agents_to_test] for _ in T_hor_to_test]
    is_ONE_equal_to_affine_LQR_store = [[ [ False for _ in range(N_random_tests)] for _ in N_agents_to_test] for _ in T_hor_to_test]
    test_counter = 0

    for test in range(N_random_tests):
        for N_agents in N_agents_to_test:
            ##########################################
            #        Test case creation              #
            #########################################
            A, B, Q, R = dyngames.LQ.generate_random_game(N_agents, n_x, n_u)
            C_x = np.vstack((np.eye(n_x), -np.eye(n_x)))
            C_u_loc = np.stack([np.vstack((np.eye(n_u), -np.eye(n_u))) for _ in range(N_agents)], axis=0)
            d_x = 100 * np.ones((2 * n_x, 1))
            d_u_loc = 100 * np.ones((N_agents, 2 * n_u, 1))
            C_u_sh = np.zeros((N_agents, 1, n_u))
            d_u_sh = np.zeros((N_agents, 1, 1))
            P = Q
            for T_hor in T_hor_to_test:
                test_counter = test_counter + 1
                ##########################################
                #             Game inizialization        #
                ##########################################
                # print("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
                # logging.info("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
                dyn_game = dyngames.LQ(A, B, Q, R, P, C_x, d_x, C_u_loc, d_u_loc, C_u_sh, d_u_sh, T_hor)
                dyn_game.set_term_cost_to_inf_hor_sol(mode="OL")
                # dyn_game.set_term_cost_to_inf_hor_sol(mode="CL", method='lyap')
                x_0 = np.ones((n_x, 1))
                x_last = np.zeros((n_x, 1)) #stores last state of the sequence
                # Store system dynamics
                A_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :] = A
                B_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :] = B
                ########################################
                ####   Compute inf-horizon solution ####
                ########################################
                P_CL, K_CL = dyn_game.solve_closed_loop_inf_hor_problem()
                P_OL, K_OL, is_OL_solved, is_OL_stable, is_P_symmetric, is_P_posdef = dyn_game.solve_open_loop_inf_hor_problem()
                is_ONE_solved_store[T_hor_to_test.index((T_hor))][N_agents_to_test.index(N_agents)][test] = is_OL_solved
                is_P_symmetric_store[T_hor_to_test.index((T_hor))][N_agents_to_test.index(N_agents)][test] = is_P_symmetric
                is_P_posdef_store[T_hor_to_test.index((T_hor))][N_agents_to_test.index(N_agents)][test] = is_P_posdef
                is_K_stable_store[T_hor_to_test.index((T_hor))][N_agents_to_test.index(N_agents)][test] = is_OL_stable
                # Verify assumption 4.9 [Monti '23]
                is_P_subspace, is_subspace_stable, is_subspace_unique = dyn_game.check_P_is_H_graph_invariant_subspace(P_OL)
                is_P_subspace_store[T_hor_to_test.index((T_hor))][N_agents_to_test.index(N_agents)][test] = is_P_subspace
                is_subspace_stable_store[T_hor_to_test.index((T_hor))][N_agents_to_test.index(N_agents)][test] = is_subspace_stable
                is_subspace_unique_store[T_hor_to_test.index((T_hor))][N_agents_to_test.index(N_agents)][test] = is_subspace_unique
                # Store inf-horizon solution
                K_CL_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)] [test, :] = K_CL
                K_OL_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :] = K_OL
                P_OL_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :] = P_OL

                is_ONE_equal_to_affine_LQR, is_ONE_and_affine_cost_to_go_equal = dyn_game.verify_ONE_is_affine_LQR()
                is_ONE_equal_to_affine_LQR_store[T_hor_to_test.index((T_hor))][N_agents_to_test.index(N_agents)][test] = is_ONE_equal_to_affine_LQR
                is_ONE_and_affine_cost_to_go_equal_store[T_hor_to_test.index((T_hor))][N_agents_to_test.index(N_agents)][test] = is_ONE_and_affine_cost_to_go_equal_store

                K_closed_form = dyn_game.solve_closed_form_MPC(T_hor, P_OL)

                for t in range(T_sim):
                    print("Test " + str(test_counter) \
                          + " of " + str(N_random_tests * len(T_hor_to_test) * len(N_agents_to_test)) \
                          + "; Timestep " + str(t) + " of " + str(T_sim))
                    logging.info("Test " + str(test_counter) \
                          + " of " + str(N_random_tests * len(T_hor_to_test) * len(N_agents_to_test)) \
                          + "; Timestep " + str(t) + " of " + str(T_sim))
                    # if t>=1:
                        # set second-to-last state to be equal to the predicted last state of previous timestep
                        # dyn_game.A_eq_loc, dyn_game.b_eq_loc_from_x, dyn_game.b_eq_loc_affine = \
                        #     dyn_game.generate_state_equality_constr(np.eye(n_x), x_last, T_hor-1)
                    game_t = dyn_game.generate_game_from_initial_state(x_0)
                    #######################################
                    #          GNE seeking                #
                    #######################################
                    # alg. initialization
                    if t==0:
                        alg = pFB_algorithm(game_t, primal_stepsize=0.0001, dual_stepsize=0.0001, x_0=None, dual_0=None)
                    else:
                        # warm start to shifted sequence
                        u_shifted = np.expand_dims(u_shifted_traj_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :, :, t-1], axis=2)
                        alg = pFB_algorithm(game_t, primal_stepsize=0.001, dual_stepsize=0.001, x_0=u_shifted, dual_0=d)
                        _, _, r, _  = alg.get_state()
                        print("Res. of shifted seq: " + str(r))
                        logging.info("Res. of shifted seq: " + str(r))
                    # alg.set_stepsize_using_Lip_const(safety_margin=.9)
                    index_storage = 0
                    avg_time_per_it = 0
                    if is_OL_solved:
                        for k in range(N_iter):
                            if k % N_it_per_residual_computation == 0:
                                # Save performance metrics
                                u_all, d, r, c = alg.get_state()
                                residual_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, index_storage, t]= r
                                # print("Timestep " + str(t) + "; Iteration " + str(k) + "; Residual: " + str(r.item()))
                                # logging.info("Iteration " + str(k) + " Residual: " + str(r.item()))
                                index_storage = index_storage + 1
                                if r.item()<=eps:
                                    break
                            #  Algorithm run
                            alg.run_once()
                        if r.item()>eps:
                            # If problem is not solved, mark it as unsolved and proceed to next test.
                            warnings.warn("Nash equilibrium not found")
                            logging.warn("Nash equilibrium not found")
                            is_game_solved_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test] = False
                            break
                        else:
                            is_game_solved_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test] = True
                    else:
                        is_game_solved_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test] = False
                        break
                    # Convert optimization variable into state and input
                    u_all, d, r, c = alg.get_state()
                    u_0 = dyn_game.get_input_timestep_from_opt_var(u_all, 0)
                    u_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :, :, t] = u_0.squeeze(2).astype(np.float32)
                    residual_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, index_storage-1,t]  = r.astype(np.float32)
                    cost_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :, t] = c.squeeze().astype(np.float32)
                    u_pred_traj_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :, :, t] = u_all.squeeze(2).astype(np.float32)
                    x_pred_traj_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :, t] = (dyn_game.T @ x_0 + np.sum(dyn_game.S @ u_all, axis=0)).squeeze().astype(np.float32)
                    u_shifted_traj_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :, :, t] = dyn_game.get_shifted_trajectory_from_opt_var(u_all, x_0).squeeze(2).astype(np.float32)

                    # Compute solution with PMP
                    u_PMP_CL, x_PMP_CL, _ = dyn_game.solve_CL_with_PMP(P_CL,
                                                np.linalg.matrix_power(A + np.sum(B @ K_CL, axis=0), T_hor) @ x_0, K_CL)
                    u_PMP_OL, x_PMP_OL, _ = dyn_game.solve_OL_with_PMP(P_OL,
                                                np.linalg.matrix_power(A + np.sum(B @ K_OL, axis=0), T_hor) @ x_0)
                    u_PMP_CL_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :, :, t] = u_PMP_CL[:,0,:].squeeze()
                    u_PMP_OL_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :, :, t] = u_PMP_OL[:,0,:].squeeze()

                    # store last state
                    x_last[:] = dyn_game.T[-n_x:, :] @ x_0 + np.sum(dyn_game.S[:, -n_x:, :] @ u_all, axis=0)

                    # Evolve state
                    x_0 = A @ x_0 + np.sum(B @ u_0, axis=0)
                    x_store[T_hor_to_test.index(T_hor)][N_agents_to_test.index(N_agents)][test, :, t] = x_0.squeeze(1).astype(np.float32)


    print("Saving results...")
    logging.info("Saving results...")
    f = open('rec_hor_consistency_result_'+ str(job_id) + ".pkl", 'wb')
    # pickle.dump([ x_store, u_store, residual_store, u_pred_traj_store, x_pred_traj_store, u_shifted_traj_store,\
    #               K_CL_store, K_OL_store, A_store, B_store,
    #               T_hor_to_test, N_agents_to_test,
    #               cost_store, u_PMP_CL_store, u_PMP_OL_store, P_OL_store,
    #               is_game_solved_store, is_ONE_solved_store, is_P_subspace_store,
    #               is_subspace_stable_store, is_subspace_unique_store], f)
    pickle.dump([ x_store, u_store, residual_store, u_pred_traj_store, x_pred_traj_store, u_shifted_traj_store,\
                  K_OL_store,
                  T_hor_to_test, N_agents_to_test,
                  cost_store, P_OL_store,
                  is_game_solved_store, is_ONE_solved_store, is_P_subspace_store,
                  is_subspace_stable_store, is_subspace_unique_store, is_ONE_equal_to_affine_LQR_store], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")



