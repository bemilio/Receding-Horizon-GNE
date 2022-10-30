import numpy as np
import torch
import pickle
from FRB_algorithm import FRB_algorithm
from GameDefinition import Game
import time
import logging
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter

torch.Tensor.ndim = property(lambda self: len(self.shape))  # Necessary to use matplotlib with tensors
matplotlib.use('TkAgg')

# def set_stepsizes(N, A_ineq_shared, algorithm='FRB'):
#     theta = 0
#     c = road_graph.edges[(0, 1)]['capacity']
#     tau = road_graph.edges[(0, 1)]['travel_time']
#     a = road_graph.edges[(0, 1)]['uncontrolled_traffic']
#     k = 0.15 * 12 * tau / c
#     L = k* (N+1) + (1 + c/tau) * (tau*road_graph.number_of_nodes())
#     if algorithm == 'FRB':
#         # L = 2*k/(4*N) * (N+a)**3 + k/N * (N+a)**2 * (N + (N/3) * (N+a) + \
#         #                      np.sqrt( ((N/3)**2) * (N+a)**2  + (2*(N**2)/3) * (N+a) + N ) )
#         delta = 2*L / (1-3*theta)
#         eigval, eigvec = torch.linalg.eig(torch.bmm(A_ineq_shared, torch.transpose(A_ineq_shared, 1, 2)))
#         eigval = torch.real(eigval)
#         alpha = 0.5/((torch.max(torch.max(eigval, 1)[0])) + delta)
#         beta = N * 0.5/(torch.sum(torch.max(eigval, 1)[0]) + delta)
#     if algorithm == 'FBF':
#         eigval, eigvec = torch.linalg.eig(torch.sum(torch.bmm(A_ineq_shared, torch.transpose(A_ineq_shared, 1, 2)), 0)  )
#         eigval = torch.real(eigval)
#         alpha = 0.5/(L+torch.max(eigval))
#         beta = 0.5/(L+torch.max(eigval))
#     return (alpha.item(), beta.item(), theta)

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    use_test_graph = True
    N_random_tests = 1
    N_agents=2   # N agents

    A_single_agent = torch.tensor([[1., 1.], [0.,1.]])
    B_single_agent = torch.tensor([[0.], [1.]])
    A = torch.broadcast_to(A_single_agent, (N_agents,2,2)) # Double integrators
    B = torch.broadcast_to(B_single_agent, (N_agents,2,1))
    n_states = A.shape[2]
    n_inputs = B.shape[2]

    #stepsizes
    alpha = 0.1
    beta = 0.1
    theta = 0.0

    # cost weights
    weight_x = 1
    weight_u = 0
    weight_terminal_cost = 0

    T_horiz_to_test= [10]
    T_simulation=20
    np.random.seed(1)
    N_iter=10000
    # containers for saved variables

    x_store = {}
    u_store = {}
    for test in range(N_random_tests):
        # Initial state
        initial_state = torch.tensor(-0.5 + 2*np.random.random_sample(size=[N_agents, n_states]))
        print("Initializing game for test " + str(test) + " out of " + str(N_random_tests))
        logging.info("Initializing game for test " + str(test) + " out of " + str(N_random_tests))
        ### Begin tests
        for T_horiz in T_horiz_to_test:
            for t in range(T_simulation):
                print("Initializing game for timestep " + str(t+1) + " out of " + str(T_simulation))
                logging.info("Initializing game for timestep " + str(t+1) + " out of " + str(T_simulation))
                game = Game(T_horiz, A, B, weight_x, weight_u, initial_state, add_terminal_cost=True,
                            add_destination_constraint=False, xi=1, problem_type="hyperadversarial", weight_terminal_cost=weight_terminal_cost)
                if t==0:
                    print("The game has " + str(N_agents) + " agents; " + str(
                        game.n_opt_variables) + " opt. variables per agent; " \
                          + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(
                        game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(
                        game.n_shared_ineq_constr) + " shared ineq. constraints")
                    logging.info("The game has " + str(N_agents) + " agents; " + str(
                        game.n_opt_variables) + " opt. variables per agent; " \
                                 + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(
                        game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(
                        game.n_shared_ineq_constr) + " shared ineq. constraints")
                    # Initialize storing
                    x_store.update({(test, T_horiz) : torch.zeros(N_agents, n_states, T_simulation)})
                    u_store.update({(test, T_horiz): torch.zeros(N_agents, n_inputs, T_simulation)})
                print("Done")
                x_store[(test, T_horiz)][:, :, t] = initial_state
                alg = FRB_algorithm(game, beta=beta, alpha=alpha, theta=theta)
                status = alg.check_feasibility()
                ### Check feasibility
                is_problem_feasible = (status == 'solved')
                if not is_problem_feasible:
                    print("the problem is not feasible")
                index_store = 0
                avg_time_per_it = 0
                ### Main iterations
                for k in range(N_iter):
                    start_time = time.time()
                    alg.run_once()
                    end_time = time.time()
                    avg_time_per_it = (avg_time_per_it * k + (end_time - start_time)) / (k + 1)
                    if k % 100 == 0:
                        x, d, r, c = alg.get_state()
                        print("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(
                            avg_time_per_it))
                        logging.info("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(
                            avg_time_per_it))
                        index_store = index_store + 1
                        if r <= 10 ** (-3):
                            break
                # store results
                x, d, r, c = alg.get_state()
                # NI_value = game.compute_Nikaido_Isoada(x)
                # print("NI value: " + str(NI_value.item()))
                initial_state = game.get_next_state_from_opt_var(x).squeeze(dim=2)
                input_gne = game.get_control_action_from_opt_var(x).squeeze(dim=2)
                u_store[(test, T_horiz)][:, :, t] = input_gne

    # traj_x1 = np.zeros((N_agents, T_simulation))
    # for i in range(N_agents):
    #     traj_x1[i, :] = x_store[(0,T_horiz_to_test[0])][i, 0, :]
    # traj_x2 = np.zeros((N_agents, T_simulation))
    # for i in range(N_agents):
    #     traj_x2[i, :] = x_store[(0, T_horiz_to_test[0])][i, 1, :]
    # plt.plot(traj_x1 )
    # plt.show(block=False)
    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result_multiperiod.pkl', 'wb')
    pickle.dump([ x_store, u_store, N_agents, T_simulation, T_horiz_to_test ], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")
