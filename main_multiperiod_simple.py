import numpy as np
import torch
import pickle
from FBF_algorithm_NON_aggregative import FBF_algorithm
from GameDefinition import Game
import time
import logging
import random
# import matplotlib
import control
control.use_numpy_matrix(flag=True, warn=True)
import sys

def set_stepsizes(N, A_ineq_shared, L, algorithm='FBF'):
    theta = 0
    if algorithm == 'FRB':
        delta = 4*L / (1-3*theta)
        eigval, eigvec = torch.linalg.eig(torch.bmm(A_ineq_shared, torch.transpose(A_ineq_shared, 1, 2)))
        eigval = torch.real(eigval)
        alpha = 0.5/((torch.max(torch.max(eigval, 1)[0])) + delta)
        beta =   0.5/(torch.sum(torch.max(eigval, 1)[0])/N + delta)
    if algorithm == 'FBF':
        eigval, eigvec = torch.linalg.eig(torch.sum(torch.bmm(A_ineq_shared, torch.transpose(A_ineq_shared, 1, 2)), 0)  )
        eigval = torch.real(eigval)
        alpha = 0.05/(L+torch.max(eigval))
        beta = 0.05/(L+torch.max(eigval))
    return (alpha.item(), beta.item(), theta)

if __name__ == '__main__':
    # Get random seed as system argument
    if len(sys.argv) < 2:
        seed = 0
        job_id=0
    else:
        seed=int(sys.argv[1])
        job_id = int(sys.argv[2])
    random.seed(seed)


    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    use_test_graph = True
    N_random_initial_states = 1
    N_agents= 5 # random.choice(range(2,8,2))  # N agents

    n_states = 4 # random.choice(range(2,6))

    # Generate random system
    is_controllable = False
    n_inputs = 2 # int(n_states) / 2
    attempt_controllable = 0

    A = torch.zeros(N_agents, n_states, n_states)
    B = torch.zeros(N_agents, n_states, n_inputs)
    for i in range(N_agents):
        while not is_controllable and attempt_controllable < 10:
            A_single_agent = -0.5 + 1 * np.random.random_sample(size=[n_states, n_states])
            B_single_agent = -0.5 + 1 * np.random.random_sample(size=[n_states, n_inputs])
            A[i,:,:]= A_single_agent
            B[i,:,:]= B_single_agent
            is_controllable = np.linalg.matrix_rank(control.ctrb(A_single_agent, B_single_agent)) == n_states
            attempt_controllable = attempt_controllable + 1
            if attempt_controllable % 10 == 0:
                n_inputs = n_inputs + 1
                print("The system was not controllable after " + str(attempt_controllable) + " attempts with  " + str(
                    n_states) + " states"+ str(
                    n_inputs) + " inputs")
                logging.info("The system was not controllable after " + str(attempt_controllable) + " attempts with  " + str(
                    n_states) + " states"+ str(
                    n_inputs) + " inputs")

    #stepsizes
    # alpha = 0.001
    # beta = 0.001
    # theta = 0.0

    # cost weights
    weight_x = 1 #np.random.random_sample()
    weight_u = 1 # np.random.random_sample()
    weight_terminal_cost = 0

    T_horiz_to_test= [2] # This is actually T+1, so for T=1 insert 2
    T_simulation=10
    N_iter=10**4
    # containers for saved variables
    # print("Warning! the coupled cost is set to zero!")
    x_store = {}
    x_traj_store = {}
    u_store = {}
    u_traj_store = {}
    cost_store = {}
    competition_evolution = {}
    has_converged = {}
    solver_problem = {}
    for test in range(N_random_initial_states):
        # Initial state
        initial_state_test = torch.tensor(1. + 0*np.random.random_sample(size=[N_agents, n_states]))
        print("Initializing game for test " + str(test) + " out of " + str(N_random_initial_states))
        logging.info("Initializing game for test " + str(test) + " out of " + str(N_random_initial_states))
        ### Begin tests
        for T_horiz in T_horiz_to_test:
            initial_state = initial_state_test # bring system back to initial state anytime we test another horizon
            for t in range(T_simulation):
                print("Initializing game for timestep " + str(t+1) + " out of " + str(T_simulation))
                logging.info("Initializing game for timestep " + str(t+1) + " out of " + str(T_simulation))
                game = Game(T_horiz, A, B, weight_x, weight_u, initial_state, add_terminal_cost=False,
                            add_destination_constraint=False, xi=1, problem_type="pairwise_quadratic", weight_terminal_cost=weight_terminal_cost)
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
                    x_traj_store.update({(test, T_horiz): torch.zeros(N_agents, n_states*T_horiz, T_simulation)})
                    u_store.update({(test, T_horiz): torch.zeros(N_agents, n_inputs, T_simulation)})
                    u_traj_store.update({(test, T_horiz): torch.zeros(N_agents, n_inputs * T_horiz, T_simulation)})
                    cost_store.update({(test, T_horiz): torch.zeros(N_agents, 1, T_simulation)})
                    competition_evolution.update({(test, T_horiz): torch.zeros(N_agents, 1, T_simulation)})
                    has_converged.update({(test, T_horiz): False})
                    solver_problem.update({(test,T_horiz): False})
                print("Done")
                x_store[(test, T_horiz)][:, :, t] = initial_state
                Lipschitz_const = N_agents*((weight_u + weight_x) + weight_terminal_cost)
                alpha, beta, theta = set_stepsizes(N_agents, game.A_ineq_shared, Lipschitz_const, algorithm='FBF')
                alg = FBF_algorithm(game, rho = alpha, tau = beta)
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
                    status = alg.run_once()
                    if status != 'solved':
                        solver_problem[(test,T_horiz)] = True
                        break
                    end_time = time.time()
                    avg_time_per_it = (avg_time_per_it * k + (end_time - start_time)) / (k + 1)
                    if k % 100 == 0:
                        x, d, r, c = alg.get_state()
                        print("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(
                            avg_time_per_it))
                        logging.info("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(
                            avg_time_per_it))
                        index_store = index_store + 1
                        if r <= 10 ** (-4):
                            break
                # store results
                x, d, r, c = alg.get_state()
                # NI_value = game.compute_Nikaido_Isoada(x)
                # print("NI value: " + str(NI_value.item()))
                initial_state = game.get_next_state_from_opt_var(x).squeeze(dim=2)
                input_gne = game.get_next_control_action_from_opt_var(x).squeeze(dim=2)
                u_store[(test, T_horiz)][:, :, t] = input_gne
                x_traj_store[(test, T_horiz)][:,:,t] = game.get_all_states_from_opt_var(x).squeeze(dim=2)
                u_traj_store[(test, T_horiz)][:, :, t] = game.get_all_control_actions_from_opt_var(x).squeeze(dim=2)
                cost_store[(test, T_horiz)][:, :, t]  = c.squeeze(dim=2)
                if t>0:
                    competition_evolution[(test, T_horiz)][:, :, t] = game.compute_competition_variation(x, last_gne)
                if torch.norm(initial_state).item() < 10**(-2):
                    has_converged[(test, T_horiz)] = True
                    break
                if torch.norm(initial_state).item() > 10 ** (2):
                    #Assume it has diverged
                    break
                last_gne = x


    print("Saving results...")
    logging.info("Saving results...")
    filename = "saved_test_result_multiperiod_" + str(job_id) + ".pkl"
    f = open(filename, 'wb')
    pickle.dump([x_store, u_store, N_agents, N_random_initial_states,
                 T_simulation, T_horiz_to_test, x_traj_store, u_traj_store,
                 cost_store, competition_evolution, has_converged, solver_problem], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")
