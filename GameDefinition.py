import torch
import numpy as np
from cmath import inf
from scipy import linalg
from operators import bestResponse

torch.set_default_dtype(torch.float64)

class Game:
    # N: number of agents
    # initial_state: matrix of size n_nodes x N. contains the initial probability distribution of each agent over the nodes.
    #type define type of game, can be congestion (J_i = ell(x) x) or hyperadversarial (J_1 = -J_2 = x_1 x_2)
    def __init__(self, T_horiz, A, B, weight_x , weight_u, initial_state, add_terminal_cost=False,
                add_destination_constraint=True, problem_type="congestion", weight_terminal_cost = 1, xi=2):
        self.N_agents = A.shape[0]
        self.weight_terminal_cost = weight_terminal_cost
        self.add_terminal_cost = add_terminal_cost
        self.add_destination_constraint = add_destination_constraint
        self.edge_time_to_index = {}
        self.node_time_to_index = {}
        self.T_horiz = T_horiz
        self.xi = xi
        self.n_states = A.shape[2]
        self.n_inputs = B.shape[2]
        self.weight_x = weight_x
        self.weight_u = weight_u
        index_x = 0
        self.problem_type=problem_type
        self.n_opt_variables = T_horiz * (self.n_states + self.n_inputs)
        if problem_type!="congestion" and problem_type!="hyperadversarial" and problem_type!="pairwise_quadratic":
            raise Exception("The type of the problem is not recognized")
        if problem_type=="pairwise_quadratic" and self.N_agents%2!=0:
            raise Exception("For pairwise quadratic games, the number of agents need be even")
        # Selection matrices
        self.Sx = torch.hstack((torch.eye((T_horiz-1)*self.n_states), \
                               torch.zeros((T_horiz-1)*self.n_states, self.n_states), \
                               torch.zeros((T_horiz-1)*self.n_states, T_horiz*self.n_inputs)))
        self.SxF = torch.hstack((torch.zeros(self.n_states, (T_horiz-1)*self.n_states), \
                                 torch.eye(self.n_states), \
                                 torch.zeros(self.n_states, T_horiz * self.n_inputs) ))
        self.Su = torch.hstack((torch.zeros(T_horiz * self.n_inputs, T_horiz * self.n_states), \
                                torch.eye(T_horiz * self.n_inputs)))
        # Local constraints
        self.A_ineq_loc, self.b_ineq_loc, self.A_eq_loc, self.b_eq_loc, self.index_soft_constraints = \
            self.define_local_constraints(T_horiz, A, B, initial_state, self.Sx, self.Su, self.SxF)
        self.n_local_ineq_constr = self.A_ineq_loc.size(1)
        self.n_local_eq_constr = self.A_eq_loc.size(1)
        # Shared constraints
        self.A_ineq_shared, self.b_ineq_shared = \
            self.define_shared_constraints(self.N_agents)
        self.n_shared_ineq_constr = self.A_ineq_shared.size(1)
        # Compute terminal cost
        Q_terminal_cost = torch.zeros([self.N_agents, self.n_states, self.n_states])
        if add_terminal_cost:
            for i in range(self.N_agents):
                Q_terminal_cost[i, :, :] = weight_terminal_cost * torch.from_numpy(linalg.solve_discrete_are(A[i, :, :], B[i, :, :],
                                                                     weight_x * torch.eye(self.n_states),
                                                                     weight_u * torch.eye(self.n_inputs)))
        # Define the game mapping as a torch custom activation function
        self.J = self.GameCost(self.N_agents, self.xi, weight_x, weight_u, Q_terminal_cost, self.Sx, self.Su, self.SxF, problem_type)
        self.F = self.GameMapping(self.N_agents,self.xi,  weight_x, weight_u, Q_terminal_cost, self.Sx, self.Su, self.SxF, problem_type)


    class GameCost(torch.nn.Module):
        def __init__(self, N_agents, xi, weight_x, weight_u, Q_terminal_cost, Sx, Su, SxF, problem_type):
            super().__init__()
            self.xi = xi
            self.weight_x = weight_x
            self.weight_u = weight_u
            self.Sx = Sx
            self.SxF = SxF
            self.Su = Su
            self.Q_terminal_cost = Q_terminal_cost
            self.problem_type = problem_type
            if problem_type=="hyperadversarial":
                n_states=self.Sx.shape[0]
                n_inputs= self.Su.shape[0]
                self.Q_adversarial_x=torch.zeros(N_agents,n_states,N_agents*n_states) #maps x to the multiplicative cost of each agent
                self.Q_adversarial_u = torch.zeros(N_agents, n_inputs, N_agents * n_inputs)
                for i in range(N_agents):
                    vector_interactions = torch.hstack((-torch.ones(i), torch.tensor(0), torch.ones(N_agents-i-1)))
                    self.Q_adversarial_x[i,:,:] = torch.kron(vector_interactions, torch.eye(n_states))
                    self.Q_adversarial_u[i, :, :] = torch.kron(vector_interactions, torch.eye(n_inputs))

        def local_cost(self, z):
            x = torch.matmul(self.Sx, z)  # broadcasted multiplication
            u = torch.matmul(self.Su, z)
            if self.problem_type == "pairwise_quadratic":
                cost_x = 0.5*self.weight_x*torch.unsqueeze(torch.pow(torch.linalg.vector_norm(x, dim=1), 2),2)
                cost_u = 0.5*self.weight_u*torch.unsqueeze(torch.pow(torch.linalg.vector_norm(u, dim=1), 2),2)
            return torch.add(cost_x, cost_u)

        def coupled_cost(self, z):
            x = torch.matmul(self.Sx, z)  # broadcasted multiplication
            u = torch.matmul(self.Su, z)
            if self.problem_type == "pairwise_quadratic":
                n_agents = x.size(0)
                # each pair has cost J_i = x_i^2 + x_{i+1}^2 + 2 (x_i x_{i+1}) ; J_{i+1}= x_i^2 + x_{i+1}^2 - 2 (x_i x_{i+1}) ;
                permutation_indexes_plus = np.zeros(int(n_agents / 2))
                permutation_indexes_minus = np.zeros(int(n_agents / 2))
                permutation_indexes_plus[:] = range(0, n_agents - 1, 2)  # indexes of agents with plus signs
                permutation_indexes_minus[:] = range(1, n_agents, 2)  # indexes of agents with minus signs
                cost_x = torch.zeros(n_agents, 1, 1)
                cost_u = torch.zeros(n_agents, 1, 1)
                n_agents = x.size(0)
                # each pair has cost J_i= x_i^2 + x_{i+1}^2 + 2 (x_i x_{i+1}) ; J_{i+1}= x_i^2 + x_{i+1}^2 - 2 (x_i x_{i+1}) ;
                permutation_indexes_plus = np.zeros(int(n_agents / 2))
                permutation_indexes_minus = np.zeros(int(n_agents / 2))
                permutation_indexes_plus[:] = range(0, n_agents - 1, 2)  # indexes of agents with plus signs
                permutation_indexes_minus[:] = range(1, n_agents, 2)  # indexes of agents with minus signs
                cost_x = torch.zeros(n_agents, 1, 1)
                cost_u = torch.zeros(n_agents, 1, 1)
                cost_x[permutation_indexes_plus, :, :] =  self.weight_x * ( \
                            torch.bmm(torch.transpose(x[permutation_indexes_plus, :, :], 1, 2),
                                      x[permutation_indexes_minus, :, :]))
                cost_x[permutation_indexes_minus, :, :] = - self.weight_x * ( \
                            torch.bmm(torch.transpose(x[permutation_indexes_plus, :, :], 1, 2),
                                      x[permutation_indexes_minus, :, :]))
                cost_u[permutation_indexes_plus, :, :] =  self.weight_u * ( \
                            torch.bmm(torch.transpose(u[permutation_indexes_minus, :, :], 1, 2),
                                      u[permutation_indexes_plus, :, :]))
                cost_u[permutation_indexes_minus, :, :] = -self.weight_u * ( \
                            torch.bmm(torch.transpose(u[permutation_indexes_plus, :, :], 1, 2),
                                      u[permutation_indexes_minus, :, :]))
            return torch.add(cost_x, cost_u)

        def forward(self, z):
            x = torch.matmul(self.Sx, z) # broadcasted multiplication
            u = torch.matmul(self.Su, z)
            xF = torch.matmul(self.SxF, z)
            term_cost = 0.5 * torch.matmul(torch.matmul(xF.transpose(1, 2), self.Q_terminal_cost), xF)
            if self.problem_type == "congestion":
                sigma_x = torch.sum(x, 0)
                sigma_x_xith = torch.pow(sigma_x, self.xi)
                sigma_u = torch.sum(u, 0)
                sigma_u_xith = torch.pow(sigma_u, self.xi)
                ell_x = self.weight_x * sigma_x_xith
                ell_u = self.weight_u * sigma_u_xith
                return torch.add(torch.add(torch.matmul(x.transpose(1,2), ell_x), torch.matmul(u.transpose(1,2), ell_u)), term_cost)
            if self.problem_type == "hyperadversarial":
                cost_x = self.weight_x *torch.bmm(torch.transpose(x, 1, 2), torch.matmul(self.Q_adversarial_x, torch.reshape(x,(-1,1))))
                cost_u = self.weight_u *torch.bmm(torch.transpose(u, 1, 2), torch.matmul(self.Q_adversarial_u, torch.reshape(u,(-1,1))))
                return torch.add(torch.add(cost_x, cost_u), term_cost)
            if self.problem_type == "pairwise_quadratic":
                return torch.add(torch.add(self.coupled_cost(z), self.local_cost(z)), term_cost)

    class GameMapping(torch.nn.Module):
        def __init__(self, N_agents, xi,weight_x, weight_u, Q_terminal_cost, Sx, Su, SxF, problem_type):
            super().__init__()
            self.xi = xi
            self.weight_x = weight_x
            self.weight_u = weight_u
            self.Sx = Sx
            self.SxF = SxF
            self.Su = Su
            self.Q_terminal_cost = Q_terminal_cost
            self.problem_type=problem_type
            if problem_type=="hyperadversarial":
                n_states=self.Sx.shape[0]
                n_inputs= self.Su.shape[0]
                self.Q_adversarial_x=torch.zeros(N_agents,n_states,N_agents*n_states) #maps x to the multiplicative cost of each agent
                self.Q_adversarial_u = torch.zeros(N_agents, n_inputs, N_agents * n_inputs)
                for i in range(N_agents):
                    # the previous agents are weighted with a +, the following agents are weighted with a minus.
                    # This way, the game is pairwise zero-sum for each agent.
                    vector_interactions = torch.hstack((-torch.ones(i), torch.tensor(0), torch.ones(N_agents-i-1)))
                    self.Q_adversarial_x[i,:,:] = torch.kron(vector_interactions, torch.eye(n_states))
                    self.Q_adversarial_u[i, :, :] = torch.kron(vector_interactions, torch.eye(n_inputs))

        def forward(self, z):
            x = torch.matmul(self.Sx, z)
            u = torch.matmul(self.Su, z)
            xF = torch.matmul(self.SxF, z)
            term_nabla = torch.matmul(self.Q_terminal_cost, xF)
            if self.problem_type == "congestion":
                sigma_x = torch.sum(x, 0)
                sigma_x_ximinone_th = torch.pow(sigma_x, self.xi - 1)
                sigma_x_xith = torch.pow(sigma_x, self.xi)
                ell_x = self.weight_x * sigma_x_xith
                nabla_ell_x = self.xi * self.weight_x * sigma_x_ximinone_th
                # \nabla_{x_i} J_i = ell(sigma) + (x_i) \nabla_{sigma} ell(sigma)
                x_grad = ell_x + torch.mul(x, nabla_ell_x)
                sigma_u = torch.sum(u, 0)
                sigma_u_ximinone_th = torch.pow(sigma_u, self.xi - 1)
                sigma_u_xith = torch.pow(sigma_u, self.xi)
                ell_u = self.weight_x * sigma_u_xith
                nabla_ell_u = self.xi * self.weight_x * sigma_u_ximinone_th
                u_grad = ell_u + torch.mul(u, nabla_ell_u)
            if self.problem_type == "hyperadversarial":
                x_grad = self.weight_x * torch.matmul(self.Q_adversarial_x, torch.reshape(x,(-1,1)))
                u_grad = self.weight_u * torch.matmul(self.Q_adversarial_u, torch.reshape(u,(-1,1)))
            if self.problem_type == "pairwise_quadratic":
                n_agents = x.size(0)
                # each pair has cost J_i= x_i^2 + x_{i+1}^2 + 2 (x_i x_{i+1}) ; J_{i+1}= x_i^2 + x_{i+1}^2 - 2 (x_i x_{i+1}) ;
                permutation_indexes_plus=np.zeros(int(n_agents/2))
                permutation_indexes_minus = np.zeros(int(n_agents/2))
                permutation_indexes_plus[:]=range(0,n_agents-1,2) # indexes of agents with plus signs
                permutation_indexes_minus[:]=range(1,n_agents,2) # indexes of agents with minus signs
                x_grad=torch.zeros(x.size())
                u_grad=torch.zeros(u.size())
                x_grad[permutation_indexes_plus,:,:] = self.weight_x * (x[permutation_indexes_plus,:,:] + x[permutation_indexes_minus,:,:])
                x_grad[permutation_indexes_minus, :, :] = self.weight_x * (
                            x[permutation_indexes_minus, :, :] - x[permutation_indexes_plus, :, :])
                u_grad[permutation_indexes_plus, :, :] = self.weight_u * (
                            u[permutation_indexes_plus, :, :] + u[permutation_indexes_minus, :, :])
                u_grad[permutation_indexes_minus, :, :] = self.weight_u * (
                        u[permutation_indexes_minus, :, :] - u[permutation_indexes_plus, :, :])
            return torch.add(torch.add(torch.matmul(torch.transpose(self.Sx, 0,1), x_grad), \
                             torch.matmul(torch.transpose(self.Su, 0,1), u_grad)), torch.matmul(torch.transpose(self.SxF, 0,1), term_nabla))

    def define_local_constraints(self, T_horiz, A, B, initial_state, Sx, Su, SxF):
        n_states = A.shape[2]
        N = self.N_agents
        # Num of local constr: evolution (Tn)
        n_local_const_eq = T_horiz * n_states
        # Constraints are in form Ax = b
        A_eq_loc_const = torch.zeros(N, n_local_const_eq, self.n_opt_variables)
        b_eq_loc_const = torch.zeros(N, n_local_const_eq, 1)
        # Num of local inequality constraints: none, set to 1 to avoid compatibility issues
        n_local_const_ineq = 1
        A_ineq_loc_const = torch.zeros(N, n_local_const_ineq, self.n_opt_variables)
        b_ineq_loc_const = torch.zeros(N, n_local_const_ineq, 1)
        n_soft_constraints = 0
        if n_soft_constraints ==0:
            index_soft_constraints = None
        else:
            index_soft_constraints = torch.zeros([N, n_soft_constraints])
        for i_agent in range(N):
            i_constr_eq = 0  # counter
            # Evolution constraint
            # x^+ = A x + B u stacked
            lower_diag_identity = torch.nn.ZeroPad2d((0,1,1,0))(torch.eye(T_horiz-1))
            vector_first_element = torch.zeros(T_horiz,1)
            vector_first_element[0] = 1
            A_evol_total = torch.kron(lower_diag_identity, A[i_agent,:,:]) - torch.eye(T_horiz*n_states)
            B_evol_total = torch.kron(torch.eye(T_horiz), B[i_agent,:,:])
            initial_state_vector = torch.matmul(A[i_agent,:,:], torch.unsqueeze(initial_state[i_agent,:], 1))
            A_eq_loc_const[i_agent,i_constr_eq:i_constr_eq +T_horiz * n_states,:] = torch.matmul(A_evol_total, torch.vstack((Sx, SxF))) + torch.matmul(B_evol_total, Su)
            b_eq_loc_const[i_agent,i_constr_eq:i_constr_eq +T_horiz * n_states,:] = -torch.kron(vector_first_element, initial_state_vector)
            i_constr_eq = T_horiz * n_states + 1
            ### Inequality constraints
            i_constr_ineq = 0
        return A_ineq_loc_const, b_ineq_loc_const, A_eq_loc_const, b_eq_loc_const, index_soft_constraints

    def define_shared_constraints(self, N):
        if self.problem_type=="congestion":
            # All aggreg. variables for which the cost is non-zero are positive. This is for monotonicity
            n_shared_ineq_constr = self.n_opt_variables
            A_ineq_shared = torch.zeros(N, n_shared_ineq_constr, self.n_opt_variables)
            b_ineq_shared = torch.zeros(N, n_shared_ineq_constr, 1)
            i_constr = 0
            for i_agent in range(N):
                A_ineq_shared[i_agent, i_constr:i_constr +self.n_opt_variables , :] = \
                    -torch.vstack((self.Sx * self.weight_x, self.SxF * self.weight_x, self.Su * self.weight_u))
            i_constr = i_constr + self.n_opt_variables
        if self.problem_type == "hyperadversarial":
            # no constraint
            n_shared_ineq_constr = 1
            A_ineq_shared = torch.zeros(N, n_shared_ineq_constr, self.n_opt_variables)
            b_ineq_shared = torch.zeros(N, n_shared_ineq_constr, 1)
            i_constr = 0
        if self.problem_type == "pairwise_quadratic":
            # no constraint
            n_shared_ineq_constr = 1
            A_ineq_shared = torch.zeros(N, n_shared_ineq_constr, self.n_opt_variables)
            b_ineq_shared = torch.zeros(N, n_shared_ineq_constr, 1)
            i_constr = 0
        return A_ineq_shared, b_ineq_shared

    def best_response(self, z):
        # Does not work
        br = bestResponse.bestResponse(z, self.J, self.F, self.A_ineq_loc, self.b_ineq_loc, self.A_eq_loc,
                                                      self.b_eq_loc, \
                                                      self.A_ineq_shared, self.b_ineq_shared)
        [sum_J_br, x_br]= br.compute()
        return sum_J_br, x_br

    def compute_Nikaido_Isoada(self, z):
        # Does not work
        cost_total = torch.sum(self.J(z), 0)
        cost_best_response, x_br = self.best_response(z)
        NI_value = cost_total - cost_best_response
        return NI_value

    def get_next_control_action_from_opt_var(self, z):
        u = torch.matmul(self.Su, z)
        return u[:,0:self.n_inputs,:]

    def get_next_state_from_opt_var(self, z):
        x = torch.matmul(self.Sx, z)
        return x[:,0:self.n_states,:]

    def get_all_states_from_opt_var(self, z):
        x = torch.matmul(self.Sx, z)
        xF = torch.matmul(self.SxF, z)
        return torch.cat((x,xF), 1)

    def get_all_control_actions_from_opt_var(self, z):
        u = torch.matmul(self.Su, z)
        return u

    # This function is used to compute a term on the Nikaido-Isoada-based lyapunov function relative to how the coupled costs evolve between GNEs
    def compute_competition_variation(self,new_gne,old_gne):
        #The cost should ACTUALLy be computed with shortened trajectories (excluding T+1).
        # However, this requires code refactoring - so we just set the last state and inputs to 0, with the assumption that the cost at 0 is 0, as a workaround
        x_old_from_1_to_T = torch.matmul(self.Sx, old_gne)  # This excludes initial state and xF
        # For the new GNE state, we need to exclude the last x (as that needs to be compared separately with the terminal cost of xF_old).
        # We then need to prepend the initial state (as that needs to be compared to x_1_old), which is exactly x_1_old
        x_0_new = x_old_from_1_to_T[:,0:self.n_states, :]
        x_new_from_1_to_T = torch.cat(( x_0_new, torch.matmul(self.Sx, new_gne)[:, :-self.n_states, :] ), dim=1 )
        # For the old GNE input, we need to cut the 0-input (as that needs not be compared, we only need to assume the cost to be pos.def.).
        # Then we exclude the last input (as that would be a non-existing value at T+1)
        u_old_from_1_to_T = torch.cat( (torch.matmul(self.Su, old_gne)[:,self.n_inputs:,:], torch.zeros(self.N_agents, self.n_inputs, 1)), dim=1 )
        # For the new GNE input, we need to exclude the last input (as that needs to be compared separately with the terminal cost of xF_old).
        u_new_from_1_to_T = torch.cat( (torch.matmul(self.Su, new_gne)[:, :-self.n_inputs, :], torch.zeros(self.N_agents, self.n_inputs, 1)), dim=1 )

        strategy_old_along_path = torch.matmul(torch.transpose(self.Sx,0,1), x_old_from_1_to_T) \
                                  + torch.matmul(torch.transpose(self.Su,0,1), u_old_from_1_to_T)
        strategy_new_along_path = torch.matmul(torch.transpose(self.Sx, 0, 1), x_new_from_1_to_T) \
                                  + torch.matmul(torch.transpose( self.Su, 0, 1), u_new_from_1_to_T)
        # Compute the value: \sum_i shared_cost_i(old played against new) - shared_cost_i(old played against old)
        variation = 0
        for i in range(self.N_agents):
            hybrid_strategy = torch.tensor(strategy_new_along_path)
            hybrid_strategy[i,:,:] = strategy_old_along_path[i,:,:]
            variation = variation + self.J.coupled_cost(hybrid_strategy)[i,:,:] - self.J.coupled_cost(strategy_old_along_path)[i,:,:]
        if variation > 0.1:
            print("Pause..")
        return variation

    #
    # def compute_optimal_cost_given_opponents(self):
    #
    # def compute_potential_local_costs(self):