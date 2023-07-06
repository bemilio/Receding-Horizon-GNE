import numpy as np
import scipy.linalg
from scipy.linalg import block_diag
from . import staticgames
from .staticgames import batch_mult_with_coulumn_stack
import control
from copy import copy

def get_multiagent_prediction_model(A,B,T_hor):
    # Generate prediction model with the convention x = col_i(col_t(x_t^i))
    # A[i,:,:] is the evolution matrix of agent i (similar notation for B)
    if len(A.shape)==2: #if there are not enough dimensions for matrix A, treat as a single-agent case
        A = copy(A)
        B = copy(B)
        A = np.expand_dims(A, axis=0)
        B = np.expand_dims(B, axis=0)
    N = A.shape[0]
    n_x = A.shape[2]
    n_u = B.shape[2]
    T_i = np.zeros([N, T_hor * n_x, n_x])
    S_i = np.zeros([N, T_hor * n_x, T_hor * n_u])

    A_pow = np.stack([np.eye(n_x) for _ in range(N)], axis=0)
    for t in range(T_hor):
        T_i[:, t * n_x:(t + 1)*n_x, :] = A_pow @ A
        if t >0:
            S_i[:, t*n_x:(t + 1)*n_x, : ] = A @ S_i[:, (t-1)*n_x:t*n_x, : ]
        S_i[:, t * n_x:(t + 1) * n_x, t * n_u: (t+1)*n_u] = B
        A_pow = A_pow @ A

    T = block_diag(*[T_i[i,:,:] for i in range(N)])
    S = block_diag(*[S_i[i,:,:] for i in range(N)])
    return S,T

def generate_random_monotone_matrix(N, n_x):
    # Produce a random square block matrix Q of dimension N*n_x where each block is n_x*n_x. The blocks on the diagonal are
    # symmetric positive definite and Q is positive definite, not symmetric.
    Q = np.random.random_sample(size=[N*n_x, N*n_x])
    for i in range(N):
        Q_i = Q[i*n_x:(i+1)*n_x, i*n_x:(i+1)*n_x]
        Q_i = (Q_i  + Q_i.T)/2
        min_eig = min(np.linalg.eigvalsh((Q_i) / 2))
        if min_eig.item() < 0:
            Q[i*n_x:(i+1)*n_x, i*n_x:(i+1)*n_x] = Q_i - min_eig * np.eye(n_x)
    min_eig = min(np.linalg.eigvalsh((Q+Q.T)/2))
    if min_eig.item() < 0:
        Q = Q - min_eig * np.eye(N*n_x)
    eps = 10 ** (-10)
    # Check just to be sure
    if min(np.linalg.eigvalsh((Q+Q.T)/2)) < -1 * eps:
        raise Exception("The game is not monotone")
    return Q

def expand_block_square_matrix_to_horizon(Q, N_blocks, n_x, T_hor, P=None):
    # It performs a "block" kronecker product with the identity for a square matrix of square matrices
    # e.g. for T_hor = 2 and Q_i = [ Q_11 Q_12
    #                                Q_21 Q_22 ]
    # it outputs [ Q_11  0     Q_12 0
    #              0     Q_11  0    Q_12
    #              Q_21  0     Q_22 0
    #              0     Q_21  0    Q_21]
    # it is used with the convention that the optimization variable is x = col_i(col_t(x_t^i))
    # If P is given, its block P_ij is put instead of the last repetition of Q_ij, in the previous example:
    # it outputs [ Q_11  0     Q_12 0
    #              0     P_11  0    P_12
    #              Q_21  0     Q_22 0
    #              0     P_21  0    P_21]
    if P is None:
        return np.vstack([np.hstack(
            [np.kron(np.eye(T_hor), Q[j * n_x:(j + 1) * n_x, k * n_x:(k + 1) * n_x]) for k in range(N_blocks)]) for j in
                   range(N_blocks)])
    else:
        pattern_Q = block_diag(np.eye(T_hor-1), 0)
        pattern_P = block_diag(np.zeros((T_hor - 1, T_hor-1)), 1)
        Q_expanded = np.vstack([np.hstack(
            [ (np.kron(pattern_Q, Q[j * n_x:(j + 1) * n_x, k * n_x:(k + 1) * n_x]) + \
             np.kron(pattern_P, P[j * n_x:(j + 1) * n_x, k * n_x:(k + 1) * n_x]) )
             for k in range(N_blocks)]) for j in range(N_blocks)])
        return Q_expanded

class LQ_decoupled:
    def __init__(self, N_agents, A, B, Q, R, P,
                 A_x_ineq_loc, b_x_ineq_loc, A_x_ineq_sh, b_x_ineq_sh,
                 A_u_ineq_loc, b_u_ineq_loc, A_u_ineq_sh, b_u_ineq_sh,
                 T_hor, test=False):
        if test == True:
            N_agents, A, B, Q, R, P, \
            A_x_ineq_loc, b_x_ineq_loc, A_x_ineq_sh, b_x_ineq_sh,\
            A_u_ineq_loc, b_u_ineq_loc, A_u_ineq_sh, b_u_ineq_sh, \
            T_hor = self.setToTestGameSetup()
        self.N_agents = N_agents
        self.n_x =  A.shape[2]
        self.n_u =  B.shape[2]
        self.T_hor = T_hor
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        n_opt_var = self.n_u*T_hor
        # J_i = \|u\|^2_Q_u^i + u' Q_u_x0^i x_0
        self.Q_u,self.Q_u_x0 = self.define_cost_functions(A, B, Q, R, P, T_hor)
        # Unconstrained formulation
        self.A_ineq_local_const, self.b_ineq_loc_from_x, self.b_ineq_loc_affine = \
            self.generate_state_input_constr(A, B, A_x_ineq_loc, b_x_ineq_loc, A_u_ineq_loc, b_u_ineq_loc, T_hor)
        self.A_ineq_shared_const, self.b_ineq_shared_from_x, self.b_ineq_shared_affine = \
            self.generate_state_input_constr(A, B, A_x_ineq_sh, b_x_ineq_sh, A_u_ineq_sh, b_u_ineq_sh, T_hor)
        self.A_eq_loc = np.zeros((N_agents, 1, n_opt_var))
        self.b_eq_loc = np.zeros((N_agents, 1, 1))

    def setToTestGameSetup(self):
        # two decoupled double integrators with no cost coupling (MPC) and no constraints
        N_agents = 2
        n_x = 2
        n_u = 1
        # A = np.stack([np.array([[1]]) for _ in range(N_agents)])
        # B = np.stack([np.array([[1]]) for _ in range(N_agents)])
        A = np.stack([np.array([[.9, 1], [0, 1]]) for _ in range(N_agents)])
        B = np.stack([np.array([[0], [1]]) for _ in range(N_agents)])
        pattern = np.zeros((N_agents, N_agents, N_agents))
        for i in range(N_agents):
            pattern[i,i,i] = 1

        Q = np.kron(pattern, np.eye(n_x))
        R = np.kron(pattern, np.eye(n_u))
        # P_i is a block matrix where the only non-zero element is on the diagonal and it is the solutions of the local DARE
        P_ii = np.stack([ scipy.linalg.solve_discrete_are(A[i], B[i], Q[i][i*n_x:(i+1)*n_x, i*n_x:(i+1)*n_x], R[i][i*n_u:(i+1)*n_u, i*n_u:(i+1)*n_u]) for i in range(N_agents) ] )
        P = np.zeros((N_agents, N_agents*n_x, N_agents*n_x))
        for i in range(N_agents):
            P[i, i*n_x:(i+1)*n_x, i*n_x:(i+1)*n_x] = P_ii[i]

        A_x_ineq_loc = np.zeros((N_agents, 1, n_x))
        A_u_ineq_loc = np.zeros((N_agents, 1, n_u))
        b_x_ineq_loc = np.zeros((N_agents, 1, 1))
        b_u_ineq_loc = np.zeros((N_agents, 1, 1))
        A_x_ineq_sh = np.zeros((N_agents, 1, n_x))
        A_u_ineq_sh = np.zeros((N_agents, 1, n_u))
        b_x_ineq_sh = np.zeros((N_agents, 1, 1))
        b_u_ineq_sh = np.zeros((N_agents, 1, 1))

        T_hor = 1

        return N_agents, A, B, Q, R, P, \
        A_x_ineq_loc, b_x_ineq_loc, A_x_ineq_sh, b_x_ineq_sh, \
        A_u_ineq_loc, b_u_ineq_loc, A_u_ineq_sh, b_u_ineq_sh, \
        T_hor


    def generate_game_from_initial_state(self, x_0):
        self.Q = self.Q_u
        # Obtain linear part of the cost from x_0. Note that the mapping x_0->cost wants x_0 = col_i(x_0^i),
        # while the mapping to the affine part constraints is agent-wise, that is, it only requires x_0^i
        self.q = batch_mult_with_coulumn_stack(self.Q_u_x0, x_0)
        self.b_ineq_local_const = self.b_ineq_loc_from_x @ x_0 + self.b_ineq_loc_affine
        self.b_ineq_local_shared = self.b_ineq_shared_from_x @ x_0 + self.b_ineq_shared_affine
        return staticgames.LinearQuadratic(self.Q, self.q, self.A_ineq_local_const, self.b_ineq_local_const, \
                                           self.A_eq_loc, self.b_eq_loc, \
                                           self.A_ineq_shared_const, self.b_ineq_local_shared)

    def define_cost_functions(self, A, B, Q, R, P, T_hor):
        # J_i = .5 \|u\|^2_{T'Q^i_all_t T} + u'S'Q^i_all_t Tx_0
        # where Q^i_all_t is block diagonal with kron(I_T, Q_i) and P_i
        # Returns Q_u[i]=T'Q^i_all_t T, Q_u_x0[i]=S'Q^i_all_t T.
        N_agents = self.N_agents
        n_x = A.shape[2]
        n_u = B.shape[2]
        Q_u = np.zeros((N_agents, n_u * N_agents * T_hor, n_u * N_agents * T_hor))
        Q_u_x0 = np.zeros((N_agents, n_u* N_agents *T_hor, n_x* N_agents))
        S, T = get_multiagent_prediction_model(A,B, T_hor)
        for i in range(N_agents):
            Q_all_t = expand_block_square_matrix_to_horizon(Q[i], N_agents, n_x, T_hor, P=P[i])
            R_all_t = expand_block_square_matrix_to_horizon(R[i], N_agents, n_u, T_hor)
            Q_u[i,:,:] = S.T @ Q_all_t @ S + R_all_t
            Q_u_x0[i,:,:] = S.T @ Q_all_t @ T
        return Q_u, Q_u_x0

    def generate_state_input_constr(self, A, B, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq, T_hor):
        N = self.N_agents
        n_state_const = A_x_ineq[0].shape[0]
        n_input_const = A_u_ineq[0].shape[0]
        n_u = B.shape[2]
        n_x = A.shape[2]
        A_ineq_all_timesteps = np.zeros((self.N_agents, T_hor * (n_state_const + n_input_const), T_hor * n_u))
        b_ineq_all_timesteps_from_x = np.zeros((self.N_agents, T_hor * (n_state_const + n_input_const), n_x))  # maps x_0 to b_ineq
        b_ineq_all_timesteps = np.zeros((self.N_agents, T_hor * (n_state_const + n_input_const), 1))
        for i in range(self.N_agents):
            # state and input constraints are:
            # kron(I_T, A_x)(S_i u_i +  T_i x0_i) <= kron(1_T, b_x)
            # kron(I_T, A_u)u_i <= kron(1_T, b_u)
            S_i, T_i = get_multiagent_prediction_model(A[i], B[i], T_hor)
            A_ineq_all_timesteps[i, 0:T_hor * n_state_const, :] = np.kron(np.eye(T_hor), A_x_ineq[i])@S_i
            A_ineq_all_timesteps[i, T_hor * n_state_const :, :] = np.kron(np.eye(T_hor), A_u_ineq[i])
            b_ineq_all_timesteps_from_x[i, 0:T_hor * n_state_const, :] = np.kron(np.eye(T_hor), A_x_ineq[i])@T_i
            b_ineq_all_timesteps[i, :, :] = np.row_stack((np.kron(np.ones((T_hor,1)), b_x_ineq[i]), np.kron(np.ones((T_hor,1)), b_u_ineq[i])))
        return A_ineq_all_timesteps, b_ineq_all_timesteps_from_x, b_ineq_all_timesteps

    def get_predicted_input_trajectory_from_opt_var(self, u_all):
        u = u_all.reshape((self.N_agents, self.T_hor, self.n_u))
        return u

    def get_first_input_from_opt_var(self, u_all):
        u = self.get_predicted_input_trajectory_from_opt_var(u_all)
        u_0 = np.expand_dims(u[:,0,:], 2)
        return u_0

    @staticmethod
    def generate_random_game(N_agents, n_states, n_inputs):
        A = np.zeros((N_agents, n_states, n_states))
        B = np.zeros((N_agents, n_states, n_inputs))
        Q_tot = generate_random_monotone_matrix(N_agents, n_states) #TODO: function that generates block matrices such that Q_ii>=0 and Q>=0
        R_tot = generate_random_monotone_matrix(N_agents, n_inputs)
        Q = np.zeros((N_agents, N_agents*n_states, N_agents*n_states ))
        R = np.zeros((N_agents, N_agents * n_inputs, N_agents * n_inputs))
        for i in range(N_agents):
            is_controllable = False
            attempt_controllable = 0
            while not is_controllable and attempt_controllable < 10:
                A_single_agent = -0.5 + 1 * np.random.random_sample(size=[n_states, n_states])
                B_single_agent = -0.5 + 1 * np.random.random_sample(size=[n_states, n_inputs])
                A[i, :, :] = A_single_agent
                B[i, :, :] = B_single_agent
                is_controllable = np.linalg.matrix_rank(control.ctrb(A_single_agent, B_single_agent)) == n_states
                attempt_controllable = attempt_controllable + 1
                if attempt_controllable % 10 == 0:
                    n_inputs = n_inputs + 1
            Q[i, i*n_states:(i+1)*n_states, :] = Q_tot[i*n_states:(i+1)*n_states, :]
            Q[i, :, :] = (Q[i, :, :] + Q[i, :, :].T)/2
            R[i, i * n_states:(i + 1) * n_states, :] = R_tot[i * n_states:(i + 1) * n_states, :]
            R[i, :, :] = (R[i, :, :] + R[i, :, :].T) / 2

        return A, B, Q, R
