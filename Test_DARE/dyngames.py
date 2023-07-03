import numpy as np
from scipy.linalg import block_diag

def get_prediction_model(A,B,T_hor):
    n_x =  A.shape(0)
    n_u = B.shape(1)
    T = np.zeros([T_hor * n_x, n_x])
    S = np.zeros([T_hor * n_x, T_hor * n_u])
    for i in range(A.shape(0)):
        A_pow = np.eye(n_x)
        for t in range(T_hor):
            T[t * n_x:(t + 1)*n_x, :] = A_pow @ A
            if t >0:
                S[t * n_x:(t + 1)*n_x, : ] = A @ S[(t-1)*n_x:(t)*n_x, : ]
            S[t * n_x:(t + 1) * n_x, t * n_u: (t+1)*n_u] = B
            A_pow = A_pow @ A
    return S,T


class LQ_decoupled:
    def __init__(self, N_agents, A, B, Q, R, P,
                 A_x_ineq_loc, b_x_ineq_loc, A_x_ineq_sh, b_x_ineq_sh,
                 A_u_ineq_loc, b_u_ineq_loc, A_u_ineq_sh, b_u_ineq_sh,
                 T_hor):
        self.N_agents = N_agents
        # J_i = \|u\|^2_Q_u^i + u' Q_u_x0^i x_0
        self.Q_u,self.Q_u_x0 = self.define_cost_functions(A, B, Q, R, P, T_hor)
        # Unconstrained formulation
        self.A_ineq_local_const, self.b_ineq_loc_from_x, self.b_ineq_loc_constant = \
            self.generate_state_input_constr(A, B, A_x_ineq_loc, b_x_ineq_loc, A_u_ineq_loc, b_u_ineq_loc, T_hor)
        self.A_ineq_shared_const, self.b_ineq_shared_from_x, self.b_ineq_shared_const = \
            self.generate_state_input_constr(A, B, A_x_ineq_sh, b_x_ineq_sh, A_u_ineq_sh, b_u_ineq_sh, T_hor)

    def generate_game_from_initial_state(self, x_0):
        self.Q = self.Q_u
        self.q = self.Q_u_x0 @ x_0
        self.b_ineq_local_const = self.b_ineq_loc_from_x @ x_0 + self.b_ineq_loc_constant

    def define_cost_functions(self, A, B, Q, R, P, T_hor):
        # J_i = .5 \|u\|^2_{T'Q^i_all_t T} + u'T'Q^i_all_t Sx_0
        # where Q^i_all_t is block diagonal with kron(I_T, Q_i) and P_i
        N_agents = self.N_agents
        A_all = block_diag(*A)
        B_all = block_diag(*B)
        n_x = A.shape(1)
        n_u = B.shape(1)
        Q_u = np.zeros((N_agents, n_u* N_agents *T_hor, n_u* N_agents *T_hor))
        Q_u_x0 = np.zeros((N_agents, n_u* N_agents *T_hor, n_x* N_agents *T_hor))
        S, T = get_prediction_model(A_all,B_all, T_hor)
        for i in range(N_agents):
            Q_all_t = np.kron(Q[i], block_diag(np.eye(T_hor-1), 0)) + np.kron(P[i], block_diag(np.zeros(T_hor-1,T_hor-1), 1))
            R_all_t = np.kron(R[i], np.eye(T_hor))
            Q_u[i,:,:] = S.T @ Q_all_t @ S + R_all_t
            Q_u_x0[i,:,:] = S.T @ Q_all_t @ T
        return Q_u, Q_u_x0

    def generate_state_input_constr(self, A, B, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq, T_hor):
        N = self.N_agents
        n_state_const = A_x_ineq[0].shape[0]
        n_input_const = A_u_ineq[0].shape[0]
        n_u = A_u_ineq.shape[1]
        n_x = A_x_ineq[0].shape[1]

        A_ineq_all_timesteps = np.zeros(self.N_agents, T_hor * (n_state_const + n_input_const), T_hor * n_u)
        b_ineq_all_timesteps_from_x = np.zeros((self.N_agents, T_hor * (n_state_const + n_input_const), n_x))  # maps x_0 to b_ineq
        b_ineq_all_timesteps = np.zeros((self.N_agents, T_hor * (n_state_const + n_input_const), 1))

        for i in range(self.N_agents):
            # state and input constraints are:
            # kron(I_T, A_x)(S_i u_i +  T_i x0_i) <= kron(1_T, b_x)
            # kron(I_T, A_u)u_i <= kron(1_T, b_u)
            T_i, S_i = get_prediction_model(A[i], B[i], T_hor)
            A_ineq_all_timesteps[i, 0:T_hor * n_state_const, :] = np.kron(np.eye(T_hor), A_x_ineq[i])*S_i
            A_ineq_all_timesteps[i, T_hor * n_state_const :, :] = np.kron(np.eye(T_hor), A_u_ineq[i])
            b_ineq_all_timesteps_from_x[i, 0:T_hor * n_state_const, :] = np.kron(np.eye(T_hor), A_x_ineq[i])*T_i
            b_ineq_all_timesteps[i, :, :] = np.row_stack(np.kron(np.ones(T_hor,1), b_x_ineq[i]), np.kron(np.ones(T_hor,1), b_u_ineq[i]))

        return A_ineq_all_timesteps, b_ineq_all_timesteps_from_x, b_ineq_all_timesteps

