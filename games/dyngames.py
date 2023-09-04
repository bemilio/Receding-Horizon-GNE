import warnings

import numpy as np
from scipy.linalg import block_diag, norm, solve_discrete_are
from . import staticgames
from .staticgames import batch_mult_with_coulumn_stack, multiagent_array
import control
from copy import copy


def get_multiagent_prediction_model(A, B, T_hor):
    # Generate prediction model with the convention x = col_i(col_t(x_t^i))
    # A[i,:,:] is the evolution matrix of agent i (similar notation for B)
    if len(B.shape) == 2:  # if there are not enough dimensions for matrix B, treat as a single-agent case
        B = copy(B)
        B = np.expand_dims(B, axis=0)
    N = B.shape[0]
    n_x = A.shape[1]
    n_u = B.shape[2]
    T = np.zeros([T_hor * n_x, n_x])
    S_i = np.zeros([N, T_hor * n_x, T_hor * n_u])
    A_pow = np.eye(n_x)
    for t in range(T_hor):
        T[t * n_x:(t + 1) * n_x, :] = A_pow @ A
        if t > 0:
            for i in range(N):
                S_i[i, t * n_x:(t + 1) * n_x, :] = A @ S_i[i, (t - 1) * n_x:t * n_x, :]
        S_i[:, t * n_x:(t + 1) * n_x, t * n_u: (t + 1) * n_u] = B
        A_pow = A_pow @ A
    # S = np.column_stack([S_i[i] for i in range(N)])
    return S_i, T


def generate_random_monotone_matrix(N, n_x, str_mon=1):
    # Produce a random square block matrix Q of dimension N*n_x where each block is n_x*n_x. The blocks on the diagonal are
    # symmetric positive definite and Q is positive definite, not symmetric.
    Q = np.random.random_sample(size=[N * n_x, N * n_x])
    for i in range(N):
        Q_i = Q[i * n_x:(i + 1) * n_x, i * n_x:(i + 1) * n_x]
        Q_i = (Q_i + Q_i.T) / 2
        min_eig = min(np.linalg.eigvalsh((Q_i) / 2))
        if min_eig.item() < 0:
            Q[i * n_x:(i + 1) * n_x, i * n_x:(i + 1) * n_x] = Q_i - min_eig * np.eye(n_x)
    min_eig = min(np.linalg.eigvalsh((Q + Q.T) / 2))
    if min_eig.item() < 0:
        Q = Q  + (str_mon - min_eig) * np.eye(N * n_x)
    eps = 10 ** (-10)
    # Check just to be sure
    if min(np.linalg.eigvalsh((Q + Q.T) / 2)) < -1 * eps:
        raise Exception("The game is not monotone")
    return Q


##todo: remove?
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
        pattern_Q = block_diag(np.eye(T_hor - 1), 0)
        pattern_P = block_diag(np.zeros((T_hor - 1, T_hor - 1)), 1)
        Q_expanded = np.vstack([np.hstack(
            [(np.kron(pattern_Q, Q[j * n_x:(j + 1) * n_x, k * n_x:(k + 1) * n_x]) + \
              np.kron(pattern_P, P[j * n_x:(j + 1) * n_x, k * n_x:(k + 1) * n_x]))
             for k in range(N_blocks)]) for j in range(N_blocks)])
        return Q_expanded


class LQ:
    """
    Linear quadratic games class
    min_{u_i} x(T)' P_i x(T) + \sum_{t=0}^{T_h-1}\sum_j x(t)' Q_i x(t) + u_i(t)' R_i u_i(t)
    s.t.      x^+ = Ax + \sum_j B_j u_j
              C_u_loc[i] * u_i(t) \leq d_u_loc[i]
              \sum_j C_u_sh[j] * u_j(t) \leq \sum_j d_u_sh[j]
              C_x_i * x(t) \leq d_x
    assume x is dimension n_x and u_j is dimension n_u for all j.
    """

    def __init__(self, A, B, Q, R, P, C_x, d_x, C_u_loc, d_u_loc, C_u_sh, d_u_sh, T_hor, test=False):
        """
        :param A: numpy array with shape (n_x * n_x)
        :param B: numpy array with shape (N * n_x * n_u)
        :param Q: numpy array with shape (N * n_x * n_x)
        :param R: numpy array with shape (N * n_u * n_u)
        :param P: numpy array with shape (N * n_x * n_x)
        :param C_x: numpy array with shape (N * m_x * n_x), m_x number of state constraints
        :param d_x: numpy array with shape (N * m_x * 1)
        :param C_u_loc: numpy array with shape (N * m_u_loc * n_x), m_u_loc number of local input constraints
        :param d_u_loc: numpy array with shape (N * m_u_loc * 1)
        :param C_u_sh: numpy array with shape (N * m_u_sh * n_x), m_u_sh number of local input constraints
        :param d_u_sh: numpy array with shape  (N * m_u_sh * 1)
        :param T_hor: horizon length (integer)
        :param test: boolean, set to true to generate a test setup (remaining inputs are ignored)
        """
        if test == True:
            A, B, Q, R, P, \
                C_x, d_x, \
                C_u_loc, d_u_loc, C_u_sh, d_u_sh, \
                T_hor = self.set_to_test_game_setup()
        self.N_agents = B.shape[0]
        self.n_x = B.shape[1]
        self.n_u = B.shape[2]
        self.T_hor = T_hor
        self.A = A
        self.B = multiagent_array(B)
        self.Q = multiagent_array(Q)
        self.R = multiagent_array(R)
        self.P = multiagent_array(P)
        self.S, self.T = get_multiagent_prediction_model(A, B, T_hor)
        # J_i = .5 u'W_iu + u'G_ix_0 + .5 x_0'H_ix_0
        self.W, self.G, self.H = self.define_cost_functions()
        # Generate constraints
        self.C_u_loc, self.d_u_loc = self.generate_input_constr(C_u_loc, d_u_loc)
        self.C_u_sh, self.d_u_sh = self.generate_input_constr(C_u_sh, d_u_sh)
        self.C_x_sh, self.D_x_sh, self.d_x_sh = self.generate_state_constr(C_x, d_x)
        self.C_eq = np.zeros((self.N_agents, 1, self.T_hor * self.n_u))  # dummy equality constraints
        self.D_eq = np.zeros((self.N_agents, 1, self.n_x))  # dummy equality constraints
        self.d_eq = np.zeros((self.N_agents, 1, 1))  # dummy equality constraints

    def set_to_test_game_setup(self):
        # two decoupled double integrators with no cost coupling (MPC) and no constraints
        N_agents = 2
        n_x = 2 #number of states per agent
        n_u = 1
        # A = np.stack([np.array([[1]]) for _ in range(N_agents)])
        # B = np.stack([np.array([[1]]) for _ in range(N_agents)])
        A = block_diag(*[np.array([[.9, 1], [0, 1]]) for _ in range(N_agents)])
        B = np.stack([np.kron(np.expand_dims(np.eye(N_agents)[:,i], 1), np.array([[0], [1]])) for i in range(N_agents)])
        pattern = np.zeros((N_agents, N_agents, N_agents))
        for i in range(N_agents):
            pattern[i, i, i] = 1
        Q = np.kron(pattern, np.eye(n_x))
        R = np.stack([np.eye(n_u) for _ in range(N_agents)])
        # P_i is a block matrix where the only non-zero element is on the diagonal and it is the solutions of the local DARE
        P_ii = np.stack([solve_discrete_are(A[i*n_x:(i+1)*n_x,i*n_x:(i+1)*n_x], B[i][i*n_x:(i+1)*n_x, :],
                            Q[i][i * n_x:(i + 1) * n_x, i * n_x:(i + 1) * n_x], R[i]) for i in range(N_agents)])
        P = np.zeros((N_agents, N_agents * n_x, N_agents * n_x))
        for i in range(N_agents):
            P[i, i * n_x:(i + 1) * n_x, i * n_x:(i + 1) * n_x] = P_ii[i]

        C_x = np.zeros((1, N_agents * n_x))
        C_u_loc = np.zeros((N_agents, 1, n_u))
        d_x = np.zeros((1, 1))
        d_u_loc = np.zeros((N_agents, 1, 1))
        C_u_sh = np.zeros((N_agents, 1, n_u))
        d_u_sh = np.zeros((N_agents, 1, 1))
        T_hor = 2
        return A, B, Q, R, P, C_x, d_x, C_u_loc, d_u_loc, C_u_sh, d_u_sh, T_hor

    def generate_game_from_initial_state(self, x_0):
        Q = self.W
        # Obtain linear part of the cost from x_0. Note that the mapping x_0->cost wants x_0 = col_i(x_0^i),
        # while the mapping to the affine part constraints is agent-wise, that is, it only requires x_0^i
        q = self.G @ x_0  # batch mult
        h = x_0.T @ self.H @ x_0 # affine part of the cost
        C = np.concatenate((self.C_u_sh, self.C_x_sh), axis=1)
        d = np.concatenate((self.d_u_sh, self.D_x_sh @ x_0 + self.d_x_sh), axis=1)
        return staticgames.LinearQuadratic(Q, q, h, self.C_u_loc, self.d_u_loc, self.C_eq, self.d_eq, C, d)

    def define_cost_functions(self):
        """
        Define the cost
        J_i = .5 u'W_iu + u'G_ix_0 + .5 x_0'H_ix_0
        where W = S'Q_iS + R_i, G_i = S'Q_iT, H_i = T'Q_iT
        S and T define the prediction model x = Tx_0 + Su
        (with abuse of notation) Q_i = blkdiag( kron(I, Q_i), P_i ) + R_i (right hand side is related to stage cost)
        R_i = [kron(I,R_ii), 0;
               0,            0] (up to permutation)
        :return: W, G, H 3-D numpy arrays which are the stacks respectively of W_i, G_i, H_i
        """
        W = np.zeros((self.N_agents, self.n_u * self.N_agents * self.T_hor, self.n_u * self.N_agents * self.T_hor))
        G = np.zeros((self.N_agents, self.n_u * self.N_agents * self.T_hor, self.n_x))
        H = np.zeros((self.N_agents, self.n_x, self.n_x))
        S = np.column_stack([self.S[i] for i in range(self.N_agents)])
        for i in range(self.N_agents):
            Q_i = block_diag(*(np.kron(np.eye(self.T_hor-1), self.Q[i]), self.P[i]))
            W[i, :, :] = S.T @ Q_i @ S
            W[i, i * self.n_u * self.T_hor:(i + 1) * self.n_u * self.T_hor, i * self.n_u * self.T_hor:(i + 1) * self.n_u * self.T_hor] = \
            W[i, i * self.n_u * self.T_hor:(i + 1) * self.n_u * self.T_hor, i * self.n_u * self.T_hor:(i + 1) * self.n_u * self.T_hor] + \
                np.kron(np.eye(self.T_hor), self.R[i])
            G[i, :, :] = S.T @ Q_i @ self.T
            H[i, :, :] = self.T.T @ Q_i @ self.T
        return W, G, H

    def generate_input_constr(self, A, b):
        n_constr = A.shape[1]
        A_all_timesteps = np.zeros((self.N_agents, self.T_hor * n_constr, self.T_hor * self.n_u))
        b_all_timesteps = np.zeros((self.N_agents, self.T_hor * n_constr, 1))
        for i in range(self.N_agents):
            A_all_timesteps[i, :, :] = np.kron(np.eye(self.T_hor), A[i])
            b_all_timesteps[i, :, :] = np.kron(np.ones((self.T_hor, 1)), b[i])
        return A_all_timesteps, b_all_timesteps

    def generate_state_constr(self, A, b):
        '''
        :param A: 2D-numpy array
        :param b: 2D-numpy array (column vectors) such that the constraints are Ax<=b
        :return: C (3D numpy array), D, p 2-D numpy arrays such that given x0 the constraints are in the form:
        \sum_j C[j]u[j] <= \sum_j D_j x0 + d_j
        '''
        n_constr = A.shape[0]
        C = np.zeros((self.N_agents, self.T_hor * n_constr, self.T_hor * self.n_u))
        D = np.zeros((self.N_agents, self.T_hor * n_constr, self.n_x))
        d = np.zeros((self.N_agents, self.T_hor * n_constr, 1))
        for i in range(self.N_agents):
            C[i, :, :] = np.kron(np.eye(self.T_hor), A) @ self.S[i]
            D[i, :, :] = -np.kron(np.eye(self.T_hor), A) @ self.T / self.N_agents
            d[i, :, :] = np.kron(np.ones((self.T_hor, 1)), b) / self.N_agents
        return C, D, d

    def generate_state_equality_constr(self, A, b, t): #TODO: This should become a SHARED equality constraint
        '''
        :param A: 2D-numpy array
        :param b: 2D-numpy array (column vectors) such that the constraints are Ax(t)=b
        :param t: timestep at which the equality constraint is enforced (integer)
        :return: nothing
        Sets the local arguments that define the equality constraints in order to enforce:
        Ax(t) = b
        '''
        # set A * x_t = b
        # by rewriting it as A * S * u = b_x - A * T * x_0
        n_constr = A.shape[0]
        C = np.zeros((self.N_agents, n_constr, self.T_hor * self.n_u))
        D = np.zeros((self.N_agents, n_constr, self.n_x))
        d = np.zeros((self.N_agents, n_constr, 1))
        for i in range(self.N_agents):
            C[i, :, :] = A[i] @ self.S[i][t * self.n_x:(t + 1) * self.n_x, :]
            D[i, :, :] = -A[i] @ self.T[i][t * self.n_x:(t + 1) * self.n_x, :]/self.N_agents
            d[i, :, :] = b[i]/self.N_agents
        self.C_eq = C
        self.D_eq = D
        self.d_eq = d

    def get_predicted_input_trajectory_from_opt_var(self, u_all):
        u = u_all.reshape((self.N_agents, self.T_hor, self.n_u))
        return u

    def get_shifted_trajectory_from_opt_var(self, u_all, x_0):
        u = u_all[:, self.n_u:]
        x_last = self.get_state_timestep_from_opt_var(u_all, x_0, self.T_hor)
        # warnings.warn("[get_shifted_trajectory_from_opt_var] For testing, the shifted sequence is being altered")
        u_shift = np.concatenate((u, self.K @ x_last), axis=1)
        return u_shift

    def get_state_timestep_from_opt_var(self, u_all, x_0, t):
        if t < 1 or t > self.T_hor:
            raise ValueError(
                "[get_state_timestep_from_opt_var] t must be >=1 and <= T_hor, where T_hor = " + str(self.T_hor))
        x = np.sum(self.S[:, (t - 1) * self.n_x: t * self.n_x, :] @ u_all, axis=0) \
            + self.T[(t - 1) * self.n_x:t * self.n_x, :] @ x_0
        return x

    def get_input_timestep_from_opt_var(self, u_all, t):
        """
        :param u_all: numpy array with shape: (N_agents, n_u*T_hor, 1) where u_all[i] is the vertical stack of u_i(t)
        :param t: integer
        :return: u numpy array with shape: (N_agents, n_u, 1) where u[i] = u_i(t)
        """
        u = self.get_predicted_input_trajectory_from_opt_var(u_all)
        u_t = np.expand_dims(u[:, t, :], 2)
        return u_t

    def solve_open_loop_inf_hor_problem(self, n_iter=1000, eps_error=10 ** (-6)):
        """
        :param n_iter: maximum number of iterations
        :param eps_error:
        :return: P, K such that A+BK gives the open loop Nash dynamics.
        """
        # Solve the asymmetric Riccati equation [Freiling '99]:
        # P_i = Q_i + A'P_i(I + \sum_j S_j P_j)^{-1} A
        # where S_j = B_jR_j^{-1}B_j'
        # by iteratively solving a Sylvester discrete-time equation
        # X = Q_i + A' X M A
        # where M = (I + \sum_j S_j P_j)^{-1}
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        n_x = self.n_x
        n_u = self.n_u
        N = self.N_agents
        K = np.zeros((N, n_u, n_x))
        P = np.zeros((N, n_x, n_x))
        # Stable initialization: cooperative optimal controller
        B_coop = np.column_stack([B[i] for i in range(N)])
        P_init, K_init = self.solve_closed_loop_inf_hor_problem()
        P[:] = P_init[:]
        K[:] = K_init[:]
        for i in range(N):
            if min(np.linalg.eigvalsh(P_init[i])) < 0:
                warnings.warn("The closed loop P is non-positive definite")
        if max(np.abs(np.linalg.eigvals(A + np.sum(B @ K_init, axis=0)))) > 1.001:
            warnings.warn("The infinite horizon CL-GNE has an unstable dynamics")
        # P_init = solve_discrete_are(A, B_coop, np.eye(n_x), np.eye(N * n_u))
        # K_init = - np.linalg.inv(np.eye(N * n_u) + B_coop.T @ P_coop @ B_coop) @ (B_coop.T @ P_coop @ A)
        # for i in range(N):
        #     K[i] = K_init[n_u*i:n_u*(i+1), :]
        #     P[i] = P_init # I think this works as far as we initialize to some P>0
        for k in range(n_iter):
            ''' 
            IMPORTANT: The commented line does not work (as in, it leads to a negative-definite solution P and unstable dynamics)
            and it takes way more iterations. 
            Why???
            '''
            # A_cl = A + np.sum(B @ np.linalg.inv(R) @ B.T3D() @ np.linalg.inv(A.T) @ (Q - P) , axis=0)
            A_cl = A + np.sum(B @ K, axis=0)
            for i in range(N):
                try:
                    P[i][:, :] = control.dlyap(A.T, A_cl.T, Q[i])  # -A.T @ X A_cl + X - Q[i] = 0
                except Exception as e:
                    print("[solve_open_loop_inf_hor_problem] An error occurred while solving the Sylvester equation:")
                    print(str(e))
            M = np.linalg.inv(np.eye(n_x) + np.sum(B @ np.linalg.inv(R) @ B.T3D() @ P, axis=0))
            K = - np.linalg.inv(R) @ B.T3D() @ P @ M @ A  # Batch multiplication
            if n_iter%10==0:
                # Test solution: Check if (9) [Freiling-Jank-Kandil '99] is satisfied
                err = 0
                M = np.linalg.inv(np.eye(n_x) + np.sum(B @ np.linalg.inv(R) @ B.T3D() @ P, axis=0))
                for i in range(N):
                    err = err + norm(Q[i] - P[i] + A.T @ P[i] @ M @ A)
                if  err < eps_error:
                    break
        if err > eps_error:
            print("[solve_open_loop_inf_hor_problem] Could not find solution")
        for i in range(N):
            if min(np.linalg.eigvalsh(P[i])) < 0:
                warnings.warn("The open loop P is non-positive definite")
        if max(np.abs(np.linalg.eigvals(A + np.sum(B @ K, axis=0)))) > 1.001:
            warnings.warn("The infinite horizon OL-GNE has an unstable dynamics")
        return P, K


    def solve_closed_loop_inf_hor_problem(self, n_iter=1000, eps_error=10 ** (-6), method='lyap'):
        """
        :param n_iter: maximum number of iterations
        :param eps_error:
        :return: P, K such that A+BK gives the open loop Nash dynamics.
        """
        # Solve the asymmetric Riccati equation:
        # P_i = Q_i + (A + \sum_{j!=i} S_j P_j)'P_i(I + \sum_j S_j P_j)^{-1} A
        # where S_j = B_jR_j^{-1}B_j'
        # by iteratively solving a Sylvester discrete-time equation
        # X = Q_i + (A + \sum_{j!=i} S_j P_j)' X M A
        # where M = (I + \sum_j S_j P_j)^{-1}
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        n_u = B.shape[2]
        n_x = B.shape[1]
        N = B.shape[0]
        P = np.zeros((N, n_x, n_x))
        K = np.zeros((N, n_u, n_x))
        # Stable initialization: cooperative optimal controller
        B_coop = np.column_stack([B[i] for i in range(N)])
        P_coop = solve_discrete_are(A, B_coop, np.eye(n_x), np.eye(N*n_u))
        K_coop = - np.linalg.inv(np.eye(N*n_u) + B_coop.T @ P_coop @ B_coop) @ (B_coop.T @ P_coop @ A)
        for i in range(N):
            K[i] = K_coop[n_u*i:n_u*(i+1), :]
            P[i] = P_coop # I think this works as far as we initialize to some P>0
        M = np.linalg.inv(np.eye(n_x) + np.sum(B @ np.linalg.inv(R) @ B.T3D() @ P, axis=0))
        A_cl = M @ A
        for k in range(n_iter):
            if method == 'lyap':
                ''' 
                Method 1 (adapted from sassano 22 and Li-Gajic): 
                solve P[i] = Q[i] + K[i]'R[i]K[i] + A_cl' P[i] A_cl by substituting
                K = - R[i]^(-1)B[i]'P[i]A_cl ;
                A_cl = (I + \sum B[j]R[j]^(-1)B[j]')^(-1) A
                Resulting in the lyapunov equation:
                A_cl P[i] A_cl + Q_bar[i] = P[i]
                where 
                Q_bar[i] = Q[i] + (B[i]' P[i] A_cl)' R[i]^(-1) (B[i]' P[i] A_cl)
                '''
                for i in range(N):
                    Q_bar = Q[i] + (B[i].T @ P[i] @ A_cl).T @ np.linalg.inv(R[i]) @ (B[i].T @ P[i] @ A_cl)
                    try:
                        P[i] = control.dlyap(A_cl.T, Q_bar, method='slycot') # transpose due to the convention in control.dlyap
                    except:
                        print("[solve_closed_loop_inf_hor_problem] An error occurred while solving the Lyapunov equation")
                M = np.linalg.inv(np.eye(n_x) + np.sum(B @ np.linalg.inv(R) @ B.T3D() @ P, axis=0))
                A_cl = M @ A
                K = - np.linalg.inv(R) @ B.T3D() @ P @ A_cl   # Batch multiplication
            elif method == 'riccati':
                ''' 
                Method 2 (adapted from sassano 22): 
                solve P[i] = Q[i] + K[i]'R[i]K[i] + A_cl' P[i] A_cl 
                by iterating the Riccati equation
                A_cl' P[i] A_cl + Q[i] + K[i]'R[i]K[i] = P[i];
                K[i] = - R[i]B[i]'P[i]A_cl 
                '''
                for i in range(N):
                    A_cl_not_i = A + np.sum(B @ K, axis=0) - B[i] @ K[i]
                    P[i] = solve_discrete_are(A_cl_not_i, B[i], Q[i], R[i])
                    K[i] = - np.linalg.inv(R[i] + B[i].T @ P[i] @ B[i]) @ B[i].T @ P[i] @ A_cl_not_i
            elif method == 'experimental':
                A_cl = A + np.sum(B @ K, axis=0)
                for i in range(N):
                    A_cl_not_i = A + np.sum(B @ K, axis=0) - B[i] @ K[i]
                    P[i][:, :] = control.dlyap(A_cl_not_i.T, A_cl.T, Q[i]) # -A_cl_not_i.T @ X A_cl + X - Q[i] = 0
                M = np.linalg.inv(np.eye(n_x) + np.sum(B @ np.linalg.inv(R) @ B.T3D() @ P, axis=0))
                K = - np.linalg.inv(R) @ B.T3D() @ P @ M @ A  # Batch multiplication
            else:
                raise ValueError("[solve_closed_loop_inf_hor_problem]: only supported methods are 'lyap', 'riccati' or 'experimental'.")
            # P[i] = control.dlyap(A_cl_not_i, M @ A, Q[i], method='slycot')
            if n_iter%10==0:
                # Test solution by verifying (6.28) in Basar book
                err = 0
                A_cl = A + np.sum(B @ K, axis=0)
                for i in range(N):
                    err = max(err, norm(P[i] - (Q[i] + K[i].T @ R[i] @ K[i] + A_cl.T @ P[i] @ A_cl)))
                    err = max(err, norm(K[i] + np.linalg.inv(R[i] + B[i].T @ P[i] @ B[i]) @ \
                                            (B[i].T @ (P[i] @ (A_cl-B[i]@K[i])))) )
                if  err < eps_error:
                    break
        if err > eps_error:
            print("[solve_closed_loop_inf_hor_problem] Could not find solution")
        # Compute optimal controllers
        return P, K

    def set_term_cost_to_inf_hor_sol(self, mode="CL", method='lyap', n_iter=10000, eps_error=10**(-6)):
        if mode=="OL":
            self.P, self.K = self.solve_open_loop_inf_hor_problem(n_iter=n_iter, eps_error=eps_error)
        elif mode=="CL":
            self.P, self.K = self.solve_closed_loop_inf_hor_problem(n_iter=n_iter,  method=method, eps_error=eps_error)
        else:
            raise ValueError("[dyngames::LQ::set_term_cost_to_inf_hor_sol] mode needs to be either 'OL' or 'CL' ")
        # Re-create cost functions with the new terminal cost
        self.W, self.G, self.H = self.define_cost_functions()

    @staticmethod
    def generate_random_game(N_agents, n_states, n_inputs):
        # A sufficient condition for the game to be monotone is Q_i=I for all i
        A = np.zeros((n_states, n_states))
        B = np.zeros((N_agents, n_states, n_inputs))
        Q = np.zeros((N_agents, n_states, n_states))
        R = np.zeros((N_agents, n_inputs, n_inputs))
        attempt_controllable = 0
        max_norm_eig = 0
        # limit maximum norm of eigenvalues so that the prediction model does not explode
        is_controllable = False
        while (is_controllable == False and attempt_controllable < 10) or max_norm_eig > 1.2:
            is_controllable = False
            A = -1 + 2 * np.random.random_sample(size=[n_states, n_states])
            B = -1 + 2 * np.random.random_sample(size=[N_agents, n_states, n_inputs])
            max_norm_eig = max(np.linalg.norm(np.expand_dims(np.linalg.eigvals(A), 1), axis=1))
            for j in range(N_agents):
                is_controllable = is_controllable or (np.linalg.matrix_rank(control.ctrb(A, B[j])) == n_states)
            attempt_controllable = attempt_controllable + 1
        if is_controllable==False:
            warnings.warn("System is not controllable")
        for i in range(N_agents):
            Q[i, :, :] =  np.eye(n_states)
            R[i, :, :] = generate_random_monotone_matrix(1, n_inputs, str_mon=1) #random pos. def. matrix
            R[i, :, :] = (R[i, :, :] + R[i, :, :].T) / 2
        return A, B, Q, R
