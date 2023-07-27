import warnings

import numpy as np
from scipy.linalg import block_diag, norm, solve_discrete_are
from . import staticgames
from .staticgames import batch_mult_with_coulumn_stack
import control
from copy import copy


def get_multiagent_prediction_model(A, B, T_hor):
    # Generate prediction model with the convention x = col_i(col_t(x_t^i))
    # A[i,:,:] is the evolution matrix of agent i (similar notation for B)
    if len(A.shape) == 2:  # if there are not enough dimensions for matrix A, treat as a single-agent case
        A = copy(A)
        B = copy(B)
        A = np.expand_dims(A, axis=0)
        B = np.expand_dims(B, axis=0)
    N = B.shape[0]
    n_x = A.shape[2]
    n_u = B.shape[2]
    T = np.zeros([T_hor * n_x, n_x])
    S_i = np.zeros([N, T_hor * n_x, T_hor * n_u])
    A_pow = np.eye(n_x)
    for t in range(T_hor):
        T[:, t * n_x:(t + 1) * n_x, :] = A_pow @ A
        if t > 0:
            for i in range(N):
                S_i[i, t * n_x:(t + 1) * n_x, :] = A @ S_i[i, (t - 1) * n_x:t * n_x, :]
        S_i[:, t * n_x:(t + 1) * n_x, t * n_u: (t + 1) * n_u] = B
        A_pow = A_pow @ A
    # S = np.column_stack([S_i[i] for i in range(N)])
    return S_i, T


def generate_random_monotone_matrix(N, n_x):
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
        Q = Q - min_eig * np.eye(N * n_x)
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
        self.n_x = A.shape[2]
        self.n_u = B.shape[2]
        self.T_hor = T_hor
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.S, self.T = get_multiagent_prediction_model(A, B, T_hor)
        # J_i = .5 u'W_iu + u'G_ix_0 + .5 x_0'H_ix_0
        self.W, self.G, self.H = self.define_cost_functions()
        # Generate constraints
        self.C_u_loc, self.d_u_loc = self.generate_input_constr(C_u_loc, d_u_loc)
        self.C_u_sh, self.d_u_sh = self.generate_input_constr(C_u_sh, d_u_sh)
        self.C_x_sh, self.D_x_sh, self.d_x_sh = self.generate_state_constr(C_x, d_x)
        self.C_eq = np.zeros((self.N_agents, 1, self.T_hor * self.n_u))  # dummy equality constraints
        self.D_eq = np.zeros((self.N_agents, 1, self.n_x))  # dummy equality constraints
        self.d_eq = np.zeros((self.N_agents, 1, self.T_hor * self.n_u))  # dummy equality constraints

    def set_to_test_game_setup(self):
        # two decoupled double integrators with no cost coupling (MPC) and no constraints
        N_agents = 2
        n_x = 2 #number of states per agent
        n_u = 1
        # A = np.stack([np.array([[1]]) for _ in range(N_agents)])
        # B = np.stack([np.array([[1]]) for _ in range(N_agents)])
        A = block_diag([np.array([[.9, 1], [0, 1]]) for _ in range(N_agents)])
        B = np.stack([np.array([[0], [1]]) for _ in range(N_agents)])
        pattern = np.zeros((N_agents, N_agents, N_agents))
        for i in range(N_agents):
            pattern[i, i, i] = 1

        Q = np.kron(pattern, np.eye(n_x))
        R = np.kron(pattern, np.eye(n_u))
        # P_i is a block matrix where the only non-zero element is on the diagonal and it is the solutions of the local DARE
        P_ii = np.stack([solve_discrete_are(A[i], B[i], Q[i][i * n_x:(i + 1) * n_x, i * n_x:(i + 1) * n_x],
                                                         R[i][i * n_u:(i + 1) * n_u, i * n_u:(i + 1) * n_u]) for i in
                         range(N_agents)])
        P = np.zeros((N_agents, N_agents * n_x, N_agents * n_x))
        for i in range(N_agents):
            P[i, i * n_x:(i + 1) * n_x, i * n_x:(i + 1) * n_x] = P_ii[i]

        C_x = np.zeros((1, N_agents * n_x))
        C_u_loc = np.zeros((N_agents, 1, n_u))
        d_x = np.zeros((1, 1))
        d_u_loc = np.zeros((N_agents, 1, 1))
        C_u_sh = np.zeros((N_agents, 1, n_u))
        d_u_sh = np.zeros((N_agents, 1, 1))
        T_hor = 1
        return A, B, Q, R, P, C_x, d_x, C_u_loc, d_u_loc, C_u_sh, d_u_sh, T_hor

    def generate_game_from_initial_state(self, x_0):
        Q = self.W
        # Obtain linear part of the cost from x_0. Note that the mapping x_0->cost wants x_0 = col_i(x_0^i),
        # while the mapping to the affine part constraints is agent-wise, that is, it only requires x_0^i
        q = self.G @ x_0  # batch mult
        C = np.concatenate((self.C_u_sh, self.C_x_sh), axis=1)
        d = np.concatenate((self.d_u_sh, self.D_x_sh @ x_0 + self.d_x_sh), axis=1)
        return staticgames.LinearQuadratic(Q, q, self.C_u_loc, self.d_u_loc, self.C_eq, self.d_eq, C, d)

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
            Q_i = block_diag((np.kron(self.T_hor, self.Q[i]), self.P[i]))
            W[i, :, :] = S.T @ Q_i @ S
            W[i, i * self.n_u * self.T_hor:(i + 1) * self.n_u * self.T_hor,
            i * self.n_u * self.T_hor:(i + 1) * self.n_u * self.T_hor] = \
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
        \sum_j C[j]u[j] <= Dx0 + d
        '''
        n_constr = A.shape[0]
        C = np.zeros((self.N_agents, self.T_hor * n_constr, self.T_hor * self.n_u))
        D = np.zeros((self.N_agents, self.T_hor * n_constr, self.n_x))
        d = np.zeros((self.N_agents, self.T_hor * n_constr, 1))
        for i in range(self.N_agents):
            C[i, :, :] = np.kron(np.eye(self.T_hor), A[i]) @ self.S[i]
            D[i, :, :] = -np.kron(np.eye(self.T_hor), A[i]) @ self.T[i]
            d[i, :, :] = np.kron(np.ones((self.T_hor, 1)), b[i])
        return C, D, d

    def generate_state_equality_constr(self, A, b, t):
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
            D[i, :, :] = -A[i] @ self.T[i][t * self.n_x:(t + 1) * self.n_x, :]
            d[i, :, :] = b[i]
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
        P = np.zeros((N, n_x, n_x))
        for k in range(n_iter):
            M = np.inv(np.eye(n_x) + np.sum(B @ P @ B.T3D(), axis=0))
            for i in range(N):
                P[i][:, :] = control.dlyap(A, M @ A, Q[i])
            if n_iter%10==0:
                # Test solution
                err = 0
                M = np.inv(np.eye(n_x) + np.sum(B @ P @ B.T3D(), axis=0))
                for i in range(N):
                    err = err + (Q - P[i] + A.T @ P[i] @ M @ A)
                if  err < eps_error:
                    break
        if err > eps_error:
            print("[solve_open_loop_inf_hor_problem] Could not find solution")
        # Compute optimal controllers
        K = - np.inv(R) @ B.T3D() @ P  # Batch multiplication
        return P, K


    def solve_closed_loop_inf_hor_problem(self, n_iter=1000, eps_error=10 ** (-6)):
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
        n_x = A.shape[2]
        N = A.shape[0]
        P = np.zeros((N, N * n_x, n_x))
        K = np.zeros((N, n_u, n_x))
        # Stable initialization: cooperative optimal controller
        B_coop = np.column_stack([B[i] for i in range(N)])
        P_coop = solve_discrete_are(A, B_coop, np.eye(n_x), np.eye(N*n_u))
        K_coop = - np.inv(np.eye(N*n_u) + B_coop.T * P_coop @ B_coop) @ (B_coop.T @ P_coop @ A)
        for i in range(N):
            K[i] = K_coop[n_u*i:n_u*(i+1), :]
        for k in range(n_iter):
            M = np.inv(np.eye(n_x) + np.sum(B @ P @ B.T3D(), axis=0))
            for i in range(N):
                A_cl_not_i = A + np.sum(B @ K, axis=0) - B[i] @ K[i]
                P[i] = control.dlyap(A_cl_not_i, M @ A, Q[i])
            K = - np.inv(R) @ B.T3D() @ P  # Batch multiplication
            if n_iter%10==0:
                # Test solution by verifying (6.28) in Basar book
                err = 0
                for i in range(N):
                    A_cl = A + np.sum(B @ K, axis=0)
                    err = max(err, norm(P[i] - (Q[i] + K[i].T @ R[i] @ K[i] + A_cl.T @ P[i] @ A_cl)))
                    err = max(err, norm(K[i] + np.linalg.inv(R[i] + B[i].T @ P[i] @ B[i]) @ \
                                            (B[i].T @ (P[i] @ (A_cl-B[i]@K[i])))) )
                if  err < eps_error:
                    break
        if err > eps_error:
            print("[solve_open_loop_inf_hor_problem] Could not find solution")
        # Compute optimal controllers
        K = - np.inv(R) @ B.T3D() @ P  # Batch multiplication
        return P, K

    # Todo: remove?
    # def solve_closed_loop_inf_hor_problem(self, n_iter=1000, eps_error=10 ** (-6)):
    #     # Iterate DARE
    #     A = self.A
    #     B = self.B
    #     Q = self.Q
    #     R = self.R
    #     n_u = B.shape[2]
    #     n_x = A.shape[2]
    #     N = A.shape[0]
    #     P = np.zeros((N, N * n_x, N * n_x))
    #     K = np.zeros((N, n_u, N * n_x))
    #     if N > 1:
    #         B_not_i = np.zeros((N, N * n_x, (N - 1) * n_u))
    #         R_not_i_not_i = np.zeros((N, (N - 1) * n_u, (N - 1) * n_u))
    #         R_i_not_i = np.zeros((N, n_u, (N - 1) * n_u))
    #         K_not_i = np.zeros((N, (N - 1) * n_u, N * n_x))
    #         K_all = np.zeros((N * n_u, N * n_x))
    #         P_err = 0
    #         K_err = 0
    #
    #         not_i_iterator = [[k for k in range(N) if k not in {i}] for i in range(N)]
    #
    #         R_i_i = np.stack([R[i][i * n_u:(i + 1) * n_u, i * n_u:(i + 1) * n_u] for i in range(N)])
    #
    #         for i in range(N):
    #             R_not_i_not_i[i][:, :] = np.vstack(
    #                 [np.hstack([R[i][j * n_u:(j + 1) * n_u, k * n_u:(k + 1) * n_u] for k in not_i_iterator[i]]) \
    #                  for j in not_i_iterator[i]])
    #
    #         for i in range(N):
    #             R_i_not_i[i][:, :] = np.hstack(
    #                 [R[i, i * n_u:(i + 1) * n_u, j * n_u:(j + 1) * n_u] for j in not_i_iterator[i]])
    #
    #         # initialize K
    #         for i in range(N):
    #             P_ii = solve_discrete_are(A[i], B[i], Q[i][i * n_x:(i + 1) * n_x, i * n_x:(i + 1) * n_x],
    #                                                    R[i][i * n_u:(i + 1) * n_u, i * n_u:(i + 1) * n_u])
    #             K[i][:, i * n_x:(i + 1) * n_x] = - np.linalg.inv(
    #                 R[i][i * n_u:(i + 1) * n_u, i * n_u:(i + 1) * n_u] + B[i].T @ P_ii @ B[i]) @ (B[i].T @ P_ii @ A[i])
    #
    #         P_err_all = np.zeros((n_iter // 10))
    #         P_evol = np.zeros((n_iter // 10))
    #         for iter in range(n_iter):
    #             P_last = np.array((P))
    #             for i in range(N):
    #                 K_not_i[i][:, :] = np.vstack([K[j] for j in not_i_iterator[i]])
    #                 P_last = np.array((P))
    #                 B_not_i = np.hstack((B_all[:, :i * n_u], B_all[:, (i + 1) * n_u:]))
    #                 P[i] = solve_discrete_are(A_all + B_not_i @ K_not_i[i],
    #                                                        B_all[:, i * n_u:(i + 1) * n_u], \
    #                                                        Q[i] + K_not_i[i].T @ R_not_i_not_i[i] @ K_not_i[i], \
    #                                                        R_i_i[i], s=(R_i_not_i[i] @ K_not_i[i]).T)
    #                 P_ii = P[i, i * n_x:(i + 1) * n_x, i * n_x:(i + 1) * n_x]
    #                 K[i] = -np.linalg.inv(R_i_i[i] + B[i].T @ P_ii @ B[i]) @ \
    #                        (B[i].T @ (P[i, i * n_x:(i + 1) * n_x, :] @ (A_all + B_not_i @ K_not_i[i])) + \
    #                         R_i_not_i[i] @ K_not_i[i])
    #                 # K[i] = -np.linalg.inv(R_i_i[i] + B_all[:, i*n_u:(i+1)*n_u].T @ P[i] @ B_all[:, i*n_u:(i+1)*n_u]) @ \
    #                 #        (B_all[:, i*n_u:(i+1)*n_u].T @ (P[i] @ (A_all  + B_not_i @ K_not_i[i]) )  + \
    #                 #         R_i_not_i[i] @ K_not_i[i] )
    #
    #             # Test solution
    #             if (iter % 10) == 0:
    #                 for i in range(N):
    #                     K_all[i * n_u:(i + 1) * n_u, :] = K[i]
    #                     K_not_i[i][:, :] = np.vstack([K[j] for j in not_i_iterator[i]])
    #                 P_err = 0
    #                 K_err = 0
    #                 for i in range(N):
    #                     B_not_i = np.hstack((B_all[:, :i * n_u], B_all[:, (i + 1) * n_u:]))
    #                     P_ii = P[i, i * n_x:(i + 1) * n_x, i * n_x:(i + 1) * n_x]
    #                     P_err = max(P_err, norm(P[i] - (
    #                                 Q[i] + K_all.T @ R[i] @ K_all + (A_all + B_all @ K_all).T @ P[i] @ (
    #                                     A_all + B_all @ K_all))))
    #                     K_err = max(K_err, norm(K[i] + np.linalg.inv(R_i_i[i] + B[i].T @ P_ii @ B[i]) @ \
    #                                             (B[i].T @ (P[i, i * n_x:(i + 1) * n_x, :] @ (
    #                                                         A_all + B_not_i @ K_not_i[i])) + \
    #                                              R_i_not_i[i] @ K_not_i[i])))
    #                     # P_test = ( Q[i] + K_not_i[i].T @ R_not_i_not_i[i] @ K_not_i[i] + K[i].T @ R_i_i[i] @ K[i] + \
    #                     #                        K[i].T @ R_i_not_i[i] @ K_not_i[i] + (K[i].T @ R_i_not_i[i] @ K_not_i[i]).T + \
    #                     #                       (A_all + B_not_i[i] @ K_not_i[i] + B_all[:, i*n_u:(i+1)*n_u] @ K[i]).T @ P[i] @ \
    #                     #                       (A_all + B_not_i[i] @ K_not_i[i] + B_all[:, i*n_u:(i+1)*n_u] @ K[i]) )
    #                     # P_test_err = norm(P[i] - P_test)
    #                     #
    #                     # K_test_err = K_all.T@ R[i]@K_all - (K[i].T@R_i_i[i]@K[i] + \
    #                     #                                      K[i].T@R_i_not_i[i]@K_not_i[i] + K_not_i[i].T@R_i_not_i[i].T@K[i] + \
    #                     #                                      K_not_i[i].T@R_not_i_not_i[i]@K_not_i[i])
    #                 P_err_all[iter // 10] = P_err
    #                 P_evol[iter // 10] = norm(P - P_last)
    #                 if P_err < eps_error and K_err < eps_error:
    #                     break
    #         if not (P_err < eps_error and K_err < eps_error):
    #             raise RuntimeError("[solve_inf_hor_problem] Could not find a solution")
    #     else:
    #         # single agent case
    #         P[0] = solve_discrete_are(A[0], B[0], Q[0], R[0])
    #         K[0] = - np.linalg.inv(R[0] + B[0].T @ P[0] @ B[0]) @ (B[0].T @ P[0] @ A[0])
    #     return P, K

    def set_term_cost_to_inf_hor_sol(self, mode="OL", n_iter=10000, eps_error=10**(-6)):
        if mode=="OL":
            self.P, self.K = self.solve_open_loop_inf_hor_problem(n_iter=10000, eps_error=10**(-6))
        elif mode=="CL":
            self.P, self.K = self.solve_closed_loop_inf_hor_problem(n_iter=10000, eps_error=10**(-6))
        else:
            raise ValueError("[dyngames::LQ::set_term_cost_to_inf_hor_sol] mode has to be 'OL' or 'CL' ")
        # Re-create cost functions with the new terminal cost
        self.W, self.G, self.H = self.define_cost_functions()

    # TODO: adapt to coupled dynamics
    @staticmethod
    def generate_random_game(N_agents, n_states, n_inputs):
        A = np.zeros((n_states, n_states))
        B = np.zeros((N_agents, n_states, n_inputs))
        Q_tot = generate_random_monotone_matrix(N_agents, n_states)
        R_tot = generate_random_monotone_matrix(N_agents, n_inputs)
        Q = np.zeros((N_agents, N_agents * n_states, N_agents * n_states))
        R = np.zeros((N_agents, N_agents * n_inputs, N_agents * n_inputs))
        for i in range(N_agents):
            is_controllable = False
            attempt_controllable = 0
            max_norm_eig = 0
            # limit maximum norm of eigenvalues so that the prediction model does not explode
            while (not is_controllable and attempt_controllable < 10) or max_norm_eig > 1.2:
                A_single_agent = -1 + 2 * np.random.random_sample(size=[n_states, n_states])
                B_single_agent = -1 + 2 * np.random.random_sample(size=[n_states, n_inputs])
                max_norm_eig = max(np.linalg.norm(np.expand_dims(np.linalg.eigvals(A_single_agent), 1), axis=1))
                A[i, :, :] = A_single_agent
                B[i, :, :] = B_single_agent
                is_controllable = np.linalg.matrix_rank(control.ctrb(A_single_agent, B_single_agent)) == n_states
                attempt_controllable = attempt_controllable + 1
                if attempt_controllable % 10 == 0:
                    n_inputs = n_inputs + 1
            Q[i, i * n_states:(i + 1) * n_states, :] = Q_tot[i * n_states:(i + 1) * n_states, :]
            Q[i, :, :] = (Q[i, :, :] + Q[i, :, :].T) / 2
            R[i, i * n_inputs:(i + 1) * n_inputs, :] = R_tot[i * n_inputs:(i + 1) * n_inputs, :]
            R[i, :, :] = (R[i, :, :] + R[i, :, :].T) / 2

        return A, B, Q, R
