import warnings

import numpy as np
from numpy.linalg import norm
from operators.backwardStep import BackwardStep

class pFB_algorithm:
    def __init__(self, game, x_0=None, dual_0=None,
                 primal_stepsize=0.001, dual_stepsize=0.001):
        self.game = game
        self.primal_stepsize = primal_stepsize
        self.dual_stepsize = dual_stepsize
        N_agents = game.N_agents
        n = game.n_opt_variables
        m = game.n_shared_ineq_constr
        if x_0 is not None:
            self.x = x_0
        else:
            self.x = np.zeros((N_agents, n, 1))
        if dual_0 is not None:
            self.dual = dual_0
        else:
            self.dual = np.zeros((m, 1))
        Q = np.zeros((N_agents, n, n)) # Local cost for proximal step is zero
        q = np.zeros((N_agents, n, 1))
        self.projection = BackwardStep(Q, q, game.A_ineq_loc, game.b_ineq_loc, game.A_eq_loc, game.b_eq_loc,1)

    def run_once(self):
        x = self.x
        dual = self.dual
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        r = self.game.F.compute(x)
        x_new, status = self.projection.compute(x - self.primal_stepsize * (r + A_i.T3D() @ self.dual  ) )
        # Dual update
        d = 2 * np.sum(A_i @ x_new - A_i @ x  - b_i, axis = 0)
        dual_new = np.maximum(dual + self.dual_stepsize *  d , np.zeros(dual.shape))
        self.x = x_new
        self.dual = dual_new

    def get_state(self):
        residual = self.compute_residual()
        cost = self.game.J.compute(self.x)
        return self.x, self.dual, residual, cost

    def compute_residual(self):
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        x = self.x
        x_transformed, status = self.projection.compute(x-self.game.F.compute(x) - A_i.T3D() @ self.dual )
        dual_transformed = np.maximum(self.dual + np.sum(A_i @ self.x - b_i, 0), np.zeros(self.dual.shape))
        residual = np.sqrt( (norm(x - x_transformed))**2 + (norm(self.dual-dual_transformed))**2 )
        return residual

    def set_stepsize_using_Lip_const(self, safety_margin=.5):
        # Converges if stepsize < 2 mu, where mu is the str. mon. of pseudograd
        mu, L = self.game.F.get_strMon_Lip_constants()
        self.primal_stepsize = safety_margin * 2 * mu/(L*L)
        self.dual_stepsize = safety_margin * 2 * mu/(L*L)


class FBF_algorithm:
    def __init__(self, game, x_0=None, dual_0=None,
                 primal_stepsize=0.001, dual_stepsize=0.001):
        self.game = game
        self.primal_stepsize = primal_stepsize
        self.dual_stepsize = dual_stepsize
        N_agents = game.N_agents
        n = game.n_opt_variables
        m = game.n_shared_ineq_constr
        if x_0 is not None:
            self.x = x_0
        else:
            self.x = np.zeros((N_agents, n, 1))
        if dual_0:
            self.dual = dual_0
        else:
            self.dual = np.zeros((m, 1))
        Q = np.zeros((N_agents, n, n)) # Local cost is zero
        q = np.zeros((N_agents, n, 1))
        self.projection = BackwardStep(Q, q, game.A_ineq_loc, game.b_ineq_loc, game.A_eq_loc, game.b_eq_loc,1)

    def run_once(self):
        x = self.x
        dual = self.dual
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        r = self.game.F.compute(x)
        x_half_fbf, status = self.projection.compute(
            x - self.primal_stepsize * (r + A_i.T3D() @ self.dual))
        d = np.sum(A_i @ x - b_i, axis=0)
        dual_half_fbf = np.maximum(dual + self.dual_stepsize * d, np.zeros(dual.shape))
        r = self.game.F.compute(x_half_fbf) - self.game.F.compute(x)
        x_new, status = self.projection.compute(
            x_half_fbf - self.primal_stepsize * (r + A_i.T3D() @  (dual_half_fbf - dual)))
        d = np.sum(A_i @ (x_half_fbf - x), axis=0)
        dual_new = dual_half_fbf + self.dual_stepsize * d
        self.x = x_new
        self.dual = dual_new

    def get_state(self):
        residual = self.compute_residual()
        cost = self.game.J.compute(self.x)
        return self.x, self.dual, residual, cost

    def compute_residual(self):
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        x = self.x
        x_transformed, status = self.projection.compute(x-self.game.F.compute(x) - A_i.T3D() @ self.dual )
        dual_transformed = np.maximum(self.dual + np.sum(A_i @ self.x - b_i, axis=0), np.zeros(self.dual.shape))
        residual = np.sqrt( (norm(x - x_transformed))**2 + (norm(self.dual-dual_transformed))**2 )
        return residual

    def set_stepsize_using_Lip_const(self, safety_margin=0.5):
        #compute stepsizes as (1/L)*safety_margin, where L is the lipschitz constant of the forward operator
        #TODO
        N = self.x.shape[0]
        # n_x = self.x.shape[1]
        # n_constr = self.game.A_ineq_shared.shape[1]
        # # In the linear-quadratic game case, the forward operator is:
        # # [ 0, A', 0;
        # #   A, L, -L;
        # #   0, L,  0 ] + F(x)
        # # Where A=diag(A_i) and L is (kron(Laplacian, n_constr)).
        # A = np.zeros((N*n_x, N*n_constr))
        # for i in range(N):
        #     A[i*n_x:(i+1)*n_x, i*n_constr:(i+1)*n_constr] = self.game.A_ineq_shared[i,:,:]
        #
        # L = np.zeros((N * n_constr, N * n_constr))
        # for i in range(N):
        #     for j in range(N):
        #         L[i * n_constr:(i + 1) * n_constr, j * n_constr:(j + 1) * n_constr] = self.game.K.L[i, j, :, :]
        # H = np.vstack((np.hstack( (np.zeros((N*n_x,N*n_x)), A.T, np.zeros((N*n_x, N*n_constr)) ) ), \
        #                np.hstack( (A, L, -L) ), \
        #                np.hstack( (np.zeros((N*n_constr, N*n_x)), L, np.zeros((N*n_constr, N*n_constr))) ) ) )
        # U, S, V = np.linalg.svd(H)
        # Lip_H = np.max(S).item()
        # mu, Lip_pseudog = self.game.F.get_strMon_Lip_constants()
        # Lip_tot = Lip_H + Lip_pseudog
        # self.primal_stepsize = safety_margin/Lip_tot
        # self.dual_stepsize = safety_margin/Lip_tot
        # self.consensus_stepsize = safety_margin/Lip_tot


