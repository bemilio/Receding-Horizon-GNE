import logging
import warnings

import scipy.linalg
import torch
from qpth.qp import QPFunction, QPSolvers
import numpy as np
import osqp
from scipy.sparse import csc_matrix
from scipy.linalg import block_diag

class BackwardStep():
    # Proximal point operator for a quadratic cost and linear set
    # min 1/2 x'Qx + x'q + alpha/2|| x-x0 ||^2 ; x\in Ax<=b
    def __init__(self, Q, q, A_ineq, b_ineq, A_eq, b_eq, alpha=1, solver='OSQP', index_soft_constraints = None, soft_const_penalty = 1000):
        super().__init__()
        N = Q.shape[0]
        n_x = Q.shape[1]
        self.Q=[]
        self.Q = csc_matrix(alpha* block_diag(*[Q[i, :, :] for i in range(N)]) + np.eye(N*n_x) )
        self.A_ineq = csc_matrix( block_diag(*[np.vstack((A_ineq[i, :, :], A_eq[i, :, :])) for i in range(N) ]) )
        self.lower = np.vstack([np.vstack((-np.inf * np.ones( (b_ineq[i,:].shape[0], 1) ), b_eq[i, :] )) for i in range(N) ] )
        self.upper = np.vstack([np.vstack((b_ineq[i,:], b_eq[i, :])) for i in range(N) ] )
        self.q = q.reshape((-1,1))
        self.alpha = alpha # inertia

    def compute(self, x):
        x_column = x.reshape((-1,1))
        q2 = self.alpha * self.q  - x_column
        is_solved = False
        solver = osqp.OSQP()
        solver.setup(P=self.Q, q=q2, A=self.A_ineq, l=self.lower, u=self.upper, verbose=False,
                warm_start=False, max_iter=10000, eps_abs=10 ** (-8), eps_rel=10 ** (-8), eps_prim_inf=10 ** (-8),
                eps_dual_inf=10 ** (-8))
        results = solver.solve()
        if results.info.status != 'solved':
            print("[BackwardStep]: OSQP did not solve correctly, OSQP status:" + results.info.status)
            logging.info("[BackwardStep]: OSQP did not solve correctly, OSQP status:" + results.info.status)
            if results.info.status == 'maximum iterations reached' or results.info.status == 'solved inaccurate':
                # Re-attempt solution by scaling the costs, sometimes this gets OSQP to unstuck
                i_attempt = 1
                while i_attempt < 1 and results.info.status != 'solved':
                    print("[BackwardStep]: Re-trying solution, attempt:" + str(i_attempt))
                    logging.info("[BackwardStep]: Re-trying solution, attempt:" + str(i_attempt))
                    solver = osqp.OSQP()
                    solver.setup(P=(i_attempt+1) * self.Q, q=(i_attempt+1) * q2, A=self.A_ineq, l=self.lower, u=self.upper, verbose=False,
                            warm_start=True, max_iter=10000, eps_abs=10 ** (-6), eps_rel=10 ** (-6),
                            eps_prim_inf=10 ** (-6),
                            eps_dual_inf=10 ** (-6))
                    results=solver.solve()
                    i_attempt = i_attempt + 1
            if results.info.status == 'solved':
                print("[BackwardStep]: QP Solved correctly")
                logging.info("[BackwardStep]: QP Solved correctly")
        y = np.transpose(results.x).reshape(x.shape)
        return y, results.info.status