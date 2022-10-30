import logging
import warnings

import scipy.linalg
import torch
from qpth.qp import QPFunction, QPSolvers
import numpy as np
import osqp
from scipy import sparse
from scipy import linalg

class BackwardStep(torch.nn.Module):
    # Proximal point operator for a quadratic cost and linear set
    # min 1/2 x'Qx + x'q + alpha/2|| x-x0 ||^2 ; x\in Ax<=b
    def __init__(self, Q, q, A_ineq, b_ineq, A_eq, b_eq, alpha=1, solver='OSQP', index_soft_constraints = None, soft_const_penalty = 1000):
        super().__init__()
        eps = 0.000001
        if solver == 'OSQP':

            Q = torch.add(alpha * Q,  torch.from_numpy(np.eye(Q.size(1))))  # batched sum
            self.Q=[]
            self.A_ineq = []
            self.b_ineq = []
            self.lower = []
            self.upper = []
            for n in range(Q.size(0)):
                if index_soft_constraints is not None:
                    n_soft_constr = index_soft_constraints.size(1)
                    self.n_soft_constraints = n_soft_constr
                    self.Q.append(
                        sparse.csr_matrix(scipy.linalg.block_diag(Q[n, :, :].numpy(), np.zeros(n_soft_constr))))
                    # With soft const in epigraphic form,
                    # A_new = [ A, [0, -I]^T;
                    #           0, -I ]
                    A_ineq_soft = np.vstack((np.hstack((A_ineq[n, :, :].numpy(), np.matrix(
                        [(k in index_soft_constraints[0, :]) * (-1.) for k in range(A_ineq.size(1))]).T)), \
                                             np.hstack(
                                                 (np.zeros((n_soft_constr, A_ineq.size(2))), -np.eye(n_soft_constr)))))
                    A_eq_soft = np.hstack((A_eq[n, :, :].numpy(), np.zeros((A_eq.size(1), n_soft_constr))))
                    self.A_ineq.append(sparse.csr_matrix(np.vstack((A_ineq_soft, A_eq_soft))))
                    self.lower.append(
                        np.vstack((-np.inf * np.ones((b_ineq.shape[1] + n_soft_constr, 1)), b_eq[n, :].numpy())))
                    self.upper.append(
                        np.vstack((b_ineq[n, :].numpy(), np.zeros((n_soft_constr, 1)), b_eq[n, :].numpy())))

                else:
                    n_soft_constr = 0
                    self.n_soft_constraints = n_soft_constr
                    self.Q.append(sparse.csr_matrix(Q[n, :, :].numpy()))
                    self.A_ineq.append(sparse.csr_matrix(np.vstack((A_ineq[n, :, :].numpy(), A_eq[n, :, :].numpy()))))
                    self.lower.append(np.vstack((-np.inf * np.ones((b_ineq.shape[1], 1)), b_eq[n, :].numpy())))
                    self.upper.append(np.vstack((b_ineq[n, :].numpy(), b_eq[n, :].numpy())))
            pad = torch.nn.ConstantPad2d((0, 0, 0, self.n_soft_constraints), soft_const_penalty)
            self.q = pad(q)
        if solver == 'QPTH':
            self.Q = torch.add(alpha *Q, torch.from_numpy(np.eye(Q.size(1))))  # batched sum
            self.q = q
            self.A_ineq = torch.cat( (torch.cat( (A_ineq, A_eq), 1) , -A_eq ), 1)
            self.b_ineq = torch.cat( (torch.cat( (b_ineq, b_eq + eps), 1), -b_eq ), 1)
            self.A_eq = torch.tensor(())
            self.b_eq = torch.tensor(())
        self.alpha = alpha # inertia

    def forward(self, x):
        pad = torch.nn.ConstantPad2d((0, 0, 0, self.n_soft_constraints), 0)
        q2 = torch.add(self.alpha * self.q,  - pad(x)) # Batched sum
        # y = QPFunction(eps=1e-6, verbose=1, maxIter=10, check_Q_spd=False)(self.Q, q2.flatten(1), self.A_ineq, self.b_ineq.flatten(1), self.A_eq, self.b_eq)
        # return y.unsqueeze(2)

        y=torch.zeros(x.size())
        flag_soft = False
        is_solved = False
        for n in range(x.size(0)):
            m = osqp.OSQP()
            q2_n = q2[n,:,:].numpy()
            # m.setup(P=self.Q[n], q=q2_n, A=self.A_ineq[n], l=self.lower[n], u=self.upper[n], verbose=False, warm_start=True, max_iter=50000, eps_abs=10**(-8), eps_rel=10**(-8))
            m.setup(P=self.Q[n], q=q2_n, A=self.A_ineq[n], l=self.lower[n], u=self.upper[n], verbose=False,
                    warm_start=True, max_iter=100000, eps_abs=10 ** (-8), eps_rel=10 ** (-8), eps_prim_inf=10 ** (-8),
                    eps_dual_inf=10 ** (-8))
            results = m.solve()
            if results.info.status != 'solved':
                print("[BackwardStep]: OSQP did not solve correctly, OSQP status:" + results.info.status)
                logging.info("[BackwardStep]: OSQP did not solve correctly, OSQP status:" + results.info.status)
                if results.info.status == 'maximum iterations reached' or results.info.status == 'solved inaccurate':
                    # Re-attempt solution by scaling the costs, sometimes this gets OSQP to unstuck
                    i_attempt = 1
                    while i_attempt < 3 and results.info.status != 'solved':
                        print("[BackwardStep]: Re-trying solution, attempt:" + str(i_attempt))
                        logging.info("[BackwardStep]: Re-trying solution, attempt:" + str(i_attempt))
                        m = osqp.OSQP()
                        m.setup(P=(i_attempt+1) * self.Q[n], q=(i_attempt+1) * q2_n, A=self.A_ineq[n], l=self.lower[n], u=self.upper[n], verbose=False,
                                warm_start=True, max_iter=100000, eps_abs=10 ** (-8), eps_rel=10 ** (-8),
                                eps_prim_inf=10 ** (-8),
                                eps_dual_inf=10 ** (-8))
                        results=m.solve()
                        if self.n_soft_constraints > 0:
                            if np.any(np.transpose(np.matrix(results.x))[-self.n_soft_constraints:, :] > 0.0001):
                                flag_soft = True

                            y[n, :, :] = torch.from_numpy(
                                np.transpose(np.matrix(results.x))[0:-self.n_soft_constraints, :])
                        else:
                            y[n, :, :] = torch.from_numpy(np.transpose(np.matrix(results.x)))
                        i_attempt = i_attempt+1
                        if results.info.status == 'solved':
                            print("[BackwardStep]: QP Solved correctly")
                            logging.info("[BackwardStep]: QP Solved correctly")


            else:
                if self.n_soft_constraints>0:
                    if np.any(np.transpose(np.matrix(results.x))[-self.n_soft_constraints:,:] >0.0001 ) :
                        flag_soft = True

                    y[n,:,:] = torch.from_numpy(np.transpose(np.matrix(results.x))[0:-self.n_soft_constraints,:] )
                else:
                    y[n,:,:] = torch.from_numpy(np.transpose(np.matrix(results.x)))
        # if flag_soft:
        #     print("[BackwardStep]: Warning: soft constraints are not satisfied, the solution is approximated")
        #     logging.info("[BackwardStep]: Warning: soft constraints are not satisfied, the solution is approximated")
        return y, results.info.status