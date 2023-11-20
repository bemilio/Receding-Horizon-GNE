import torch
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import SR1 # Hessian approx.

class bestResponse():
    def __init__(self, x, J, F, A_ineq, b_ineq, A_eq, b_eq, A_shared_ineq, b_shared_ineq):
        self.N = A_ineq.shape[0]
        self.n_local_opt_var = A_ineq.shape[2]
        self.n_opt_var = self.N * self.n_local_opt_var
        n_local_ineq_const = A_ineq.shape[1]
        n_local_eq_const = A_eq.shape[1]
        n_shared_ineq_const = A_shared_ineq.shape[1]
        n_constraints = self.N * (n_local_ineq_const + n_local_eq_const + n_shared_ineq_const)
        self.A_ineq_all = torch.block_diag(*A_ineq).numpy()
        self.A_eq_all = torch.block_diag(*A_eq).numpy()
        self.A_shared_all = torch.block_diag(*A_shared_ineq).numpy()
        self.b_up_ineq_all= np.reshape(b_ineq[:,:,0].numpy(), (self.N * n_local_ineq_const))
        self.b_low_ineq_all= -np.inf*np.ones(self.N * n_local_ineq_const) #check
        self.b_up_eq_all = np.reshape(b_eq[:,:,0].numpy(), (self.N * n_local_eq_const))
        self.b_low_eq_all = np.reshape(b_eq[:,:,0].numpy(), (self.N * n_local_eq_const))
        b_shared_local = b_shared_ineq - torch.bmm(A_shared_ineq, x) #turn shared constraints into local constraints given the opponents
        self.b_up_shared_all = np.reshape(b_shared_ineq[:,:,0].numpy(), (self.N * n_shared_ineq_const))
        self.b_low_shared_all = -np.inf*np.ones(self.N * n_shared_ineq_const)
        self.x_current = x
        self.J = J
        self.F = F
        self.lin_constr = LinearConstraint( np.vstack((self.A_ineq_all, self.A_eq_all, self.A_shared_all)), \
                                            np.concatenate((self.b_low_ineq_all, self.b_low_eq_all, self.b_low_shared_all)), \
                                            np.concatenate((self.b_up_ineq_all, self.b_up_eq_all, self.b_up_shared_all)) )

    def cost(self, x):
        y=torch.zeros(self.N, self.N, self.n_local_opt_var, 1) # stack of copies of (x_i, \bar{x}_-i) where \bar{x} is the current state
        x_resh = x.reshape(self.N,self.n_local_opt_var) #each element in the first line is an agent
        costs_local = torch.zeros(self.N,1)
        for i in range(self.N):
            y[i,:,:] = self.x_current
            y[i,i,:] = torch.unsqueeze(torch.from_numpy(x_resh[i,:]), 1)
            costs_local[i] = self.J(y[i,:,:])[i]
        return torch.sum(costs_local,0).numpy()

    def gradient(self, x):
        y = torch.zeros(self.N, self.N, self.n_local_opt_var, 1)  # stack of copies of (x_i, \bar{x}_-i) where \bar{x} is the current state
        x_resh = x.reshape(self.N, self.n_local_opt_var)  # each element in the first line is an agent
        grad_local = torch.zeros(self.N, self.n_local_opt_var, 1)
        for i in range(self.N):
            y[i, :, :] = self.x_current
            y[i, i, :] = torch.unsqueeze(torch.from_numpy(x_resh[i, :]), 1)
            grad_local[i,:] = self.F(y[i,:,:])[i]
        return np.reshape(grad_local[:, :, 0].numpy(), (self.N *self.n_local_opt_var))

    def compute(self):
        x_np = self.x_current.numpy().reshape(1, self.n_opt_var).squeeze()
        res = minimize(self.cost, x_np, method='trust-constr', jac=self.gradient, hess=SR1(),
                       constraints=self.lin_constr,
                       options={'verbose': 1})
        x_sol = torch.from_numpy(res.x.reshape(self.N, self.n_local_opt_var,1 ))
        return res.fun, x_sol