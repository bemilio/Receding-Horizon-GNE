import networkx
import torch
import numpy as np
import networkx as nx

torch.set_default_dtype(torch.float64)

class multiagent_array(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def T3D(self):
        return self.transpose((0,2,1))

def batch_mult_with_coulumn_stack(Q, x):
    # Let Q be a stack of matrices, i.e. Q[i]\in\R^{nxm}
    # and x be a stack of vectors, i.e. x[i]\in\R^{m_i} s.t. \sum m_i = m.
    # This function does a batch multiplication of Q_i with col(x_i)
    return Q @ np.broadcast_to(x.reshape(-1, 1), (x.shape[0], x.shape[0] * x.shape[1], 1))

class LinearQuadratic:
    # Define game with quadratic cost and lin constraints where each agent has the same number of opt. variables
    # Parameters:
    # Q \in R^N*(N*n_x)*(N*n_x),  where Q[i,:,:] is the matrix that define the (quadratic) cost for agent i
    # q \in R^N*(N*n_x),  where q[i,:] is the affine part of the cost for agent i
    # A_loc \in R^N*n_m*n_x,  where A_shared[i,:,:] defines the local contribution to the shared ineq. constraints
    # b_loc \in R^N*n_x,  where b_shared[i,:] is the affine part of the shared ineq. constraints
    # A_shared \in R^N*n_m*n_x,  where A_shared[i,:,:] defines the local contribution to the shared ineq. constraints
    # b_shared \in R^N*n_x,  where b_shared[i,:] is the affine part of the shared ineq. constraints
    # The game is in the form:
    # \sum .5 x' Q_i x + q_i'x
    # s.t. \sum_i A_shared_i x_i <= \sum_i b_shared_i
    #             A_loc_i x_i <= b_loc_i
    def __init__(self, Q, q, A_ineq_loc, b_ineq_loc, A_eq_loc, b_eq_loc, A_ineq_shared, b_ineq_shared, communication_graph=None, test=False):
        if test:
            N, n_opt_var, Q, q, A_ineq_shared, b_ineq_shared, \
                A_ineq_loc, b_ineq_loc, A_eq_loc, b_eq_loc, communication_graph = self.setToTestGameSetup()

        Q = multiagent_array(Q)
        q = multiagent_array(q)
        self.N_agents = Q.shape[0]
        if communication_graph is None:
            communication_graph = nx.complete_graph(self.N_agents)
        self.n_opt_variables = Q.shape[1]//self.N_agents
        # Local constraints
        self.A_ineq_loc = multiagent_array(A_ineq_loc)
        self.b_ineq_loc = multiagent_array(b_ineq_loc)
        self.n_loc_ineq_constr = self.A_ineq_loc.shape[1]
        self.A_eq_loc = multiagent_array(A_eq_loc)
        self.b_eq_loc = multiagent_array(b_eq_loc)
        self.n_loc_eq_constr = self.A_ineq_loc.shape[1]
        # Shared constraints
        self.A_ineq_shared= multiagent_array(A_ineq_shared)
        self.b_ineq_shared = multiagent_array(b_ineq_shared)
        self.n_shared_ineq_constr = self.A_ineq_shared.shape[1]
        # Define the game mapping and cost
        self.F = self.GameMapping(Q, q)
        self.J = self.GameCost(Q, q)
        # self.K = self.Consensus(communication_graph) #TODO

    class GameCost():
        def __init__(self, Q, q):
            self.Q = Q
            self.q = q

        def compute(self, x):
            x_repeated = multiagent_array(np.broadcast_to(x.reshape(-1,1), (x.shape[0],x.shape[0]*x.shape[1],1))) # x_repeated[i, :, :] = col_i(x_i)
            cost = x_repeated.T3D() @ (self.Q @ x_repeated + self.q)
            return cost

    class GameMapping():
        def __init__(self, Q, q):
            N = Q.shape[0]
            n_x = Q.shape[1]//N
            aux_mat = np.kron(np.expand_dims(np.eye(N), axis = 2).transpose((0,2,1)), np.eye(n_x)) # For each agent i, selects the row of Q relative to agent i
            self.Q = aux_mat @ Q
            self.q = aux_mat @ q

        def compute(self, x):
            pgrad = batch_mult_with_coulumn_stack(self.Q, x) + self.q
            return pgrad

        def get_strMon_Lip_constants(self):
            #TODO
            raise NotImplementedError("[LinearQuadraticFullInfo::get_strMon_Lip_constants] Not implemented")

    def setToTestGameSetup(self):
        # Solutions are x_2 = x_1, x_1<=0, x_2<=0
        Q = np.zeros((2, 2, 2))
        Q[0, 0, 1] = -1
        Q[0, 1, 0] = -1
        # Q[0, 0, 0] = .01
        # Q[0, 1, 1] = .01
        Q[1, 1, 0] = 1
        Q[1, 0, 1] = 1
        # Q[1, 0, 0] = .01
        # Q[1, 1, 1] = .01
        q = np.zeros((2, 2, 1))
        A_shared_ineq = np.zeros((2, 1, 1))
        A_shared_ineq[0, 0, 0] = -1
        A_shared_ineq[1, 0, 0] = 1
        b_shared_ineq = np.zeros((2, 1, 1))
        A_loc_ineq = np.zeros((2, 1, 1))
        b_loc_ineq = np.zeros((2, 1, 1))
        A_eq_loc = np.zeros((2, 1, 1))
        b_eq_loc = np.zeros((2, 1, 1))
        n_opt_var = 1
        N = 2
        communication_graph = nx.complete_graph(2)
        return N, n_opt_var, Q, q, A_shared_ineq, b_shared_ineq, A_loc_ineq, b_loc_ineq, A_eq_loc, b_eq_loc, communication_graph

    def get_strMon_Lip_constants_eq_constraints(self):
        #TODO
        raise NotImplementedError("[LinearQuadraticFullInfo::get_strMon_Lip_constants_eq_constraints] Not implemented")
        # N=self.N_agents
        # ist_of_A_i = [self.A_eq_shared[i, :, :] for i in range(N)]
        # list_of_A_i = [self.A_eq_shared[i, :, :] for i in range(N)]
        # A = torch.column_stack(list_of_A_i)
        # A_square = torch.matmul(A, torch.transpose(A, 0, 1))
        # mu_A = torch.min(torch.linalg.eigvals(A_square).real)
        # L_A = torch.sqrt(torch.max(torch.linalg.eigvals(A_square).real))
        # return mu_A, L_A

class AggregativePartialInfo:
    # Define distributed aggregative game where each agent has the same number of opt. variables
    # Parameters:
    # Q \in R^N*n_x*n_x,  where Q[i,:,:] is the matrix that define the (quadratic) local cost
    # q \in R^N*n_x,  where q[i,:] is the affine part of the local cost
    # C \in R^N*n_s*n_x,  where C[i,:,:] is the matrix that define the local contribution to the aggregation
    # The aggregative variable is sigma = \sum (1/N) C_i x_i
    # D \in R^N*n_s*n_x,  where D[i,:,:] is the matrix that define the influence of the aggregation to the agent
    # A_shared \in R^N*n_m*n_x,  where A_shared[i,:,:] defines the local contribution to the shared eq. constraints
    # b_shared \in R^N*n_x,  where b_shared[i,:] is the affine part of the shared eq. constraints
    #### WARNING: D_iC_i should be symmetric!!
    # The game is in the form:
    # \sum .5 x_i' Q_i x_i + q_i'x_i + (1/N)(D_i x_i)'Cx
    # s.t. \sum_i A_shared_i x_i = \sum_i b_shared_i
    def __init__(self, N, communication_graph, Q, q, C, D, A_loc, b_loc, A_shared, b_shared, A_sel_positive_vars, gamma_barr=10, test=False):
        if test:
            N, n_opt_var, Q, c, Q_sel, c_sel, A_shared, b_shared, \
                A_eq_loc, A_ineq_loc, b_eq_loc, b_ineq_loc, communication_graph = self.setToTestGameSetup()
        self.N_agents = N
        self.n_opt_variables = Q.size(1)
        self.n_agg_variables = C.size(1)
        # Local constraints
        self.A_eq_loc = A_loc
        self.b_eq_loc = b_loc
        self.n_loc_eq_constr = self.A_eq_loc.size(1)
        # Shared constraints
        self.A_eq_shared= A_shared
        self.b_eq_shared = b_shared
        self.n_shared_eq_constr = self.A_eq_shared.size(1)
        # Selection matrix for variables that need be positive
        self.A_sel_positive_vars = A_sel_positive_vars
        self.gamma_barr = gamma_barr
        # Define the (nonlinear) game mapping as a torch custom activation function
        self.F = self.GameMapping(Q, q, C, D, A_sel_positive_vars, gamma_barr)
        self.J = self.GameCost(Q, q, C, D) # TODO: includ here barrier function
        # Define the consensus operator
        # self.K = self.Consensus(communication_graph, self.n_shared_eq_constr)
        # Define the adjacency operator
        self.W = self.Adjacency(communication_graph)
        # Define the operator which computes the locally-estimated aggregation
        self.S = self.Aggregation(C)

    class GameCost(torch.nn.Module):
        def __init__(self, Q, q, C, D):
            super().__init__()
            self.Q = Q
            self.q = q
            self.C = C
            self.D = D
            self.N = Q.size(0)

        def forward(self, x):
            N = self.N
            agg = torch.sum(torch.bmm(self.C, x), dim=0).unsqueeze(0).repeat(N,1,1)
            cost = torch.bmm(x.transpose(1,2), torch.bmm(self.Q, x) + self.q) + (1/N)*torch.bmm(torch.transpose(torch.bmm(self.D,x),1,2), agg)
            return cost

    class GameMapping(torch.nn.Module):
        def __init__(self, Q, q, C, D, A_sel_positive_vars, gamma_barr):
            super().__init__()
            self.Q = Q
            self.q = q
            self.C = C
            self.D = D
            self.N = Q.size(0)
            self.n_x = Q.size(1)
            self.A_sel_positive_vars = A_sel_positive_vars
            self.gamma_barr = gamma_barr

        def forward(self, x, agg=None):
            # Optional argument agg allows to provide the estimated aggregation (Partial information)
            N = self.N
            if agg is None:
                agg = torch.mean(torch.bmm(self.C, x), dim=0).unsqueeze(0).repeat(N,1,1)
            #Force positive variables via barrier function. #TODO: clean this up!
            barrier = torch.maximum( -torch.div(1,torch.bmm(self.A_sel_positive_vars, x)), -self.gamma_barr * torch.ones(x.size()))
            # F = Qx + q + (1/N)*(D_i'Cx + C_i'*D_i*x_i)
            pgrad = barrier + torch.bmm(self.Q, x) + self.q + (1 / N) * (
                        torch.bmm(torch.transpose(self.D, 1, 2), agg) + torch.bmm(torch.transpose(self.C,1,2), torch.bmm(self.D, x)))
            return pgrad

        def get_strMon_Lip_constants(self):
            # Return strong monotonicity and Lipschitz constant
            # Define the matrix that defines the pseudogradient mapping
            # F = Mx +m, where M = diag(Q_i) + diag(C_i'D_i) + col(D_i'C)
            N = self.Q.size(0)
            n_x = self.Q.size(2)
            diagonal_elements = self.Q + (1/N)*torch.bmm(torch.transpose(self.C,1,2), self.D)
            diagonal_elements_list = [diagonal_elements[i,:,:] for i in range(N)]
            Q_mat = torch.block_diag(*diagonal_elements_list)
            for i in range(N):
                for j in range(N):
                    Q_mat[i*n_x:(i+1)*n_x, j*n_x:(j+1)*n_x] = Q_mat[i*n_x:(i+1)*n_x, j*n_x:(j+1)*n_x] + \
                                                              torch.matmul(torch.transpose(self.D[i,:,:],0,1), self.C[j,:,:])
            U,S,V = torch.linalg.svd(Q_mat)
            return torch.min(S).item(), torch.max(S).item()

    class Consensus(torch.nn.Module):
        def __init__(self, communication_graph, N_dual_variables):
            super().__init__()
            # Convert Laplacian matrix to sparse tensor
            L = networkx.laplacian_matrix(communication_graph).tocoo()
            values = L.data
            rows = L.row
            cols = L.col
            indices = np.vstack((rows, cols))
            L = L.tocsr()
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            L_torch = torch.zeros(L.shape[0],L.shape[1], 1, 1)
            for i in rows:
                for j in cols:
                    L_torch[i,j,0,0] = L[i,j]
            # TODO: understand why sparse does not work
            # self.L = L_torch.to_sparse_coo()
            self.L = L_torch

        def forward(self, x):
            n_x = x.size(1)
            L_expanded = torch.kron(torch.eye(n_x).unsqueeze(0).unsqueeze(0), self.L)
            return torch.sum(torch.matmul(L_expanded, x), dim=1) # This applies the laplacian matrix to each of the dual variables

    class Adjacency(torch.nn.Module):
        def __init__(self, communication_graph):
            super().__init__()
            # Convert Laplacian matrix to sparse tensor
            W = networkx.adjacency_matrix(communication_graph).tocoo()
            values = W.data
            rows = W.row
            cols = W.col
            indices = np.vstack((rows, cols))
            W = W.tocsr()
            N=W.shape[0]
            W_torch = torch.zeros(N,N,1,1)
            for i in rows:
                for j in cols:
                    W_torch[i,j,0,0] = W[i,j]
            # TODO: understand why sparse does not work
            # self.L = L_torch.to_sparse_coo()
            self.W = W_torch

        def forward(self, x):
            n_x = x.size(1)
            W_expanded = torch.kron(torch.eye(n_x).unsqueeze(0).unsqueeze(0), self.W)
            return torch.sum(torch.matmul(W_expanded, x), dim=1) # This applies the adjacency matrix to each of the dual variables

    class Aggregation(torch.nn.Module):
        def __init__(self, C):
            super().__init__()
            self.C = C

        def forward(self, x):
            return torch.bmm(self.C,x)

    def setToTestGameSetup(self):
        raise NotImplementedError("[GameAggregativePartInfo:setToTestGameSetup] Test game not implemented")

    def get_strMon_Lip_constants_eq_constraints(self):
        N=self.N_agents
        ist_of_A_i = [self.A_eq_shared[i, :, :] for i in range(N)]
        list_of_A_i = [self.A_eq_shared[i, :, :] for i in range(N)]
        A = torch.column_stack(list_of_A_i)
        A_square = torch.matmul(A, torch.transpose(A, 0, 1))
        mu_A = torch.min(torch.linalg.eigvals(A_square).real)
        L_A = torch.sqrt(torch.max(torch.linalg.eigvals(A_square).real))
        return mu_A, L_A