import numpy as np
import osqp
from scipy.sparse import csc_matrix
from scipy.linalg import block_diag
import logging

def solve_qp(Q, q, A, l, u):
    m = osqp.OSQP()
    m.setup(csc_matrix(Q), q.squeeze(), csc_matrix(A), l, u)
    results = m.solve()
    if results.info.status != 'solved':
        print("[solve_qp]: OSQP did not solve correctly, OSQP status:" + results.info.status)
        logging.info("[solve_qp]: OSQP did not solve correctly, OSQP status:" + results.info.status)

    y = np.expand_dims(np.transpose(results.x), 1)
    return y, results.info.status