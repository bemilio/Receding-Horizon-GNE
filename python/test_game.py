from games.staticgames import LinearQuadratic, GenericVI
from algorithms.GNE_centralized import pFB_algorithm, FBF_algorithm
import logging
import numpy as np

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    eps = 10**(-4)
    use_test_game = True  # trigger 2-players sample zero-sum monotone game
    if use_test_game:
        print("WARNING: test game will be used.")
        logging.info("WARNING: test game will be used.")
    # parameters
    N_iter = 10000

    # game = LinearQuadratic(None, None, None, None, None, None, None, None, test=True)
    game = GenericVI(None, None, None, None, None, None, None, None, test=True)
    x_0 = np.zeros((2,1,1))
    x_0[0,0,0] = 2
    x_0[1,0,0] = 5
    alg_1 = pFB_algorithm(game, x_0=x_0, primal_stepsize=0.1, dual_stepsize=0.1)
    alg_2 = FBF_algorithm(game, x_0=x_0, primal_stepsize=0.1, dual_stepsize=0.1)

    for k in range(N_iter):
        alg_1.run_once()
        alg_2.run_once()
        if k % 10 ==0:
            x_pfb, dual, residual, cost = alg_1.get_state()
            # print("pFB: x = " + str(x_pfb), "; dual =" + str(dual), "; residual =" + str(residual))
            x_fbf, dual, residual, cost = alg_2.get_state()
            # print("FBF: x = " + str(x_fbf), "; dual =" + str(dual), "; residual =" + str(residual))
    if abs(x_pfb[0].item() - x_pfb[1].item()) < eps and x_pfb[0].item()<=0:
        print("Test centralized pFB: PASSED")
    else:
        print("Test centralized pFB: FAILED")
    if abs(x_fbf[0].item() - x_fbf[1].item()) < eps and x_fbf[0].item()<=0:
        print("Test centralized FBF: PASSED")
    else:
        print("Test centralized FBF: FAILED")