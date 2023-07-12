import matplotlib as mpl
import seaborn as sns
import pandas as pd
from cmath import inf

mpl.interactive(True)
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "STIXGeneral",
    "font.serif": ["Computer Modern Roman"],
})
import numpy as np
import pickle
import torch
import os
from scipy.linalg import norm
from games.staticgames import batch_mult_with_coulumn_stack

f = open('results/rec_hor_consistency_result_0.pkl', 'rb')


# # directory = '/Users/...'
# n_files = 0
# filepaths = []
# for filename in os.listdir(directory):
#     filepath_candidate = os.path.join(directory, filename)
#     if os.path.isfile(filepath_candidate) and filepath_candidate.endswith('.pkl'):
#         filepaths.append(filepath_candidate)
# f = open(filepaths[0], 'rb')
# Retrieve common info
x_store, u_store, residual_store, u_pred_traj_store, x_pred_traj_store, u_shifted_traj_store, K_store, A_store, B_store,\
                  T_hor_to_test, N_agents_to_test = pickle.load(f)

# u_pred_traj_store is [ [ np.zeros((N_random_tests, N_agents, T_hor * n_u, T_sim)) for N_agents in N_agents_to_test ] for T_hor in T_hor_to_test ]
# u_store is [ [ np.zeros((N_random_tests, N_agents, n_u, T_sim)) for N_agents in N_agents_to_test ] for _ in T_hor_to_test ]
N_tested_T_hor = len(T_hor_to_test)
N_tested_N_agents = len(N_agents_to_test)
N_random_tests = u_pred_traj_store[0][0].shape[0]
n_u = u_store[0][0].shape[2]
T_sim = u_pred_traj_store[0][0].shape[3]

# x_store is (N_random_tests, N_agents, n_x, T_sim)
n_x = x_store[0][0].shape[2]

eps = 10**(-4)

fig, ax = plt.subplots(1, figsize=(5 * 1, 1.8 * 2), layout='constrained', sharex=True)

diff_predicted_trajectory_and_shifted = np.zeros((N_random_tests, N_tested_T_hor, N_tested_N_agents))


# Difference between predicted sequence and shifted sequence:
# \max_t \| u_{: | t+1} -  [ u_{t+1: | t}, K x_{t+T | t} \|
for i_T_hor in range(N_tested_T_hor):
    for i_N_agents in range(N_tested_N_agents):
        for i_random_test in range(N_random_tests):
            for t in range(T_sim-1):
                diff_predicted_trajectory_and_shifted[i_random_test, i_T_hor, i_N_agents] = \
                    max(diff_predicted_trajectory_and_shifted[i_random_test, i_T_hor, i_N_agents],
                    norm(u_pred_traj_store[i_T_hor][i_N_agents][i_random_test, :, :, t+1 ] - \
                         u_shifted_traj_store[i_T_hor][i_N_agents][i_random_test, :, :, t ] ) )

# Difference between predicted next input and actual input:
# \max_t \| u_{t+1 | t+1} - u_{t+1 | t} \|
diff_predicted_next_input_and_actual = np.zeros((N_random_tests, N_tested_T_hor, N_tested_N_agents))
for i_T_hor in range(N_tested_T_hor):
    for i_N_agents in range(N_tested_N_agents):
        for i_random_test in range(N_random_tests):
            for t in range(T_sim-1):
                diff_predicted_next_input_and_actual[i_random_test, i_T_hor, i_N_agents] = \
                    max(diff_predicted_next_input_and_actual[i_random_test, i_T_hor, i_N_agents],
                    norm(u_store[i_T_hor][i_N_agents][i_random_test, :, :, t+1 ] - \
                         u_pred_traj_store[i_T_hor][i_N_agents][i_random_test, :, n_u:2*n_u, t ] ) )

# Difference between first input and input computed using K:
diff_first_input_and_LQR = np.zeros((N_random_tests, N_tested_T_hor, N_tested_N_agents))
for i_T_hor in range(N_tested_T_hor):
    for i_N_agents in range(N_tested_N_agents):
        for i_random_test in range(N_random_tests):
            for t in range(T_sim-1):
                K = K_store[i_T_hor][i_N_agents][i_random_test,:]
                x = x_store[i_T_hor][i_N_agents][i_random_test, :, :, t ]
                diff_last_input_and_LQR[i_random_test, i_T_hor, i_N_agents] = \
                    max(diff_last_input_and_LQR[i_random_test, i_T_hor, i_N_agents],
                    norm(u_store[i_T_hor][i_N_agents][i_random_test, :, :, t+1 ] - \
                        batch_mult_with_coulumn_stack(K, x) ) )

# Difference between last input and input computed using K:
diff_last_input_and_LQR = np.zeros((N_random_tests, N_tested_T_hor, N_tested_N_agents))
for i_T_hor in range(N_tested_T_hor):
    for i_N_agents in range(N_tested_N_agents):
        for i_random_test in range(N_random_tests):
            for t in range(T_sim-1):
                K = K_store[i_T_hor][i_N_agents][i_random_test,:]
                x_second_last = x_pred_traj_store[i_T_hor][i_N_agents][i_random_test, :, -2*n_x:-n_x, t ]
                u_last = u_pred_traj_store[i_T_hor][i_N_agents][i_random_test, :, -n_u:, t ]
                diff_last_input_and_LQR[i_random_test, i_T_hor, i_N_agents] = \
                    max(diff_last_input_and_LQR[i_random_test, i_T_hor, i_N_agents],
                        norm(u_last- batch_mult_with_coulumn_stack(K, x_second_last) ) )



fig, ax = plt.subplots(4, figsize=(5 * 1, 1.8 * 3), layout='constrained', sharex=True)

ax[0].boxplot(np.amax(diff_predicted_trajectory_and_shifted, axis = 2))
ax[0].set(ylabel=r'$\max_t \|{\mathbf{u}}_{t+1} -  {\mathbf{u}}^{\mathrm{shift}}_{t}\|$' )
ax[0].set(xlabel=r'horizon' )

ax[1].boxplot(np.amax(diff_predicted_next_input_and_actual, axis = 2))
ax[1].set(ylabel=r'$\max_t \| u_{t+1 | t+1} - u_{t+1 | t} \| $' )
ax[1].set(xlabel=r'horizon' )
# ax[1].set_xticks(np.arange(len(T_hor_to_test)))
# ax[1].set_xticklabels(T_hor_to_test)

ax[2].boxplot(np.amax(diff_first_input_and_LQR, axis = 2))
ax[2].set(ylabel=r'$\max_t \| u_{t | t} - K x_{t} \| $' )
ax[2].set(xlabel=r'horizon' )

ax[3].boxplot(np.amax(diff_last_input_and_LQR, axis = 2))
ax[3].set(ylabel=r'$\max_t \| u_{t+T | t} - K x_{t+T|t} \| $' )
ax[3].set(xlabel=r'horizon' )

# ax[1].set_xticks(np.arange(len(T_hor_to_test)))
ax[2].set_xticks(range(1,N_tested_T_hor+1),T_hor_to_test)
plt.savefig('results/figures/rec_hor_consistency_vs_horizon.png', dpi=600)

fig, ax = plt.subplots(4, figsize=(5 * 1, 1.8 * 3), layout='constrained', sharex=True)

ax[0].boxplot(np.amax(diff_predicted_trajectory_and_shifted, axis = 1))
ax[0].set(ylabel=r'$\max_t \|{\mathbf{u}}_{t+1} -  {\mathbf{u}}^{\mathrm{shift}}_{t}\|$' )
ax[0].set(xlabel=r'$N$ agents' )

ax[1].boxplot(np.amax(diff_predicted_next_input_and_actual, axis = 1))
ax[1].set(ylabel=r'$\max_t \| u_{t+1 | t+1} - u_{t+1 | t} \| $' )
ax[1].set(xlabel=r'$N$ agents' )
# ax[1].set_xticks(np.arange(len(T_hor_to_test)))

ax[2].boxplot(np.amax(diff_first_input_and_LQR, axis = 1))
ax[2].set(ylabel=r'$\max_t \| u_{t | t} - K x_{t} \| $' )
ax[2].set(xlabel=r'$N$ agents' )
ax[2].set_xticks(range(1,N_tested_N_agents+1),N_agents_to_test)

ax[3].boxplot(np.amax(diff_last_input_and_LQR, axis = 1))
ax[3].set(ylabel=r'$\max_t \| u_{t+T | t} - K x_{t+T|t} \| $' )
ax[3].set(xlabel=r'$N$ agents' )

plt.savefig('results/figures/rec_hor_consistency_vs_N_agents.png', dpi=600)


### Plot residuals
fig, ax = plt.subplots(1, figsize=(5 * 1, 1.8 * 2), layout='constrained', sharex=True)

length_residuals = residual_store[0][0].shape[1]
all_residuals = np.zeros((N_tested_T_hor * N_tested_N_agents * N_random_tests * T_sim, length_residuals))
index = 0
for i_T_hor in range(N_tested_T_hor):
    for i_N_agents in range(N_tested_N_agents):
        for i_random_test in range(N_random_tests):
            for t in range(T_sim):
                all_residuals[index, :] = residual_store[i_T_hor][i_N_agents][i_random_test,:,t]
                index = index + 1

ax.loglog(range(1, 100*length_residuals, 100), np.mean(all_residuals, axis = 0))
ax.fill_between(range(1, 100*length_residuals, 100), np.amax(all_residuals, axis = 0), np.amin(all_residuals, axis = 0), alpha=0.2)
ax.set(ylabel=r'Residual' )
ax.set(xlabel=r'Iteration' )
ax.set_ylim([10**(-6), 10])

plt.show(block=False)

print("done")