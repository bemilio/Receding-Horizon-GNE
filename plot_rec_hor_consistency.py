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

load_results_from_single_file = False
if load_results_from_single_file:
    f = open('results/rec_hor_consistency_result_0.pkl', 'rb')
    x_store, u_store, residual_store, u_pred_traj_store, x_pred_traj_store, u_shifted_traj_store, \
        K_CL_store, K_OL_store, A_store, B_store, \
        T_hor_to_test, N_agents_to_test, status_store = pickle.load(f)
    f.close()
else:
    # Load all files in a directory
    directory = r"C:\Users\ebenenati\surfdrive - Emilio Benenati@surfdrive.surf.nl\TUDelft\Simulations\Receding_horizon_games\02_oct_23\Results"
    N_files = 0
    for filename in os.listdir(directory):
        if filename.find('.pkl')>=0:
            N_files=N_files+1 #count all files
    N_tests=0
    for filename in os.listdir(directory):
        if filename.find('.pkl')>=0:
            f=open(directory+"\\"+filename, 'rb')
            #               x_store, u_store, residual_store, u_pred_traj_store, x_pred_traj_store, u_shifted_traj_store,
            #               K_CL_store, K_OL_store, A_store, B_store,
            #               T_hor_to_test, N_agents_to_test,
            #               cost_store, u_PMP_CL_store, u_PMP_OL_store, P_OL_store,
            #               is_game_solved_store, is_ONE_solved_store, is_P_subspace_store,
            #               is_subspace_stable_store, is_subspace_unique_store
            # x_store, u_store, residual_store, u_pred_traj_store, u_shifted_traj_store,
            # T_hor_to_test, N_agents_to_test, cost_store,
            # is_game_solved_store, is_ONE_solved_store, is_P_subspace_store,
            # is_subspace_stable_store, is_subspace_unique_store, is_ONE_equal_to_affine_LQR_store
            x_store_file, u_store_file, residual_store_file, u_pred_traj_store_file, u_shifted_traj_store_file, \
                T_hor_to_test, N_agents_to_test, cost_store_file,\
                is_game_solved_file, is_ONE_solved_file, is_P_subspace_file,\
                is_subspace_stable_file, is_subspace_unique_file, is_ONE_equal_to_affine_LQR_file = pickle.load(f)
            if N_tests == 0:
                x_store = x_store_file
                u_store = u_store_file
                residual_store = residual_store_file
                u_pred_traj_store = u_pred_traj_store_file
                u_shifted_traj_store = u_shifted_traj_store_file
                cost_store = cost_store_file
                is_game_solved_store = is_game_solved_file
                is_ONE_solved_store = is_ONE_solved_file
                is_P_subspace_store = is_P_subspace_file
                is_subspace_stable_store = is_subspace_stable_file
                is_subspace_unique_store = is_subspace_unique_file
                is_ONE_equal_to_affine_LQR_store = is_ONE_equal_to_affine_LQR_file
            else:
                for T_hor_idx in range(len(T_hor_to_test)):
                    for N_agents_idx in range(len(N_agents_to_test)):
                        x_store[T_hor_idx][N_agents_idx] = np.concatenate((x_store[T_hor_idx][N_agents_idx], x_store_file[T_hor_idx][N_agents_idx]), axis=0)
                        u_store[T_hor_idx][N_agents_idx] = np.concatenate((u_store[T_hor_idx][N_agents_idx], u_store_file[T_hor_idx][N_agents_idx]), axis=0)
                        residual_store[T_hor_idx][N_agents_idx] = np.concatenate((residual_store[T_hor_idx][N_agents_idx], residual_store_file[T_hor_idx][N_agents_idx]), axis=0)
                        u_pred_traj_store[T_hor_idx][N_agents_idx] = np.concatenate((u_pred_traj_store[T_hor_idx][N_agents_idx], u_pred_traj_store_file[T_hor_idx][N_agents_idx]), axis=0)
                        u_shifted_traj_store[T_hor_idx][N_agents_idx] = np.concatenate((u_shifted_traj_store[T_hor_idx][N_agents_idx], u_shifted_traj_store_file[T_hor_idx][N_agents_idx]), axis=0)
                        cost_store[T_hor_idx][N_agents_idx] = np.concatenate((cost_store[T_hor_idx][N_agents_idx], cost_store_file[T_hor_idx][N_agents_idx]), axis=0)
                        is_game_solved_store[T_hor_idx][N_agents_idx] = np.concatenate((is_game_solved_store[T_hor_idx][N_agents_idx], is_game_solved_file[T_hor_idx][N_agents_idx]), axis=0)
                        is_ONE_solved_store[T_hor_idx][N_agents_idx] = np.concatenate((is_ONE_solved_store[T_hor_idx][N_agents_idx], is_ONE_solved_file[T_hor_idx][N_agents_idx]), axis=0)
                        is_P_subspace_store[T_hor_idx][N_agents_idx] = np.concatenate((is_P_subspace_store[T_hor_idx][N_agents_idx], is_P_subspace_file[T_hor_idx][N_agents_idx]), axis=0)
                        is_subspace_stable_store[T_hor_idx][N_agents_idx] = np.concatenate((is_subspace_stable_store[T_hor_idx][N_agents_idx], is_subspace_stable_file[T_hor_idx][N_agents_idx]), axis=0)
                        is_subspace_unique_store[T_hor_idx][N_agents_idx] = np.concatenate((is_subspace_unique_store[T_hor_idx][N_agents_idx], is_subspace_unique_file[T_hor_idx][N_agents_idx]), axis=0)
                        is_ONE_equal_to_affine_LQR_store[T_hor_idx][N_agents_idx] = np.concatenate((is_ONE_equal_to_affine_LQR_store[T_hor_idx][N_agents_idx], is_ONE_equal_to_affine_LQR_file[T_hor_idx][N_agents_idx]), axis=0)
            N_tests_file = x_store[0][0].shape[0]
            N_tests = N_tests + N_tests_file
print("Files loaded, computing values to plot...")

# u_pred_traj_store is [ [ np.zeros((N_random_tests, N_agents, T_hor * n_u, T_sim)) for N_agents in N_agents_to_test ] for T_hor in T_hor_to_test ]
# u_store is [ [ np.zeros((N_random_tests, N_agents, n_u, T_sim)) for N_agents in N_agents_to_test ] for _ in T_hor_to_test ]
N_tested_T_hor = len(T_hor_to_test)
N_tested_N_agents = len(N_agents_to_test)
N_random_tests = u_pred_traj_store[0][0].shape[0]
n_u = u_store[0][0].shape[2]
T_sim = u_pred_traj_store[0][0].shape[3]

# x_store is (N_random_tests, n_x, T_sim)
n_x = x_store[0][0].shape[1]

eps = 10**(-4)

diff_predicted_trajectory_and_shifted = np.zeros((N_random_tests, N_tested_T_hor, N_tested_N_agents))

# Difference between predicted sequence and shifted sequence:
# \max_t \| u_{: | t+1} -  [ u_{t+1: | t}, K x_{t+T | t} \|
for i_T_hor in range(N_tested_T_hor):
    for i_N_agents in range(N_tested_N_agents):
        for i_random_test in range(N_random_tests):
            if (is_game_solved_store[i_T_hor][i_N_agents][i_random_test]):
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
            if (is_game_solved_store[i_T_hor][i_N_agents][i_random_test]):
                for t in range(T_sim-1):
                    diff_predicted_next_input_and_actual[i_random_test, i_T_hor, i_N_agents] = \
                        max(diff_predicted_next_input_and_actual[i_random_test, i_T_hor, i_N_agents],
                        norm(u_store[i_T_hor][i_N_agents][i_random_test, :, :, t+1 ] - \
                             u_pred_traj_store[i_T_hor][i_N_agents][i_random_test, :, n_u:2*n_u, t ] ) )

# verify that, if assumptions are satisfied, then ONE is found
sassano_assumptions_are_verified = np.zeros(N_tested_T_hor * N_tested_N_agents * N_random_tests, dtype=bool)
is_ONE_solved_vector = np.zeros(N_tested_T_hor * N_tested_N_agents * N_random_tests, dtype=bool)
idx=0
for i_T_hor in range(N_tested_T_hor):
    for i_N_agents in range(N_tested_N_agents):
        for i_random_test in range(N_random_tests):
            sassano_assumptions_are_verified[idx] = is_P_subspace_store[i_T_hor][i_N_agents][i_random_test] and \
                                                    is_subspace_stable_store[i_T_hor][i_N_agents][i_random_test] and \
                                                    is_subspace_unique_store[i_T_hor][i_N_agents][i_random_test]
            idx = idx+1



# Difference between first input and input computed using K OL:
# diff_first_input_and_OL = np.zeros((N_random_tests, N_tested_T_hor, N_tested_N_agents))
# for i_T_hor in range(N_tested_T_hor):
#     for i_N_agents in range(N_tested_N_agents):
#         for i_random_test in range(N_random_tests):
#             if (status_store[i_T_hor][i_N_agents][i_random_test] == 'solved'):
#                 for t in range(T_sim-1):
#                     K = K_OL_store[i_T_hor][i_N_agents][i_random_test,:]
#                     x = x_store[i_T_hor][i_N_agents][i_random_test, :, t ]
#                     diff_first_input_and_OL[i_random_test, i_T_hor, i_N_agents] = \
#                         max(diff_first_input_and_OL[i_random_test, i_T_hor, i_N_agents],
#                             norm(u_store[i_T_hor][i_N_agents][i_random_test, :, :, t+1 ] - K @ x ))

# Difference between last input and input computed using K OL:
# diff_last_input_and_OL = np.zeros((N_random_tests, N_tested_T_hor, N_tested_N_agents))
# for i_T_hor in range(N_tested_T_hor):
#     for i_N_agents in range(N_tested_N_agents):
#         for i_random_test in range(N_random_tests):
#             if (status_store[i_T_hor][i_N_agents][i_random_test] == 'solved'):
#                 for t in range(T_sim-1):
#                     K = K_OL_store[i_T_hor][i_N_agents][i_random_test,:]
#                     x_second_last = x_pred_traj_store[i_T_hor][i_N_agents][i_random_test, -2*n_x:-n_x, t ]
#                     u_last = u_pred_traj_store[i_T_hor][i_N_agents][i_random_test, :, -n_u:, t ]
#                     diff_last_input_and_OL[i_random_test, i_T_hor, i_N_agents] = \
#                         max(diff_last_input_and_OL[i_random_test, i_T_hor, i_N_agents], norm(u_last - K @ x_second_last))

# Difference between first input and input computed using K CL:

# diff_first_input_and_CL = np.zeros((N_random_tests, N_tested_T_hor, N_tested_N_agents))
# for i_T_hor in range(N_tested_T_hor):
#     for i_N_agents in range(N_tested_N_agents):
#         for i_random_test in range(N_random_tests):
#             if (status_store[i_T_hor][i_N_agents][i_random_test] == 'solved'):
#                 for t in range(T_sim-1):
#                     K = K_CL_store[i_T_hor][i_N_agents][i_random_test,:]
#                     x = x_store[i_T_hor][i_N_agents][i_random_test, :, t ]
#                     diff_first_input_and_CL[i_random_test, i_T_hor, i_N_agents] = \
#                         max(diff_first_input_and_CL[i_random_test, i_T_hor, i_N_agents],
#                             norm(u_store[i_T_hor][i_N_agents][i_random_test, :, :, t+1 ] - K @ x ))

# Difference between last input and input computed using K CL:
# diff_last_input_and_CL = np.zeros((N_random_tests, N_tested_T_hor, N_tested_N_agents))
# for i_T_hor in range(N_tested_T_hor):
#     for i_N_agents in range(N_tested_N_agents):
#         for i_random_test in range(N_random_tests):
#             if (status_store[i_T_hor][i_N_agents][i_random_test] == 'solved'):
#                 for t in range(T_sim-1):
#                     K = K_CL_store[i_T_hor][i_N_agents][i_random_test,:]
#                     x_second_last = x_pred_traj_store[i_T_hor][i_N_agents][i_random_test, -2*n_x:-n_x, t ]
#                     u_last = u_pred_traj_store[i_T_hor][i_N_agents][i_random_test, :, -n_u:, t ]
#                     diff_last_input_and_CL[i_random_test, i_T_hor, i_N_agents] = \
#                         max(diff_last_input_and_CL[i_random_test, i_T_hor, i_N_agents], norm(u_last - K @ x_second_last))

# Cumulative cost over time:
cumulative_cost = np.zeros((N_random_tests, N_tested_T_hor, N_tested_N_agents, T_sim))
for i_T_hor in range(N_tested_T_hor):
    for i_N_agents in range(N_tested_N_agents):
        for i_random_test in range(N_random_tests):
            if (is_game_solved_store[i_T_hor][i_N_agents][i_random_test]):
                for t in range(T_sim-1):
                    cumulative_cost[i_random_test, i_T_hor, i_N_agents, t] = np.sum(cost_store[i_T_hor][i_N_agents], axis=1)[i_random_test, t]

print("Values computed, plotting...")
fig, ax = plt.subplots(6, figsize=(5 * 1, 1.8 * 6), layout='constrained', sharex=True)

ax[0].boxplot(np.amax(diff_predicted_trajectory_and_shifted, axis = 2))
ax[0].set(ylabel=r'$\max_t \|{\mathbf{u}}_{t+1} -  {\mathbf{u}}^{\mathrm{shift}}_{t}\|$' )
ax[0].set(xlabel=r'horizon' )

ax[1].boxplot(np.amax(diff_predicted_next_input_and_actual, axis = 2))
ax[1].set(ylabel=r'$\max_t \| u_{t+1 | t+1} - u_{t+1 | t} \| $' )
ax[1].set(xlabel=r'horizon' )
# ax[1].set_xticks(np.arange(len(T_hor_to_test)))
# ax[1].set_xticklabels(T_hor_to_test)

# ax[2].boxplot(np.amax(diff_first_input_and_OL, axis = 2))
# ax[2].set(ylabel=r'$\max_t \| u_{t | t} - K^{\mathrm{OL}} x_{t} \| $' )
# ax[2].set(xlabel=r'horizon' )
#
# ax[3].boxplot(np.amax(diff_last_input_and_OL, axis = 2))
# ax[3].set(ylabel=r'$\max_t \| u_{t+T | t} - K^{\mathrm{OL}} x_{t+T|t} \| $' )
# ax[3].set(xlabel=r'horizon' )
#
# ax[4].boxplot(np.amax(diff_first_input_and_CL, axis = 2))
# ax[4].set(ylabel=r'$\max_t \| u_{t | t} - K^{\mathrm{CL}} x_{t} \| $' )
# ax[4].set(xlabel=r'horizon' )
#
# ax[5].boxplot(np.amax(diff_last_input_and_CL, axis = 2))
# ax[5].set(ylabel=r'$\max_t \| u_{t+T | t} - K^{\mathrm{CL}} x_{t+T|t} \| $' )
# ax[5].set(xlabel=r'horizon' )

# ax[1].set_xticks(np.arange(len(T_hor_to_test)))
ax[2].set_xticks(range(1,N_tested_T_hor+1),T_hor_to_test)
plt.savefig('results/figures/rec_hor_consistency_vs_horizon.png', dpi=600)

fig, ax = plt.subplots(4, figsize=(5 * 1, 1.8 * 4), layout='constrained', sharex=True)

ax[0].boxplot(np.amax(diff_predicted_trajectory_and_shifted, axis = 1))
ax[0].set(ylabel=r'$\max_t \|{\mathbf{u}}_{t+1} -  {\mathbf{u}}^{\mathrm{shift}}_{t}\|$' )
ax[0].set(xlabel=r'$N$ agents' )

ax[1].boxplot(np.amax(diff_predicted_next_input_and_actual, axis = 1))
ax[1].set(ylabel=r'$\max_t \| u_{t+1 | t+1} - u_{t+1 | t} \| $' )
ax[1].set(xlabel=r'$N$ agents' )
# ax[1].set_xticks(np.arange(len(T_hor_to_test)))

ax[2].boxplot(np.amax(diff_first_input_and_OL, axis = 1))
ax[2].set(ylabel=r'$\max_t \| u_{t | t} - K^{OL} x_{t} \| $' )
ax[2].set(xlabel=r'$N$ agents' )
ax[2].set_xticks(range(1,N_tested_N_agents+1),N_agents_to_test)

ax[3].boxplot(np.amax(diff_last_input_and_OL, axis = 1))
ax[3].set(ylabel=r'$\max_t \| u_{t+T | t} - K^{OL} x_{t+T|t} \| $' )
ax[3].set(xlabel=r'$N$ agents' )

ax[2].boxplot(np.amax(diff_first_input_and_CL, axis = 1))
ax[2].set(ylabel=r'$\max_t \| u_{t | t} - K^{CL} x_{t} \| $' )
ax[2].set(xlabel=r'$N$ agents' )
ax[2].set_xticks(range(1,N_tested_N_agents+1),N_agents_to_test)

ax[3].boxplot(np.amax(diff_last_input_and_CL, axis = 1))
ax[3].set(ylabel=r'$\max_t \| u_{t+T | t} - K^{CL} x_{t+T|t} \| $' )
ax[3].set(xlabel=r'$N$ agents' )


plt.savefig('results/figures/rec_hor_consistency_vs_N_agents.png', dpi=600)

### Plot costs over time
fig, ax = plt.subplots(1, figsize=(5 * 1, 1.8 * 2), layout='constrained', sharex=True)
cumulative_cost_difference = (cumulative_cost[:, :, :, 1:] - cumulative_cost[:, :, :, :-1]).reshape(-1, cumulative_cost.shape[3]-1)
plt.plot(range(T_sim-1), np.mean(cumulative_cost_difference, axis=0))
plt.fill_between(range(T_sim-1), np.amax(cumulative_cost_difference, axis = 0), np.amin(cumulative_cost_difference, axis = 0), alpha=0.2)
ax.set_xticks(range(T_sim-1))
ax.set(ylabel=r'$\sum_i J^*_{t+1}-J^*_{t}$' )
ax.set(xlabel=r't' )

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