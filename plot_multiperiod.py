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

f = open('saved_test_result_multiperiod_0.pkl', 'rb')
# # directory = '/Users/ebenenati/surfdrive/TUDelft/Simulations/Receding_horizon_games/23_nov_22/Results/'
# n_files = 0
# filepaths = []
# for filename in os.listdir(directory):
#     filepath_candidate = os.path.join(directory, filename)
#     if os.path.isfile(filepath_candidate) and filepath_candidate.endswith('.pkl'):
#         filepaths.append(filepath_candidate)
# f = open(filepaths[0], 'rb')
# Retrieve common info
x_store, u_store, N_agents, N_random_tests, T_simulation, T_horiz_to_test, \
x_traj_store, u_traj_store, cost_store, competition_evolution, has_converged, solver_problem = pickle.load(f)

######## First batch of experiments: no dependence on timestep ############
## Test 1: If T is large enough, does the competition evolution -> 0?

fig, ax = plt.subplots(1, figsize=(5 * 1, 1.8 * 2), layout='constrained', sharex=True)
N_datapoints = N_random_tests * len(T_horiz_to_test) * len(filepaths)
list_tests = [None] *N_datapoints
list_file = [None] * N_datapoints
list_T_horiz = [None] *N_datapoints
list_timestep = [None] *N_datapoints
list_value = [None] *N_datapoints
list_has_converged = [None] * N_datapoints
index_datapoint =0

for filepath_index in range(len(filepaths)):
    f = open(filepaths[filepath_index], 'rb')
    x_store, u_store, N_agents, N_random_tests, T_simulation, T_horiz_to_test, \
    x_traj_store, u_traj_store, cost_store, competition_evolution, has_converged, solver_problem = pickle.load(f)
    f.close()
    for T_horiz in T_horiz_to_test:
        for test in range(N_random_tests):
            # Normalize by average value over all horizons (that have not diverged)
            normalizing_factor = 0
            for T_2 in T_horiz_to_test:
                if has_converged[(test, T_2)] and not solver_problem[(test, T_2)]:
                    normalizing_factor = normalizing_factor + torch.norm(competition_evolution[(test, T_2)]).item()
            list_tests[index_datapoint] = test
            list_T_horiz[index_datapoint] = T_horiz
            list_file[index_datapoint] = filepath_index
            list_has_converged[index_datapoint] = has_converged[test, T_horiz]
            if has_converged[(test, T_horiz)] and not solver_problem[(test, T_horiz)]:
                list_value[index_datapoint] = torch.norm(competition_evolution[(test, T_horiz)]).item()
                if (list_value[index_datapoint] > 0.5) and (T_horiz>10):
                    print("Pause...")
            index_datapoint = index_datapoint + 1

competition_evol_dataframe = pd.DataFrame(list(zip(list_tests, list_timestep, list_T_horiz, list_value, list_has_converged, list_file)),
                                          columns=['test', 't', 'T horizon', 'Value', 'Has converged', 'File'])
sns.boxplot(data=competition_evol_dataframe, x='T horizon', y='Value', palette="bright")
ax.set(ylabel=r'$\sum_t \|\sum_i g_i(x_{t}^{i*}, x_{t+1}^{i*}) - g_i(x_{t}^{i*}, x_{t}^{i*})\|$' )
ax.set(xlabel='T Horizon')
plt.grid()
plt.show(block=False)
fig.savefig('1_competition_v_horizon.pdf')

### Test 2: Does long horizon imply convergence?
fig, ax = plt.subplots(1, figsize=(5 * 1, 1.8 * 2), layout='constrained', sharex=True)
sns.countplot(data=competition_evol_dataframe, x='T horizon', hue='Has converged')
ax.set(ylabel=r'N. of diverging trajectories' )
ax.set(xlabel='T Horizon')
plt.grid()
plt.show(block=False)
fig.savefig('2_convergence_v_horizon.pdf')



######## Second batch of experiments WITH dependence on timestep ############

#### Test 3: Is the NI-based candidate Lyapunov an indicator of stability?
fig, ax = plt.subplots(1, figsize=(5 * 1, 1.8 * 2), layout='constrained', sharex=True)
N_datapoints = N_random_tests * len(T_horiz_to_test) * len(filepaths)
list_tests = [None] *N_datapoints
list_T_horiz = [None] *N_datapoints
list_timestep = [None] *N_datapoints
list_lyap_variat = [None] *N_datapoints
list_compet_evol = [None] *N_datapoints
list_sign_lyap_var = [None] *N_datapoints
list_has_converged = [None] * N_datapoints
list_file = [None] * N_datapoints
index_datapoint =0
for filepath_index in range(len(filepaths)):
    f = open(filepaths[filepath_index], 'rb')
    x_store, u_store, N_agents, N_random_tests, T_simulation, T_horiz_to_test, \
    x_traj_store, u_traj_store, cost_store, competition_evolution, has_converged, solver_problem = pickle.load(f)
    f.close()
    for T_horiz in T_horiz_to_test:
        for test in range(N_random_tests):
            list_tests[index_datapoint] = test
            list_T_horiz[index_datapoint] = T_horiz

            lyap_var = torch.diff(torch.sum(cost_store[(test, T_horiz)][:, 0, :], dim=0))
            list_lyap_variat[index_datapoint] = '+' if torch.any(lyap_var>0).item() else '-'
            # list_compet_evol[index_datapoint]  = torch.sum(competition_evolution[(test, T_horiz)][:, 0, t]).item()
            # list_sign_lyap_var[index_datapoint] = '+' if torch.sign(lyap_var).item()>0 else '-'
            list_has_converged[index_datapoint] = has_converged[test, T_horiz]
            list_file[index_datapoint] = int(filepath_index)
            index_datapoint = index_datapoint + 1
dependence_on_t_dataframe = pd.DataFrame(list(zip(list_tests, list_file, list_timestep,
                                                  list_T_horiz, list_lyap_variat,
                                                  list_has_converged)),
                                         columns=['test',  'File', 't', 'T horizon', 'Lyap. variation',
                                         'Has converged'])
fig, ax = plt.subplots(1, figsize=(5 * 1, 1.8 * 2), layout='constrained', sharex=True)
sns.countplot(data=dependence_on_t_dataframe, x='Has converged', hue='Lyap. variation')
ax.set(ylabel=r'N. of occurrences' )
ax.set(xlabel='Has converged?')
plt.grid()
plt.show(block=False)
fig.savefig('3_lyap_decrease_v_convergence.pdf')





############### other plots: evolution of system with weird lyapunov evolution
fig, ax = plt.subplots(3, figsize=(5 * 1, 1.8 * 3), layout='constrained', sharex=True)
critical_dataf = dependence_on_t_dataframe.loc[((dependence_on_t_dataframe['Lyap. variation']=='-') &
                               (dependence_on_t_dataframe['Has converged']==False))].head(1)
critical_filename = filepaths[int(critical_dataf['File'].item())]
critical_test = int(critical_dataf['test'].item())
critical_T_horiz = int(critical_dataf['T horizon'].item())
f = open(critical_filename, 'rb')
x_store, u_store, N_agents, N_random_tests, T_simulation, T_horiz_to_test, \
x_traj_store, u_traj_store, cost_store, competition_evolution, has_converged, solver_problem = pickle.load(f)
f.close()
traj_x_norm = np.zeros((N_agents, T_simulation))
for i in range(N_agents):
    traj_x_norm[i, :] = torch.norm(x_store[(critical_test, critical_T_horiz)][i, :, :], dim=0)
traj_u_norm = np.zeros((N_agents, T_simulation))
for i in range(N_agents):
    traj_u_norm[i, :] = torch.norm(u_store[(critical_test, critical_T_horiz)][i, :, :], dim=0)
lyap_var = torch.diff(torch.sum(cost_store[(critical_test, critical_T_horiz)][:, 0, :], dim=0))
ax[0].plot(traj_x_norm[:,:].transpose())
ax[0].set(ylabel='x norm')
ax[0].grid()
ax[1].plot(traj_u_norm[:,:].transpose())
ax[1].set(ylabel='u norm')
ax[1].grid()
ax[2].plot(lyap_var)
ax[2].set(ylabel='$\Delta$ V')
ax[2].grid()
plt.savefig('0-sample_evolution.pdf')
plt.show(block=False)

## Figure 2: sum of cost (candidate Lyap.)
fig, ax = plt.subplots(1, figsize=(5 * 1, 1.8 * 2), layout='constrained', sharex=True)
N_datapoints = N_random_tests * len(T_horiz_to_test) * T_simulation
list_tests = [None] *N_datapoints
list_T_horiz = [None] *N_datapoints
list_timestep = [None] *N_datapoints
list_value = [None] *N_datapoints
index_datapoint =0
for T_horiz in T_horiz_to_test:
    for t in range(T_simulation-1):
        for test in range(N_random_tests):
            list_tests[index_datapoint] = test
            list_T_horiz[index_datapoint] = T_horiz
            list_timestep[index_datapoint] = t
            list_value[index_datapoint] = (torch.sum(cost_store[(test, T_horiz)][:, 0, t+1]) - torch.sum(cost_store[(test, T_horiz)][:, 0, t])).item()
            index_datapoint = index_datapoint + 1

cost_dataframe = pd.DataFrame(list(zip(list_tests, list_timestep, list_T_horiz, list_value)),
                                columns=['test', 't', 'T horizon', 'Value'])
sns.lineplot(data=cost_dataframe, x='t', y='Value', hue='T horizon',palette="bright")
ax.set(ylabel='$$\sum_i J_i(x_{t+1}^*) - J_i(x_{t}^{*})$$')
ax.set(xlabel='t')
plt.ylim([-10,10])
plt.grid()
plt.savefig('1-cost.pdf')
plt.show(block=False)

## Figure 3: Competition evolution (term in candidate lyap. fun.)
fig, ax = plt.subplots(1, figsize=(5 * 1, 1.8 * 2), layout='constrained', sharex=True)
N_datapoints = N_random_tests * len(T_horiz_to_test) * T_simulation
list_tests = [None] *N_datapoints
list_T_horiz = [None] *N_datapoints
list_timestep = [None] *N_datapoints
list_value = [None] *N_datapoints
index_datapoint =0
for T_horiz in T_horiz_to_test:
    for t in range(1,T_simulation):
        for test in range(N_random_tests):
            list_tests[index_datapoint] = test
            list_T_horiz[index_datapoint] = T_horiz
            list_timestep[index_datapoint] = t
            list_value[index_datapoint] = torch.sum(competition_evolution[(test, T_horiz)][:, 0, t]).item()
            index_datapoint = index_datapoint + 1

competition_evol_dataframe = pd.DataFrame(list(zip(list_tests, list_timestep, list_T_horiz, list_value)),
                                columns=['test', 't', 'T horizon', 'Value'])
sns.lineplot(data=competition_evol_dataframe, x='t', y='Value', hue='T horizon',palette="bright")
ax.set(ylabel='$$\sum_i g_i(x_{t}^{i*}, x_{t+1}^{i*}) - g_i(x_{t}^{i*}, x_{t}^{i*}) $$')
ax.set(xlabel='t')
plt.grid()
plt.ylim([-10,10])
plt.show(block=False)
fig.savefig('2_competition.pdf')
print("Done")
