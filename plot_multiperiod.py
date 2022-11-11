import matplotlib as mpl
import seaborn as sns
import pandas as pd
from cmath import inf

mpl.interactive(True)
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
import numpy as np
import pickle
import torch

f = open('saved_test_result_multiperiod_total.pkl', 'rb')
x_store, u_store, N_agents, N_random_tests, T_simulation, T_horiz_to_test, x_traj_store,\
            u_traj_store, cost_store, competition_evolution =  pickle.load(f)
f.close()


fig, ax = plt.subplots(3, figsize=(5 * 1, 1.8 * 3), layout='constrained', sharex=True)
traj_x1 = np.zeros((N_agents, T_simulation))
for i in range(N_agents):
    traj_x1[i, :] = x_store[(0, T_horiz_to_test[-1])][i, 0, :]
traj_x2 = np.zeros((N_agents, T_simulation))
for i in range(N_agents):
    traj_x2[i, :] = x_store[(0, T_horiz_to_test[-1])][i, 1, :]
traj_u = np.zeros((N_agents, T_simulation))
for i in range(N_agents):
    traj_u[i, :] = u_store[(0, T_horiz_to_test[-1])][i, 0, :]
ax[0].plot(traj_x1[:,:].transpose())
ax[0].set(ylabel='x_1 (pos.)')
ax[0].grid()
ax[1].plot(traj_x2[:,:].transpose())
ax[1].set(ylabel='x_2 (vel.)')
ax[1].grid()
ax[2].plot(traj_u[:,:].transpose())
ax[2].set(ylabel='u (accel.)')
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
plt.show(block=False)
plt.savefig('2_competition.pdf')
print("Done")
