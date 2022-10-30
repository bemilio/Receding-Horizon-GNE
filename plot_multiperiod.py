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

f = open('saved_test_result_multiperiod.pkl', 'rb')
## Data structure:
## visited_nodes: dictionary with keys (index test, T_horiz) whose elements are tensors with dimension (N_agents, N_vehicles, T_horiz+1)
x_store, u_store, N_agents, T_simulation, T_horiz_to_test =  pickle.load(f)
f.close()


fig, ax = plt.subplots(3, figsize=(5 * 1, 1.8 * 2), layout='constrained', sharex=True)
traj_x1 = np.zeros((N_agents, T_simulation))
for i in range(N_agents):
    traj_x1[i, :] = x_store[(0, T_horiz_to_test[0])][i, 0, :]
traj_x2 = np.zeros((N_agents, T_simulation))
for i in range(N_agents):
    traj_x2[i, :] = x_store[(0, T_horiz_to_test[0])][i, 1, :]
traj_u = np.zeros((N_agents, T_simulation))
for i in range(N_agents):
    traj_u[i, :] = u_store[(0, T_horiz_to_test[0])][i, 0, :]
ax[0].plot(traj_x1[:,:].transpose())
ax[0].set(ylabel='x_1 (pos.)')
ax[1].plot(traj_x2[:,:].transpose())
ax[1].set(ylabel='x_2 (vel.)')
ax[2].plot(traj_u[:,:].transpose())
ax[2].set(ylabel='u (accel.)')
plt.show(block=False)


print("Done")
