import matplotlib as mpl
import seaborn as sns
import pandas as pd
from cmath import inf
from Utilities.generate_permutations import generate_permutations
from Utilities.multinomial_factor import multinomial_factor
from operator import itemgetter


mpl.interactive(True)
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
import networkx as nx
import numpy as np
import pickle
import torch
from Utilities.plot_agent_route import plot_agent_route

# f = open('/Users/ebenenati/surfdrive/TUDelft/Simulations/MDP_traffic_nonlinear/29_aug_22/saved_test_result.pkl', 'rb')
f = open('saved_test_result.pkl', 'rb')
## Data structure:
## x: Tensor with dimension (n. random tests, N agents, n variables)
## dual: Tensor with dimension (n. random tests, n constraints, )
## Residual: Tensor with dimension (n. random tests, K iterations)
x, dual, residual, cost, road_graph, edge_time_to_index, node_time_to_index, T_horiz, initial_junctions, final_destinations, \
congestion_baseline_stored, cost_baseline_stored = pickle.load(f)
f.close()

N_tests = x.size(0)
N_agents = x.size(1)
N_opt_var = x.size(2)
N_iter = residual.size(1)
N_edges = road_graph.number_of_edges()
N_vertices = road_graph.number_of_nodes()

torch.Tensor.ndim = property(lambda self: len(self.shape))  # Necessary to use matplotlib with tensors

#Plot # 0: road graph
# Toggle for multiple graphs
# fig, ax = plt.subplots(T_horiz, 2, figsize=(5, 10 * 1.4), layout='constrained')
# # fig, ax = plt.subplots(1, 1, figsize=(5, 2.6), layout='constrained')
# pos = nx.get_node_attributes(road_graph, 'pos')
#
# # Toggle for multiple grrpahs
# for t in range(T_horiz):
# # for t in range(1):
#     colors = []
#     edgelist = []
#     nodes = []
#     for edge in road_graph.edges:
#         if not edge[0] == edge[1]:
#             color_edge = np.matrix([0])
#             congestion = torch.sum(x[0, :, edge_time_to_index[(edge, t)]], 0) / N_agents
#             color_edge = congestion
#             colors.append(int(256 * color_edge))
#             edgelist.append(edge)
#     for node in road_graph.nodes:
#         nodes.append(node)
#     pos = nx.kamada_kawai_layout(road_graph)
#     plt.sca(ax[t, 0])
#     nx.draw_networkx_nodes(road_graph, pos=pos, nodelist=nodes, node_size=120, node_color='cyan')
#     nx.draw_networkx_labels(road_graph, pos)
#     # Toggle to color edges based on congestion
#     nx.draw_networkx_edges(road_graph, pos=pos, edge_color=colors, edgelist=edgelist, edge_cmap=plt.cm.cool,
#                            connectionstyle='arc3, rad = 0.1')
    # nx.draw_networkx_edges(road_graph, pos=pos, edgelist=edgelist,
    #                        connectionstyle='arc3, rad = 0.2')
# Draw baseline
# for t in range(T_horiz):
#     colors = []
#     edgelist = []
#     nodes = []
#     for edge in road_graph.edges:
#         if not edge[0] == edge[1]:
#             color_edge = np.matrix([0])
#             if (edge, t) in congestion_baseline_stored[0].keys():
#                 congestion = congestion_baseline_stored[0][(edge, t)]
#             else:
#                 congestion = 0
#             color_edge = congestion
#             colors.append(int(256 * color_edge))
#             edgelist.append(edge)
#     for node in road_graph.nodes:
#         nodes.append(node)
#     pos = nx.kamada_kawai_layout(road_graph)
#     plt.sca(ax[t, 1])
#     nx.draw_networkx_nodes(road_graph, pos=pos, nodelist=nodes, node_size=150, node_color='blue')
#     nx.draw_networkx_labels(road_graph, pos)
#     nx.draw_networkx_edges(road_graph, pos=pos, edge_color=colors, edgelist=edgelist, edge_cmap=plt.cm.cool,
#                            connectionstyle='arc3, rad = 0.1')
#
# plt.show(block=False)
#
# plt.savefig('0_graph.png')



## Plot # 1: Residuals
print("Plotting residual...")
fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained')
ax.plot(np.arange(0, N_iter * 10, 10), torch.max(residual[:, :], dim=0).values, "tab:cyan", linewidth=.5 )
ax.plot(np.arange(0, N_iter * 10, 10), torch.min(residual[:, :], dim=0).values, "tab:cyan", linewidth=.5 )
plt.fill_between(np.arange(0, N_iter * 10, 10), torch.min(residual[:, :], dim=0).values, torch.max(residual[:, :], dim=0).values, color="tab:cyan", alpha=0.3, label="Min-max residual")
ax.plot(np.arange(0, N_iter * 10, 10), torch.sum(residual[:, :], 0)/N_tests, "tab:orange", label="Average residual" )

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)
plt.savefig('1_residual.png')
plt.savefig('1_residual.pdf')

plt.show()



### Plot #2 : congestion in tests vs. shortest path

# Preliminaries: generate realization of probabilistic controller
N_vehicles_per_agent = 10**3

visited_nodes = torch.zeros((N_tests, N_agents, N_vehicles_per_agent, T_horiz+1))
count_edge_taken_at_time = {} # count how many vehicles pass by the edge at time t
for test in range(N_tests):
    for edge in road_graph.edges:
        for t in range(T_horiz):
            count_edge_taken_at_time.update({(test, edge, t): 0})

for test in range(N_tests):
    for i_agent in range(N_agents):
        visited_nodes[test, i_agent, :, 0] = initial_junctions[test][i_agent] * torch.ones((N_vehicles_per_agent, 1)).flatten()
    for t in range(T_horiz):
        for i_agent in range(N_agents):
            for vehicle in range(N_vehicles_per_agent):
                starting_node = visited_nodes[test, i_agent, vehicle, t].int().item()
                if t==0:
                    prob_starting_node=1 #probability of initial condition
                else:
                    prob_starting_node = max(x[test, i_agent, node_time_to_index[(starting_node, t)]], 0)
                if prob_starting_node < 0.00001:
                    print("Warning: a vehicle ended up in a very unlikely node")
                conditioned_prob = np.zeros(road_graph.number_of_nodes())
                for end_node in range(road_graph.number_of_nodes()):
                    if (starting_node, end_node) in road_graph.edges:
                        conditioned_prob[end_node] =  max(x[test, i_agent, edge_time_to_index[(( starting_node, end_node), t)] ], 0) / prob_starting_node
                    else:
                        conditioned_prob[end_node] = 0
                if sum(conditioned_prob) -1 >=0.0001:
                    print("Warning: conditioned prob. does not sum to 1, but to " + sum(conditioned_prob))
                conditioned_prob = conditioned_prob/sum(conditioned_prob) #necessary just for numerical tolerancies
                next_visited_node = np.random.choice(range(road_graph.number_of_nodes()), p=conditioned_prob)
                visited_nodes[test, i_agent, vehicle, t + 1] = next_visited_node
                count_edge_taken_at_time.update( {( test, (starting_node, next_visited_node), t): count_edge_taken_at_time[(test, (starting_node, next_visited_node), t)] +1 } )

## x: Tensor with dimension (n. random tests, N agents, n variables)

print("Plotting congestion comparison...")
# bar plot of maximum edge congestion compared to constraint
fig, ax = plt.subplots(figsize=(5 * 1, 3.6 * 1), layout='constrained')
congestions = torch.zeros(N_tests, N_edges)
congestion_baseline = torch.zeros(N_tests, N_edges)

for i_test in range(N_tests):
    i_edges = 0
    for edge in road_graph.edges:
        if not edge[0] == edge[1]:
            congestions[i_test, i_edges] = 0
            for t in range(T_horiz):
                # Toggle these two to change from expected value to realized value
                relative_congestion = (torch.sum(x[i_test, :, edge_time_to_index[(edge, t)]], 0) / N_agents) \
                                      / road_graph[edge[0]][edge[1]]['limit_roads']
                # relative_congestion = ( (count_edge_taken_at_time[(test, edge, t)]/N_vehicles_per_agent) / N_agents) \
                #                           / road_graph[edge[0]][edge[1]]['limit_roads']
                if relative_congestion>1.002:
                    print("Constraint not satisfied at test " + str(i_test)  )
                congestions[i_test, i_edges] = max(relative_congestion, congestions[i_test, i_edges])
                if (edge, t) in congestion_baseline_stored[i_test].keys():
                    relative_congestion_baseline = congestion_baseline_stored[i_test][(edge, t)] / \
                                                   road_graph[edge[0]][edge[1]]['limit_roads']
                    congestion_baseline[i_test, i_edges] = max(relative_congestion_baseline,
                                                               congestion_baseline[i_test, i_edges])
            i_edges = i_edges + 1
congestions = congestions[:, 0:i_edges]  # ignore self-loops
congestion_baseline = congestion_baseline[:, 0:i_edges]
congestion_dataframe = pd.DataFrame(columns=['test', 'edge', 'method', 'value'])
for i_test in range(N_tests):
    for edge in range(i_edges):
        s_row = pd.DataFrame([[i_test, edge, 'Proposed', congestions[i_test, edge].item()]],
                             columns=['test', 'edge', 'method', 'value'])
        congestion_dataframe = congestion_dataframe.append(s_row)
        s_row = pd.DataFrame([[i_test, edge, 'Baseline', congestion_baseline[i_test, edge].item()]],
                             columns=['test', 'edge', 'method', 'value'])
        congestion_dataframe = congestion_dataframe.append(s_row)

if N_tests >= 2:
    ax.axhline(y=1, linewidth=2, color='red', label="Road limit")
    ax.axhline(y= 0.1/0.2, linewidth=2, color='gold', linestyle='--', label="Free-flow limit")
    ax = sns.boxplot(x="edge", y="value", hue="method", data=congestion_dataframe, palette="muted", medianprops={'color': 'lime', 'lw': 2})
    ax.grid(True)
else:
    ax.axhline(y=1, linewidth=1, color='red', label="Road limit")
    ax.axhline(y= 0.05/0.15, linewidth=1, color='orange', linestyle='--', label="Free-flow limit")
    ax.bar([k - 0.2 for k in range(congestions.size(1))], congestions.flatten(), width=0.4, align='center',
           label="Proposed")
    ax.bar([k + 0.2 for k in range(congestion_baseline.size(1))], congestion_baseline.flatten(), width=0.4,
           align='center',
           label="Naive")

plt.legend(prop={'size': 8})
ax.set_xlabel(r'Edge')
ax.set_ylabel(r'Congestion')
ax.set_ylim([-0.1, 1.5])
plt.savefig('2_comparison_congestions.png')
plt.savefig('2_comparison_congestions.pdf')

plt.show(block=False)

### Plot #3 : traversing time in tests vs. shortest path

## Computed cost function with exact expected congestion
# Generate vectors of length N that sum to 4
# print("Computing exact cost... this will take a while")
# k_all = generate_permutations(4, N_agents)
# expected_sigma_4th = torch.zeros(N_tests, N_opt_var) # For consistency, include also the variables associated to nodes (with no congestion)
# cost_exact = torch.zeros(N_tests,N_agents)
# for i_test in range(N_tests):
#     for edge in road_graph.edges:
#         for t in range(T_horiz):
#             for i_perm in range(k_all.size(0)):
#                 k = k_all[i_perm, :]
#                 prod_p = 1
#                 factor = multinomial_factor(4,k)
#                 for index_agent in range(N_agents):
#                     if k[index_agent]>0.001 : # if the corresponding factor is non-zero
#                         if x[i_test, index_agent, edge_time_to_index[(edge, t)]].item() < 0.00001: # save on some computation if the current factor is 0
#                             prod_p = 0
#                             break
#                         else:
#                             prod_p = prod_p * x[i_test, index_agent, edge_time_to_index[(edge, t)]].item()
#                 expected_sigma_4th[i_test, edge_time_to_index[(edge, t)]] = \
#                     expected_sigma_4th[i_test, edge_time_to_index[(edge, t)]] + factor * prod_p
#
# for i_agent in range(N_agents):
#     for i_test in range(N_tests):
#         for edge in road_graph.edges:
#             for t in range(T_horiz):
#                 congestion = road_graph[edge[0]][edge[1]]['travel_time'] * ( 1 +  0.15 * expected_sigma_4th[i_test, edge_time_to_index[(edge, t)]] / \
#                                                                              ((N_agents * road_graph[edge[0]][edge[1]]['capacity'])**4))
#                 cost_partial = x[i_test, i_agent, edge_time_to_index[(edge, t)]] * congestion
#                 cost_exact[i_test, i_agent] = cost_exact[i_test, i_agent] + cost_partial
# print("Computed, plotting...")
##

fig, ax = plt.subplots(figsize=(5 * 1, 1.8 * 1), layout='constrained')

social_cost = torch.zeros(N_tests)
social_cost_baseline = torch.zeros(N_tests)
for i_test in range(N_tests):
    for edge in road_graph.edges:
        for t in range(T_horiz):
            uncontrolled_traffic_edge = road_graph[edge[0]][edge[1]]['uncontrolled_traffic']
            sigma = (count_edge_taken_at_time[(i_test, edge, t)]/N_vehicles_per_agent)/N_agents
            cost_edge = road_graph[edge[0]][edge[1]]['travel_time'] * ( 1 + 0.15 * ( (sigma + (uncontrolled_traffic_edge / N_agents) )/ \
                                                            (road_graph[edge[0]][edge[1]]['capacity']) )**4)
            if (edge, t) in congestion_baseline_stored[i_test].keys():
                sigma_baseline = congestion_baseline_stored[i_test][(edge, t)]
            else:
                sigma_baseline=0
            cost_edge_baseline = road_graph[edge[0]][edge[1]]['travel_time'] * (
                        1 + 0.15 * ((sigma_baseline+(uncontrolled_traffic_edge / N_agents) ) / \
                        (road_graph[edge[0]][edge[1]]['capacity']) )** 4)
            cost_partial = sigma * cost_edge
            cost_partial_baseline = sigma_baseline * cost_edge_baseline

            social_cost[i_test] = social_cost[i_test] + cost_partial
            social_cost_baseline[i_test] = social_cost_baseline[i_test] + cost_partial_baseline

# costs_baseline = torch.zeros(N_tests, N_agents)
# for i_test in range(N_tests):
#     costs_baseline[i_test, :] = cost_baseline_stored[i_test]
# social_cost_baseline = torch.sum(costs_baseline, 1)/ N_agents
costs_dataframe = pd.DataFrame(columns=['test', 'value'])
for i_test in range(N_tests):
    s_row = pd.DataFrame([[i_test,
                           (social_cost_baseline[i_test].item() - social_cost[i_test].item()) / social_cost_baseline[
                               i_test].item()]], columns=['test', 'value'])
    costs_dataframe = costs_dataframe.append(s_row)

ax = sns.boxplot(x="value", data=costs_dataframe, palette="Set3")
ax.set_ylim([-1, 1])
ax.grid(True)

# ax.axhline(y=0, linewidth=1, color='red')

ax.set_xlabel(r'$\frac{\sum_i J_i(x_b) - J_i(x^{\star})}{\sum_i J_i(x_b)}$ ')

plt.savefig('3_comparison_social_welfare.png')
plt.savefig('3_comparison_social_welfare.pdf')

plt.show(block=False)

### Plot #4 : expected value congestion error vs. # of agents

# Preliminaries: generate realization of probabilistic controller FOR ALL NUMBER OF VEHICLES PER AGENT
N_vehicles_per_agent = [ 10**1, 10**2, 10**3]
N_tests_sample_size = np.size(N_vehicles_per_agent)

cost_edges_expected = torch.zeros(N_tests_sample_size, N_tests, N_edges, T_horiz)
cost_edges_real = torch.zeros(N_tests_sample_size, N_tests, N_edges, T_horiz)

count_edge_taken_at_time = {} # count how many vehicles pass by the edge at time t
for index_n in range(N_tests_sample_size):
    for test in range(N_tests):
        for edge in road_graph.edges:
            for t in range(T_horiz):
                count_edge_taken_at_time.update({(index_n, test, edge, t): 0})

for index_n in range(N_tests_sample_size):
    n = N_vehicles_per_agent[index_n]
    visited_nodes = torch.zeros((N_tests, N_agents, n, T_horiz + 1))
    for i_test in range(N_tests):
        for i_agent in range(N_agents):
            visited_nodes[i_test, i_agent, :, 0] = initial_junctions[i_test][i_agent] * torch.ones((n, 1)).flatten()
        for t in range(T_horiz):
            for i_agent in range(N_agents):
                for vehicle in range(n):
                    starting_node = visited_nodes[i_test, i_agent, vehicle, t].int().item()
                    if t==0:
                        prob_starting_node=1 #probability of initial condition
                    else:
                        prob_starting_node = max(x[i_test, i_agent, node_time_to_index[(starting_node, t)]], 0)
                    if prob_starting_node < 0.00001:
                        print("Warning: a vehicle ended up in a very unlikely node")
                    conditioned_prob = np.zeros(road_graph.number_of_nodes())
                    for end_node in range(road_graph.number_of_nodes()):
                        if (starting_node, end_node) in road_graph.edges:
                            conditioned_prob[end_node] =  max(x[i_test, i_agent, edge_time_to_index[(( starting_node, end_node), t)] ], 0) / prob_starting_node
                        else:
                            conditioned_prob[end_node] = 0
                    if sum(conditioned_prob) -1 >=0.0001:
                        print("Warning: conditioned prob. does not sum to 1, but to " + str(sum(conditioned_prob)))
                    conditioned_prob = conditioned_prob/sum(conditioned_prob) #necessary just for numerical tolerancies
                    next_visited_node = np.random.choice(range(road_graph.number_of_nodes()), p=conditioned_prob)
                    visited_nodes[i_test, i_agent, vehicle, t + 1] = next_visited_node
                    count_edge_taken_at_time.update( {(index_n, i_test, (starting_node, next_visited_node), t): count_edge_taken_at_time[(index_n, i_test, (starting_node, next_visited_node), t)] +1 } )

    for i_test in range(N_tests):
        i_edges = 0
        for edge in road_graph.edges:
            for t in range(T_horiz):
                uncontrolled_traffic_edge = road_graph[edge[0]][edge[1]]['uncontrolled_traffic']
                sigma_expected = (torch.sum(x[i_test, :, edge_time_to_index[(edge, t)]], 0) / N_agents)
                sigma_real = ((count_edge_taken_at_time[
                                            (index_n, i_test, edge, t)] / n) / N_agents)
                cost_edges_expected[index_n, i_test, i_edges, t] = road_graph[edge[0]][edge[1]]['travel_time'] * (
                            1 + 0.15 * ((sigma_expected + (uncontrolled_traffic_edge/N_agents) ) ** 4) / \
                            ((road_graph[edge[0]][edge[1]]['capacity']) ** 4))
                cost_edges_real[index_n, i_test, i_edges, t] = road_graph[edge[0]][edge[1]]['travel_time'] * (
                        1 + 0.15 * ((sigma_real + (uncontrolled_traffic_edge/N_agents) ) ** 4) / \
                        (( road_graph[edge[0]][edge[1]]['capacity']) ** 4))
                if np.abs(cost_edges_expected[index_n, i_test, i_edges, t]-cost_edges_real[index_n, i_test, i_edges, t])>0.1 and index_n==2:
                    print("pause")
            i_edges = i_edges+1

cost_dataframe_comparison = pd.DataFrame(columns=['n_vehicles', 'sample'])
for index_n in range(N_tests_sample_size):
    n = N_vehicles_per_agent[index_n]
    for i_test in range(N_tests):
        for i_edge in range(N_edges):
            for t in range(T_horiz):
                if cost_edges_expected[index_n, i_test, i_edge, t].item() >= 0.01:
                    cost_diff = np.abs(cost_edges_expected[index_n, i_test, i_edge, t].item()-cost_edges_real[index_n, i_test, i_edge, t].item())
                    s_row = pd.DataFrame([[n, cost_diff ]],
                                         columns=['n_vehicles', 'sample'])
                    cost_dataframe_comparison = cost_dataframe_comparison.append(s_row)

ax = sns.boxplot(x='n_vehicles', y='sample', data=cost_dataframe_comparison, palette="muted")
ax.set_xlabel("Vehicles per agent")
ax.set_ylabel(r'$ \sum_{e\in\mathcal E, t} | \ell(\sigma^{e, t}) - \ell(\hat\sigma^{e, t})|$')
ax.set_yscale('log')
ax.set_ylim([-0.1, 5])
ax.grid(True)
plt.savefig('4_comparison_expected_congestion.png')
plt.savefig('4_comparison_expected_congestion.pdf')

plt.show(block="False")


print("Done")
