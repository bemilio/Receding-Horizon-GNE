function [game] = definePowerSystemGame(G)
%% TODO!!

% Parameters from Venkat thesis page 311
D = [ 3; 0.275; 2; 2.75]; 
R_f = [0.03; 0.07; 0.04; 0.03];
M_a = [4; 40; 35; 10];
T_ch = [5; 10; 20; 10];
T_G = [4; 25; 15; 5];
T_ij = [2.54; 1.5; 2.5];

N = G.numnodes;
E = G.numedges;
n_node_states = 3;
n_edge_states = 1;
n_x = N*n_node_states + E*n_edge_states;
n_u = 1; %generator reference point for each node

A_nodes = zeros(N * n_node_states, n_x);
A_edges = zeros(E * n_edge_states, n_x);

Incid_mat = G.incidence;

for i=1:N
    node_indexes = (i-1)*n_node_states+1: i*n_node_states;
    A_nodes(node_indexes, node_indexes) = [ -D(i)/M_a(i),       1/M_a(i),  -1/M_a(i) ;
                                            0,                 -1/T_ch(i), 1/T_ch(i) ;
                                            -1/(R_f(i)*T_G(i)), 0,         -1/T_G(i)];
    A_nodes(node_indexes, N*n_node_states +1:end) = kron(Incid_mat(i,:), [1/M_a(i);0;0]); 
end
for e=1:E
    Incid_mat_T = Incid_mat';
    A_edges(e, 1:N * n_node_states) = kron(Incid_mat_T(e,:), [T_ij(e),0,0]);
end
game.A = [A_nodes; A_edges];

game.B = zeros(n_x, n_u, N);
for i=1:N
    node_indexes = (i-1)*n_node_states+1: i*n_node_states;
    game.B(node_indexes,:,i) = [ 0; 0; 1/T_G(i)];
end


game.Q = zeros(n_x, n_x, N);
game.R = zeros(n_u, n_u, N);
% NOTE: this code needs to be modified for graphs that are not line graphs
for i=1:N
    % Agent 1 has a penalty on omega_1, the remaining agents have a penalty
    % both on omega_i and an edge power flow
    index_omega = (i-1) * n_node_states + 1;
    index_tie_line = N*n_node_states+ (i-1);
    if i==1
        game.Q(index_omega,index_omega,i) = 5; 
    else
        game.Q(index_omega,index_omega,i) = 5;
        game.Q(index_tie_line,index_tie_line,i) = 5; 
    end
    game.R(:,:,i) = 1;
end
game.n_x = n_x;
game.n_u = n_u;
game.N = N; %one agent per node

end

