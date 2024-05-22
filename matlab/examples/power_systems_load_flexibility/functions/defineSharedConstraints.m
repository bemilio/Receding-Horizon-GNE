function [C_u_sh, d_u_sh, C_u_x0_sh] = defineSharedConstraints(T, game, p)

% sum_i  C_u_sh(:,:,i) u_i + C_u_x0_sh(:,:,i) x_0 <= d_u_sh
% In this example, cumulative net injection is within limits
% sum_i  load_i(t) + u_bi(t) - v_to_P(i) * v_g(t)   <=  max_L
% where load_i(t) = u_di(t) + load_ref(i)
% u_di(t) is the i-th deferral of consumption, load_ref is the nominal load
% u_bi(t) is the power drawn from the battery
% v_g(t) is the speed of the generator
 
n_agent_states = 3; % Generator speed, battery charge, deferral of consumption
n_x = game.N*n_agent_states;
n_u = 3; 


%% Shared input constraints
% sum_i C_u(:,:,i) * u_i(t) <= sum_i d(:,:,i)

n_constraints = 2*n_u*T; % box constraints
C_u = zeros(n_constraints, n_u, N);
C_x_0 = zeros(n_constraints, n_x);
d = zeros(n_constraints, 1, N);
for i=1:N
    C_u(:,:,i) = kron(eye(game.T),  [eye(n_u); -eye(n_u)]);
    d(:,:,i) = kron(ones(game.T,1), [p.max_u; -p.min_u]);
end

%% State constraints

%% Mixed input-state constraints

C_u_sh = zeros(2*T, T*n_u, game.N);
C_u_x0_sh = zeros(2*T, n_x, game.N);
d_u_sh = ones(2*T,1,game.N);

max_load_traslated = p.max_cum_load - sum(p.load); 
min_load_traslated = p.min_cum_load - sum(p.load); 


predmod = genPredModel(game.A,game.B,T);

for i=1:game.N
    index_ith_states = n_agent_states * (i-1) + 1 : n_agent_states * i;
    state_constr = zeros(1, n_x);
    state_constr(index_ith_states) = [-1*p.v_to_P(i), 0, 0]; % constraint matrix on the i-th agent state at time t
    C_u_x0_sh(:,:,i) = [kron(eye(T), state_constr) * predmod.T;
                        -1*kron(eye(T), state_constr) * predmod.T];
    input_constr = [0, 1, 1];
    C_u_sh(:,:,i) = [kron(eye(T), input_constr) + kron(eye(T), state_constr) * predmod.S(:,:,i);
                     -1*(kron(eye(T), input_constr) + kron(eye(T), state_constr) * predmod.S(:,:,i))];
    d_u_sh(:,:,i) = [kron(ones(T,1), max_load_traslated/game.N);
                     -1*kron(ones(T,1), min_load_traslated/game.N)];
end

end

