function [u_0, x, u] = solveImplFinHorCL(game,T, x_0)

params.use_ws = false;
params.use_initialization = false;
params.linear = true;
params.open_loop = false;
params.debug_plot = false;
params.contact = false;

B = [];
for i=1:game.N
    B = [B, game.B(:,:,i)];
end
c = zeros(game.n_x, 1);
FF = [c, game.A, B];

% The next line is a bodge. This solver works for games with decoupled
% dynamics and constraints and it requires state constraints to be assigned
% to each agent. It looks like it does not work if one of the
% state constraints is assigned to an agent that cannot make it satisfied
% (even though the same constraints is assigned to other agents that indeed
% make it satisfied). We split the constraints as if the dynamics were
% decoupled, even though it is not.
n_constr_per_agent = size(game.C_x,1)/game.N; 

% define inequality constraints. If I am not mistaken by reading the
% totally messy code from Laine, the inequality constraints are collected
% in G, which is a cell array indexed in time and agent. For each t,i then
% Gti:=G{t,i} is a matrix where each line g:=Gti(p,:) is a constraint of
% the kind 
% g(1) <= (maybe >=?) g(2:)'* [x; u_i]
% except for index T+1 which is a terminal constraint and does not include
% constraints on the input.

for i=1:game.N
    % Gcx = [-min_x; max_x]; % affine part of constraints on x 
    % Gcu = [-min_u; max_u]; % affine part of constraints on x 
    % Gx = [-eye(game.n_x); eye(game.n_x)];
    % Gu = [-eye(game.n_u); eye(game.n_u)];
    constr_indexes = 1+(i-1)*n_constr_per_agent: i*n_constr_per_agent;
    G{1,i} = [ [game.d_x(constr_indexes,:); game.d_u_loc(:,:,i)] , ...
               blkdiag(-game.C_x(constr_indexes,:), -game.C_u_loc(:,:,i)) ]; 
    for t = 1:T
        R_all = zeros(game.N * game.n_u);
        R_all(1+(i-1)*game.n_u:i*game.n_u, 1+(i-1)*game.n_u:i*game.n_u ) = game.R(:,:,i);
        Q{t,i} = blkdiag(0, game.Q(:,:,i), R_all); % the 0-block is because the first column (?) is for the linear part of the cost;
        G{t,i} = G{1,i}; % constant constraints
        H{t,i} = zeros(0,1+game.n_x+game.n_u); % no equality constr
        F{t} = FF;
        working_set{t,i} = zeros(size(G{1,i},1),1);
    end
    Q{T+1,i} = blkdiag(0, game.P_cl(:,:,i)); % the 0-block is because the first column (?) is for the linear part of the cost
    G{T+1,i} = 0*[game.d_x(constr_indexes,:), -game.C_x(constr_indexes,:)]; 
    H{T+1,i} = zeros(0,game.n_x+1);
    working_set{T+1,i} = zeros(size(G{1,i},1),1);
    m{i} = game.n_u;
end

[x, u] = active_set_lq_game_solver(F,H,G,Q,game.N,T,m,x_0,working_set, false, params);
u = cell2mat(u);

u_0 = u(1:game.n_u, :);

end