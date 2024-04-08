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

% define inequality constraints
for i=1:game.N
    Gcx = [-min_x; max_x]; % affine part of constraints on x 
    Gcu = [-min_u; max_u]; % affine part of constraints on x 
    Gx = [-eye(n_x); eye(n_x)];
    Gu = [-eye(n_u); eye(n_u)];
    G{1,i} = [ [Gcx;Gcu], blkdiag(Gx, Gu) ]; 
    for t = 1:T
        R_all = zeros(game.N * game.n_u);
        R_all(1+(i-1)*game.n_u:i*game.n_u, 1+(i-1)*game.n_u:i*game.n_u ) = game.R(:,:,i);
        Q{t,i} = blkdiag(0, game.Q(:,:,i), R_all); % the 0-block is because the first column (?) is for the linear part of the cost;
        G{t,i} = G{1,i}; % constant constraints
        H{t,i} = zeros(0,1+game.n_x+game.n_u); % no equality constr
        F{t} = FF;
        working_set{t,i} = zeros(size(G{1,i},1),1);
    end
    Q{T+1,i} = blkdiag(0, game.P(:,:,i)); % the 0-block is because the first column (?) is for the linear part of the cost
    G{T+1,i} = [Gcx, Gx];
    H{T+1,i} = zeros(0,game.n_x+1);
    working_set{T+1,i} = zeros(size(G{1,i},1),1);
    m{i} = game.n_u;
end

[x, u] = active_set_lq_game_solver(F,H,G,Q,game.N,T,m,x_0,working_set, false, params);
u = cell2mat(u);

u_0 = u(1:game.n_u, :);

end