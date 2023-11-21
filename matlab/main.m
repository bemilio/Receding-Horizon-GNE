clear all
clc
close all
addpath(genpath(pwd))

n_x = 3;
n_u = 2;
N = 2;
T = 3;
T_sim = 10;

max_x = 1;
min_x = -1;
max_u = 1;
min_u = -1;

sparse_density=0.4;
posdef=0.5;
game = generateRandomGame(n_x, n_u, N, sparse_density, posdef);
[P, K, isInfHorStable] = solveInfHorCL(game, 1000, 10^(-6));

game.min_x = min_x * ones(n_x,1);
game.max_x = max_x * ones(n_x,1);
game.min_u = min_u * ones(n_u,1);
game.max_u = max_u * ones(n_u,1);

X_f = computeTerminalSetCL(game, K);

game.P = P;

expmpc = solveFinHorCL(game, T);

x = zeros(n_x, 1, T_sim + 1);
is_init_state_reachable = false;
x_0 = generateReachableInitState(game, expmpc, X_f, T);
x(:, :, 1) = x_0;

u = zeros(n_u, 1, N, T_sim);
u_full_trajectory = zeros(n_u*T, 1, N, T_sim);
u_shifted = zeros(n_u*T, 1, N, T_sim);

for t=1:T_sim
    for i=1:N
        u(:, 1, i, t) = expmpc{i}.evaluate(x(:,t));
        u_full_trajectory(:,1,i,t) = expmpc{i}.optimizer.feval(x(:,t), 'primal');
    end
    x(:,:,t+1) = evolveState(x(:,:,t), game.A, game.B, u(:, :, :, t), 1);
    % Retrieve last state, used for computing the shifted trajectory
    x_T = evolveState(x(:,:,t), game.A, game.B, u_full_trajectory(:,:,:,t), T);
    for i=1:N
        u_shifted(:,1,i,t) = [u_full_trajectory(n_u+1:end,1,i,t); K(:,:,i) * x_T ];
    end
end 



% END script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x_0 = generateReachableInitState(game, expmpc, X_f, T)
    is_init_state_reachable = false;
    
    n_x = expmpc{1}.nx;
    n_u = expmpc{1}.nu;
    N = length(expmpc);

    while ~is_init_state_reachable 
        x_0 = game.min_x + (diag(game.max_x - game.min_x) * rand(n_x, 1));
        u_full_trajectory = zeros(game.n_u*T, 1, N);
        for i=1:N
            u_full_trajectory(:,:,i) = expmpc{i}.optimizer.feval(x_0, 'primal');
        end
        x_T = evolveState(x_0, game.A, game.B, u_full_trajectory, T);
        is_init_state_reachable = X_f.contains(x_T);    
    end
end

function x_T = evolveState(x_0, A, B, u, T)
    x_T = x_0;
    n_u = size(u,1)/T;
    for tau=1:T
        x_T = A * x_T + sum(pagemtimes(B, u(1+(tau-1)*n_u:tau*n_u, :, :)), 3);
    end
end

