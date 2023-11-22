clear all
clc
close all
addpath(genpath(pwd))

diary

seed = 1;
rng(seed); 

n_x = 3;
n_u = 2;
N = 3;
T = 3;
T_sim = 20;

max_x = 1;
min_x = -1;
max_u = 1;
min_u = -1;
N_tests = 2;

sparse_density=0.4;
posdef=0.5;

u = zeros(n_u, 1, N, T_sim, N_tests);
u_full_trajectory = zeros(n_u*T, 1, N, T_sim, N_tests);
u_shifted = zeros(n_u*T, 1, N, T_sim, N_tests);
workers_complete = 0;

parfor test = 1:N_tests
    disp( "Initializing worker " + num2str(test) )
    u_test = zeros(n_u, 1, N, T_sim);
    u_full_trajectory_test = zeros(n_u*T, 1, N, T_sim);
    u_shifted_test = zeros(n_u*T, 1, N, T_sim);

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
    
    for t=1:T_sim
        for i=1:N
            u_test(:, 1, i, t)  = expmpc{i}.evaluate(x(:,t));
            try
                u_full_trajectory_test(:,1,i,t) = expmpc{i}.optimizer.feval(x(:,t), 'primal');
            catch 
               disp("Value of x: " + x(:,t))
            end
        end
        x(:,:,t+1) = evolveState(x(:,:,t), game.A, game.B, u_test(:, :, :, t), 1);
        % Retrieve last state, used for computing the shifted traje ctory
        x_T = evolveState(x(:,:,t), game.A, game.B, u_full_trajectory_test(:,:,:,t), T);
        for i=1:N
            u_shifted_test(:,1,i,t) = [u_full_trajectory_test(n_u+1:end, 1, i, t); K(:,:,i) * x_T ];
        end
    end 
    u_full_trajectory(:,:,:,:, test) = u_full_trajectory_test;
    u(:,:,:,:, test) = u_test;
    u_shifted(:,:,:,:, test) = u_shifted_test;
    disp("Worker " + num2str(test) + " complete!")
%    workers_complete = workers_complete + 1;
%    disp("Workers complete: " + num2str(workers_complete) + " of " + num2str(N_tests))
end

save("f_NE_consistency_result", "u_shifted", "u", "u_full_trajectory");
disp( "Job complete" )

% END script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


