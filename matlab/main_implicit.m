% Solver of CL-NE with implicit solution based on Laine

clear all
clc
close all
addpath(genpath(pwd))

diary

seed = 1;
rng(seed); 

n_x = 4;
n_u = 3;
N = 5;
T = 4;
T_sim = 10;

max_x = kron(ones(n_x, 1), 0.5);
min_x = kron(ones(n_x, 1),-0.5);
max_u = kron(ones(n_u, 1),10);
min_u = kron(ones(n_u, 1),-10);
N_tests = 1;

sparse_density=0.4;
posdef=0.5;

u_full_trajectory = zeros(n_u*T, N, T_sim, N_tests);
u_shifted = zeros(n_u*T, N, T_sim, N_tests);
workers_complete = 0;

for test = 1:N_tests
    disp( "Initializing worker " + num2str(test) )
    u_cl = zeros(n_u, 1, N, T_sim);
    u_full_traj_cl = zeros(n_u*T, 1, N, T_sim);
    u_shift_cl = zeros(n_u*T, 1, N, T_sim);

    u_ol = zeros(n_u, 1, N, T_sim);
    u_full_traj_ol = zeros(n_u*T, 1, N, T_sim);
    u_shift_ol = zeros(n_u*T, 1, N, T_sim);

    game = genIntegratorsGame(n_x, n_u, N, 0.9);

    % game = generateRandomGame(n_x, n_u, N, 1, 0.1);
    [game.P_cl, game.K_cl, isInfHorStable_cl] = solveInfHorCL(game, 1000, 10^(-6));
    [game.P_ol, game.K_ol, isInfHorStable_ol] = solveInfHorOL(game, 1000, 10^(-6));
    
    [game.C_x, game.d_x, game.C_u_loc, game.d_u_loc] = defineBoxConstraints(n_x, n_u, ...
        min_x, max_x, min_u, max_u, N);

    [game.C_u_sh, game.d_u_sh] = defineDummySharedInputConstraints(n_u, N);

    % X_f = computeTerminalSetCL(game, K);
    game.VI_generator = computeVIGenerator(game, T);
        
    x_cl = zeros(n_x, 1, T_sim + 1);
    x_ol = zeros(n_x, 1, T_sim + 1);
    is_init_state_reachable = false;
    x_0 = min_x + (diag(max_x - min_x) * rand(n_x, 1));
    %x_0 = generateReachableInitState(game, expmpc, X_f, T);
    x_cl(:, :, 1) = x_0;
    x_ol(:, :, 1) = x_0;
    [~, ~, A_sh, ~, ~, ~] = game.VI_generator(x_0);
    n_sh_constraints = size(A_sh,1);
    dual = zeros(n_sh_constraints, 1);
    for t=1:T_sim
        [VI.J, VI.F, VI.A_sh, VI.b_sh, VI.A_loc, VI.b_loc, VI.n_x, VI.N]...
            = game.VI_generator(x_ol(:,t));
        if t>1
            u_ol_warm_start = u_shift_ol(:,:, :, t-1);
            u_cl_warm_start = u_shift_cl(:,:, :, t-1);
        else
            u_ol_warm_start = zeros(n_u * T, 1, N);
            u_cl_warm_start = zeros(n_u * T, 1, N); %unused
        end
        dual_warm_start = dual;
        [u_full_traj_ol(:,:,:,t), dual] = solveVICentrFB(VI, 10^5, 10^(-4), ...
            0.001, 0.001, u_ol_warm_start, dual_warm_start);
        u_ol(:,:,:,t) = u_full_traj_ol(1:n_u,:,:,t);
        x_ol(:,:,t+1) = evolveState(x_ol(:,:,t), game.A, game.B, u_ol(:, :,:, t), 1, n_u);
        [u_cl(:, :,:, t), ~, u_full_traj_cl(:,:,:,t)] = solveImplFinHorCL(game, T, x_cl(:,:,t));
        x_cl(:,:,t+1) = evolveState(x_cl(:,:,t), game.A, game.B, u_cl(:, :,:, t), 1, n_u);
        % Retrieve last state, used for computing the shifted trajectory
        x_ol_T = evolveState(x_ol(:,:,t), game.A, game.B, u_full_traj_ol(:,:,:,t), T, n_u);
        x_cl_T = evolveState(x_cl(:,:,t), game.A, game.B, u_full_traj_cl(:,:,:,t), T, n_u);
        for i=1:N
            u_shift_ol(:,:,i,t) = [u_full_traj_ol(n_u+1:end, :, i, t); game.K_ol(:,:,i) * x_ol_T ];
            u_shift_cl(:,:,i,t) = [u_full_traj_cl(n_u+1:end, :, i, t); game.K_cl(:,:,i) * x_cl_T ];
        end
    end 

    disp("Worker " + num2str(test) + " complete!")
%    workers_complete = workers_complete + 1;
%    disp("Workers complete: " + num2str(workers_complete) + " of " + num2str(N_tests))
end



save("f_NE_implicit_consistency_result", "u_shifted", "u", "u_full_trajectory");
disp( "Job complete" )

% END script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


