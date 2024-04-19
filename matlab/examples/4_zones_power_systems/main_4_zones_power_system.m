%  4-zones power system distributed control based on Venkat, Hiskens,
%  Rawlings, Wright 2008

clear all
clc
close all
addpath(genpath('../../')) % add all folders in the root folder
rmpath(genpath('../')) % remove folders of the other examples (function names are conflicting)
addpath(genpath(pwd)) % re-add the subfolders of this example


diary

seed = 1;
rng(seed); 

G = generateGraph();

n_x = 3 * G.numnodes + G.numedges ; % {For each node: speed of inertia,
% mechanical power, steam valve position. For each edge: power flow
n_u = 1; % reference at the governor
N = G.numnodes; % 4
T = 6;
T_sim = 50;

N_tests = 1000;

u_full_trajectory = zeros(n_u*T, N, T_sim, N_tests);
u_shifted = zeros(n_u*T, N, T_sim, N_tests);

u_cl = zeros(n_u, 1, N, T_sim);
u_full_traj_cl = zeros(n_u*T, 1, N, T_sim);
u_shift_cl = zeros(n_u*T, 1, N, T_sim);

u_ol = zeros(n_u, 1, N, T_sim);
u_full_traj_ol = zeros(n_u*T, 1, N, T_sim);
u_shift_ol = zeros(n_u*T, 1, N, T_sim);

game = definePowerSystemGame(G);

% game = generateRandomGame(n_x, n_u, N, 1, 0.1);
[game.P_cl, game.K_cl, isInfHorStable_cl] = solveInfHorCL(game, 1000, 10^(-6));
[game.P_ol, game.K_ol, isInfHorStable_ol] = solveInfHorOL(game, 1000, 10^(-6));

[game.C_x, game.d_x, game.C_u_loc, game.d_u_loc] = defineConstraints(N, n_x, n_u);

[game.C_u_sh, game.d_u_sh] = defineDummySharedInputConstraints(n_u, N);
    
x_cl = zeros(n_x, 1, T_sim + 1);
x_ol = zeros(n_x, 1, T_sim + 1);
% is_init_state_reachable = false;
x_0 = randn(n_x,1);% min_x + (diag(max_x - min_x) * rand(n_x, 1));
X_f_cl = computeTerminalSetCL(game);
x_0 = 0.1 * X_f_cl.project(x_0).x;
is_cl_solved = zeros(N_tests,1);
is_ol_solved = zeros(N_tests,1);
for test = 1:N_tests
    disp( "Test " + num2str(test) )
    % x_0 = generateReachableInitState(game, expmpc, X_f, T);
    x_cl(:, :, 1) = x_0;
    x_ol(:, :, 1) = x_0;
    
    if isInfHorStable_ol
        game.VI_generator = computeVIGenerator(game, T);
        [~, ~, A_sh, ~, ~, ~] = game.VI_generator(x_0);
        n_sh_constraints = size(A_sh,1);
        dual = zeros(n_sh_constraints, 1);
    end
    
    for t=1:T_sim
        if t>1
            u_ol_warm_start = u_shift_ol(:,:, :, t-1);
            u_cl_warm_start = u_shift_cl(:,:, :, t-1);
        else
            u_ol_warm_start = zeros(n_u * T, 1, N);
            u_cl_warm_start = zeros(n_u * T, 1, N); %unused
        end
        %% Solve open-loop MPC problem
        if isInfHorStable_ol
            [VI.J, VI.F, VI.A_sh, VI.b_sh, VI.A_loc, VI.b_loc, VI.n_x, VI.N]...
                = game.VI_generator(x_ol(:,t));
            dual_warm_start = dual;
            [u_full_traj_ol(:,:,:,t), dual, res, solved(t)] = solveVICentrFB(VI, 10^5, 10^(-4), ...
                0.001, 0.001, u_ol_warm_start, dual_warm_start);
            u_ol(:,:,:,t) = u_full_traj_ol(1:n_u,:,:,t);
            x_ol(:,:,t+1) = evolveState(x_ol(:,:,t), game.A, game.B, u_ol(:, :,:, t), 1, n_u);
            x_ol_T = evolveState(x_ol(:,:,t), game.A, game.B, u_full_traj_ol(:,:,:,t), T, n_u);
            for i=1:N
                u_shift_ol(:,:,i,t) = [u_full_traj_ol(n_u+1:end, :, i, t); game.K_ol(:,:,i) * x_ol_T ];
            end
        end
        %% Solve closed-loop MPC problem
        if isInfHorStable_cl
            [u_cl(:, :,:, t), ~, u_full_traj_cl(:,:,:,t)] = solveImplFinHorCL(game, T, x_cl(:,:,t));
            x_cl(:,:,t+1) = evolveState(x_cl(:,:,t), game.A, game.B, u_cl(:, :,:, t), 1, n_u);
            % Retrieve last state, used for computing the shifted trajectory
            x_cl_T = evolveState(x_cl(:,:,t), game.A, game.B, u_full_traj_cl(:,:,:,t), T, n_u);
            if ~X_f_cl.contains(x_cl_T)
                is_cl_solved = 0;
            else
                is_cl_solved = 1;
            end
            for i=1:N
                u_shift_cl(:,:,i,t) = [u_full_traj_cl(n_u+1:end, :, i, t); game.K_cl(:,:,i) * x_cl_T ];
            end
        end
    end 

    disp("Worker " + num2str(test) + " complete!")
%    workers_complete = workers_complete + 1;
%    disp("Workers complete: " + num2str(workers_complete) + " of " + num2str(N_tests))
end




% save("f_NE_implicit_consistency_result", "u_shifted", "u", "u_full_trajectory");
disp( "Job complete" )

% END script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


