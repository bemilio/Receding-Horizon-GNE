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
T = 13;
T_sim = 100;

N_tests = 100;

u_full_trajectory = zeros(n_u*T, N, T_sim, N_tests);
u_shifted = zeros(n_u*T, N, T_sim, N_tests);

u_cl = zeros(n_u, 1, N, T_sim,N_tests);
u_full_traj_cl = zeros(n_u*T, 1, N, T_sim);
u_shift_cl = zeros(n_u*T, 1, N, T_sim);

u_ol = zeros(n_u, 1, N, T_sim,N_tests);
u_full_traj_ol = zeros(n_u*T, 1, N, T_sim);
u_shift_ol = zeros(n_u*T, 1, N, T_sim);

% Baseline 
u_bl = zeros(n_u, 1, N, T_sim);
u_full_traj_bl = zeros(n_u*T, 1, N, T_sim);
u_shift_bl = zeros(n_u*T, 1, N, T_sim);

game = definePowerSystemGame(G);
% Baseline game used for baseline, with no terminal cost/controller
game_bl = definePowerSystemGame(G);

% game = generateRandomGame(n_x, n_u, N, 1, 0.1);
[game.P_cl, game.K_cl, isInfHorStable_cl] = solveInfHorCL(game, 1000, 10^(-6));
[game.P_ol, game.K_ol, isInfHorStable_ol] = solveInfHorOL(game, 1000, 10^(-6));

game_bl.P_cl = zeros(n_x, n_x, N);
game_bl.K_cl = zeros(n_u, n_x, N);
game_bl.P_ol = zeros(n_x, n_x, N);
game_bl.K_ol = zeros(n_u, n_x, N);

[game.C_x, game.d_x, game.C_u_loc, game.d_u_loc] = defineConstraints(N, n_x, n_u);
[game_bl.C_x, game_bl.d_x, game_bl.C_u_loc, game_bl.d_u_loc] = defineConstraints(N, n_x, n_u);

[game.C_u_sh, game.d_u_sh] = defineDummySharedInputConstraints(n_u, N);
[game_bl.C_u_sh, game_bl.d_u_sh] = defineDummySharedInputConstraints(n_u, N);

x_cl = zeros(n_x, 1, T_sim + 1, N_tests);
x_ol = zeros(n_x, 1, T_sim + 1, N_tests);
x_bl = zeros(n_x, 1, T_sim + 1, N_tests);
% is_init_state_reachable = false;
X_f_cl = computeTerminalSetCL(game);
% is_test_valid_cl = zeros(N_tests,1);
% is_test_valid_ol = zeros(N_tests,1);
err_shift = zeros(N_tests,1);
x_0 = zeros(n_x, 1, N_tests);

norms_x_0_to_test = [.1, .5, 1, 5, 10]; % P-norm of initial state relative to radius of Xf
test = 1;
while test<N_tests + 1
    disp( "Test " + num2str(test) )
    % x_0 = generateReachableInitState(game, expmpc, X_f, T);
    %x_0 = randn(n_x,1);
    r_x_0 = X_f_cl.d * norms_x_0_to_test(1+mod(test, length(norms_x_0_to_test)));
    x_0(:,:,test) = randVecWithGivenPNorm(X_f_cl.P,r_x_0);
    % x_0(:,:,test) = [kron(ones(N,1),[.01;0;0]); zeros(G.numedges,1)];
    x_cl(:, :, 1, test) = x_0(:,:,test);
    x_ol(:, :, 1, test) = x_0(:,:,test);
    x_bl(:,:,1, test) = x_0(:,:,test);
        
    if isInfHorStable_ol
        game.VI_generator = computeVIGenerator(game, T);
        [~, ~, A_sh, ~, ~, ~] = game.VI_generator(x_0);
        n_sh_constraints = size(A_sh,1);
        dual = zeros(n_sh_constraints, 1);
    end
    
    for t=1:T_sim
        if t>1
            u_ol_warm_start = u_shift_ol(:,:, :, t-1);
            % u_cl_warm_start = u_shift_cl(:,:, :, t-1);
            % u_bl_warm_start = u_shift_bl(:,:, :, t-1);
        else
            u_ol_warm_start = zeros(n_u * T, 1, N);
            % u_cl_warm_start = zeros(n_u * T, 1, N);
            % u_bl_warm_start = zeros(n_u * T, 1, N); 
        end
        %% Solve open-loop MPC problem
        if isInfHorStable_ol
            [VI.J, VI.F, VI.A_sh, VI.b_sh, VI.A_loc, VI.b_loc, VI.n_x, VI.N]...
                = game.VI_generator(x_ol(:,:,t,test));
            dual_warm_start = dual;
            [u_full_traj_ol(:,:,:,t), dual, res, solved(t)] = solveVICentrFB(VI, 10^5, 10^(-4), ...
                0.001, 0.001, u_ol_warm_start, dual_warm_start);
            u_ol(:,:,:,t,test) = u_full_traj_ol(1:n_u,:,:,t);
            x_ol(:,:,t+1,test) = evolveState(x_ol(:,:,t,test), game.A, game.B, u_ol(:, :,:, t), 1, n_u);
            x_ol_T = evolveState(x_ol(:,:,t,test), game.A, game.B, u_full_traj_ol(:,:,:,t), T, n_u);
            for i=1:N
                u_shift_ol(:,:,i,t) = [u_full_traj_ol(n_u+1:end, :, i, t); game.K_ol(:,:,i) * x_ol_T ];
            end
        end
        %% Solve closed-loop MPC problem 
        if isInfHorStable_cl 
            [u_cl(:, :,:, t,test), ~, u_full_traj_cl(:,:,:,t)] = solveImplFinHorCL(game, T, x_cl(:,:,t,test));
            x_cl(:,:,t+1,test) = evolveState(x_cl(:,:,t,test), game.A, game.B, u_cl(:, :,:, t,test), 1, n_u);
            % Retrieve last state, used for computing the shifted trajectory
            x_cl_T = evolveState(x_cl(:,:,t,test), game.A, game.B, u_full_traj_cl(:,:,:,t), T, n_u);
            % if ~X_f_cl.contains(x_cl_T)
            %     is_test_valid_cl(test) = false;
            % else
            %     is_test_valid_cl(test) = true;
            % end
            for i=1:N
                u_shift_cl(:,:,i,t) = [u_full_traj_cl(n_u+1:end, :, i, t); game.K_cl(:,:,i) * x_cl_T ];
            end
            % Baseline
            % if is_test_valid_cl(test)
            [u_bl(:, :,:, t,test), ~, u_full_traj_bl(:,:,:,t)] = solveImplFinHorCL(game_bl, T, x_bl(:,:,t,test));
            x_bl(:,:,t+1,test) = evolveState(x_bl(:,:,t,test), game_bl.A, game_bl.B, u_bl(:, :,:,t,test), 1, n_u);
            % end
        end
    end 
    % err_shift(test) = 0;
    % look for test where the baseline is unstable
    % unstable_cl(test) = max(x_cl(:,:,T_sim,test)) > 1;
    % unstable_bl(test) = max(x_bl(:,:,T_sim,test)) > 1;
    % if is_test_valid_cl(test) && ~unstable_cl(test) && unstable_bl(test)
    %     interesting_test = test;
    % end
    % if is_test_valid_cl(test)
    test = test+1;
    % end
end


% for test = 1:N_tests
%     unstable_cl(test) = max(max(x_cl(:,:,:,test))) > 10;
%     unstable_bl(test) = max(max(x_bl(:,:,:,test))) > 10;
% end
save("workspace_variables.mat", "x_ol", "x_cl", "x_bl", "u_ol", "u_cl", "u_bl", "X_f_cl", "norms_x_0_to_test")
plot_4_zones_power_system

% save("f_NE_implicit_consistency_result", "u_shifted", "u", "u_full_trajectory");
disp( "Job complete" )

% END script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


