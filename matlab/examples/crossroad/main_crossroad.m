clear all
clc
close all

addpath(genpath('../../')) % add all folders in the root folder
rmpath(genpath('../')) % remove folders of the other examples (function names are conflicting)
addpath(genpath(pwd)) % re-add the subfolders of this example


run_cl = false;

seed = 1;
rng(seed); 
eps = 10^(-4);

N = 15; 
% state For each agent: position error, speed error. 
% %Note: the position error of the leading vehicle is constant 0 (dummy state)
n_x = 2 * N; 
n_u = 1; % acceleration
T = 10;
T_sim = 400;

N_tests = 1;

u_full_trajectory = zeros(n_u*T, N, T_sim, N_tests);
u_shifted = zeros(n_u*T, N, T_sim, N_tests);

u_ol = zeros(n_u, 1, N, T_sim,N_tests);
u_full_traj_ol = zeros(n_u*T, 1, N, T_sim);
u_shift_ol = zeros(n_u*T, 1, N, T_sim);

param = defineCrossroadGameParameters(N);
game = defineCrossroadGame(N, param);

[game.P_ol, game.K_ol, isInfHorStable_ol] = solveInfHorOL(game, 1000, 10^(-6));

% [game.C_x, game.d_x, ...
%     game.C_u_loc, game.d_u_loc,...
%     game.C_u_sh,  game.d_u_sh,...
%     game.C_u_mix, game.C_x_mix, game.d_mix] = defineConstraints(N, param);

X_f_ol = computeTerminalSetOL(game);

x_ol = zeros(n_x, 1, T_sim + 1, N_tests);

err_shift = zeros(N_tests,1);
%% Create an initial state in position-velocity coordinates and convert 
% % Position is relative with the first agent and meas. unit is meters
x_0_p(1) = 0;
for i=2:N
    % initialize the agents to be distant d_min plus something
    x_0_p(i) =  x_0_p(i-1) - param.d_min(1) - .2*param.d_min(1)*rand;
end
x_0_v = param.min_speed ...
      + (rand(N,1)) .*min(param.max_speed - param.min_speed);

x_0 = convertPosVelToState(x_0_p, x_0_v, param.v_des_1, param.d_des, param.headway_time, param.crossing_dir, param.conflicts_dict);

test = 1;
%% Run tests
while test<N_tests + 1
    disp( "Test " + num2str(test) )
    x_ol(:, :, 1, test) = x_0(:,:,test);
    if isInfHorStable_ol
        game.VI_generator = computeVIGenerator(game, T);
        [~, ~, A_sh, ~, ~, ~] = game.VI_generator(x_0(:,:,test)); % Computed here just to get the number of shared constrains and initialize the dual
        n_sh_constraints = size(A_sh,1);
        dual = zeros(n_sh_constraints, 1);
    end
    t_OL_assumpt_satisfied = zeros(N_tests,1);
    for t=1:T_sim
        disp("timestep = " + num2str(t));
        if t>1
            u_ol_warm_start = u_shift_ol(:,:, :, t-1);
        else
            u_ol_warm_start = zeros(n_u * T, 1, N);
        end

        %% Solve open-loop MPC problem
        if isInfHorStable_ol
            [VI.F, VI.A_sh, VI.b_sh, VI.A_loc, VI.b_loc, VI.n_x, VI.N, VI.Q, VI.q]...
                = game.VI_generator(x_ol(:,:,t,test));
            dual_warm_start = dual;
            [u_full_traj_ol(:,:,:,t), res, solved(t)] = solveVICentrDR(VI, 10^6, eps, ...
                0.5, eye(VI.N * VI.n_x), u_ol_warm_start);
            u_full_traj_ol(:,:,:,t) = u_full_traj_ol(:,:,:,t);
            u_ol(:,:,:,t,test) = u_full_traj_ol(1:n_u,:,:,t);
            x_ol(:,:,t+1,test) = evolveState(x_ol(:,:,t,test), game.A, game.B, u_ol(:, :,:, t), 1, n_u);
            x_ol_T = evolveState(x_ol(:,:,t,test), game.A, game.B, u_full_traj_ol(:,:,:,t), T, n_u);
            inf_hor_ol_input = pagemtimes(game.K_ol,x_ol_T);
            for i=1:N
                u_shift_ol(:,:,i,t) = [u_full_traj_ol(n_u+1:end, :, i, t); inf_hor_ol_input(:,:,i)];
            end
        end
    end 
    test = test+1;
end

save("workspace_variables.mat", "x_ol", "u_ol", "param")
plot_crossroad

disp( "Job complete" )

% END script



