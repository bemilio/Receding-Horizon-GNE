%  4-zones power system distributed control based on Venkat, Hiskens,
%  Rawlings, Wright 2008

clear all
clc
close all

addpath(genpath('../../')) % add all folders in the root folder
rmpath(genpath('../')) % remove folders of the other examples (function names are conflicting)
addpath(genpath(pwd)) % re-add the subfolders of this example


run_cl = true;
use_cl_surrogate = true;

seed = 1;
rng(seed); 
eps = 10^(-4);

N = 5; 
% state For each agent: position error, speed error. 
% %Note: the position error of the leading vehicle is constant 0 (dummy state)
n_x = 2 * N; 
n_u = 1; % acceleration
T = 10;
T_sampl = 1;
T_sim = 200;

N_tests = 1;

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

param = defineVehiclePlatooningGameParameters(N);
game = defineVehiclePlatooningGame(N, param);

[game.P_cl, game.K_cl, isInfHorStable_cl] = solveInfHorCL(game, 1000, 10^(-6));
[game.P_ol, game.K_ol, isInfHorStable_ol] = solveInfHorOL(game, 1000, 10^(-6));

[game.C_x, game.d_x, ...
    game.C_u_loc, game.d_u_loc,...
    game.C_u_sh,  game.d_u_sh,...
    game.C_u_mix, game.C_x_mix, game.d_mix] = defineConstraints(N, param);

X_f_ol = computeTerminalSetOL(game);

x_cl = zeros(n_x, 1, T_sim + 1, N_tests);
x_ol = zeros(n_x, 1, T_sim + 1, N_tests);
x_bl = zeros(n_x, 1, T_sim + 1, N_tests);

% X_f_cl = computeTerminalSetCL(game);

err_shift = zeros(N_tests,1);
%% Create an initial state in position-velocity coordinates and convert 
% % Position is relative with the first agent and meas. unit is meters
x_0_p = (0:-90:-90*(N-1))' +  [0;15*randn(N-1,1)];
x_0_v = (param.max_speed - param.min_speed)/2 + .3* param.min_speed + (.5-rand(N,1)) .* min(param.max_speed - param.min_speed);

x_0 = convertPosVelToState(x_0_p, x_0_v, param.v_des_1, param.d_des, param.headway_time);

test = 1;
%% Run tests
while test<N_tests + 1
    disp( "Test " + num2str(test) )
    % x_0 = 10*randn(n_x,1);
    % x_0(1) = 0;
    x_cl(:, :, 1, test) = x_0(:,:,test);
    x_ol(:, :, 1, test) = x_0(:,:,test);
    x_bl(:,:,1, test) = x_0(:,:,test);
        
    if isInfHorStable_ol
        game.VI_gen_OL = computeVIGenerator(game, T);
        [~, ~, A_sh, ~, ~, ~] = game.VI_gen_OL(x_0(:,:,test)); % Computed here just to get the number of shared constrains and initialize the dual
        n_sh_constraints = size(A_sh,1);
        dual_ol = zeros(n_sh_constraints, 1);
        if use_cl_surrogate
            game.VI_gen_surrogate_CL = computeVIGeneratorSurrogateCL(game,T);
            [~, ~, A_sh_cl, ~, ~, ~] = game.VI_gen_surrogate_CL(x_0(:,:,test)); % Computed here just to get the number of shared constrains and initialize the dual
            n_sh_constraints_cl = size(A_sh_cl,1);
            dual_cl = zeros(n_sh_constraints_cl,1);
        end
    end
    t_OL_assumpt_satisfied = zeros(N_tests,1);
    for t=1:T_sim
        disp("timestep = " + num2str(t));
        if t>1
            u_ol_warm_start = u_shift_ol(:,:, :, t-1);
            u_cl_warm_start = u_shift_cl(:,:, :, t-1);
        else
            u_ol_warm_start = zeros(n_u * T, 1, N);
            u_cl_warm_start = zeros(n_u * T, 1, N);
        end

        %% Solve open-loop MPC problem
        if isInfHorStable_ol
            [VI.J, VI.F, VI.A_sh, VI.b_sh, VI.A_loc, VI.b_loc, VI.n_x, VI.N]...
                = game.VI_gen_OL(x_ol(:,:,t,test));
            dual_warm_start = dual_ol;
            [u_full_traj_ol(:,:,:,t), dual_ol, res, solved(t)] = solveVICentrFB(VI, 10^6, eps, ...
                0.2, 0.2, u_ol_warm_start, dual_warm_start);
            u_full_traj_ol(:,:,:,t) = u_full_traj_ol(:,:,:,t);
            u_ol(:,:,:,t,test) = u_full_traj_ol(1:n_u,:,:,t);
            x_ol(:,:,t+1,test) = evolveState(x_ol(:,:,t,test), game.A, game.B, u_ol(:, :,:, t), 1, n_u);
            x_ol_T = evolveState(x_ol(:,:,t,test), game.A, game.B, u_full_traj_ol(:,:,:,t), T, n_u);
            inf_hor_ol_input = pagemtimes(game.K_ol,x_ol_T);
            for i=1:N
                u_shift_ol(:,:,i,t) = [u_full_traj_ol(n_u+1:end, :, i, t); inf_hor_ol_input(:,:,i)];
            end
            % if t==1
                % This is actually not enough
                % is_test_valid_ol(test) = all(all(pagemtimes(game.C_u_loc, inf_hor_ol_input) <= game.d_u_loc - eps)) ...
                %                     & all(sum(pagemtimes(game.C_u_sh, inf_hor_ol_input), 3) <= game.d_u_sh-eps) ...
                %                     & all(sum(pagemtimes(game.C_u_mix, inf_hor_ol_input), 3) + game.C_x_mix * x_ol_T <= game.d_mix-eps) ...
                %                     & all( game.C_x * x_ol_T <= game.d_x-eps);
            % end
            if checkTerminalConditionOL(x_ol_T, X_f_ol) && t_OL_assumpt_satisfied(test) == 0
                t_OL_assumpt_satisfied(test) = t;
            end

        end
        %% Solve closed-loop MPC problem 
        if isInfHorStable_cl && run_cl
            
            if use_cl_surrogate
                [VI.J, VI.F, VI.A_sh, VI.b_sh, VI.A_loc, VI.b_loc, VI.n_x, VI.N]...
                                = game.VI_gen_surrogate_CL(x_cl(:,:,t,test));
                dual_warm_start = dual_cl;
                [u_full_traj_cl(:,:,:,t), dual_cl, res, solved(t)] = solveVICentrFB(VI, 10^6, eps, ...
                    0.2, 0.2, u_cl_warm_start, dual_warm_start);
            else
                [~, ~, u_full_traj_cl(:,:,:,t)] = solveImplFinHorCL(game, T, x_cl(:,:,t,test));
            end
            u_cl(:,:,:,t,test) = u_full_traj_cl(1:n_u,:,:,t);
            x_cl(:,:,t+1,test) = evolveState(x_cl(:,:,t,test), game.A, game.B, u_cl(:, :,:, t,test), 1, n_u);
            % Retrieve last state, used for computing the shifted trajectory
            x_cl_T = evolveState(x_cl(:,:,t,test), game.A, game.B, u_full_traj_cl(:,:,:,t), T, n_u);
            % if ~X_f_cl.contains(x_cl_T)
            %     is_test_valid_cl(test) = false;
            % else
            %     is_test_valid_cl(test) = true;
            % end
            for i=1:N
                inf_hor_cl_input = game.K_cl(:,:,i) * x_cl_T;
                u_shift_cl(:,:,i,t) = [u_full_traj_cl(n_u+1:end, :, i, t); inf_hor_cl_input];
            end
            % Baseline
            %% ToDo
        end
    end 
    err_shift(test) = 0;
    for t=t_OL_assumpt_satisfied(test):T_sim-1
        for i=1:N
            err_shift(test) = max(err_shift(test), norm(u_shift_ol(:,:,i,t) - u_full_traj_ol(:,:,i, t+1)));
        end
    end
    disp("err test = " + num2str(err_shift(test)))
    test = test+1;
end

save("workspace_variables.mat", "x_ol", "x_cl", "x_bl", "u_ol", "u_cl", "u_bl")
plot_vehicle_platooning

disp( "Job complete" )

% END script



