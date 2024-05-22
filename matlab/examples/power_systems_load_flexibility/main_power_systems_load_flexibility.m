%  4-zones power system distributed control based on Venkat, Hiskens,
%  Rawlings, Wright 2008

clear all
clc
close all

addpath(genpath('../../')) % add all folders in the root folder
rmpath(genpath('../')) % remove folders of the other examples (function names are conflicting)
addpath(genpath(pwd)) % re-add the subfolders of this example

seed = 1;
rng(seed); 

N = 2;

run_cl = false;

n_x = 3 * N ; % {For each agent: Generator speed, battery charge state, deferral of consumption
n_u = 3; % ramp-up of generator, injected battery power, instantenuous load curbing
T = 5;
T_sim = 20;

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

% Loads 
load = 10*ones(N,1);
collective_load_ref = .5 * sum(load);

param = defineLoadFlexGameParameters(N, load, collective_load_ref);

game = defineLoadFlexGame(N, param);

[game.P_cl, game.K_cl, isInfHorStable_cl] = solveInfHorCL(game, 1000, 10^(-6));
[game.P_ol, game.K_ol, isInfHorStable_ol] = solveInfHorOL(game, 1000, 10^(-6));

[game.C_x, game.d_x, ...
    game.C_u_loc, game.d_u_loc,...
    game.C_u_sh, game.d_u_sh,...
    game.C_u_mix, game.C_x_mix, game.d_mix,...
    game.offset_x, game.offset_u] = defineConstraints(N, param, game);

x_cl = zeros(n_x, 1, T_sim + 1, N_tests);
x_ol = zeros(n_x, 1, T_sim + 1, N_tests);
x_bl = zeros(n_x, 1, T_sim + 1, N_tests);

% X_f_cl = computeTerminalSetCL(game);

err_shift = zeros(N_tests,1);
x_0 = zeros(n_x, 1, N_tests);

norms_x_0_to_test = [10]; % P-norm of initial state relative to radius of Xf
test = 1;
while test<N_tests + 1
    disp( "Test " + num2str(test) )
    %x_0 = randn(n_x,1);
    x_0(:,:,test) = kron(ones(N,1),[.5; param.max_b_charge(1)/2; 0]);
    % x_0(:,:,test) = [kron(ones(N,1),[.01;0;0]); zeros(G.numedges,1)];
    x_cl(:, :, 1, test) = x_0(:,:,test);
    x_ol(:, :, 1, test) = x_0(:,:,test);
    x_bl(:,:,1, test) = x_0(:,:,test);
        
    if isInfHorStable_ol
        game.VI_generator = computeVIGenerator(game, T);
        [~, ~, A_sh, ~, ~, ~] = game.VI_generator(x_0(:,:,test) - game.offset_x); % Computed here just to get the number of shared constrains and initialize the dual
        n_sh_constraints = size(A_sh,1);
        dual = zeros(n_sh_constraints, 1);
    end
    
    for t=1:T_sim
        if t>1
            u_ol_warm_start = u_shift_ol(:,:, :, t-1) - repmat(game.offset_u, T, 1, 1);
        else
            u_ol_warm_start = zeros(n_u * T, 1, N);
        end
        %% Solve open-loop MPC problem
        if isInfHorStable_ol
            [VI.J, VI.F, VI.A_sh, VI.b_sh, VI.A_loc, VI.b_loc, VI.n_x, VI.N]...
                = game.VI_generator(x_ol(:,:,t,test) - game.offset_x);
            dual_warm_start = dual;
            [u_full_traj_ol(:,:,:,t), dual, res, solved(t)] = solveVICentrFB(VI, 10^5, 10^(-4), ...
                0.001, 0.001, u_ol_warm_start, dual_warm_start);
            u_full_traj_ol(:,:,:,t) = u_full_traj_ol(:,:,:,t) + repmat(game.offset_u, T, 1, 1);
            u_ol(:,:,:,t,test) = u_full_traj_ol(1:n_u,:,:,t);
            x_ol(:,:,t+1,test) = evolveState(x_ol(:,:,t,test), game.A, game.B, u_ol(:, :,:, t), 1, n_u);
            x_ol_T = evolveState(x_ol(:,:,t,test), game.A, game.B, u_full_traj_ol(:,:,:,t), T, n_u);
            for i=1:N
                inf_hor_ol_input = game.K_ol(:,:,i) * (x_ol_T - game.offset_x) + game.offset_u(:,:,i);
                u_shift_ol(:,:,i,t) = [u_full_traj_ol(n_u+1:end, :, i, t); inf_hor_ol_input];
                if t==1
                    inf_hor_ol_without_offset = game.K_ol(:,:,i) * (x_ol_T - game.offset_x);
                    is_test_valid_ol(test) = all(all(pagemtimes(game.C_u_loc, inf_hor_ol_without_offset)<= game.d_u_loc )) ...
                                        & all(sum(pagemtimes(game.C_u_sh, inf_hor_ol_input), 3) <= game.d_u_sh) ...
                                        & all(sum(pagemtimes(game.C_u_mix, inf_hor_ol_input), 3) + game.C_x_mix * (x_ol_T - game.offset_x) <= game.d_mix) ...
                                        & all( game.C_x * (x_ol_T - game.offset_x) <= game.d_x);
                end
            end
        end
        %% Solve closed-loop MPC problem 
        if isInfHorStable_cl && run_cl
            [~, ~, u_full_traj_cl(:,:,:,t)] = solveImplFinHorCL(game, T, x_cl(:,:,t,test));
            u_full_traj_cl(:,:,:,t) = u_full_traj_cl(:,:,:,t) + repmat(game.offset_u, T, 1, 1);
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
                inf_hor_cl_input = game.K_cl(:,:,i) * (x_cl_T - game.offset_x) + game.offset_u(:,:,i);
                u_shift_cl(:,:,i,t) = [u_full_traj_cl(n_u+1:end, :, i, t); inf_hor_cl_input];
            end
            % Baseline
            %% ToDo
        end
    end 
    err_shift(test) = 0;
    % if is_test_valid_ol(test)
        for t=1:T_sim-1
            for i=1:N
                err_shift(test) = max(err_shift(test), norm(u_shift_ol(:,:,i,t) - u_full_traj_ol(:,:,i, t+1)));
            end
        end
        disp("err test = " + num2str(err_shift(test)))
    % end
    test = test+1;
end

save("workspace_variables.mat", "x_ol", "x_cl", "x_bl", "u_ol", "u_cl", "u_bl")
plot_load_flex

disp( "Job complete" )

% END script



