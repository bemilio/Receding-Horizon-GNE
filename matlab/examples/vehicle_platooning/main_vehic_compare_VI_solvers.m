clear all
clc
close all

addpath(genpath('../../')) % add all folders in the root folder
rmpath(genpath('../')) % remove folders of the other examples (function names are conflicting)
addpath(genpath(pwd)) % re-add the subfolders of this example


run_cl = false;

seed = 1;
rng(seed); 
eps = 10^(-3);

N = 5; 
% state For each agent: position error, speed error. 
% %Note: the position error of the leading vehicle is constant 0 (dummy state)
n_x = 2 * N; 
n_u = 1; % acceleration
T = 10;
T_sim = 100;
N_tests = 10;
n_iter_VI_solver = 10^5; % max iterations VI solvers


param = defineVehiclePlatooningGameParameters(N);
game = defineVehiclePlatooningGame(N, param);

[game.P_ol, game.K_ol, isInfHorStable_ol] = solveInfHorOL(game, 1000, 10^(-6));

[game.C_x, game.d_x, ...
    game.C_u_loc, game.d_u_loc,...
    game.C_u_sh,  game.d_u_sh,...
    game.C_u_mix, game.C_x_mix, game.d_mix] = defineConstraints(N, param);

X_f_ol = computeTerminalSetOL(game);

% Initialize containers for computed variables
u_FB = zeros(n_u, 1, N, T_sim, N_tests);
u_DR = zeros(n_u, 1, N, T_sim, N_tests);
x_FB = zeros(n_x, 1, T_sim + 1, N_tests);
x_DR = zeros(n_x, 1, T_sim + 1, N_tests);
res_FB = zeros(n_iter_VI_solver, T_sim, N_tests);
res_DR = zeros(n_iter_VI_solver, T_sim, N_tests);



u_shift = zeros(n_u*T, 1, N);

% X_f_cl = computeTerminalSetCL(game);

err_shift = zeros(N_tests,1);

%% Run tests
if isInfHorStable_ol
for test=1:N_tests

    disp( "Test " + num2str(test) )

    %% Create a random initial state in position-velocity coordinates 
    x_0_p = (0:-90:-90*(N-1))' +  [0;15*randn(N-1,1)];
    x_0_v = (param.max_speed - param.min_speed)/2 + param.min_speed...
                             + (.5-rand(N,1)) .* min(param.max_speed - param.min_speed);
    
    x_0 = convertPosVelToState(x_0_p, x_0_v, param.v_des_1, param.d_des, param.headway_time);

    x_FB(:, :, 1, test) = x_0;
    x_DR(:, :, 1, test) = x_0;
    % Initialize VIs at x_0
    game.VI_generator = computeVIGenerator(game, T);
    [~, ~, A_sh, ~, ~, ~] = game.VI_generator(x_0); 
    n_sh_constraints = size(A_sh,1);
    dual = zeros(n_sh_constraints, 1);

    %% Run open-loop MPC problem with FB algorithm
    for t=1:T_sim
        disp("timestep = " + num2str(t));

        % Warm-start at previous solution
        if t>1
            u_warm_start = u_shift;
        else
            u_warm_start = zeros(n_u * T, 1, N);
        end
        dual_warm_start = dual;
        
        % Initialize VI
        [VI.F, VI.A_sh, VI.b_sh, VI.A_loc, VI.b_loc, VI.n_x, VI.N, VI.Q, VI.q]...
            = game.VI_generator(x_FB(:,:,t,test));

        % Solve VI with FB
        [u_full_traj, dual, res_FB(:, t, test), ~] = solveVICentrFB(VI, n_iter_VI_solver, eps, ...
            0.2, 0.2, u_warm_start, dual_warm_start);

        % Extract first input of sequence
        u_FB(:,:,:,t,test) = u_full_traj(1:n_u,:,:);

        % Compute next state
        x_FB(:,:,t+1,test) = evolveState(x_FB(:,:,t,test), game.A, game.B, u_full_traj, 1, n_u);

        % Compute shifted sequence
        x_T = evolveState(x_FB(:,:,t,test), game.A, game.B, u_full_traj, T, n_u);
        inf_hor_ol_input = pagemtimes(game.K_ol,x_T);
        for i=1:N
            u_shift(:,1,i) = [u_full_traj(n_u+1:end, :, i); inf_hor_ol_input(:,:,i)];
        end
    end
    
    %% Run open-loop MPC problem with DR algorithm
    for t=1:T_sim
        
        % Warm-start at previous solution
        if t>1
            u_warm_start = u_shift;
        else
            u_warm_start = zeros(n_u * T, 1, N);
        end
        
        % Initialize VI
        [VI.F, VI.A_sh, VI.b_sh, VI.A_loc, VI.b_loc, VI.n_x, VI.N, VI.Q, VI.q]...
            = game.VI_generator(x_DR(:,:,t,test));

        % Solve VI with DR
        [u_full_traj, res_DR(:, t, test), ~] = solveVICentrDR(VI, n_iter_VI_solver, eps, ...
                0.5, eye(VI.N * VI.n_x), u_warm_start);

        % Extract first input of sequence
        u_DR(:,:,:,t,test) = u_full_traj(1:n_u,:,:);

        % Compute next state
        x_DR(:,:,t+1,test) = evolveState(x_DR(:,:,t,test), game.A, game.B, u_full_traj, 1, n_u);

        % Compute shifted sequence
        x_T = evolveState(x_DR(:,:,t,test), game.A, game.B, u_full_traj, T, n_u);
        inf_hor_ol_input = pagemtimes(game.K_ol,x_T);
        for i=1:N
            u_shift(:,1,i) = [u_full_traj(n_u+1:end, :, i); inf_hor_ol_input(:,:,i)];
        end
    end
end
end

% save("workspace_variables.mat", "x_FB", "x_cl", "x_bl", "u_ol_FB", "u_cl", "u_bl", "distance_state_reg_attraction")


%% Plot residuals
% Compute mean, min, and max of the residual at first timestep across test dimensions
meanDRres = mean(squeeze(res_DR(:,1,:)), 2); % Average over both test dimensions
minDRres = min(squeeze(res_DR(:,1,:)), [], 2); % Minimum across test dimensions
maxDRres = max(squeeze(res_DR(:,1,:)), [], 2); % Maximum across test dimensions
meanFBres = mean(squeeze(res_FB(:,1,:)), 2); % Average over both test dimensions
minFBres = min(squeeze(res_FB(:,1,:)), [], 2); % Minimum across test dimensions
maxFBres = max(squeeze(res_FB(:,1,:)), [], 2); % Maximum across test dimensions

t = 1:n_iter_VI_solver; 

%% Plot residual over time
figure;
hold on;
% Plot DR
fill([t, fliplr(t)], [maxDRres', fliplr(minDRres')], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Shaded area
plot(t, meanDRres, 'b', 'LineWidth', 1); % Mean line
% Plot FB
fill([t, fliplr(t)], [maxFBres', fliplr(minFBres')], [1, 0.5, 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Shaded area
plot(t, meanFBres, 'Color', [1, 0.5, 0], 'LineWidth', 1); % Mean line

hold off;

% Labels
xlabel('Iteration');
ylabel('Residual');
legend({'D-R (Min-Max Range)', 'D-R (Mean)', 'F-B (Min-Max Range)', 'F-B (Mean)'}, 'Location', 'Best');
grid on;

set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')


%% Plot time to convergence in a box plot

% Flatten test dimensions into one (n_iterations x N_tests*T_sim,)
all_res_FB = reshape(res_FB, n_iter_VI_solver, []); 
all_res_DR = reshape(res_DR, n_iter_VI_solver, []); 


% Find the first time index where each test falls below the threshold
firstBelowThreshold_FB = nan(1, N_tests*T_sim); % Preallocate
firstBelowThreshold_DR = nan(1, N_tests*T_sim); % Preallocate

for testIdx = 1:N_tests*T_sim
    belowIdx_FB = find(all_res_FB(:, testIdx) < eps, 1, 'first'); % First occurrence
    belowIdx_DR = find(all_res_DR(:, testIdx) < eps, 1, 'first'); % First occurrence
    if ~isempty(belowIdx_FB)
        firstBelowThreshold_FB(testIdx) = belowIdx_FB; % Store the first time it went below threshold
    end
    if ~isempty(belowIdx_DR)
        firstBelowThreshold_DR(testIdx) = belowIdx_DR; % Store the first time it went below threshold
    end
end

% Convert to column vectors
firstBelowThreshold_FB = firstBelowThreshold_FB(:);
firstBelowThreshold_DR = firstBelowThreshold_DR(:);

% Grouping variable for boxplot
group = [ones(size(firstBelowThreshold_FB)); 2 * ones(size(firstBelowThreshold_DR))]; % 1 for first threshold, 2 for second
values = [firstBelowThreshold_FB; firstBelowThreshold_DR]; % Combine both datasets
print('lineplot_residuals.png', '-dpng', '-r600');  % Save with 600 dpi resolution

% Create figure
figure;
hold on;

% Use 'boxchart' for color customization
b1 = boxchart(group, values); % Orange

% Labels and title
xticks([1 2]);
xticklabels({['F-B'], ['D-R']});
ylabel('Convergence timestep');
grid on;
set(gca, 'YScale', 'log')

hold off;
print('boxplot_convergence.png', '-dpng', '-r600');  % Save with 600 dpi resolution

disp( "Job complete" )

% END script



