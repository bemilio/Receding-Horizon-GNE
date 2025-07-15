clear all
clc
close all

addpath(genpath('../../')) % add all folders in the root folder
rmpath(genpath('../')) % remove folders of the other examples (function names are conflicting)
addpath(genpath(pwd)) % re-add the subfolders of this example

seed = 1;
rng(seed); 
eps = 10^(-4);

N = 5; 
% state For each agent: position error, speed error. 
% %Note: the position error of the leading vehicle is constant 0 (dummy state)
n_x = 2 * N; 
n_u = 1; % acceleration
T_hor_to_test = [2, 5, 10, 15, 20, 30];
N_tests = 100;

param = defineCrossroadGameParameters(N);
game = defineCrossroadGame(N, param);

mu = zeros(N_tests*length(T_hor_to_test), 1);
tested_T = zeros(N_tests*length(T_hor_to_test), 1);
is_stable = false(N_tests*length(T_hor_to_test), 1);
is_solved = false(N_tests,1);
x0 = zeros(n_x,1);

for test=1:N_tests
    % Personalize objective
    for i=1:N
        game.Q(:,:,i) = diag([zeros((i-1)*2,1); 3*rand(2,1); zeros((N-i)*2,1)]);
        %game.Q(:,:,i) = rand * eye(n_x);
        game.R(:,:,i) = 10*rand;
    end
    [game.P_ol, game.K_ol, solved, isInfHorStable_ol] = solveInfHorOL(game, 1000, 10^(-6));
    is_solved(test) = solved && isInfHorStable_ol;
    if solved
        for T_idx = 1:length(T_hor_to_test)
            T = T_hor_to_test(T_idx);
            VI_generator = computeVIGenerator(game, T);
            [~,~,~,~,~,~,~,~,Q_mat] = VI_generator(x0);
            tested_T((test-1)*length(T_hor_to_test) + T_idx) = T;
            is_stable((test-1)*length(T_hor_to_test) + T_idx) = isInfHorStable_ol;
            mu((test-1)*length(T_hor_to_test) + T_idx) = min(eig(Q_mat + Q_mat'));
        end
    else
        for T_idx = 1:length(T_hor_to_test)
            mu((test-1)*length(T_hor_to_test) + T_idx) = nan;
        end
    end
end



%% Box Plot

% Remove NaNs from R and corresponding entries from N
mu_clean = mu(is_solved);
tested_T_clean = tested_T(is_solved);

% Create boxplot
figure;
boxplot(mu_clean, tested_T_clean, 'Whisker', Inf);
grid on

xlabel('Horizon','FontSize', 12, 'Interpreter', 'latex');
ylabel('Monotonicity constant','FontSize', 12, 'Interpreter', 'latex');
fig = gcf;
set(fig, 'Position', [10, 10, 500, 230]); % Define figure size [left, bottom, width, height]
drawnow;  % Force rendering
exportgraphics(fig, 'box_plot_monotonicity.png', 'Resolution', 600);


%% Compute percentage of solved

solved_percent = sum(is_solved) / N_tests * 100;
