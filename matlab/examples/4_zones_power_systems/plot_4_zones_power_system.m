% Create figure
close all
figure;

time = 1:T_sim+1;

%% Plot 1: boxplot of maximum vector norms
% Step 1: Compute the maximum absolute values for each column
max_norm_x_cl = zeros(N_tests,1);
max_norm_x_bl = zeros(N_tests,1);
for k = 1:N_tests
    for t=1:T_sim
        max_norm_x_cl(k) = max(max_norm_x_cl(k), norm(x_cl(:,1,t,k)));
        max_norm_x_bl(k) = max(max_norm_x_bl(k), norm(x_bl(:,1,t,k)));
    end
end
combinedVectors = [max_norm_x_cl; max_norm_x_bl];

group = [repmat({'With terminal cost'}, length(max_norm_x_cl), 1); repmat({'Without terminal cost'}, length(max_norm_x_bl), 1)];

figure;
boxplot(combinedVectors, group);
ylabel('$ \max_{t\in\mathcal{T}} \|x[t]\|$', 'Interpreter', 'latex');
set(gca, 'YScale', 'log');
ylim([1e-2, 1e7]);
grid on
savefig('boxplot_comparison');
print('boxplot_comparison', '-dpng', '-r600');  % Save with 300 dpi resolution

%% Plot %2: histogram of percentage of stable initial states vs. P-norm 
norm_x_0 = zeros(N_tests,1);
is_test_stable = zeros(N_tests,1);
is_test_stable_bl = zeros(N_tests,1);
stability_threshold = 1;
norm_P = @(x) sqrt(x'*X_f_cl.P*x);
for test=1:N_tests
    norm_x_0(test) = norm_P(x_cl(:,1,1,test))/ X_f_cl.d; % norm relative to terminal set radius
    is_test_stable(test) = norm(x_cl(:,1,end,test))<stability_threshold;
    is_test_stable_bl(test) = norm(x_bl(:,1,end,test))<stability_threshold;
end
% Calculate percentages for each bin
bins_edges = [0,norms_x_0_to_test];
percentages_stable = zeros(1, length(norms_x_0_to_test));
percentages_stable_bl = zeros(1, length(norms_x_0_to_test));
for i = 1:length(norms_x_0_to_test)
    % Indices for each bin in the first array of each couple
    binIndices = norm_x_0 >= bins_edges(i)-0.01 & norm_x_0 < bins_edges(i + 1) + 0.01;
    % Calculate percentages of stable trajectories
    if any(binIndices)
        percentages_stable(i) = sum(is_test_stable(binIndices)) / sum(binIndices) * 100;
    end
    if any(binIndices)
        percentages_stable_bl(i) = sum(is_test_stable_bl(binIndices)) / sum(binIndices) * 100;
    end
end


% Combine percentages for plotting
percentages = [percentages_stable; percentages_stable_bl]';

% Plot the histogram
figure;
bar(percentages, 'grouped');
xlabel('$\|x_0\|_{P}/r$', 'Interpreter', 'latex');
ylabel('Stable trajectories (%)', 'Interpreter', 'latex');
legend({'With term. cost', 'No term. cost'}, 'Interpreter', 'latex', 'Location','southwest');

xticks(1:length(norms_x_0_to_test));
xticklabels(arrayfun(@(i) sprintf('%.2f', bins_edges(i + 1)), 1:5, 'UniformOutput', false));
grid on;
set(gcf, 'Position', [100, 100, 500, 230]); % Define figure size [left, bottom, width, height]
savefig('histogram_stability');
print('histogram_stability', '-dpng', '-r600');  % Save with 300 dpi resolution

%% Plot 3: delta omega
% First subplot
figure,
delta_omega1 = zeros(length(time),1);
delta_omega1(:) = x_cl(1,1,:, test);
subplot(2,2,1);
plot(time, delta_omega1);
xlabel('Time (s)');
ylabel('\Delta \omega_1');
title('\Delta \omega_1 vs. Time');

% Second subplot
subplot(2,2,2);
delta_omega2 = zeros(length(time),1);
delta_omega2(:) = x_cl(4,1,:,test);
plot(time, delta_omega2);
xlabel('Time (s)');
ylabel('\Delta P_{ref}_1');
title('\Delta P_{ref}_1 vs. Time');

% Third subplot
subplot(2,2,3);
delta_omega_1_bl(:) = x_bl(1,1,:,test);
plot(time, delta_omega_1_bl);
xlabel('Time (s)');
% ylabel('\Delta P_{tie}^{12}');
% title('\Delta P_{tie}^{12} vs. Time');
% 
% Fourth subplot
subplot(2,2,4);
delta_omega2(:) = x_bl(4,1,:,test);
plot(time, delta_omega2);
xlabel('Time (s)');
% ylabel('\Delta P_{ref}_2');
% title('\Delta P_{ref}_2 vs. Time');