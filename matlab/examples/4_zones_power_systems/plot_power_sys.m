% Create figure
close all
figure;

time = 1:T_sim+1;
test = interesting_test;

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

%% Plot 2: delta omega
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