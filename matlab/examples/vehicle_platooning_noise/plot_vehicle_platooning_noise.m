close all

% Assuming u_ol is of size (n_u, 1, N, T_sim, N_tests)
[n_u, ~, N, T_sim, N_tests] = size(u_ol);

n_x_per_agent = 2;
p_ol = zeros(T_sim, N);
v_ol = zeros(T_sim, N);
p_cl = zeros(T_sim, N);
v_cl = zeros(T_sim, N);
%% Conversion from error variables to actual speed/position
for t = 1:T_sim
    [p_ol(t,:), v_ol(t,:)] = convertStateToPosVel(x_ol(:,1,t),...
                param.v_des_1, param.d_des, param.headway_time);
end
if run_cl
    for t = 1:T_sim
        [p_cl(t,:), v_cl(t,:)] =...
            convertStateToPosVel(x_ol(:,1,t),...
                param.v_des_1, param.d_des, param.headway_time);
    end
end


%% Boxplot of input deviation from nominal when noise is introduced to term. cost
figure;

% Get the number of variance values tested

% Extract baseline values (test 1)
baseline = x_ol(:,:,:,1); % Size: (n_x, 1, T_sim)

% Compute the norm of the state sequence
baseline_norm = norm(baseline(:)); 

% Initialize matrix to store normalized sums
norm_ratios = zeros(N_tests-1, 1);

% Loop over tests (excluding test 1)
for test_idx = 2:100
    % Compute the difference from baseline
    diff = x_ol(:,:,:,test_idx) - baseline;

    % Compute vector 2-norm
    diff_norm = norm(diff(:)); % Size: (n_u*N, 1)

    % Normalize by baseline norm (element-wise division)
    normalized_diff = 100*diff_norm ./ baseline_norm; % Size: (n_u*N, 1)
    if max(normalized_diff) > 10
        disp("pause")
    end
    % Sum over all n_u*N elements to get a single scalar
    norm_ratios(test_idx-1) = max(normalized_diff);
end

% Group results by parameter
num_tested_variance = length(variance_to_test);
grouped_data = cell(num_tested_variance, 1);
for i = 1:num_tested_variance
    grouped_data{i} = norm_ratios(test_variance(2:end) == variance_to_test(i));
end


% Convert to box plot format
grouped_data = cell2mat(grouped_data'); %transpose?

% Create box plot
figure;
boxplot(grouped_data, variance_to_test*100, 'Labels', string(variance_to_test*100));
xlabel('$\mathrm{variance}/\mathrm{max}_i(\|P_i^{\mathrm{OL}}\|_\infty)$ (\%)', 'Interpreter','latex');
ylabel('$\|x - \hat{x}\|/\|\hat{x}\|$ (\%)', 'Interpreter','latex');
grid on;
% Adjust figure size (wider x-axis, shorter y-axis)
set(gcf, 'Position', [100, 100, 800, 300]); % [left, bottom, width, height]
ylim([0 20]);
print('box_plot_noise_P.png', '-dpng', '-r600');  % Save with 600 dpi resolution


%% 
test_to_plot = 4;
n_x_per_agent = 2;
p_ol = zeros(T_sim, N);
v_ol = zeros(T_sim, N);
p_cl = zeros(T_sim, N);
v_cl = zeros(T_sim, N);
%% Conversion from error variables to actual speed/position
for t = 1:T_sim
    [p_ol(t,:), v_ol(t,:)] = convertStateToPosVel(x_ol(:,1,t, test_to_plot),...
                param.v_des_1, param.d_des, param.headway_time);
end

% Plot the sequence
figure; 
ax1 = subplot(3,1,1); % 3 rows, 1 column, first subplot
hold on
colors = lines(N);
x = linspace(0, (T_sim - 1) * param.T_sampl, T_sim);
desired_position = 0; % We iteratively sum the desired distance of each agent to compute this value
for i=1:N
    selectedColor = colors(i, :);
    plot(x, p_ol(:,i), '-', 'DisplayName', "Agent " + num2str(i), 'Color',selectedColor, 'LineWidth',1.5);
    if i~=1
        % Plot desired position
        desired_position = desired_position + param.d_des(i) + param.headway_time(i) * param.v_des_1;
        yline(desired_position, 'Color', selectedColor, 'LineStyle', ':', 'LineWidth', 2,'HandleVisibility', 'off');
        % Plot the safety distance
        % Define the lower boundary for the shaded area
        p_ol_lower = p_ol(:,i) - 20 - v_ol(:,i) * param.headway_time(i);
        % Define the x and y coordinates for the shaded area
        x_fill = [x, fliplr(x)];
        y_fill = [p_ol(:,i)',fliplr(p_ol_lower')];
        fill(x_fill, y_fill, colors(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none','HandleVisibility', 'off');
    end
end 
hold on
if run_cl
    plot(p_cl, '-','DisplayName', "CL-NE");
end
ylabel('$p_i$ (m)', 'Interpreter','latex');
grid on;
set(gca, 'XTickLabel', []); % Remove x-tick labels from the top subplot
legend

% Add label (a)
text(1.02, 0.5, '(a)', 'Units', 'normalized', 'FontSize', 12, 'Interpreter', 'latex');


%% plot velocity over time
ax2 = subplot(3,1,2); % 3 rows, 1 column, second subplot
hold on
indexes_position = 1:n_x_per_agent:n_x;
indexes_speed = 2:n_x_per_agent:n_x;
for i=1:N
    selectedColor = colors(i, :);
    plot(x,v_ol(:,i), '-', 'DisplayName', "position",'DisplayName', "Agent " + num2str(i), 'Color',selectedColor,'LineWidth',1.5);
end
yline(param.max_speed(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
yline(param.min_speed(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
yline(param.v_des_1, 'Color', colors(1, :), 'LineStyle', ':', 'LineWidth', 2,'HandleVisibility', 'off');
xlabel('$t$', 'Interpreter','latex');
ylabel('$v_i$ (m/s)', 'Interpreter','latex');
grid on
% Add label (b)
text(1.02, 0.5, '(b)', 'Units', 'normalized', 'FontSize', 12, 'Interpreter', 'latex');

