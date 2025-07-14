close all

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

% Plot the sequence
figure; 
ax1 = subplot(2,1,1); % 2 rows, 1 column, second subplot
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
%TODO: plot closed loop
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

%% Plot distance from region of attraction
ax3 = subplot(3,1,3); % 3 rows, 1 column, third subplot
plot(x, distance_state_reg_attraction, 'LineWidth',1.5, 'Color', "k");
grid on
ylabel('dist(𝑥(𝑇), 𝕏ᵒˡ)');
xlabel('$t$', 'Interpreter','latex');
% Add label (c)
text(1.02, 0.5, '(c)', 'Units', 'normalized', 'FontSize', 12, 'Interpreter', 'latex');


%% Adjust subplots
% Link the x-axes
linkaxes([ax1, ax2, ax3], 'x');

% Remove the gap between the plots
gap = 0.05; % Small gap between the plots

% Adjust positions
pos2 = get(ax2, 'Position');
pos3 = get(ax3, 'Position');

pos2(2) = pos3(2) + pos3(4) + gap; % Move ax2 above ax3
set(ax2, 'Position', pos2);

pos1 = get(ax1, 'Position');
pos1(2) = pos2(2) + pos2(4) + gap; % Move ax1 above ax2
set(ax1, 'Position', pos1);



print('pos_velocity_dist_to_Xf.png', '-dpng', '-r600');  % Save with 600 dpi resolution


