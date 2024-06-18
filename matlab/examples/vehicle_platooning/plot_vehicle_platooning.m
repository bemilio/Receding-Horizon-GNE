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
            convertStateToPosVel(x_cl(:,1,t),...
                param.v_des_1, param.d_des, param.headway_time);
    end
end

% Plot the sequence
figure; 
ax1 = subplot(3,1,1); % 2 rows, 1 column, second subplot
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

ylabel('$p_i$ (m)', 'Interpreter','latex');
grid on;
set(gca, 'XTickLabel', []); % Remove x-tick labels from the top subplot
legend

ax2 = subplot(3,1,2); % 2 rows, 1 column, second subplot
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

ax3 = subplot(3,1,3); % 2 rows, 1 column, second subplot
hold on
indexes_position = 1:n_x_per_agent:n_x;
indexes_speed = 2:n_x_per_agent:n_x;
for i=1:N
    selectedColor = colors(i, :);
    plot(x,squeeze(u_ol(:,:,i,:)), '-', 'DisplayName', "position",'DisplayName', "Agent " + num2str(i), 'Color',selectedColor,'LineWidth',1.5);
end
yline(param.max_acc(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
yline(param.min_acc(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
xlabel('$t$', 'Interpreter','latex');
ylabel('$u_i$ (m/s)', 'Interpreter','latex');
grid on

% Link the x-axes
linkaxes([ax1, ax2], 'x');
% Remove the gap between the plots
pos1 = get(ax1, 'Position');
pos2 = get(ax2, 'Position');
gap = 0.05; % Small gap between the plots
pos1(2) = pos2(2) + pos2(4) + gap;
set(ax1, 'Position', pos1);

print('pos_velocity.png', '-dpng', '-r600');  % Save with 600 dpi resolution


if run_cl
    figure; 
    ax1 = subplot(3,1,1); % 2 rows, 1 column, second subplot
    hold on
    colors = lines(N);
    x = linspace(0, (T_sim - 1) * param.T_sampl, T_sim);
    desired_position = 0; % We iteratively sum the desired distance of each agent to compute this value
    for i=1:N
        selectedColor = colors(i, :);
        plot(x, p_cl(:,i), '-', 'DisplayName', "Agent " + num2str(i), 'Color',selectedColor, 'LineWidth',1.5);
        if i~=1
            % Plot desired position
            desired_position = desired_position + param.d_des(i) + param.headway_time(i) * param.v_des_1;
            yline(desired_position, 'Color', selectedColor, 'LineStyle', ':', 'LineWidth', 2,'HandleVisibility', 'off');
            % Plot the safety distance
            % Define the lower boundary for the shaded area
            p_cl_lower = p_cl(:,i) - 20 - v_cl(:,i) * param.headway_time(i);
            % Define the x and y coordinates for the shaded area
            x_fill = [x, fliplr(x)];
            y_fill = [p_cl(:,i)',fliplr(p_cl_lower')];
            fill(x_fill, y_fill, colors(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none','HandleVisibility', 'off');
        end
    end 
    hold on
    
    ylabel('$p_i$ (m)', 'Interpreter','latex');
    grid on;
    set(gca, 'XTickLabel', []); % Remove x-tick labels from the top subplot
    legend
    
    ax2 = subplot(3,1,2); % 2 rows, 1 column, second subplot
    hold on
    indexes_position = 1:n_x_per_agent:n_x;
    indexes_speed = 2:n_x_per_agent:n_x;
    for i=1:N
        selectedColor = colors(i, :);
        plot(x,v_cl(:,i), '-', 'DisplayName', "position",'DisplayName', "Agent " + num2str(i), 'Color',selectedColor,'LineWidth',1.5);
    end
    yline(param.max_speed(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
    yline(param.min_speed(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
    yline(param.v_des_1, 'Color', colors(1, :), 'LineStyle', ':', 'LineWidth', 2,'HandleVisibility', 'off');
    xlabel('$t$', 'Interpreter','latex');
    ylabel('$v_i$ (m/s)', 'Interpreter','latex');
    grid on
    
    % Link the x-axes
    linkaxes([ax1, ax2], 'x');
    % Remove the gap between the plots
    pos1 = get(ax1, 'Position');
    pos2 = get(ax2, 'Position');
    gap = 0.05; % Small gap between the plots
    pos1(2) = pos2(2) + pos2(4) + gap;
    set(ax1, 'Position', pos1);
    
    ax3 = subplot(3,1,3); % 2 rows, 1 column, second subplot
    hold on
    indexes_position = 1:n_x_per_agent:n_x;
    indexes_speed = 2:n_x_per_agent:n_x;
    for i=1:N
        selectedColor = colors(i, :);
        plot(x,squeeze(u_cl(:,:,i,:)), '-', 'DisplayName', "position",'DisplayName', "Agent " + num2str(i), 'Color',selectedColor,'LineWidth',1.5);
    end
    yline(param.max_acc(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
    yline(param.min_acc(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
    xlabel('$t$', 'Interpreter','latex');
    ylabel('$u_i$ (m/s)', 'Interpreter','latex');
    grid on


    print('pos_velocity_cl.png', '-dpng', '-r600');  % Save with 600 dpi resolution
end

