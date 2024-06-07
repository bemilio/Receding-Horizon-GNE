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
hold on
colors = lines(N);
x = linspace(0, (T_sim - 1) * param.T_sampl, T_sim);
for i=1:N
    selectedColor = colors(i, :);
    plot(x, p_ol(:,i), '-o', 'DisplayName', "Agent " + num2str(i), 'Color',selectedColor);
    if i~=1
        % Plot the safety distance
        % Define the lower boundary for the shaded area
        p_ol_lower = p_ol(:,i) - 20 - v_ol(:,i) * param.headway_time(i);
        % Define the x and y coordinates for the shaded area
        x_fill = [x, fliplr(x)];
        y_fill = [p_ol(:,i)',fliplr(p_ol_lower')];
        fill(x_fill, y_fill, colors(i,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none','HandleVisibility', 'off');
    end
end 
%TODO: plot closed loop
hold on
if run_cl
    plot(p_cl, '-o','DisplayName', "CL-NE");
end

xlabel('t');
ylabel('$p_i$', 'Interpreter','latex');
grid on;
legend
print('position.png', '-dpng', '-r600');  % Save with 600 dpi resolution

figure;
hold on
indexes_position = 1:n_x_per_agent:n_x;
indexes_speed = 2:n_x_per_agent:n_x;
for i=1:N
    selectedColor = colors(i, :);
    plot(v_ol(:,i), '-o', 'DisplayName', "position",'DisplayName', "Agent " + num2str(i), 'Color',selectedColor);
end
yline(param.max_speed(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
yline(param.min_speed(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
xlabel('t');
ylabel('$v_i$', 'Interpreter','latex');
grid on
legend
print('position.png', '-dpng', '-r600');  % Save with 600 dpi resolution

