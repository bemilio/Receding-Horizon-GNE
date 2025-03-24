close all

n_x_per_agent = 2;

%% Conversion from error variables to actual speed/position
% Compute relative positions w.r.t. leading vehicles, and absolute velocity
for t = 1:T_sim
    [p(t,:), v(t,:)] = convertStateToPosVel(x_ol(:,1,t), param.v_des_1, param.d_des, param.headway_time, param.crossing_dir, param.conflicts_dict);
end

% Convert relative position to absolute position by integrating the leading
% vehicle's velocity
p_abs(1, :) = x_0_p';
for t=1:T_sim-1
    for i=1:N
        j = agentBefore(i, param.crossing_dir, param.conflicts_dict);
        % Integrate speed + accel.
        p_abs(t+1,i) = p_abs(t, i) +  v(t,i) * param.T_sampl + squeeze(u_ol(1,1,i,t) - param.K(:,:,i) * x_ol(:,:,t)) * (param.T_sampl^2)/2;
    end
end
% Taslate everything so that all the vehicles appear in the picture
% p = p+min(x_0_p);


%% Plot roads
% Create figure
figure;
hold on;
axis equal;

% Define vehicle and road sizes
vehicleLength = 5;
road_length = 160; %m 
roadWidth = 8;
laneOffset = roadWidth/2; % Offset from the centerline to the right lane

vehicleWidth = 3;

road_top = road_length/2;
road_bottom = -road_length/2;
road_right = road_length/2;
road_left = -road_length/2;

xlim([road_left road_right]);
ylim([road_bottom road_top]);


% Draw horizontal road
fill([road_left road_right road_right road_left], [-roadWidth -roadWidth roadWidth roadWidth], [0.3 0.3 0.3], 'EdgeColor', 'none');

% Draw vertical road
fill([-roadWidth roadWidth roadWidth -roadWidth], [road_bottom road_bottom road_top road_top], [0.3 0.3 0.3], 'EdgeColor', 'none');

% Add dashed centerlines
for x = road_left:2:road_right
    plot([x x+1], [0 0], 'w--', 'LineWidth', 2);
end

for y = road_bottom:2:road_top
    plot([0 0], [y y+1], 'w--', 'LineWidth', 2);
end

% Create vehicles 
yPos = -laneOffset; % Move vehicle to the right lane
vehicleColor = 'r'; % Red car
for i=1:N
    origin{i} = param.crossing_dir{i}(1); % N, S, E, W
    destination{i} = param.crossing_dir{i}(2); % N, S, E, W
end
for i=1:N 
    if origin{i}=='W'
        vehicle(i) = fill(road_left + [-vehicleLength/2 vehicleLength/2 vehicleLength/2 -vehicleLength/2], ... %X coord.
                                      [-vehicleWidth/2 -vehicleWidth/2 vehicleWidth/2 vehicleWidth/2] -laneOffset, ... %Y coord.
                                      vehicleColor);
    end 
    if origin{i} == 'N'
        vehicle(i) = fill( [-vehicleWidth/2 vehicleWidth/2 vehicleWidth/2 -vehicleWidth/2] - laneOffset, ... % X-coordinates
                           road_top + [-vehicleLength/2 -vehicleLength/2 vehicleLength/2 vehicleLength/2], ... % Y-coordinates
                           vehicleColor);
    end

    if origin{i} == 'E'
        vehicle(i) = fill(road_right + [-vehicleLength/2 vehicleLength/2 vehicleLength/2 -vehicleLength/2], ... %X coord.
                                       [-vehicleWidth/2 -vehicleWidth/2 vehicleWidth/2 vehicleWidth/2] + laneOffset, ... %Y coord.
                                       vehicleColor);
    end
    if origin{i} == 'S'
        vehicle(i) = fill( [-vehicleWidth/2 vehicleWidth/2 vehicleWidth/2 -vehicleWidth/2] + laneOffset, ... % X-coordinates
                           road_bottom + [-vehicleLength/2 -vehicleLength/2 vehicleLength/2 vehicleLength/2], ... % Y-coordinates
                           vehicleColor);
    end
    
end

% Video setup
videoFilename = 'vehicle_animation.mp4';
video = VideoWriter(videoFilename, 'MPEG-4');
open(video);

% GIF setup
gifFilename = 'vehicle_animation.gif';

% Animation loop
for t = 1:T_sim
    for i=1:N
        if origin{i}=='W' && destination{i} == 'E'
            set(vehicle(i), 'XData', road_left + [-vehicleLength/2 vehicleLength/2 vehicleLength/2 -vehicleLength/2] + p_abs(t,i));
        end
        if origin{i}=='N' && destination{i} == 'S'
            set(vehicle(i), 'YData', road_top + [-vehicleLength/2 -vehicleLength/2 vehicleLength/2 vehicleLength/2] - p_abs(t,i));
        end
        if origin{i}=='E' && destination{i} == 'W'
            set(vehicle(i), 'XData', road_right + [-vehicleLength/2 vehicleLength/2 vehicleLength/2 -vehicleLength/2] - p_abs(t,i));
        end

        if origin{i}=='S' && destination{i} == 'N'
            set(vehicle(i), 'YData', road_bottom + [-vehicleLength/2 -vehicleLength/2 vehicleLength/2 vehicleLength/2] + p_abs(t,i));
        end
    end
    drawnow;

    % Capture the frame
    frame = getframe(gcf);

    % Write frame to video
    writeVideo(video, frame);
    
    % Convert frame to an image for GIF
    img = frame2im(frame);
    [imind, cm] = rgb2ind(img, 256);

    % Write frame to GIF
    timestep_frame = 0.05;

    if t == 1
        imwrite(imind, cm, gifFilename, 'gif', 'LoopCount', inf, 'DelayTime', timestep_frame);
    else
        imwrite(imind, cm, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', timestep_frame);
    end
    
    pause(timestep_frame); % Control speed of animation

end

% Close video file
close(video);

disp(['Video saved as ', videoFilename]);
disp(['GIF saved as ', gifFilename]);

hold off;

%% Plot the sequence
figure; 
ax1 = subplot(3,1,1); % 3 rows, 1 column, first subplot
hold on
colors = lines(N);
x = linspace(0, (T_sim - 1) * param.T_sampl, T_sim);
desired_position = 0; % We iteratively sum the desired distance of each agent to compute this value
for i=1:N
    selectedColor = colors(i, :);
    plot(x, p_abs(:,i), '-', 'DisplayName', "Agent " + num2str(i), 'Color',selectedColor, 'LineWidth',1.5);
    if i~=N
        % Plot the safety distance:

        % Define the lower boundary for the shaded area
        p_ol_lower = p_abs(:,i) - param.d_min(i+1) - v(:,i+1) * param.headway_time(i+1);

        % Define the x and y coordinates for the shaded area
        x_fill = [x, fliplr(x)];
        y_fill = [p_abs(:,i)',fliplr(p_ol_lower')];

        % Plot shaded area
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
    plot(x,v(:,i), '-', 'DisplayName', "position",'DisplayName', "Agent " + num2str(i), 'Color',selectedColor,'LineWidth',1.5);
end
yline(param.max_speed(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
yline(param.min_speed(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
yline(param.v_des_1, 'Color', colors(1, :), 'LineStyle', ':', 'LineWidth', 2,'HandleVisibility', 'off');
xlabel('$t$', 'Interpreter','latex');
ylabel('$v_i$ (m/s)', 'Interpreter','latex');
grid on
% Add label (b)
text(1.02, 0.5, '(b)', 'Units', 'normalized', 'FontSize', 12, 'Interpreter', 'latex');


%% Adjust subplots
% Link the x-axes
linkaxes([ax1, ax2, ax3], 'x');

% Remove the gap between the plots
gap = 0.05; % Small gap between the plots

% Adjust positions
pos2 = get(ax2, 'Position');

pos2(2) = pos3(2) + pos3(4) + gap; % Move ax2 above ax3
set(ax2, 'Position', pos2);

pos1 = get(ax1, 'Position');
pos1(2) = pos2(2) + pos2(4) + gap; % Move ax1 above ax2
set(ax1, 'Position', pos1);

print('pos_velocity_dist_to_Xf.png', '-dpng', '-r600');  % Save with 600 dpi resolution

