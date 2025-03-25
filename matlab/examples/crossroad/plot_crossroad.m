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

% Taslate everything so that there is only one vehicle at the beginning of
% the animation
if max(max(p_abs(1,:)))>0
    p_abs = p_abs - max(max(p_abs(1,:)));
end

%% Plot roads


% Create figure
figure;
hold on;
axis equal;

road_length = 100; %m 
roadWidth = 8; % Width of the road

% Define vehicle size
vehicleWidth = 2;
vehicleLength = 2.5*vehicleWidth;

% Road 1: A smooth curved road using a cubic Bézier curve
resolution=1000;
t = linspace(0, 1, resolution+1); % Parameter for curve
 % control points for Road 1. The first and last elements are the initial
 % and final point of the road, respectively
P1 = [[0, -road_length/2] - (road_length/10)*rand(1,2);  
      (road_length/10)*randn(1,2);  
      (road_length/10)*randn(1,2);   
      [0, road_length/2 + 2*vehicleLength] + (road_length/10)*rand(1,2)]; 
                        
 
road1 = bezierCurve(P1, t);

% Compute road angle (used to compute vehicle orientation)
dx = diff(road1(:,1)); dy = diff(road1(:,2));
angles_road1 = atan2(dy, dx); 

% Compute normal vectors to shift the vehicle to the right lane
laneOffset = roadWidth / 4; % Move the vehicle half a lane to the right
normals_road1 = [-dy, dx]; % Perpendicular vector to the road
normals_road1 = normals_road1 ./ vecnorm(normals_road1, 2, 2) * laneOffset; % Normalize & scale

% Road 2: Another curved road using a cubic Bézier curve
P2 = [[-road_length/2 - 2*vehicleLength, 0] - (road_length/10)*rand(1,2);  
      (road_length/10)*randn(1,2);  
      (road_length/10)*randn(1,2);   
      [road_length/2 + 2*vehicleLength, 0] + (road_length/10)*rand(1,2)]; 

% Modify the second control point so that the midpoint of the two curves is
% equal
midpoint = bezierCurve(P1, .5);
P2(2, 1) = (midpoint(1) - 0.125*P2(1,1) - 0.375*P2(3,1)  - 0.125*P2(4,1) )/ 0.375;
P2(2, 2) = (midpoint(2) - 0.125*P2(1,2) - 0.375*P2(3,2)  - 0.125*P2(4,2) )/ 0.375;

road2 = bezierCurve(P2, t);

% Compute road angle (used to compute behicle orientation)
dx = diff(road2(:,1)); dy = diff(road2(:,2));
angles_road2 = atan2(dy, dx);

% Compute normal vectors to shift the vehicle to the right lane
normals_road2 = [-dy, dx]; % Perpendicular vector to the road
normals_road2 = normals_road2 ./ vecnorm(normals_road2, 2, 2) * laneOffset; % Normalize & scale

% Plot the roads
drawRoad(road1, roadWidth, [0.4 0.4 0.4]); % Dark gray asphalt
drawRoad(road2, roadWidth, [0.4 0.4 0.4]);


% Create vehicles 
yPos = -laneOffset; % Move vehicle to the right lane
colors = lines(N); % vehicle and plot colors
for i=1:N
    origin{i} = param.crossing_dir{i}(1); % N, S, E, W
    destination{i} = param.crossing_dir{i}(2); % N, S, E, W
end
for i=1:N 
    if origin{i}=='W'
        % Find the position of the beginning of the road, offset to the
        % right lane
        vehicle_x = road2(1, 1) - normals_road2(1,1);
        vehicle_y = road2(1, 2) - normals_road2(1,2);

        % Get the coordinates of the polygon defining the vehicle
        vehicle_shape = vehicleShape(angles_road2(1), vehicleLength, vehicleWidth);
        vehicle(i) = fill(vehicle_x + vehicle_shape(1,:), ... %X coord.
                          vehicle_y + vehicle_shape(2,:), ... %Y coord.
                                      colors(i,:));
    end 
    if origin{i} == 'N'
        % Find the position of the beginning of the road, offset to the
        % right lane
        vehicle_x = road1(end, 1) + normals_road1(end,1);
        vehicle_y = road1(end, 2) + normals_road1(end,2);

        % Get the coordinates of the polygon defining the vehicle
        vehicle_shape = vehicleShape(angles_road1(end), vehicleLength, vehicleWidth);
        vehicle(i) = fill(vehicle_x + vehicle_shape(1,:), ... %X coord.
                          vehicle_y + vehicle_shape(2,:), ... %Y coord.
                                      colors(i,:));
    end

    if origin{i} == 'E'
        % Find the position of the beginning of the road, offset to the
        % right lane
        vehicle_x = road2(end, 1) + normals_road2(end,1);
        vehicle_y = road2(end, 2) + normals_road2(end,2);

        % Get the coordinates of the polygon defining the vehicle
        vehicle_shape = vehicleShape(angles_road2(end), vehicleLength, vehicleWidth);
        vehicle(i) = fill(vehicle_x + vehicle_shape(1,:), ... %X coord.
                          vehicle_y + vehicle_shape(2,:), ... %Y coord.
                                      colors(i,:));
    end
    if origin{i} == 'S'
        % Find the position of the beginning of the road, offset to the
        % right lane
        vehicle_x = road1(1, 1) - normals_road1(1,1);
        vehicle_y = road1(1, 2) - normals_road1(1,2);

        % Get the coordinates of the polygon defining the vehicle
        vehicle_shape = vehicleShape(angles_road1(1), vehicleLength, vehicleWidth);
        vehicle(i) = fill(vehicle_x + vehicle_shape(1,:), ... %X coord.
                          vehicle_y + vehicle_shape(2,:), ... %Y coord.
                                      colors(i,:));
    end
    
end
axis off;
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
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i) - vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );

                vehicle_x = road2(road_progress, 1) - normals_road2(road_progress,1);
                vehicle_y = road2(road_progress, 2) - normals_road2(road_progress,2);
        
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_road2(road_progress), vehicleLength, vehicleWidth);
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
            end
        end
        if origin{i}=='N' && destination{i} == 'S'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i) - vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the
                % right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = road1(end-road_progress, 1) + normals_road1(end-road_progress,1);
                vehicle_y = road1(end-road_progress, 2) + normals_road1(end-road_progress,2);
        
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_road1(end-road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
            end
        end
        if origin{i}=='E' && destination{i} == 'W'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i)- vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = road2(end-road_progress, 1) + normals_road2(end-road_progress,1);
                vehicle_y = road2(end-road_progress, 2) + normals_road2(end-road_progress,2);
        
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_road2(end-road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
            end
                    
        end

        if origin{i}=='S' && destination{i} == 'N'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i)- vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the beginning of the road, offset to the
                % right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = road1(road_progress, 1) - normals_road1(road_progress,1);
                vehicle_y = road1(road_progress, 2) - normals_road1(road_progress,2);
        
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_road1(road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
            end
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

    % If we are in the middle of the simulation, save the frame in an image
    if t==ceil(T_sim/4)
        set(gcf, 'Color', 'w');
        print('frame_crossroad.png', '-dpng', '-r600');  % Save with 600 dpi resolution
    end
    
    pause(timestep_frame); % Control speed of animation

end

% Close video file
close(video);

disp(['Video saved as ', videoFilename]);
disp(['GIF saved as ', gifFilename]);

hold off;

%% Position plot

% 1st plot: Position w.r.t vehicle ahead
figure; 
ax1 = subplot(2,1,1); % 2 rows, 1 column, first subplot
hold on
x = linspace(0, (T_sim - 1) * param.T_sampl, T_sim);
desired_position = 0; % We iteratively sum the desired distance of each agent to compute this value
for i=1:N
    selectedColor = colors(i, :);
    j=agentBefore(i,param.crossing_dir, param.conflicts_dict);
    if ~isempty(j)
        plot(x, p(:,j)-p(:,i), '-', 'DisplayName', "Agent " + num2str(i), 'Color',selectedColor, 'LineWidth',1.5);
        yline(param.d_min(i), 'Color', [1 0.1 0.1], 'LineStyle', '--', 'LineWidth', 2,'HandleVisibility', 'off');
        yline(param.d_des(i), 'Color', colors(1, :), 'LineStyle', ':', 'LineWidth', 2,'HandleVisibility', 'off');
    end
end 
ylabel('$p_j(t) - p_i(t)$ (m)', 'Interpreter','latex');
grid on;
set(gca, 'XTickLabel', []); % Remove x-tick labels from the top subplot
legend

% Add label (a)
text(1.02, 0.5, '(a)', 'Units', 'normalized', 'FontSize', 12, 'Interpreter', 'latex');

%% plot velocity over time


ax2 = subplot(2,1,2); % 3 rows, 1 column, 3rd subplot
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
linkaxes([ax1, ax2], 'x');

% Remove the gap between the plots
gap = 0.05; % Small gap between the plots

% Adjust positions
pos2 = get(ax2, 'Position');

pos1 = get(ax1, 'Position');
pos1(2) = pos2(2) + pos2(4) + gap; % Move ax1 above ax2
set(ax1, 'Position', pos1);

print('pos_velocity_dist_to_Xf.png', '-dpng', '-r600');  % Save with 600 dpi resolution


