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
cd(fileparts(mfilename('fullpath'))); % Move to path of this script
if ~exist('Figures', 'dir') 
    % Create Figures subfolder
    mkdir('Figures');
end

% Create figure
fig = figure;
set(gcf, 'Color', 'w');
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0)); % Remove extra padding
set(fig, 'PaperPositionMode', 'auto'); % Adjusts paper size to figure
hold on;
axis equal;

road_length = 100; %m 
roadWidth = 15; % Width of the road

% Define vehicle size
vehicleWidth = 1.5;
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
                        
 
roadSN = bezierCurve(P1, t);

% Compute road angle (used to compute vehicle orientation) and
%  normal vectors to shift the vehicle to the right lane
laneOffset = roadWidth / 6; % Move the vehicle half a lane to the right
[normals_roadSN, tangent_vecsSN, angles_roadSN] = computeRoadAngleNormalTangent(roadSN);
normals_roadSN = normals_roadSN * laneOffset; % scale 

% Road 2: Another curved road using a cubic Bézier curve
P2 = [[-road_length/2 - 2*vehicleLength, 0] - (road_length/4)*rand(1,2);  
      ([-road_length/2 - 2*vehicleLength, 0] + [road_length/2 + 2*vehicleLength, 0])/2 + (road_length/10)*randn(1,2);  
      ([-road_length/2 - 2*vehicleLength, 0] + [road_length/2 + 2*vehicleLength, 0])/2 + (road_length/10)*randn(1,2); 
      [road_length/2 + 2*vehicleLength, 0] + (road_length/4)*rand(1,2)]; 

% Modify the second control point so that the midpoint of the two curves is
% equal
midpoint = bezierCurve(P1, .5);
P2(2, 1) = (midpoint(1) - 0.125*P2(1,1) - 0.375*P2(3,1)  - 0.125*P2(4,1) )/ 0.375;
P2(2, 2) = (midpoint(2) - 0.125*P2(1,2) - 0.375*P2(3,2)  - 0.125*P2(4,2) )/ 0.375;

roadWE = bezierCurve(P2, t);

% Compute road angle (used to compute vehicle orientation) and
%  normal vectors to shift the vehicle to the right lane
[normals_roadWE, tangent_vecsWE, angles_roadWE] = computeRoadAngleNormalTangent(roadWE);
normals_roadWE = normals_roadWE * laneOffset; % scale 

% Compute roads that blend the two original roads
[roadWN, roadNE, roadES, roadSW] = blendRoads(roadSN, roadWE, .4, .6);

% Compute normal vectors and road angles
% WN
[normals_roadWN, tangent_vecsWN, angles_roadWN] = computeRoadAngleNormalTangent(roadWN);
normals_roadWN = normals_roadWN * laneOffset; % scale 
% NE
[normals_roadNE, tangent_vecsNE, angles_roadNE] = computeRoadAngleNormalTangent(roadNE);
normals_roadNE = normals_roadNE * laneOffset; % scale 
% ES
[normals_roadES, tangent_vecsES, angles_roadES] = computeRoadAngleNormalTangent(roadES);
normals_roadES = normals_roadES * laneOffset; % scale 
% SW
[normals_roadSW, tangent_vecsSW, angles_roadSW] = computeRoadAngleNormalTangent(roadSW);
normals_roadSW = normals_roadSW * laneOffset; % scale 

% Plot the roads
drawRoad(roadSN, roadWidth, [0.7 0.7 0.7]); % Dark gray asphalt
drawRoad(roadWE, roadWidth, [0.7 0.7 0.7]);
drawRoad(roadWN, roadWidth, [0.7 0.7 0.7]);
drawRoad(roadNE, roadWidth, [0.7 0.7 0.7]);
drawRoad(roadES, roadWidth, [0.7 0.7 0.7]);
drawRoad(roadSW, roadWidth, [0.7 0.7 0.7]);


% Add dashed line along car path
plot(roadSN(1:end-1,1)+normals_roadSN(:,1), roadSN(1:end-1,2)+normals_roadSN(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])
plot(roadSN(1:end-1,1)-normals_roadSN(:,1), roadSN(1:end-1,2)-normals_roadSN(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])

plot(roadWE(1:end-1,1)+normals_roadWE(:,1), roadWE(1:end-1,2)+normals_roadWE(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])
plot(roadWE(1:end-1,1)-normals_roadWE(:,1), roadWE(1:end-1,2)-normals_roadWE(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])

plot(roadWN(1:end-1,1)+normals_roadWN(:,1), roadWN(1:end-1,2)+normals_roadWN(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])
plot(roadWN(1:end-1,1)-normals_roadWN(:,1), roadWN(1:end-1,2)-normals_roadWN(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])

plot(roadNE(1:end-1,1)+normals_roadNE(:,1), roadNE(1:end-1,2)+normals_roadNE(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])
plot(roadNE(1:end-1,1)-normals_roadNE(:,1), roadNE(1:end-1,2)-normals_roadNE(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])

plot(roadES(1:end-1,1)+normals_roadES(:,1), roadES(1:end-1,2)+normals_roadES(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])
plot(roadES(1:end-1,1)-normals_roadES(:,1), roadES(1:end-1,2)-normals_roadES(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])

plot(roadSW(1:end-1,1)+normals_roadSW(:,1), roadSW(1:end-1,2)+normals_roadSW(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])
plot(roadSW(1:end-1,1)-normals_roadSW(:,1), roadSW(1:end-1,2)-normals_roadSW(:,2), 'Linestyle', ':', 'LineWidth', 1, 'Color', [0.6 0.6 0.6])

% Add white mid line
plot(roadSN(:,1), roadSN(:,2), 'LineStyle','--', 'LineWidth', 2, 'Color','w')
plot(roadWE(:,1), roadWE(:,2), 'LineStyle','--', 'LineWidth', 2, 'Color','w')


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
        vehicle_x = roadWE(1, 1) - normals_roadWE(1,1);
        vehicle_y = roadWE(1, 2) - normals_roadWE(1,2);

        % Get the coordinates of the polygon defining the vehicle
        vehicle_shape = vehicleShape(angles_roadWE(1), vehicleLength, vehicleWidth);
        vehicle(i) = fill(vehicle_x + vehicle_shape(1,:), ... %X coord.
                          vehicle_y + vehicle_shape(2,:), ... %Y coord.
                                      colors(i,:));
    end 
    if origin{i} == 'N'
        % Find the position of the beginning of the road, offset to the
        % right lane
        vehicle_x = roadSN(end, 1) + normals_roadSN(end,1);
        vehicle_y = roadSN(end, 2) + normals_roadSN(end,2);

        % Get the coordinates of the polygon defining the vehicle
        vehicle_shape = vehicleShape(angles_roadSN(end), vehicleLength, vehicleWidth);
        vehicle(i) = fill(vehicle_x + vehicle_shape(1,:), ... %X coord.
                          vehicle_y + vehicle_shape(2,:), ... %Y coord.
                                      colors(i,:));
    end

    if origin{i} == 'E'
        % Find the position of the beginning of the road, offset to the
        % right lane
        vehicle_x = roadWE(end, 1) + normals_roadWE(end,1);
        vehicle_y = roadWE(end, 2) + normals_roadWE(end,2);

        % Get the coordinates of the polygon defining the vehicle
        vehicle_shape = vehicleShape(angles_roadWE(end), vehicleLength, vehicleWidth);
        vehicle(i) = fill(vehicle_x + vehicle_shape(1,:), ... %X coord.
                          vehicle_y + vehicle_shape(2,:), ... %Y coord.
                                      colors(i,:));
    end
    if origin{i} == 'S'
        % Find the position of the beginning of the road, offset to the
        % right lane
        vehicle_x = roadSN(1, 1) - normals_roadSN(1,1);
        vehicle_y = roadSN(1, 2) - normals_roadSN(1,2);

        % Get the coordinates of the polygon defining the vehicle
        vehicle_shape = vehicleShape(angles_roadSN(1), vehicleLength, vehicleWidth);
        vehicle(i) = fill(vehicle_x + vehicle_shape(1,:), ... %X coord.
                          vehicle_y + vehicle_shape(2,:), ... %Y coord.
                                      colors(i,:));
    end
    
end
axis off;
% Video setup
videoFilename = 'vehicle_animation.avi';
video = VideoWriter(videoFilename, 'Motion JPEG AVI');
open(video);

% GIF setup
gifFilename = 'vehicle_animation.gif';
shadow_handle = cell(N,1);
% Animation loop
idx_figure = 1;
for t = 1:T_sim
    for i=1:N
        if origin{i}=='W' && destination{i} == 'E'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i) - vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );

                vehicle_x = roadWE(road_progress, 1) - normals_roadWE(road_progress,1);
                vehicle_y = roadWE(road_progress, 2) - normals_roadWE(road_progress,2);
        
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadWE(road_progress), vehicleLength, vehicleWidth);
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = min(road_progress + 10*ceil(v(t,i)), length(roadWE));
                road_section = roadWE(road_progress:arrow_tip_idx,:) - normals_roadWE(road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
            end
        end

        if origin{i}=='W' && destination{i} == 'N'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i) - vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );

                vehicle_x = roadWN(road_progress, 1) - normals_roadWN(road_progress,1);
                vehicle_y = roadWN(road_progress, 2) - normals_roadWN(road_progress,2);

                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadWN(road_progress), vehicleLength, vehicleWidth);
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = min(road_progress + 10*ceil(v(t,i)), length(roadWE));
                road_section = roadWN(road_progress:arrow_tip_idx,:) - normals_roadWN(road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
            end
        end

        if origin{i}=='W' && destination{i} == 'S'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i) - vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );

                vehicle_x = roadSW(end-road_progress, 1) + normals_roadSW(end-road_progress,1);
                vehicle_y = roadSW(end-road_progress, 2) + normals_roadSW(end-road_progress,2);

                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadSW(end-road_progress), vehicleLength, vehicleWidth);
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = max(length(roadSW) - road_progress - 10*ceil(v(t,i)), 1);
                road_section = roadSW(arrow_tip_idx:end-road_progress,:) + normals_roadSW(end-road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));

            else
                set(vehicle(i), 'Visible', 'off'); 
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
            end
        end

        if origin{i}=='N' && destination{i} == 'E'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i) - vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the
                % right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = roadNE(road_progress, 1) - normals_roadNE(road_progress,1);
                vehicle_y = roadNE(road_progress, 2) - normals_roadNE(road_progress,2);
        
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadNE(road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = min(road_progress + 10*ceil(v(t,i)), length(roadNE));
                road_section = roadNE(road_progress:arrow_tip_idx,:) - normals_roadNE(road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
            end
        end

        if origin{i}=='N' && destination{i} == 'W'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i) - vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the
                % right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = roadWN(end-road_progress, 1) + normals_roadWN(end-road_progress,1);
                vehicle_y = roadWN(end-road_progress, 2) + normals_roadWN(end-road_progress,2);
                
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadWN(end-road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = max(length(roadSW) - road_progress - 10*ceil(v(t,i)), 1);
                road_section = roadWN(arrow_tip_idx:end-road_progress,:) + normals_roadWN(end-road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));

            else
                set(vehicle(i), 'Visible', 'off'); 
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
            end
        end

        if origin{i}=='N' && destination{i} == 'S'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i) - vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the
                % right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = roadSN(end-road_progress, 1) + normals_roadSN(end-road_progress,1);
                vehicle_y = roadSN(end-road_progress, 2) + normals_roadSN(end-road_progress,2);
                
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadSN(end-road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = max(length(roadSN) - road_progress - 10*ceil(v(t,i)), 1);
                road_section = roadSN(arrow_tip_idx:end-road_progress,:) + normals_roadSN(end-road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));
                
            else
                set(vehicle(i), 'Visible', 'off'); 
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
            end
        end
        if origin{i}=='E' && destination{i} == 'W'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i)- vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = roadWE(end-road_progress, 1) + normals_roadWE(end-road_progress,1);
                vehicle_y = roadWE(end-road_progress, 2) + normals_roadWE(end-road_progress,2);
        
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadWE(end-road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = max(length(roadWE) - road_progress - 10*ceil(v(t,i)), 1);
                road_section = roadWE(arrow_tip_idx:end-road_progress,:) + normals_roadWE(end-road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
            end
                    
        end

        if origin{i}=='E' && destination{i} == 'N'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i)- vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = roadNE(end-road_progress, 1) + normals_roadNE(end-road_progress,1);
                vehicle_y = roadNE(end-road_progress, 2) + normals_roadNE(end-road_progress,2);
        
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadNE(end-road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = max(length(roadNE) - road_progress - 10*ceil(v(t,i)), 1);
                road_section = roadNE(arrow_tip_idx:end-road_progress,:) + normals_roadNE(end-road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
            end
        end

        if origin{i}=='E' && destination{i} == 'S'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i)- vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the vehicle, offset to the right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = roadES(road_progress, 1) - normals_roadES(road_progress,1);
                vehicle_y = roadES(road_progress, 2) - normals_roadES(road_progress,2);
        
                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadES(road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = min(road_progress + 10*ceil(v(t,i)), length(roadNE));
                road_section = roadES(road_progress:arrow_tip_idx,:) - normals_roadES(road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));
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
                vehicle_x = roadSN(road_progress, 1) - normals_roadSN(road_progress,1);
                vehicle_y = roadSN(road_progress, 2) - normals_roadSN(road_progress,2);

                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadSN(road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = min(road_progress + 10*ceil(v(t,i)), length(roadSN));
                road_section = roadSN(road_progress:arrow_tip_idx,:) - normals_roadSN(road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
            end
        end

        if origin{i}=='S' && destination{i} == 'E'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i)- vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the beginning of the road, offset to the
                % right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = roadES(end - road_progress, 1) + normals_roadES(end-road_progress,1);
                vehicle_y = roadES(end - road_progress, 2) + normals_roadES(end-road_progress,2);

                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadES(end-road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = max(length(roadES) - road_progress - 10*ceil(v(t,i)), 1);
                road_section = roadES(arrow_tip_idx:end-road_progress,:) + normals_roadES(end-road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));
            else
                set(vehicle(i), 'Visible', 'off'); 
            end
        end

        if origin{i}=='S' && destination{i} == 'W'
            if p_abs(t,i)+vehicleLength/2<road_length && p_abs(t,i)- vehicleLength/2>0
                set(vehicle(i), 'Visible', 'on'); 

                % Find the position of the beginning of the road, offset to the
                % right lane
                road_progress = ceil((p_abs(t,i)/road_length) *resolution );
                vehicle_x = roadSW(road_progress, 1) - normals_roadSW(road_progress,1);
                vehicle_y = roadSW(road_progress, 2) - normals_roadSW(road_progress,2);

                % Get the coordinates of the polygon defining the vehicle
                vehicle_shape = vehicleShape(angles_roadSW(road_progress), vehicleLength, vehicleWidth);
                
                % Re-draw the vehicle
                set(vehicle(i), 'XData', vehicle_x + vehicle_shape(1,:));
                set(vehicle(i), 'YData', vehicle_y + vehicle_shape(2,:));

                % Draw shadow on the road with length proportional to speed
                % Take a piece of road in front of the vehicle
                arrow_tip_idx = min(road_progress + 10*ceil(v(t,i)), length(roadSW));
                road_section = roadSW(road_progress:arrow_tip_idx,:) - normals_roadSW(road_progress,:);
                % Delete any precedingly drawn shadow
                if ~isempty(shadow_handle{i})
                    delete(shadow_handle{i});
                end
                % Draw the piece of road in front of the vehicle
                shadow_handle{i} = drawRoad(road_section, vehicleWidth/4, colors(i,:));
                
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
    if mod(t, T_sim/20) == 0
        print("Figures/frame_crossroad_" + num2str(idx_figure) + ".png", '-dpng', '-r600');  % Save with 600 dpi resolution
        idx_figure = idx_figure + 1;
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
fig = figure; 
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

set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0)); % Remove extra padding
set(fig, 'PaperPositionMode', 'auto'); % Adjusts paper size to figure
print('Figures/pos_velocity_dist_to_Xf.png', '-dpng', '-r600');  % Save with 600 dpi resolution

%% Plot number of iterations to convergence

fig = figure;
stairs(x, iter_to_convergence_DR,'marker', '+', 'LineWidth', 1);
if exist("iter_to_convergence_FB", 'var')
    hold on
    stairs(x, iter_to_convergence_FB,'marker', '+', 'LineWidth', 1);
end
grid on
xlabel('$t$', 'Interpreter','latex');
ylabel('\# iterations', 'Interpreter','latex');
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0)); % Remove extra padding
set(fig, 'PaperPositionMode', 'auto'); % Adjusts paper size to figure
legend("DR", "FB")

print('Figures/num_iter_to_convergence.png', '-dpng', '-r600');  % Save with 600 dpi resolution


