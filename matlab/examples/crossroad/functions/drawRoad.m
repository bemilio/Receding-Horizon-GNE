% Function to draw a filled road area
function handle = drawRoad(road, width, color)
    % Calculate normal vectors for width
    dx = diff(road(:,1)); dy = diff(road(:,2));
    norms = [dy, -dx]; % Perpendicular vector
    norms = norms ./ vecnorm(norms, 2, 2) * (width / 2); % Normalize & scale

    % Extend road points to create a filled polygon
    leftEdge = road(1:end-1, :) + norms;
    rightEdge = road(1:end-1, :) - norms;
    roadPatch = [leftEdge; flipud(rightEdge)];
    handle = fill(roadPatch(:,1), roadPatch(:,2), color, 'EdgeColor', 'none');
    %drawDashedLine(road); BROKEN
end


function drawDashedLine(road)
    for i = 1:10:length(road)-10
        plot(road(i:i+5, 1), road(i:i+5, 2), 'w--', 'LineWidth', 2);
    end
end