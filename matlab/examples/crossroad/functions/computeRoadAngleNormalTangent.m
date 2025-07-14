function [normal_vecs, tangent_vecs, angles] = computeRoadAngleNormalTangent(road)

dx = diff(road(:,1)); 
dy = diff(road(:,2));

angles = atan2(dy, dx);

normal_vecs = [-dy, dx]; % Perpendicular vector to the road
normal_vecs = normal_vecs ./ vecnorm(normal_vecs, 2, 2); % Normalize & scale

tangent_vecs = [dx, dy]; 
tangent_vecs = tangent_vecs / norm(tangent_vecs); % Normalize

end

