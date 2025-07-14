function [x] = convertPosVelToState(p, v, v_1, d, h, crossing_dir, conflicts_dict)
% Input: position-velocity coordinates,
% desired speed of leading agent v_1, desired distance from previous agent d,
% desired time delay from previous sgent h. 
% The input position is with respect to the leading agent.

N = length(p);
n_x_per_agent = 2;
n_x = N * n_x_per_agent;
x = zeros(n_x,1);
indexes_position = 1:n_x_per_agent:n_x;
indexes_speed = 2:n_x_per_agent:n_x;
for i=1:N
    j = agentBefore(i, crossing_dir, conflicts_dict);
    if ~isempty(j)
        x(indexes_position(i)) = p(j) - p(i) - d(i) - h(i) * v(i);
        x(indexes_speed(i)) = v(j) - v(i);
    else %leading vehicle
        x(indexes_position(i)) = 0; %dummy state
        x(indexes_speed(i)) = v_1 - v(i);
    end
end

end

