function [x] = convertPosVelToState(p, v, v_1, d, h)
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
    if i>1
        x(indexes_position(i)) = p(i-1) - p(i) - d(i) - h(i) * v(i);
        x(indexes_speed(i)) = v(i-1) - v(i);
    else
        x(indexes_position(i)) = 0; %dummy state
        x(indexes_speed(i)) = v_1 - v(1);
    end
end

end

