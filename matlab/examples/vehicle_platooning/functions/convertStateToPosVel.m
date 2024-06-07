function [p,v] = convertStateToPosVel(x, v_1, d, h)
% Converts error variables to position and velocity. Input: state vector x,
% desired speed of leading agent v_1, desired distance from previous agent d,
% desired time delay from previous sgent h. 
% The resulting position is with respect to the leading agent.


n_x = length(x);
n_x_per_agent = 2; %position error var, speed error var
N = n_x/n_x_per_agent;
p = zeros(N,1);
v = zeros(N,1);
indexes_position = 1:n_x_per_agent:n_x;
indexes_speed = 2:n_x_per_agent:n_x;
%% Conversion from error variables to actual speed/position
for i=1:N
    telesc_sum_operator = horzcat(ones(1,i), zeros(1,N-i)); % Defined to apply a telescopic sum
    v(i) = v_1 - telesc_sum_operator * x(indexes_speed);
end

for i=1:N
    % Defined to apply a telescopic sum. The  state of agent 1is excluded as it
    % is a dummy state
    telesc_sum_operator = horzcat(0, ones(1,i-1), zeros(1,N-i));  
    p(i) = telesc_sum_operator * (x(indexes_position)+ d + h.*v);
end

end

