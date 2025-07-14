function [p,v] = convertStateToPosVel(x, v_1, d, h, crossing_dir, conflicts_dict)
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
% v_i = v_1ref - e_i^v - \sum{agents_blocking_i} e_j^v
% where e_j^v is the state (error variable) associated to the velocity of j
% v_1ref is the reference velocity for the leading agents
% and {agents_blocking_i} is the set that contains the j-th agent that i has
% to maintain distance from, the agent from which j has to maintain distance
% from, and so on
for i=1:N
    % j-th element is 1 if the i-th agent has to wait for j before passing 
    agents_blocking{i} = zeros(1,N);
    % Find the agent that i has to wait for before passing
    j = agentBefore(i, crossing_dir, conflicts_dict);
    while ~isempty(j)
        % Include j in the set of agents that i has to wait for
        agents_blocking{i}(j) = 1;
        % Find the agent that j has to wait before passing
        j = agentBefore(j, crossing_dir, conflicts_dict);
    end
    selector_agent_i = horzcat(zeros(1,i-1), 1, zeros(1,N-i));

    v(i) = v_1 - (agents_blocking{i} + selector_agent_i) * x(indexes_speed);
end

for i=1:N
    j = agentBefore(i, crossing_dir, conflicts_dict);
    if ~isempty(j)
        p(i) = p(j) - x(2*(i-1)+1) - d(i) - h(i)*v(i);
    end
end

end

