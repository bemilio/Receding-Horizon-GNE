function [p] = defineVehiclePlatooningGameParameters(N)

%% Define Parameters 
% for details see IFAC conference Shi/Lazar 2017

n_u = 1;
n_agent_states = 2;
n_x = n_agent_states * N;

p.T_sampl = .1;
p.headway_time = ones(N,1); % this term is used to make safety distance dependend on speed, seconds. see paper.
p.headway_time(1) = 0;
p.v_des_1 = 30; %desired speed of leading vehicle, m/s
p.max_speed = 40*ones(N,1); % upper speed limit, m/s
p.min_speed = 10*ones(N,1); % lower speed limit, m/s
p.d_des = 100*ones(N,1); % Desired safety distance, not dependent on speed, m
p.d_des(1) = 0;
p.d_min = 20*ones(N,1); % Minimum safety distance, m
p.max_acc = 30* ones(N,1); % Max acceleration, m/s^2
p.min_acc = -30* ones(N,1); % Max acceleration, m/s^2

%% Define pre-stabilizing controller
p.K = zeros(n_u,n_x,N);
gain = 0.1;
for i=1:N
    % Local proportional controller
    indexes = (i-1)*n_agent_states+1: i*n_agent_states;
    p.K(:,indexes, i) = gain*[1, 1]; % Proportional controller on position
end

end

