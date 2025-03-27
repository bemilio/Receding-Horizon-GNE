function [p] = defineCrossroadGameParameters(N)

%% Define Parameters 
% for details see IFAC conference Shi/Lazar 2017

n_u = 1;
n_agent_states = 2;
n_x = n_agent_states * N;

p.T_sampl = .1;
p.headway_time = 0*ones(N,1); % this term is used to make safety distance dependend on speed, seconds. see paper.
p.headway_time(1) = 0;
p.v_des_1 = 6; %desired speed of leading vehicle, m/s
p.max_speed = 10*ones(N,1); % upper speed limit, m/s
p.min_speed = 1*ones(N,1); % lower speed limit, m/s
p.d_des = 15*ones(N,1); % Desired safety distance, not dependent on speed, m
p.d_des(1) = 0;
p.d_min = 10*ones(N,1); % Minimum safety distance, m
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

% Define all directions that one can take on the crossroad
p.all_crossing_dir = ["WE", "WN", "WS", "NW", "NE", "NS", "EW", "EN", "ES", "SN", "SW", "SE" ]; 

% For every direction that one can take, define which other directions one
% has to take care of
p.crossings_conflicts = {
        ["WE", "WN", "WS", "NE", "NS", "ES", "SN", "SW", "SE"], % Trajectories conflicting with WE
        ["WE", "WN", "WS", "NE", "NS", "EW", "EN", "ES", "SN", "SW"], % Trajectories conflicting with WN
        ["WE", "WN", "WS", "NS", "ES"], % Trajectories conflicting with WS

        ["NW", "NE", "NS", "EW", "SW"], % Trajectories conflicting with NW
        ["WE", "WN", "NW", "NE", "NS", "EW", "ES", "SN", "SW", "SE"], % Trajectories conflicting with NE
        ["WE", "WN", "WS", "NW", "NE", "NS", "EW", "ES", "SW"], % Trajectories conflicting with NS

        ["WN", "NW", "NE", "NS", "EW", "EN", "ES", "SN", "SW"] % Trajectories conflicting with EW
        ["WN", "EW", "EN", "ES", "SN"] % Trajectories conflicting with EN
        ["WE", "WN", "WS", "NE", "NS", "EW", "EN", "ES", "SN", "SW"] % Trajectories conflicting with ES

        ["WE", "WN", "NE", "EW", "EN", "ES", "SN", "SW", "SE"] % Trajectories conflicting with SN
        ["WE", "WN", "NW", "NE", "NS", "EW", "ES", "SN", "SW", "SE"] % Trajectories conflicting with SW
        ["WE", "NE", "SN", "SW", "SE"] % Trajectories conflicting with SE
    };


% Define desired directions that agents take on the crossroad
p.crossing_dir = randsample(p.all_crossing_dir, N, true);
% p.crossing_dir = ["WE", "EW", "SN", "SN", "NS", "NS", "WE", "SN", "SN", "WE"];

% Create dictionary that maps from direction on crossroad to all
% conflicting directions
p.conflicts_dict = containers.Map(p.all_crossing_dir, p.crossings_conflicts);

end

