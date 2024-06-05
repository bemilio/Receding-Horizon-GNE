function [game] = defineVehiclePlatooningGame(N, p)



%% Create matrices 
n_agent_states = 2; % position, speed
n_x = N*n_agent_states;
n_u = 1; % Acceleration 

%% Define pre-stabilizing controller
K = zeros(n_u,n_x,N);
gain = 0.1;
for i=1:N
    % Local proportional controller
    indexes = (i-1)*n_agent_states+1: i*n_agent_states;
    K(:,indexes, i) = gain*[1, 0]; % Proportional controller on position
end


%% Define dynamics
game.B = zeros(n_x, n_u, N);
for i=1:N
    indexes = (i-1)*n_agent_states+1: i*n_agent_states;
    indexes_succ = i*n_agent_states+1: (i+1)*n_agent_states;
    game.B(indexes,:,i) = [ -p.headway_time*p.T_sampl- p.T_sampl^2/2; 
                            -p.T_sampl];
    if i<N
        game.B(indexes_succ,:,i) = [p.T_sampl^2/2; 
                                    p.T_sampl];
    end
end

game.A = zeros(n_x, n_x);

for i=1:N
    indexes = (i-1)*n_agent_states+1: i*n_agent_states;
    game.A(indexes, indexes) = [ 1, p.T_sampl;
                                 0,       1];
end
% Include the pre-stabilizing controllers
game.A = game.A + sum(pagemtimes(game.B, K), 3);
if max(abs(eig(game.A)))>1
    warning("[defineVehiclePlatooningGame] The pre-stabilizing controllers do not stabilize the system. The OL-NE will likely not work.")
end


game.Q = zeros(n_x, n_x, N);
game.R = zeros(n_u, n_u, N);
for i=1:N
    indexes = (i-1)*n_agent_states+1: i*n_agent_states;
    game.Q(:, :, i) = 1 * eye(n_x); 
    % game.Q(indexes, indexes, i) = eye(n_agent_states); % Big weight on my state
    game.R(:,:,i)= 1;
end
game.n_x = n_x;
game.n_u = n_u;
game.N = N; 


end