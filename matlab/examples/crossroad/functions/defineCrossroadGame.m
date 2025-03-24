function [game] = defineCrossroadGame(N, p)



% Define basic constants
n_agent_states = 2; % position, speed
n_x = N*n_agent_states;
n_u = 1; % Acceleration 

% Store the costants in the game instance
game.n_x = n_x;
game.n_u = n_u;
game.N = N; 



%% Define dynamics and objectives
game.A = zeros(n_x, n_x);
game.B = zeros(n_x, n_u, N);

game.Q = zeros(n_x, n_x, N);
game.R = zeros(n_u, n_u, N);

for i=1:N
    
    
    % Find agent that i has to wait before passing
    j = agentBefore(i, p.crossing_dir, p.conflicts_dict);

    % Indexes of states associated to agent i
    indexes = (i-1)*n_agent_states+1: i*n_agent_states;

    if ~isempty(j)
        game.B(indexes,:,i) = [ -p.headway_time(i)*p.T_sampl- p.T_sampl^2/2; 
                                -p.T_sampl];
        % The input of agent j also affects i
        game.B(indexes,:,j) = [p.T_sampl^2/2; p.T_sampl];
    else
        % if the agent does not have to wait for anyone, then the
        % "position" state is zero (leading agent)
        game.B(indexes,:,i) = [ 0; -p.T_sampl];
    end

    if ~isempty(j)
        game.A(indexes, indexes) = [ 1, p.T_sampl;
                                     0,        1];
    else % leading agent
        % Position error of the leading agent is a dummy state, but A needs
        % to be non-singular. Remember to set the initial state to 0.
        game.A(indexes, indexes) = [ 0.1,       0;
                                     0,         1];
    end

    game.Q(:, :, i) = 1 * eye(n_x); 
    % game.Q(indexes, indexes, i) = eye(n_agent_states); % Big weight on my state
    game.R(:,:,i)= 1;
end
% Include the pre-stabilizing controllers
game.A = game.A + sum(pagemtimes(game.B, p.K), 3);
if max(abs(eig(game.A)))>1
    warning("[defineVehiclePlatooningGame] The pre-stabilizing controllers do not stabilize the system. The OL-NE will likely not work.")
end

%% Define constraints

% State constraints:
% Cap on velocity and distance between vehicles, in the
% error-variable coordinate system. It is derived via the telescopic series.
% v(i) < v_max(i) Becomes sum_{j\in{agents_blocking_i} } -e_v(i) <= v_max(i) - v_des(1)
% where e_v(i) is the state associated to the speed error for ag. i
% and {agents_blocking_i} is the set that contains the j-th agent that i has
% to maintain distance fro, the agent from which j has to maintain distance
% from, and so on
% p(i-1) - p(i) > d_min becomes 
% -e_p(i) - h * sum_{j\in{agents_blocking_i}} e_v(i) < - d_min(i) + d_des(i) + h*v_des(1)
game.C_x = zeros(3*N, n_x); % For each agent: max-min speed, safety distance (not for agent 1, replaced by dummy constraint)
game.d_x = zeros(3*N, 1);
for i = 1:N
    % j-th element is 1 if the i-th agent has to wait for j before passing
    % or if j==i
    if i==5
        disp("pause")
    end
    agents_blocking_i = zeros(1,N);
    % Find the agent that i has to wait for before passing
    j = agentBefore(i, p.crossing_dir, p.conflicts_dict);
    while ~isempty(j)
        % Include j in the set of agents that i has to wait for
        agents_blocking_i(j) = 1;
        % Find the agent that j has to wait before passing
        j = agentBefore(j, p.crossing_dir, p.conflicts_dict);
    end
    selector_agent_i = horzcat(zeros(1,i-1), 1, zeros(1,N-i));
    % Speed constraint
    game.C_x(3*(i-1)+1:3*(i-1)+2,:) = kron(agents_blocking_i + selector_agent_i, [0, -1; ...
                                                                                  0, 1]);
    game.d_x(3*(i-1)+1:3*(i-1)+2,:) = [p.max_speed(i) - p.v_des_1;
                                  p.v_des_1 - p.min_speed(i)];
    % Position constraint 
    if ~isempty(agentBefore(i, p.crossing_dir, p.conflicts_dict)) % True if the vehicle is not leading
        game.C_x(3*(i-1)+3,:) = p.headway_time(i) * kron(agents_blocking_i, [0, -1]) + ... 
                                                    kron(selector_agent_i,  [-1, 0]);
        game.d_x(3*(i-1)+3,:) = -p.d_min(i) + p.d_des(i) + p.headway_time(i) * p.v_des_1;
    else
        % dummy constraint, it makes indexing easier than excluding it
        game.C_x(3*(i-1)+3,:) = zeros(1,n_x);  % included just for readability
        game.d_x(3*(i-1)+3,:) = 0;
    end
end

%% Local input constraints
% Dummy. Box constraints on input are included as mixed input-state
% constraints to account for the pre-stabilizing controller
for i = 1:N
    min_u(:,:,i) = p.min_acc(i);
    max_u(:,:,i) = p.max_acc(i);
end
game.C_u_loc = zeros(2*n_u, n_u, N);
game.d_u_loc = zeros(2*n_u, 1, N);
for i=1:N
    game.C_u_loc(:,:,i) = [-eye(n_u); eye(n_u)];
    game.d_u_loc(:,:,i) = [-min_u(:,:,i); max_u(:,:,i)]; 
end

%% Shared input constraints
% sum_i C_u(:,:,i) * u_i(t) <= d(:,:,i)
% Dummy constraints
game.C_u_sh = zeros(1, n_u, N);
game.d_u_sh = zeros(1, 1);

%% Mixed input-state constraints - dummy
game.C_u_mix = zeros(2*N,n_u, N);
game.C_x_mix = zeros(2*N,n_x);
game.d_mix = zeros(2*N,1);


end
