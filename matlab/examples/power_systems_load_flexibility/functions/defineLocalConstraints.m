function [C_x, d_x, C_u, d_u] = defineLocalConstraints(N, p, x_offs, u_offs)
    n_u = 3;
    n_agent_states = 3;
    n_x = n_agent_states*N; % N decoupled systmes

    min_x = zeros(n_x, 1);
    max_x = zeros(n_x, 1);
    min_u = zeros(n_u, 1, N);
    max_u = zeros(n_u, 1, N);
    cumulative_load_flexibility = rand(N);
    instantaneous_load_flexibility = .2*rand(N);

    for i = 1:N
        indexes = (i-1)*n_agent_states+1: i*n_agent_states;
        min_x(indexes,:) = [ 0; % minimum speed of generators 
                             0; % minimum battery charge
                             p.min_load_delay(i)] ... %load deferral limit
                             - x_offs(indexes);  % traslate to reference point
        max_x(indexes,:) = [1; % max generator speed
                            p.max_b_charge(i); % max battery charge
                            cumulative_load_flexibility(i)] ...
                            - x_offs(indexes); %traslate to reference point
        min_u(:,:,i) = [-p.max_g_rate(i); % minimum acceleration of generators 
                        -p.max_b_rate(i); % minimum battery charging rate 
                        -instantaneous_load_flexibility(i)] ...
                        - u_offs(:,:,i);
        max_u(:,:,i) = [p.max_g_rate(i); % maximum acceleration of generators 
                        p.max_b_rate(i); % maximum battery charging rate 
                        instantaneous_load_flexibility(i)] ...
                        - u_offs(:,:,i);
        % TODO: maximum load (mixed state-input)
    end

    d_x = [-min_x; max_x]; % affine part of constraints on x 
    C_x = [-eye(n_x); eye(n_x)];
    C_u = zeros(2*n_u, n_u, N);
    d_u = zeros(2*n_u, 1, N);
    for i=1:N
        C_u(:,:,i) = [-eye(n_u); eye(n_u)];
        d_u(:,:,i) = [-min_u(:,:,i); max_u(:,:,i)]; 
    end
end

