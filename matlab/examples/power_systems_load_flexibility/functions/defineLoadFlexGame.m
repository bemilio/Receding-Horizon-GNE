function [game] = defineLoadFlexGame(N, p)


%% Compute weights 
load_setpoint = sum(p.load) - p.collective_load_ref;
q_g = zeros(N,N,N); % Q-matrices only with generator weights
q_b = zeros(N,1); % weight of battery state deviation (local cost)
q_d = zeros(N,1); % weight of load delay (local cost)
for i=1:N
    for k=1:N
        for j=1:N
            q_g(j,k,i) = p.k_g(i) * p.v_to_P(j) * p.v_to_P(k);
        end
    end
    q_b(i) = p.k_b(i);
    q_d(i) = p.k_d(i);

    v_offset(i) = load_setpoint / (N * p.v_to_P(i));
    b_offset(i) = p.q_des(i);
    d_offset(i) = 0;
end


%% Create matrices 

n_agent_states = 3; % Generator speed, battery charge, deferral of consumption
n_x = N*n_agent_states;
n_u = 3; 

game.A = zeros(n_x, n_x);

for i=1:N
    indexes = (i-1)*n_agent_states+1: i*n_agent_states;
    game.A(indexes, indexes) = [ p.a_g(i), 0,      0;
                                 0,        p.a_b(i), 0;
                                 0,        0,      p.a_d(i)];
end

game.B = zeros(n_x, n_u, N);
for i=1:N
    indexes = (i-1)*n_agent_states+1: i*n_agent_states;
    game.B(indexes,:,i) = [ 1, 0,        0; 
                            0, p.b_b(i), 0;
                            0, 0,        1];
end

game.Q = zeros(n_x, n_x, N);
game.R = zeros(n_u, n_u, N);
for i=1:N
    for k=1:N
        indexes_k = (k-1)*n_agent_states+1: k*n_agent_states;
        for j=1:N
            indexes_j = (j-1)*n_agent_states+1: j*n_agent_states;
            if k==j
                game.Q(indexes_j, indexes_k, i) = [q_g(j,j,i), 0,      0;
                                                   0,          q_b(j), 0;
                                                   0,          0,      q_d(j)]; 
            else
                % Cost coupling in velocity generator
                game.Q(indexes_j, indexes_k, i) = [q_g(j,k,i), 0, 0;
                                                   0,          0, 0;
                                                   0,          0, 0]; 
            end
        end
    end
    game.R(:,:,i)= [p.r_g(i), 0,        0;
                    0,        p.r_b(i), 0;
                    0,        0,        p.r_d(i)];
end
game.n_x = n_x;
game.n_u = n_u;
game.N = N; 
game.offset_x = zeros(n_x,1);
game.offset_u = zeros(n_u,1,N);
for i=1:N
    indexes = (i-1)*n_agent_states+1: i*n_agent_states;
    game.offset_x(indexes) = [v_offset(i); 
                              b_offset(i);
                              d_offset(i)];
end

game.offset_u = targetSelector(game.A, game.B, game.R, game.offset_x);

end

