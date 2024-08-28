function [C_x, d_x, ...
          C_u_loc, d_u_loc,...
          C_u_sh, d_u_sh,...
          C_u_mix, C_x_mix, d_mix] = defineConstraints(N, p)
    n_u = 1;
    n_agent_states = 2;
    n_x = n_agent_states*N; % N decoupled systmes
    %% State constraints
    % Just a cap on velocity and distance between vehicles, in the
    % error-variable coordinate system. It is derived via the telescopic
    % series.
    % v(i) < v_max(i) Becomes sum_{j=1}^i -e_v(i) <= v_max(i) = v_des(1)
    % where e_v(i) is the state associated to the speed error for ag. i
    % p(i-1) - p(i) > d_min  becomes 
    % -e_p(i) - h * sum_{j=1}^i e_v(i) < - d_min(i) + d_des(i) + h*v_des(1)
    C_x = zeros(3*N, n_x); % For each agent: max-min speed, safety distance (not for agent 1, replaced by dummy constraint)
    d_x = zeros(3*N, 1);
    for i = 1:N
        sel_vector_speed = horzcat(ones(1,i), zeros(1,N-i)); % Defined to apply a telescopic sum
        sel_vector_position = horzcat(zeros(1,i-1), 1, zeros(1,N-i));
        C_x(3*(i-1)+1:3*(i-1)+2,:) = kron(sel_vector_speed, [0, -1; ...
                                                             0, 1]);
        d_x(3*(i-1)+1:3*(i-1)+2,:) = [p.max_speed(i) - p.v_des_1;
                                      p.v_des_1 - p.min_speed(i)];
        if i~=1
            % C_x(3*(i-1)+3,:) = p.headway_time(i) * kron(sel_vector_speed, [0, -1]) + ... 
            %                    kron(sel_vector_position, [-1, 0]);
            C_x(3*(i-1)+3,:) = kron(sel_vector_position, [-1, 0]);
            d_x(3*(i-1)+3,:) = -p.d_min(i) + p.d_des(i); % + p.headway_time(i) * p.v_des_1;
        else
            % dummy constraint, it makes indexing easier than excluding it
            C_x(3*(i-1)+3,:) = zeros(1,n_x);  % included just for readability
            d_x(3*(i-1)+3,:) = 0;
        end
    end

    %% Local input constraints
    % Dummy. Box constraints on input are included as mixed input-state
    % constraints to account for the pre-stabilizing controller
    for i = 1:N
        min_u(:,:,i) = p.min_acc(i);
        max_u(:,:,i) = p.max_acc(i);
    end
    C_u_loc = zeros(2*n_u, n_u, N);
    d_u_loc = zeros(2*n_u, 1, N);
    for i=1:N
        C_u_loc(:,:,i) = [-eye(n_u); eye(n_u)];
        d_u_loc(:,:,i) = [-min_u(:,:,i); max_u(:,:,i)]; 
    end

    %% Shared input constraints
    % sum_i C_u(:,:,i) * u_i(t) <= d(:,:,i)
    % Dummy constraints
    C_u_sh = zeros(1, n_u, N);
    d_u_sh = zeros(1, 1);

    %% Mixed input-state constraints
    % Box constraints on the acceleration. They are mixed input-state
    % because of the pre-stabilizing controller
    % C_x * x(t) + sum_i C_u(:,:,i) * u_i(t) <= d(:,:,i)

    % Kx + u <= u_max
    C_u_mix = zeros(2*N,n_u, N);
    C_x_mix = zeros(2*N,n_x);
    d_mix = zeros(2*N,1);
    % for i=1:N
    %     indexes = (i-1)*2+1:i*2;
    %     C_x_mix(indexes,:) = [p.K(:,:,i);
    %                           -p.K(:,:,i)];
    %     C_u_mix(indexes,:,i) = [1; 
    %                             -1];
    %     d_mix(indexes,:) = [p.max_acc(i); 
    %                         -p.min_acc(i)];
    % end

end

