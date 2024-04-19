function [C_x, d_x, C_u, d_u] = defineConstraints(N, x_ref, u_ref)
    n_x = 3*N; % N decoupled systmes
    n_u = 3;
    min_x = zeros(n_x, 1);
    max_x = zeros(n_x, 1);
    min_u = zeros(n_u, 1, N);
    max_u = zeros(n_u, 1, N);
    cumulative_load_flexibility = rand(N);
    instantaneous_load_flexibility = .2*rand(N);
    for i = 1:N
        min_x((i-1)*(n_x/N)+1:i*(n_x/N),:) = [ 0; % minimum speed of generators 
                         0; % minimum battery charge
                         -cumulative_load_flexibility(i)] ... %load deferral limit
                         - x_ref(:,:,i);  % traslate to reference point
        max_x((i-1)*(n_x/N)+1:i*(n_x/N),:) = [1; % max generator speed
                                              1; % max battery charge
                                              cumulative_load_flexibility(i)] ...
                                              - x_ref(:,:,i); %traslate to reference point
        min_u(:,:,i) = [0; % minimum acceleration of generators 
                        -1; % minimum battery charging rate 
                        -instantaneous_load_flexibility(i)] ...
                        - u_ref(:,:,i);
        max_u(:,:,i) = [1; % maximum acceleration of generators 
                        1; % maximum battery charging rate 
                        instantaneous_load_flexibility(i)] ...
                        - u_ref(:,:,i);
        
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

