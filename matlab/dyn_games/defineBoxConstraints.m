function [C_x, d_x, C_u, d_u] =  defineBoxConstraints(n_x, n_u, min_x, max_x, min_u, max_u, N)
    d_x = [-min_x; max_x]; % affine part of constraints on x 
    C_x = [-eye(n_x); eye(n_x)];
    C_u = zeros(2*n_u, n_u, N);
    d_u = zeros(2*n_u, 1, N);
    for i=1:N
        C_u(:,:,i) = [-eye(n_u); eye(n_u)];
        d_u(:,:,i) = [-min_u; max_u]; % affine part of constraints on x 
    end
end