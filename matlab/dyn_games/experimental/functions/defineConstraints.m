function [C_x, d_x, C_u, d_u] = defineConstraints(N, n_x, n_u)

    min_u = -ones(n_u, 1, N); 
    max_u = ones(n_u, 1, N);
    min_u(:,:,2)=0.1*min_u(:,:,2);
    max_u(:,:,2)=0.1*max_u(:,:,2);
    
    d_x = [1]; % dummy
    C_x = zeros(1,n_x); % dummy
    C_u = zeros(2*n_u, n_u, N);
    d_u = zeros(2*n_u, 1, N);
    for i=1:N
        C_u(:,:,i) = [-eye(n_u); eye(n_u)];
        d_u(:,:,i) = [-min_u(:,:,i); max_u(:,:,i)]; 
    end
end

