function x_T = evolveState(x_0, A, B, u, T, n_u)
    x_T = x_0;
    N = size(u,3);
    for tau=1:T
        x_T = A * x_T;
        for i=1:N
            x_T = x_T + B(:,:,i) * u(1+(tau-1)*n_u:tau*n_u, i);
        end
    end
end
