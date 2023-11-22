function x_T = evolveState(x_0, A, B, u, T)
    x_T = x_0;
    n_u = size(u,1)/T;
    for tau=1:T
        x_T = A * x_T + sum(pagemtimes(B, u(1+(tau-1)*n_u:tau*n_u, :, :)), 3);
    end
end
