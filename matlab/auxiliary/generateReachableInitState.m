function x_0 = generateReachableInitState(game, expmpc, X_f, T)
    is_init_state_reachable = false;
    
    n_x = expmpc{1}.nx;
    n_u = expmpc{1}.nu;
    N = length(expmpc);

    while ~is_init_state_reachable 
        x_0 = game.min_x + (diag(game.max_x - game.min_x) * rand(n_x, 1));
        u_full_trajectory = zeros(game.n_u*T, 1, N);
        for i=1:N
            u_full_trajectory(:,:,i) = expmpc{i}.optimizer.feval(x_0, 'primal');
        end
        x_T = evolveState(x_0, game.A, game.B, u_full_trajectory, T);
        is_init_state_reachable = X_f.contains(x_T);    
    end
end