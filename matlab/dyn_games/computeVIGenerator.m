function [VIgen] = computeVIGenerator(game,T_hor)
%COMPUTEVIGENERATOR returns a function that maps from state to the VI which
% characterizes the open-loop Nash equilibrium problem
    predmod = genPredModel(game.A, game.B, T_hor);
    [W, G] = defineVIMatrices(game, T_hor, predmod);
    % create the stacked matrices that define the constraints over the entire
    % horizon
    % C_u_loc(:,:,i) * u_i <= d_u_loc(:,:,i)
    [C_u_loc_all, d_u_loc_all] = generateInputConstr(game.C_u_loc, game.d_u_loc, T_hor);
    % \sum_i C_u_sh(:,:,i) * u_i <=sum_i d_u_sh(:,:,i)
    [C_u_sh, d_u_sh] = generateSharedInputConstr(game.C_u_sh, game.d_u_sh, T_hor);
    % \sum_i C_x(:,:,i) * u_i <= sum_i D_x(:,:,i) * x_0 + d_x(:,:,i)
    [C_x, D_x, d_x] = generateStateConstr(predmod, game.C_x, game.d_x, T_hor); 
    [C_mix, D_mix, d_mix] = generateMixedConstr(predmod, game.C_x_mix, game.C_u_mix, game.d_mix, T_hor);
    VIgen = @(x_0) genVIFromInitialState(...
        W,G,...
        C_u_loc_all,d_u_loc_all,...
        C_u_sh, d_u_sh,...
        C_x,D_x,d_x,...
        C_mix, D_mix, d_mix,...
        x_0, game.N, game.n_u, T_hor);
end

function [W, G] = defineVIMatrices(game, T_hor, predmod)
% Define the matrices W,G such that
% F(x,u) = W_i u + G_ix_0
% The matrices are defined as:
% W_i = S'Q_iS + R_i, G_i = S'Q_iT
% S and T define the prediction model x = Tx_0 + Su
% (with abuse of notation) Q_i = blkdiag( kron(I, Q_i), P_i ) + R_i 
% R_i = [kron(I,R_ii), 0;
%        0,            0] (up to permutation)
% :return: W, G, 3-D numpy arrays which are the stacks respectively of W_i, G_i
W = zeros(game.n_u * game.N * T_hor, game.n_u * game.N * T_hor, game.N);
G = zeros(game.n_u * game.N * T_hor, game.n_x, game.N);
% H = zeros(game.n_x, game.n_x, game.N);
S_all = reshape(predmod.S, game.n_x * T_hor, game.N * game.n_u * T_hor); % horizontal stack of all S(:,:,i)
for i =1:game.N
    Q_i = blkdiag(kron(eye(T_hor-1), game.Q(:,:,i)), game.P_ol(:,:,i));
    W(:, :, i) = S_all' * Q_i * S_all;
    W(1+(i-1)*game.n_u*T_hor : i*game.n_u*T_hor, 1+(i-1)*game.n_u*T_hor : i*game.n_u*T_hor, i) = ...
        W(1+(i-1)*game.n_u*T_hor : i*game.n_u*T_hor, 1+(i-1)*game.n_u*T_hor : i*game.n_u*T_hor, i) + ...
        kron(eye(T_hor), game.R(:,:,i));
    G(:, :, i) = S_all' * Q_i * predmod.T;
    % H(:, :, i) = predmod.T' * Q_i * predmod.T;
end

end



function [C_all, d_all] = generateInputConstr(C, d, T_hor)
    n_constr = size(C, 1);
    n_u = size(C, 2);
    N = size(C, 3);
    C_all = zeros(T_hor * n_constr, T_hor * n_u, N);
    d_all = zeros(T_hor * n_constr, 1, N);
    for i=1:N
        C_all(:, :, i) = kron(eye(T_hor), C(:,:,i));
        d_all(:, :, i) = kron(ones(T_hor, 1), d(:,:,i));
    end
end

function [C_all, d_all] = generateSharedInputConstr(C, d, T_hor)
    n_constr = size(C, 1);
    n_u = size(C, 2);
    N = size(C, 3);
    C_all = zeros(T_hor * n_constr, T_hor * n_u, N);
    d_all = zeros(T_hor * n_constr, 1, N);
    for i=1:N
        C_all(:, :, i) = kron(eye(T_hor), C(:,:,i));
        d_all(:, :, i) = kron(ones(T_hor, 1), d(:,:)/N);
    end
end


function [C_all, D_all, d_all] = generateStateConstr(predmod, C, d, T_hor)
    n_constr = size(C, 1);
    n_x = size(predmod.T, 2);
    N = size(predmod.S, 3);
    n_u = size(predmod.S, 2)/T_hor;
    C_all = zeros(T_hor * n_constr, T_hor * n_u, N);
    D_all = zeros(T_hor * n_constr, n_x, N);
    d_all = zeros(T_hor * n_constr, 1, N);
    for i=1:N
        C_all(:, :, i) = kron(eye(T_hor), C) * predmod.S(:,:,i);
        D_all(:, :, i) = -kron(eye(T_hor), C) * predmod.T / N;
        d_all(:, :, i) = kron(ones(T_hor, 1), d) / N;
    end
end

function [C_all, D_all, d_all] = generateMixedConstr(predmod, C_x, C_u, d, T_hor)
    % C_x * x(t) + sum_i C_u(:,:,i) * u_i(t) <= d
    n_constr = size(C_x, 1);
    n_x = size(predmod.T, 2);
    N = size(predmod.S, 3);
    n_u = size(predmod.S, 2)/T_hor;
    D_all = zeros(T_hor * n_constr, n_x, N);
    d_all = zeros(T_hor * n_constr, 1, N);
    C_all = zeros(T_hor * n_constr, T_hor * n_u, N);
    for i=1:N
        C_all(:, :, i) = kron(eye(T_hor), C_u(:,:,i));
        % The first row of the pred. mod. is associated to x(1), while the
        % first input is u(0). Need to redefine the prediction model.
        S_realigned = [ zeros(n_x, n_u * T_hor); predmod.S(1:end-n_x,:,i) ]; 
        T_realigned = [ eye(n_x); predmod.T(1:end-n_x,:) ]; 
        C_all(:, :, i) = C_all(:, :, i) + kron(eye(T_hor), C_x) * S_realigned;
        D_all(:, :, i) = -kron(eye(T_hor), C_x) * T_realigned / N;
        d_all(:, :, i) = kron(ones(T_hor, 1), d) / N;
    end
end


function [F, A_sh, b_sh, A_loc, b_loc, n_x, N, Q_mat, g_mat] = genVIFromInitialState( ...
                                        W,G,...
                                        C_u_loc,d_u_loc, ...
                                        C_u_sh,d_u_sh, ...
                                        C_x,D_x,d_x, ...
                                        C_mixed, D_mixed, d_mixed, ...
                                        x_0, N, n_u, T_hor)    

    % given an n*m*p array createa a np * m * p array, where each page of
    % the new array is the column stack of all pages of the original array
    rep = @(u) repmat(reshape(u, [n_u * N * T_hor,1] ), 1,1,N); % For each i, take the rows associated to agent i in W(:,:,i)
    sel_mat = zeros(n_u*T_hor, n_u * N * T_hor, N);  
    for i=1:N
        sel_mat(:, (i-1)*n_u*T_hor+1:i*n_u*T_hor,i) = eye(n_u*T_hor);
    end
    
    Q = pagemtimes(sel_mat, W);
    g = pagemtimes(sel_mat, pagemtimes(G, x_0));
    F = @(u) pagemtimes(Q, rep(u)) + g;

    % Generate matrices for alternative comput. of F as Q*u + g
    Q_mat = zeros(T_hor * N * n_u, T_hor * N * n_u);
    g_mat = zeros(T_hor * N * n_u, 1);
    for i=1:N
        Q_mat( (i-1) * n_u * T_hor + 1:i*n_u* T_hor, :) = Q(:,:,i);
        g_mat( (i-1) * n_u * T_hor + 1:i*n_u* T_hor, :) = g(:,:,i);
    end
    if min(eig(Q_mat + Q_mat')) <= -eps
        warning("The OL-NE VI is not monotone: the minimum eigenvalue is %.2d for the mapping matrix", min(eig(Q_mat + Q_mat')))
        Q_mat
    else
        fprintf("The OL-NE VI is monotone with min eigenvalue %.2d for the mapping matrix", min(eig(Q_mat + Q_mat')))
    end
    
    n_sh_const_u = size(C_u_sh,1);
    n_const_x = size(C_x,1);
    n_const_mix = size(C_mixed,1);
    A_sh = zeros(n_sh_const_u + n_const_x + n_const_mix, n_u*T_hor,N);
    b_sh = zeros(n_sh_const_u + n_const_x + n_const_mix, 1,N);
    d_x0 = pagemtimes(D_x, x_0);
    d_mixed_x0 = pagemtimes(D_mixed, x_0);
    A_loc = C_u_loc;
    b_loc = d_u_loc;
    for i=1:N
        A_sh(:,:,i) = [C_u_sh(:,:,i); 
                       C_x(:,:,i); 
                       C_mixed(:,:,i)];
        b_sh(:,:,i) = [d_u_sh(:,:,i); 
                       d_x0(:,:,i) + d_x(:,:,i); 
                       d_mixed_x0(:,:,i) + d_mixed(:,:,i)];
    end
    n_x = n_u * T_hor;

    % Check if feasible
    A_all = [];
    b_all = [];
    for i=1:N
        A_all = blkdiag(A_all, A_loc(:,:,i));
        b_all = [b_all; b_loc(:,:,i)];
    end
    A_all = [A_all;
             reshape(A_sh, [size(A_sh,1), size(A_sh,2) * size(A_sh, 3)])];
    b_all = [b_all; sum(b_sh, 3)];
    options = optimoptions('quadprog','Display','off');
    [~,~,exit_flag] = quadprog(eye(n_x*N), zeros(n_x*N,1), A_all, b_all, [],[],[],[],[],options);
    if exit_flag<0
        error("[ol-NE] The VI is infeasible")
    end

end