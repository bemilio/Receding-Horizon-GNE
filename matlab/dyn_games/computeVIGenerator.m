function [VIgen] = computeVIGenerator(game,T_hor)
%COMPUTEVIGENERATOR returns a function that maps from state to the VI which
% characterizes the open-loop Nash equilibrium problem

predmod = genPredModel(game.A, game.B, T_hor);
W, G, H = defineVICostFunction(game, T_hor, predmod);
% create the stacked matrices that define the constraints over the entire
% horizon
% C_u_loc(:,:,i) * u_i <= d_u_loc(:,:,i)
C_u_loc, d_u_loc = generateInputConstr(game.C_u_loc, game.d_u_loc, T_hor);
% \sum_i C_u_sh(:,:,i) * u_i <=sum_i d_u_sh(:,:,i)
C_u_sh, d_u_sh = generateInputConstr(game.C_u_sh, game.d_u_sh, T_hor);
% \sum_i C_x(:,:,i) * u_i <= sum_i D_x(:,:,i) * x_0 + d_x(:,:,i)
C_x, D_x, d_x = generateStateConstr(predmod, game.C_x, game.d_x, T_hor); 
VIgen = @(x_0) genVIFromInitialState(W,G,H,C_u_loc,d_u_loc,C_u_sh,d_u_sh,C_x,D_x,d_x,x_0);
end

function [W, G, H] = defineVICostFunction(game, T_hor, predmod)

% Define the cost
% J_i = .5 u'W_iu + u'G_ix_0 + .5 x_0'H_ix_0
% where W = S'Q_iS + R_i, G_i = S'Q_iT, H_i = T'Q_iT
% S and T define the prediction model x = Tx_0 + Su
% (with abuse of notation) Q_i = blkdiag( kron(I, Q_i), P_i ) + R_i (right hand side is related to stage cost)
% R_i = [kron(I,R_ii), 0;
%        0,            0] (up to permutation)
% :return: W, G, H 3-D numpy arrays which are the stacks respectively of W_i, G_i, H_i
W = zeros(game.n_u * game.N * T_hor, game.n_u * game.N * T_hor, game.N);
G = zeros(game.n_u * game.N * T_hor, game.n_x, game.N);
H = zeros(game.n_x, game.n_x, game.N);
S_all = reshape(predmod.S, game.n_x * T_hor, game.N * game.n_u * T_hor); % horizontal stack of all S(:,:,i)
for i =1:game.N
    Q_i = blkdiag(kron(eye(T_hor-1), game.Q(:,:,i)), game.P(:,:,i));
    W(:, :, i) = S_all' * Q_i * S_all;
    W(1+(i-1)*game.n_u*T_hor : i*game.n_u*T_hor, 1+(i-1)*game.n_u*T_hor : i*game.n_u*T_hor, i) = ...
        W(1+(i-1)*game.n_u*T_hor : i*game.n_u*T_hor, 1+(i-1)*game.n_u*T_hor : i*game.n_u*T_hor, i) + ...
        kron(eye(T_hor), game.R(:,:,i));
    G(:, :, i) = S_all' * Q_i * predmod.T;
    H(:, :, i) = predmod.T' * Q_i * predmod.T;
end

end

function [C_all, d_all] = generateInputConstr(C, d, T_hor)
    n_constr = shape(C, 1);
    n_u = shape(C, 2);
    N = shape(C, 3);
    C_all = zeros(T_hor * n_constr, T_hor * n_u, N);
    d_all = np.zeros(T_hor * n_constr, 1, N);
    for i=1:N
        C_all(:, :, i) = kron(eye(T_hor), C(:,:,i));
        d_all(:, :, i) = kron(ones(T_hor, 1), d(:,:,i));
    end
end


function [C_all, D_all, d_all] = generateStateConstr(predmod, C, d, T_hor)
    n_constr = shape(C, 1);
    n_x = shape(C, 2);
    N = shape(C, 3);
    n_u = shape(predmod.S, 2)/T_hor;
    C_all = zeros(T_hor * n_constr, T_hor * n_u, N);
    D_all = zeros(T_hor * n_constr, n_x, N);
    d_all = zeros(T_hor * n_constr, N);
    for i=1:N
        C_all(:, :, i) = kron(eye(T_hor), C) * predmod.S(:,:,i);
        D_all(:, :, i) = -kron(eye(T_hor), C) * predmod.T / N;
        d_all(:, :, i) = kron(ones(T_hor, 1), d) / N;
    end
end

function [J,F] = genVIFromInitialState(W,G,H,C_u_loc,d_u_loc,C_u_sh,d_u_sh,C_x,D_x,d_x,x_0)
    % given an n*m*p array createa a np * m * p array, where each page of
    % the new array is the column stack of all pages of the original array
    rep = @(u) repmat(reshape(u, [shape(u,1) * shape(u,3),1] ), 1,1,shape(u,3)); 
    h = .5 * x_0'*H*x_0;
    g = pagemtimes(G, x_0);
    J = @(u) .5 * pagemtimes(T3D(rep(u)), pagemtimes(W,rep(u))) + ...
        pagemtimes(T3D(rep(u)), g) + h;
    % page-multiply sel_mat to a np * m * p array so that the resulting
    % n*m*p array contains in the i-th page the p rows associated to agent i
    sel_mat = zeros(shape(u,1), shape(u,1) * shape(u,3), shape(u,3) );
    for i=1:N
    sel_mat = kron(T3D(eye(N)), eye(n_x));
    F = @(u) W*u + 

end