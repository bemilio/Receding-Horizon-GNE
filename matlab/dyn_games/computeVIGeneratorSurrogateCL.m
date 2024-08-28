function [VIgen] = computeVIGeneratorSurrogateCL(game,T_hor)
%COMPUTEVIGENERATOR returns a function that maps from state to the VI which
% characterizes the surrogate CL-NE problem
    predmod = genPredModelSurrogateCL(game.A, game.B, game.K_cl, T_hor);
    [W, G, H] = defineVICostFunction(game, T_hor, predmod);
    % create the stacked matrices that define the constraints over the entire
    % horizon
    % C_u_loc(:,:,i) * u_i <= d_u_loc(:,:,i)
    [C_u_loc_all, d_u_loc_all] = generateInputConstr(game.C_u_loc, game.d_u_loc, T_hor);
    % \sum_i C_u_sh(:,:,i) * u_i <=sum_i d_u_sh(:,:,i)
    [C_u_sh, D_u_sh, d_u_sh] = generateSharedInputConstr(predmod, game.C_u_sh, game.d_u_sh, game.K_cl, T_hor);
    % \sum_i C_x(:,:,i) * u_i <= sum_i D_x(:,:,i) * x_0 + d_x(:,:,i)
    [C_x, D_x, d_x] = generateStateConstr(predmod, game.C_x, game.d_x, T_hor); 
    [C_mix, D_mix, d_mix] = generateMixedConstr(predmod, game.C_x_mix, game.C_u_mix, game.d_mix, T_hor);
    VIgen = @(x_0) genVIFromInitialState(...
        W,G,H,...
        C_u_loc_all,d_u_loc_all,...
        C_u_sh, D_u_sh, d_u_sh,...
        C_x,D_x,d_x,...
        C_mix, D_mix, d_mix,...
        x_0, game.N, game.n_u, T_hor);
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
H = zeros(game.n_x, game.n_x, game.N); % TODO: define H.
for i =1:game.N
    S_all = reshape(predmod.S{i}, game.n_x * T_hor, game.N * game.n_u * T_hor); % horizontal stack of all S(:,:,i)
    Q_i = blkdiag(kron(eye(T_hor-1), game.Q(:,:,i)), game.P_cl(:,:,i));
    W(:, :, i) = S_all' * Q_i * S_all;
    indexes_u_i = 1+(i-1)*game.n_u*T_hor : i*game.n_u*T_hor;
    W(indexes_u_i, indexes_u_i, i) = ...
        W(indexes_u_i, indexes_u_i, i) + kron(eye(T_hor), game.R(:,:,i));
    G(:, :, i) = S_all' * Q_i * predmod.T{i};
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


function [C_all, D_all, d_all ] = generateSharedInputConstr(predmod, C, d, K, T_hor)
    % For each agent, the predicted input of the remaining agent is Kx. 
    % Thus, each agent has a different set of constraints.
    % For t=0:
    % \sum_i C_i u_i[0] <= d
    % For t >=1, for all i:
    % C_i u_i[t] + sum_{j~i} C_j K_j x_j[t] <= d
    % Where x_k is the j-th prediction of the state given by the pred. mod.
    n_u = size(C, 2);
    N = size(C, 3);
    n_constr = size(C, 1); 
    n_x = size(predmod.T{1}, 2);
    % At t=0: same constraint for all agents. For every successive
    % timestep: each agent introduces a different set of shared constr.
    C_all = zeros(n_constr + (T_hor-1) * N * n_constr, T_hor * n_u, N);
    D_all = zeros(n_constr + (T_hor-1) * N * n_constr, n_x, N);
    d_all = zeros(n_constr + (T_hor-1) * N * n_constr, 1, N);
    % Constraints for first timestep
    C_all(1:n_constr, 1:n_u, :) = C(:,:,:);
    D_all(1:n_constr, :, :) = zeros(n_constr, n_x, N);
    for i=1:N
        d_all(1:n_constr,:,i) = d(:,:,:)/N;
    end
    % Constraints for remaining timestep
    for i=1:N
        index_constr = n_constr + (i-1)*(T_hor-1)*n_constr + 1 : ...
                       n_constr + i*(T_hor-1)*n_constr;
        S_i = predmod.S{i}(1:end-n_x,1:end-n_u,i);
        CK_i = kron(eye(T_hor-1), sum(pagemtimes(C,K),3)-C(:,:,i)*K(:,:,i)); % kron(I, sum_{j~i} C_jK_j)
        C_all(index_constr, n_u+1:end, i) = kron(eye(T_hor-1), C(:,:,i)) + CK_i * S_i;
        for j=1:N
            if j~=i
                S_j = predmod.S{i}(1:end-n_x,1:end-n_u,j);
                C_all(index_constr, n_u+1:end, j) = CK_i * S_j; 
            end
            D_all(index_constr, :, j) = -CK_i * predmod.T{i}(1:end-n_x,:)/N; %Notice that T is indexed in i (not on ) because this is the i-th state prediction.
            d_all(index_constr, :, j) = kron(ones(T_hor-1, 1), d(:,:)/N);
        end
    end
end




function [C_all, D_all, d_all] = generateStateConstr(predmod, C, d, T_hor)
    n_constr = size(C, 1);
    n_x = size(C, 2);
    N = size(predmod.S{1}, 3);
    n_u = size(predmod.S{1}, 2)/T_hor;
    % There is one set of constraint for each agent, as the state
    % prediction is different for each agent.
    C_all = zeros(N * T_hor * n_constr, T_hor * n_u, N);
    D_all = zeros(N * T_hor * n_constr, n_x, N);
    d_all = zeros(N * T_hor * n_constr, 1, N);
    for i=1:N
        index_constr = (i-1)*T_hor*n_constr + 1 : ...
                       i*T_hor*n_constr;
        for j=1:N
            C_all(index_constr, :, j) = kron(eye(T_hor), C) * predmod.S{i}(:,:,j);
            D_all(index_constr, :, j) = -kron(eye(T_hor), C) * predmod.T{i} / N;
            d_all(index_constr, :, j) = kron(ones(T_hor, 1), d) / N;
        end
    end
end

function [C_all, D_all, d_all] = generateMixedConstr(predmod, C_x, C_u, d, T_hor)
% TODO
    n_x = size(predmod.T{1}, 2);
    N = size(predmod.S{1}, 3);
    n_u = size(predmod.S{1}, 2)/T_hor;
    C_all = zeros(1, T_hor * n_u, N);
    D_all = zeros(1, n_x, N);
    d_all = ones(1, 1, N);

    % C_x * x(t) + sum_i C_u(:,:,i) * u_i(t) <= d
    % n_constr = size(C_x, 1);
    % n_x = size(predmod.T, 2);
    % N = size(predmod.S, 3);
    % n_u = size(predmod.S, 2)/T_hor;
    % D_all = zeros(T_hor * n_constr, n_x, N);
    % d_all = zeros(T_hor * n_constr, 1, N);
    % C_all = zeros(T_hor * n_constr, T_hor * n_u, N);
    % for i=1:N
    %     C_all(:, :, i) = kron(eye(T_hor), C_u(:,:,i));
    %     % The first row of the pred. mod. is associated to x(1), while the
    %     % first input is u(0). Need to redefine the prediction model.
    %     S_realigned = [ zeros(n_x, n_u * T_hor); predmod.S(1:end-n_x,:,i) ]; 
    %     T_realigned = [ eye(n_x); predmod.T(1:end-n_x,:) ]; 
    %     C_all(:, :, i) = C_all(:, :, i) + kron(eye(T_hor), C_x) * S_realigned;
    %     D_all(:, :, i) = -kron(eye(T_hor), C_x) * T_realigned / N;
    %     d_all(:, :, i) = kron(ones(T_hor, 1), d) / N;
    % end
end

% TODO: REWRITE FROM HERE!

function [J,F, A_sh, b_sh, A_loc, b_loc, n_x, N] = genVIFromInitialState( ...
                                        W,G,H,...
                                        C_u_loc,d_u_loc, ...
                                        C_u_sh, D_u_sh, d_u_sh, ...
                                        C_x,D_x,d_x, ...
                                        C_mixed, D_mixed, d_mixed, ...
                                        x_0, N, n_u, T_hor)    

    % given an n*m*p array createa a np * m * p array, where each page of
    % the new array is the column stack of all pages of the original array
    rep = @(u) repmat(reshape(u, [n_u * N * T_hor,1] ), 1,1,N); 
    for i=1:N
        h(i) = .5*x_0'*H(:,:,i)*x_0;
    end
    g = pagemtimes(G, x_0);
    J = @(u) .5 * pagemtimes(T3D(rep(u)), pagemtimes(W,rep(u))) + ...
        pagemtimes(T3D(rep(u)), g) + h;
    % For each i, take the rows associated to agent i in W(:,:,i)
    sel_mat = zeros(n_u*T_hor, n_u * N * T_hor, N);  
    for i=1:N
        sel_mat(:, (i-1)*n_u*T_hor+1:i*n_u*T_hor,i) = eye(n_u*T_hor);
    end
    Q = pagemtimes(sel_mat, W);
    F = @(u) pagemtimes(Q, rep(u)) + pagemtimes(sel_mat, g);
    
    n_sh_const_u = size(C_u_sh,1);
    n_const_x = size(C_x,1);
    n_const_mix = size(C_mixed,1);
    A_sh = zeros(n_sh_const_u + n_const_x + n_const_mix, n_u*T_hor,N);
    b_sh = zeros(n_sh_const_u + n_const_x + n_const_mix, 1,N);
    d_u_x0 = pagemtimes(D_u_sh, x_0);
    d_x0 = pagemtimes(D_x, x_0);
    d_mixed_x0 = pagemtimes(D_mixed, x_0);
    A_loc = C_u_loc;
    b_loc = d_u_loc;
    for i=1:N
        A_sh(:,:,i) = [C_u_sh(:,:,i); 
                       C_x(:,:,i); 
                       C_mixed(:,:,i)];
        b_sh(:,:,i) = [d_u_sh(:,:,i)+d_u_x0(:,:,i); 
                       d_x0(:,:,i)  + d_x(:,:,i); 
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
    if exit_flag~=1
        error("[surrogate cl-NE] The VI is infeasible")
    end

end