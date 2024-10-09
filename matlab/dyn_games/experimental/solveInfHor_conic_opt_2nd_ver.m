function [X, K, isStable, A_cl] = solveInfHor_conic_opt_2nd_ver(game, n_iter, eps_err, K_warm)

A = game.A;
B = game.B;
Q = game.Q;
R = game.R;
n_x = game.n_x;
n_u = game.n_u;
N = game.N;
B_all = [];
Q_all = zeros(n_x, n_x);
Sigma = eye(n_x);

for i=1:N
    B_all = [B_all, B(:,:,i)];
    Q_all = Q_all + Q(:,:,i);
end
if ~is_stabilizable(A, B_all)
    warning("[solveInfHorCL]: stabilizability assump. not satisfied")
end
if ~is_detectable(A, Q_all)
    warning("[solveInfHorCL]: detectability assump. not satisfied")
end

Gph = digraph(triu(ones(N),1)); % Complete graph with directed edges
G = Gph.incidence;
N_edges = numedges(Gph);

F = generate_mapping(Q,R,A,B);

stepsize = .01;

Ly_A = @(X) A * X + X * A';
Ly_A_inv = @(Y) lyap(A, -Y); % solves Ly_op(A,X) = Y
Ly_B = @(L) pagemtimes(B, L) + T3D(pagemtimes(B,L)); 

L = zeros(n_u,n_x,N);
K = zeros(n_u,n_x,N);
Z = zeros(n_u,n_u,N);

if exist('K_warm', 'var')
    X = lyap( (A + sum(pagemtimes(B, K_warm), 3) ), Sigma);
    for i=1:N
        L(:,:,i) = K_warm(:,:,i) * X;
        Z(:,:,i) = K_warm(:,:,i) * X * K_warm(:,:,i)';
        K(:,:,i) = K_warm(:,:,i);
    end
    % Sanity check
    err = 0;
    for i=1:N
        A_cl_i = A + sum(pagemtimes(B, K_warm), 3) - B(:,:,i) * K_warm(:,:,i);
        [~, K_br_i, ~] = icare(A_cl_i, B(:,:,i), Q(:,:,i), R(:,:,i));
        K_br_i = -K_br_i;
        err = max(err, norm(K_warm(:,:,i) - K_br_i));
    end
    constr_sat = A*X + X * A' + sum(pagemtimes(B,L),3) + sum(pagemtimes(B,L),3)' + Sigma;
    disp("Warm start error est.=  " + num2str(err) + "; Lyap. eq. error= " + num2str(norm(constr_sat)) )
end

for k=1:n_iter
    [L_grad, Z_grad] = F(L,Z);
    L_half = L - stepsize*L_grad;
    Z_half = Z - stepsize*Z_grad;
    [L_half,Z_half] = BackwardStep(L_half,Z_half, A, B, Sigma);

    [L_grad_half, Z_grad_half]  = F(L_half,Z_half);

    L_new = L_half + stepsize * (L_grad - L_grad_half);
    Z_new = Z_half + stepsize * (Z_grad - Z_grad_half);

    L = L_new;
    Z = Z_new;
    

    X = - Ly_A_inv(sum(Ly_B(L), 3) + Sigma );
    for i=1:N
        K(:,:,i) = L(:,:,i)/X;
    end

    if mod(k,30)==0
        % Test solution by checking distance from best responsed            
        err = 0;
        for i=1:N
            A_cl_i = A + sum(pagemtimes(B, K), 3) - B(:,:,i) * K(:,:,i);
            [~, K_br_i, ~] = icare(A_cl_i, B(:,:,i), Q(:,:,i), R(:,:,i));
            K_br_i = -K_br_i;
            err = max(err, norm(K(:,:,i) - K_br_i));
        end
        disp("iteration " + num2str(k) + " of " + num2str(n_iter))
        disp("Error est.=  " + num2str(err))
        constr_sat = A*X + X * A' + sum(pagemtimes(B,L),3) + sum(pagemtimes(B,L),3)' + Sigma 
    end
end


if  err > eps_err
    warning("[solveInfHorCL] Could not find infinite horizon CL-NE")
end
A_cl = A + sum(pagemtimes(B, K), 3);
if max(real(eig(A + sum(pagemtimes(B, K), 3)))) > 0.0001
    warning("The infinite horizon CL-GNE has an unstable dynamics")
    isStable = false;
else
    isStable=true;
end

end

function F = generate_mapping(Q,R,A,B)
    n_x = size(A, 1);
    n_u = size(B,2);
    N_agents = size(B,3);
    Pi = com_mat(n_u, n_x); 
    A_hat = kron(eye(n_x), A) + kron(A, eye(n_x));
    B_hat = zeros(n_x * n_x, n_x * n_u, N_agents);
    for j = 1:N_agents
        B_hat(:,:,j) = kron(eye(n_x), B(:,:,j)) + kron(B(:,:,j), eye(n_x)) * Pi;
    end
    F = @operator;

    function [L_p, Z_p] = operator(L,Z)
        % given an n*m*p array createa a np * m * p array, where each page of
        % the new array is the column stack of all pages of the original array
        % v_stack_pages = @(A) reshape(permute(A, [1, 3, 2]), [], size(A, 2));

        L_p = zeros(size(L));
        for i=1:N_agents
            L_p(:,:,i) = -vec_inv( (A_hat\B_hat(:,:,i))' * vec(Q(:,:,i)) , n_u, n_x );
        end

        Z_p = zeros(size(Z));
        for i=1:N_agents
            Z_p(:,:,i) = R(:,:,i);
        end

    end
end

function [L_p, Z_p] = BackwardStep(L,Z, A, B, Sigma) 
    % Projection into the local feasibility set.
    N = size(L,3);
    n_x = size(L, 2);
    n_u = size(L,1);
    L_p = zeros(size(L));
    Z_p = zeros(size(Z));

    L_sdp = sdpvar(n_u, n_x, N);
    Z_sdp = sdpvar(n_u, n_u, N); % every slice in the first two dim. is symmetric!
    X = sdpvar(n_x, n_x); % Symmetric

    ops = sdpsettings('verbose',0);
    J = 0;
    % pagemtimes does not support sdpvar, thus I use a loop to implement
    % the lyapunov operator of A and B_i.
    Ly_A = A * X + X * A';
    Ly_B = zeros(n_x, n_x);
    for i=1:N
        Ly_B = Ly_B + B(:,:,i)*L_sdp(:,:,i) + (B(:,:,i)*L_sdp(:,:,i))';
    end

    constr = [X>=eps*eye(size(X,1));
              Ly_A + Ly_B + Sigma==0];
    for i=1:N
        J = J + trace((L_sdp(:,:,i) - L(:,:,i)) * (L_sdp(:,:,i) - L(:,:,i))') ...
              + trace((Z_sdp(:,:,i) - Z(:,:,i)) * (Z_sdp(:,:,i) - Z(:,:,i))');
        M = [Z_sdp(:,:,i),  L_sdp(:,:,i); 
             L_sdp(:,:,i)', X];
        constr = [constr; M>=0];
    end
    optimize(constr, J, ops);

    L_p = value(L_sdp);
    Z_p = value(Z_sdp);

end


function P = com_mat(m, n) 
    %% P * vec(M) = vec(M')
    % determine permutation applied by K
    A = reshape(1:m*n, m, n);
    v = reshape(A', 1, []);
    
    % apply this permutation to the rows (i.e. to each column) of identity matrix
    P = eye(m*n);
    P = P(v,:);
end

