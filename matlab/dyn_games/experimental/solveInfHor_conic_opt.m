function [X, K, isStable, A_cl] = solveInfHor_conic_opt(game, n_iter, eps_err)
K = zeros(game.n_u, game.n_x, game.N);
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

F = generate_mapping(Q,R,A,B,Sigma, G);

stepsize = .01;
L = zeros(n_u,n_x,N);
X = zeros(n_x,n_x,N);
for i=1:N
    X(:,:,i) = eye(n_x);
end
Z = zeros(n_u,n_u,N);
dual = zeros(n_x, n_x,N);
aux = zeros(N_edges*n_x,n_x);
for k=1:n_iter
    [L_grad, X_grad, Z_grad, dual_grad, aux_grad] = F(L,X,Z,dual,0*aux);
    L_half = L - stepsize*L_grad;
    X_half = X - stepsize*X_grad;
    Z_half = Z - stepsize*Z_grad;
    [L_half,X_half,Z_half] = BackwardStep(L_half,X_half,Z_half);
    dual_half = dual + stepsize * dual_grad;
    aux_half = aux + stepsize*aux_grad;

    [L_grad_half, X_grad_half, Z_grad_half, dual_grad_half, aux_grad_half]  = ...
        F(L_half,X_half,Z_half,dual_half,0*aux_half);

    L_new = L_half + stepsize * (L_grad - L_grad_half);
    X_new = X_half + stepsize * (X_grad - X_grad_half);
    Z_new = Z_half + stepsize * (Z_grad - Z_grad_half);
    dual_new = dual_half + stepsize * (dual_grad_half - dual_grad);
    aux_new = aux_half + stepsize * (aux_grad_half - aux_half);

    L = L_new;
    X = X_new;
    Z = Z_new;
    dual = dual_new;
    aux = aux_new;

    if mod(k,30)==0
        % Test solution by checking distance from best response
        for i=1:N
            K(:,:,i) = L(:,:,i)/X(:,:,i);
        end            
        err = 0;
        for i=1:N
            A_cl_i = A + sum(pagemtimes(B, K), 3) - B(:,:,i) * K(:,:,i);
            [~, K_br_i, ~] = icare(A_cl_i, B(:,:,i), Q(:,:,i), R(:,:,i));
            K_br_i = -K_br_i;
            err = max(err, norm(K(:,:,i) - K_br_i));
        end
        disp("iteration " + num2str(k) + " of " + num2str(n_iter))
        disp("Error est.=  " + num2str(err))
        constr_sat = A*X(:,:,1) + X(:,:,1) * A' + sum(pagemtimes(B,L),3) + sum(pagemtimes(B,L),3)' + Sigma 
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



function F = generate_mapping(Q,R,A,B,Sigma, G)

    F = @operator;

    function [L_p, X_p, Z_p, dual_p, aux_p] = operator(L,X,Z, dual, aux)

        % given an n*m*p array createa a np * m * p array, where each page of
        % the new array is the column stack of all pages of the original array
        v_stack_pages = @(A) reshape(permute(A, [1, 3, 2]), [], size(A, 2));

        X_inv = arrayfun(@(i) inv(X(:,:,i)), 1:size(X,3), 'UniformOutput', false);
        X_inv = cat(3, X_inv{:});

        N_agents = size(L,3);
        X_p = zeros(size(X));
        for i=1:N_agents
          A_cl_not_i = A + sum(pagemtimes(pagemtimes(B,L), X_inv), 3) - B(:,:,i) * L(:,:,i) * X_inv(:,:,i);
          X_p(:,:,i) = Q(:,:,i) + ...
              (A_cl_not_i' * dual(:,:,i)) + (dual(:,:,i) * A_cl_not_i) + ...
              kron(G(i,:), eye(size(X,1)))*aux ; % MISSING DERIVATIVE OF CONSTRAINT OF AGENT j WRT INVERSE OF Xi!
          % X_p(:,:,i) = Q(:,:,i) + ...
          %     (A' * dual(:,:,i)) + (dual(:,:,i) * A) + ...
          %     kron(G(i,:), eye(size(X,1)))*aux ;
        end
        
        L_p = zeros(size(L));
        for i=1:N_agents
            % for j=1:N_agents
                % L_p(:,:,i) = L_p(:,:,i) + B(:,:,i)'*(dual(:,:,j) + dual(:,:,j)');
            % end
            L_p(:,:,i) = B(:,:,i)'*(dual(:,:,i) + dual(:,:,i)');
        end

        Z_p = zeros(size(Z));
        for i=1:N_agents
            Z_p(:,:,i) = R(:,:,i);
        end
    
        dual_p = zeros(size(dual));
        for i=1:N_agents
            A_cl_not_i = A + sum(pagemtimes(pagemtimes(B,L), X_inv), 3) - B(:,:,i) * L(:,:,i) * X_inv(:,:,i);
            dual_p(:,:,i) = A_cl_not_i*X(:,:,i) + X(:,:,i) * A_cl_not_i' + ...
                            B(:,:,i)*L(:,:,i) + (B(:,:,i)*L(:,:,i))'...
                            + Sigma ;
            % dual_p(:,:,i) = A*X(:,:,i) + X(:,:,i) * A' + ...
                            % sum(pagemtimes(B,L),3) + sum(pagemtimes(B,L),3)'...
                            % + Sigma ;
        end
        %aux_p = aux;
        aux_p = kron(G', eye(size(X,1))) * v_stack_pages(X);
    end
end

function [L_p, X_p, Z_p] = BackwardStep(L,X,Z) 
    % Projection into the local feasibility set.
    N = size(L,3);
    L_p = zeros(size(L));
    X_p = zeros(size(X));
    Z_p = zeros(size(Z));
    ops = sdpsettings('verbose',0);
    for i=1:N
        L_i = L(:,:,i);
        X_i = X(:,:,i);
        Z_i = Z(:,:,i);
        L_sdp = sdpvar(size(L_i,1), size(L_i,2)); % rectangular
        X_sdp = sdpvar(size(X_i,1), size(X_i,2)); % Symmetric by default
        Z_sdp = sdpvar(size(Z_i,1), size(Z_i,2)); % Symmetric by default
        M = [Z_sdp , L_sdp; 
             L_sdp', X_sdp];
        constr = [X_sdp>=eps*eye(size(X_sdp,1)), M>=0];
        optimize(constr,  trace((X_sdp - X_i) * (X_sdp - X_i)') +...
                          trace((L_sdp - L_i) * (L_sdp - L_i)') +...
                          trace((Z_sdp - Z_i) * (Z_sdp - Z_i)'), ops);

        L_p(:,:,i) = value(L_sdp);
        X_p(:,:,i) = value(X_sdp);
        Z_p(:,:,i) = value(Z_sdp);
    end

end

