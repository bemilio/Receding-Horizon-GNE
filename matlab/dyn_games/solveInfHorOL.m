function [P,K, isStable] = solveInfHorOL(game, n_iter, eps_err)
%SOLVEINFHORCL 

P = zeros(game.n_x, game.n_x, game.N);
K = zeros(game.n_u, game.n_x, game.N);
A = game.A;
B = game.B;
Q = game.Q;
R = game.R;
invertMat = @(mat) inv(mat); % for batched inversion
Rinv = arrayfun(invertMat, R, 'UniformOutput', false);
bmm = @(A,B) pagemtimes(A, B); %shorter name for batched matrix mult
n_x = game.n_x;
n_u = game.n_u;
N = game.N;

% Stable initialization: cooperative optimal controller
B_coop = reshape(B, n_x, n_u * N ); % stack B horizontally
P_init, K_init = idare(A, B_coop, eye(n_x), eye(N*n_u));
for i=1:N
    K(:,:,i) = - K_init(1+(i-1)*n_u:i*n_u, :);
end

for k=1:n_iter 
    A_cl = A + np.sum(pagemtimes(B, K), 3);
    for i=1:N
        try
            P(:,:,i) = sylvester(inv(A'), A_cl, inv(A') * Q(:,:,i));  % solves -A.T @ X A_cl + X - Q[i] = 0
        catch 
            disp("[solveInfHorOL] An error occurred while solving the Sylvester equation")
        end
    end
    M = inv(eye(n_x) + sum( bmm(bmm(bmm(B, Rinv), T3D(B)), P), 3));
    K = - bmm(Rinv, bmm(bmm(bmm(T3D(B), P), M), A) );
    if mod(n_iter,10)==0
        % Test solution: Check if (9) [Freiling-Jank-Kandil '99] is satisfied
        err = 0;
        M = inv(eye(n_x) + sum( bmm(bmm(bmm(B, Rinv), T3D(B)), P), 3));
        for i = 1:N
            err = err + norm(Q(:,:,i) - P(:,:,i) + A.T * P(:,:,i) * M * A);
        end
        if  err < eps_error
            break
        end
    end
end
%TODO:complete
if err > eps_error
    print("[solve_open_loop_inf_hor_problem] Could not find solution")
end
for i=1:N
    if min(eig(P(:,:,i))) < 0
        warnings.warn("The open loop P is non-positive definite")
    end
end
if max(abs(eig(A + np.sum(pagemtimes(B, K), 3)))) > 1.001
    warnings.warn("The infinite horizon OL-GNE has an unstable dynamics")
end

end