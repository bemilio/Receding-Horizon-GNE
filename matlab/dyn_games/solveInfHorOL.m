function [P,K, isStable] = solveInfHorOL(game, n_iter, eps_err)
%SOLVEINFHORCL 

%% Check if basic assumptions are satisfied
if min(abs(eig(game.A)))<eps_err
    warning("[solveInfHorOL] The matrix A appears singular")
end
for i=1:game.N
    if ~is_stabilizable(game.A, game.B(:,:,i))
        warning("[solveInfHorOL] The system is not stabilizable for agent " + num2str(i))
    end
    if ~is_detectable(game.A, game.Q(:,:,i))
        warning("[solveInfHorOL] The system is not detectable for agent " + num2str(i))
    end
end

P = zeros(game.n_x, game.n_x, game.N);
K = zeros(game.n_u, game.n_x, game.N);
A = game.A;
B = game.B;
Q = game.Q;
R = game.R;
n_x = game.n_x;
n_u = game.n_u;
N = game.N;

Rinv = zeros(size(R));
for i=1:N
    Rinv(:,:,i) = eye(n_u)/R(:,:,i);
end

A_T_inv = eye(n_x)/(A');


% Stable initialization: cooperative optimal controller
B_coop = reshape(B, n_x, n_u * N ); % stack B horizontally
[~, K_init] = idare(A, B_coop, eye(n_x), eye(N*n_u));
for i=1:N
    K(:,:,i) = - K_init(1+(i-1)*n_u:i*n_u, :);
end
all_good = true;
for k=1:n_iter 
    A_cl = A + sum(pagemtimes(B, K), 3);
    for i=1:N
        try
            P(:,:,i) = sylvester(A_T_inv, -A_cl, A_T_inv * Q(:,:,i));  % solves -A.T @ X A_cl + X - Q[i] = 0
        catch e
            disp("[solveInfHorOL] An error occurred while solving the Sylvester equation: " + e.message)
            all_good = false;
        end
    end
    M = eye(n_x)/(eye(n_x) + sum( pagemtimes(pagemtimes(pagemtimes(B, Rinv), T3D(B)), P), 3));
    K = - pagemtimes(Rinv, pagemtimes(pagemtimes(pagemtimes(T3D(B), P), M), A) );
    if mod(n_iter,10)==0
        % Test solution: Check if (9) [Freiling-Jank-Kandil '99] is satisfied
        err = 0;
        M = eye(n_x)/(eye(n_x) + sum( pagemtimes(pagemtimes(pagemtimes(B, Rinv), T3D(B)), P), 3));
        for i = 1:N
            err = err + norm(Q(:,:,i) - P(:,:,i) + A' * P(:,:,i) * M * A);
        end
        if  err < eps_err
            break
        end
    end
    if ~all_good
        break
    end
end
%TODO:complete
if err > eps_err
    disp("[solve_open_loop_inf_hor_problem] Could not find solution")
end
for i=1:N
    if min(eig(P(:,:,i))) < 0
        warning("The open loop P is non-positive definite")
    end
end
game.A_ol = A + sum(pagemtimes(B, K), 3);

if max(abs(eig(A + sum(pagemtimes(B, K), 3)))) > 1.001
    warning("The infinite horizon OL-GNE has an unstable dynamics")
    isStable = false;
else
    isStable=true;
end

end