function [P,K, isStable] = solveInfHorCL(game, n_iter, eps_err)
%SOLVEINFHORCL 

P = zeros(game.n_x, game.n_x, game.N);
K = zeros(game.n_u, game.n_x, game.N);
A = game.A;
B = game.B;
Q = game.Q;
R = game.R;
n_x = game.n_x;
n_u = game.n_u;
N = game.N;
% (adapted from sassano 22): 
% solve P[i] = Q[i] + K[i]'R[i]K[i] + A_cl' P[i] A_cl 
% b iterating the Riccati equation
% A_cl' P[i] A_cl + Q[i] + K[i]'R[i]K[i] = P[i];
% K[i] = - R[i]B[i]'P[i]A_cl 
for k=1:n_iter
    for i=1:N
        A_cl_not_i = A + sum(pagemtimes(B, K), 3) - B(:,:, i)*K(:,:,i);
        P(:,:,i) = idare(A_cl_not_i, B(:,:,i), Q(:,:,i), R(:,:,i));
        K(:,:,i) = - inv(R(:,:,i) + B(:,:,i)' * P(:,:,i) * B(:,:,i)) * ...
            B(:,:,i)' * P(:,:,i) * A_cl_not_i;
    end
    if mod(n_iter,20)==0
        % Test solution by verifying (6.28) in Basar book
        err = 0;
        A_cl = A + sum(pagemtimes(B, K), 3);
        for i=1:N
            err = max(err, norm(P(:,:,i) - ...
                (Q(:,:,i) + K(:,:,i)' * R(:,:,i) * K(:,:,i)...
                + A_cl' * P(:,:,i) * A_cl)));
            err = max(err, norm(K(:,:,i) +  ...
                (R(:,:,i) + B(:,:,i)' * P(:,:,i) * B(:,:,i))\ ...
                (B(:,:,i)' * (P(:,:,i) * (A_cl - B(:,:,i)*K(:,:,i)))) ) );
        end
        if  err < eps_err
            break
        end
    end
end
if  err > eps_err
    warning("[solveInfHorCL] Could not find infinite horizon CL-NE")
end
for i=1:N
    if min(eig(P(:,:,i))) < -eps_err
        warning("The closed-loop P is non-positive definite")
    end
end
if max(abs(eig(A + sum(pagemtimes(B, K), 3)))) > 1.0001
    warning("The infinite horizon CL-GNE has an unstable dynamics")
    isStable = false;
else
    isStable=true;
end

end